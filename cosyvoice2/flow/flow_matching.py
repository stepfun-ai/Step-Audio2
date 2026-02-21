# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List
import onnxruntime
import torch
import torch.nn.functional as F

from cosyvoice2.flow.decoder_dit import DiT
from cosyvoice2.utils.mask import make_pad_mask


def get_data_ptr(tensor: torch.Tensor, dummy_buffer: torch.Tensor):
    if tensor.numel() == 0:
        return dummy_buffer.data_ptr()
    else:
        return tensor.contiguous().data_ptr()


"""
Inference wrapper
"""
class CausalConditionalCFM(torch.nn.Module):
    def __init__(self, estimator: DiT, inference_cfg_rate:float=0.7):
        super().__init__()
        self.estimator = estimator
        self.inference_cfg_rate = inference_cfg_rate
        self.out_channels = estimator.out_channels
         # a maximum of 600s
        self.register_buffer('rand_noise', torch.randn([1, self.out_channels, 50 * 600]), persistent=False)

        self.register_buffer('dummy_buffer', torch.zeros(1), persistent=False)

    def scatter_cuda_graph(self, enable_cuda_graph: bool):
        if enable_cuda_graph:
            self.estimator._init_cuda_graph_all()

    def forward_estimator(self, x, mask, mu, t, spks, cond):
        if isinstance(self.estimator, torch.nn.Module):
            return self.estimator(x, mask, mu, t, spks, cond)
        else:
            [estimator, stream], trt_engine = self.estimator.acquire_estimator()
            # NOTE need to synchronize when switching stream
            torch.cuda.current_stream().synchronize()
            batch_size = x.size(0)
            with stream:
                estimator.set_input_shape('x', (batch_size, 80, x.size(2)))
                estimator.set_input_shape('mask', (batch_size, 1, x.size(2)))
                estimator.set_input_shape('mu', (batch_size, 80, x.size(2)))
                estimator.set_input_shape('t', (batch_size,))
                estimator.set_input_shape('spks', (batch_size, 80))
                estimator.set_input_shape('cond', (batch_size, 80, x.size(2)))
                data_ptrs = [x.contiguous().data_ptr(),
                             mask.contiguous().data_ptr(),
                             mu.contiguous().data_ptr(),
                             t.contiguous().data_ptr(),
                             spks.contiguous().data_ptr(),
                             cond.contiguous().data_ptr(),
                             x.data_ptr()]
                for i, j in enumerate(data_ptrs):
                    estimator.set_tensor_address(trt_engine.get_tensor_name(i), j)
                # run trt engine
                assert estimator.execute_async_v3(torch.cuda.current_stream().cuda_stream) is True
                torch.cuda.current_stream().synchronize()
            self.estimator.release_estimator(estimator, stream)
            return x

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t_in = torch.zeros([x.shape[0] * 2], device=x.device, dtype=x.dtype)

        assert self.inference_cfg_rate > 0, 'inference_cfg_rate better > 0'

        # constant during denoising
        mask_in = torch.cat([mask, mask], dim=0)
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        
        for step in range(1, len(t_span)):
            x_in = torch.cat([x, x], dim=0)
            t_in.fill_(t)

            dphi_dt = self.forward_estimator(
                x_in, mask_in,
                mu_in, t_in,
                spks_in,
                cond_in,
            )

            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return x

    @torch.inference_mode()
    def forward(self, mu, mask, spks, cond, n_timesteps=10, temperature=1.0):
        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        # cosine scheduling
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span, mu, mask, spks, cond)


    def forward_estimator_chunk(self, x, mu, t, spks, cond, cnn_cache, att_cache):
        if isinstance(self.estimator, torch.nn.Module):
            dphi_dt, this_new_cnn_cache, this_new_att_cache = self.estimator.forward_chunk(
                x = x,
                mu = mu,
                t = t,
                spks = spks,
                cond = cond,
                cnn_cache = cnn_cache,
                att_cache = att_cache,
            )
            return dphi_dt, this_new_cnn_cache, this_new_att_cache
        else:
            [estimator, stream], trt_engine = self.estimator.acquire_estimator()
            # NOTE need to synchronize when switching stream
            torch.cuda.current_stream().synchronize()
            batch_size = x.size(0)
            with stream:
                estimator.set_input_shape('x', (batch_size, 80, x.size(2)))
                estimator.set_input_shape('mu', (batch_size, 80, x.size(2)))
                estimator.set_input_shape('t', (batch_size,))
                estimator.set_input_shape('spks', (batch_size, 80))
                estimator.set_input_shape('cond', (batch_size, 80, x.size(2)))
                estimator.set_input_shape('cnn_cache', cnn_cache.shape)
                estimator.set_input_shape('att_cache', att_cache.shape)
                new_cnn_cache = torch.empty_like(cnn_cache)
                new_att_cache_shape = list(att_cache.shape)
                new_att_cache_shape[3] += x.size(2)
                new_att_cache = torch.empty(new_att_cache_shape, device=att_cache.device, dtype=x.dtype)
                data_ptrs = [x.contiguous().data_ptr(),
                             mu.contiguous().data_ptr(),
                             t.contiguous().data_ptr(),
                             spks.contiguous().data_ptr(),
                             cond.contiguous().data_ptr(),
                             cnn_cache.contiguous().data_ptr(),
                             get_data_ptr(att_cache, self.dummy_buffer),
                             x.data_ptr(),
                             new_cnn_cache.data_ptr(),
                             get_data_ptr(new_att_cache, self.dummy_buffer)]

                for i, j in enumerate(data_ptrs):
                    estimator.set_tensor_address(trt_engine.get_tensor_name(i), j)
                # run trt engine
                assert estimator.execute_async_v3(torch.cuda.current_stream().cuda_stream) is True
                torch.cuda.current_stream().synchronize()
            self.estimator.release_estimator(estimator, stream)

            return x, new_cnn_cache, new_att_cache


    def solve_euler_chunk(self, 
                          x:torch.Tensor, 
                          t_span:torch.Tensor, 
                          mu:torch.Tensor, 
                          spks:torch.Tensor, 
                          cond:torch.Tensor, 
                          cnn_cache:torch.Tensor=None,
                          att_cache:torch.Tensor=None,
                          ):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
            cnn_cache: shape (n_time, depth, b, c1+c2, 2)
            att_cache: shape (n_time, depth, b, nh, t, c * 2)
        """
        assert self.inference_cfg_rate > 0, 'cfg rate should be > 0'
        
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t_in = torch.zeros([x.shape[0] * 2], device=x.device, dtype=x.dtype)

        # setup initial cache
        if cnn_cache is None:
            cnn_cache = [None for _ in range(len(t_span)-1)]
            cnn_cache = torch.zeros((len(t_span)-1, 16, x.shape[0] * 2, 1024, 2), device=x.device, dtype=x.dtype)
        if att_cache is None:
            att_cache = [None for _ in range(len(t_span)-1)]
            att_cache = torch.empty((len(t_span)-1, 16, x.shape[0] * 2, 8, 0, 128), device=x.device, dtype=x.dtype)
        # next chunk's cache at each timestep
        new_cnn_caches = []
        new_att_caches = []

        if att_cache[0] is not None:
            last_att_len = att_cache.shape[4]
        else:
            last_att_len = 0

        # constant during denoising
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        for step in range(1, len(t_span)):
            # torch.cuda.memory._record_memory_history(max_entries=100000)
            # torch.cuda.memory._record_memory_history(max_entries=100000)
            this_att_cache = att_cache[step-1]
            this_cnn_cache = cnn_cache[step-1]

            x_in = x.repeat(2, 1, 1)
            t_in.fill_(t)

            dphi_dt, this_new_cnn_cache, this_new_att_cache = self.forward_estimator_chunk(
                x = x_in,
                mu = mu_in,
                t = t_in,
                spks = spks_in,
                cond = cond_in,
                cnn_cache = this_cnn_cache,
                att_cache = this_att_cache,
            )

            dphi_dt, cfg_dphi_dt = dphi_dt.chunk(2, dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

            new_cnn_caches.append(this_new_cnn_cache)
            new_att_caches.append(this_new_att_cache)
        
        cnn_cache = torch.stack(new_cnn_caches)
        att_cache = torch.stack(new_att_caches)
        return x, cnn_cache, att_cache
    
    @torch.inference_mode()
    def forward_chunk(self, 
                      mu:torch.Tensor, 
                      spks:torch.Tensor, 
                      cond:torch.Tensor,
                      n_timesteps:int=10, 
                      temperature:float=1.0, 
                      cnn_cache:torch.Tensor=None,
                      att_cache:torch.Tensor=None,
                      ):
        """
        Args:
            mu(torch.Tensor): shape (b, c, t)
            spks(torch.Tensor): shape (b, 192)
            cond(torch.Tensor): shape (b, c, t)
            cnn_cache: shape (n_time, depth, b, c1+c2, 2)
            att_cache: shape (n_time, depth, b, nh, t, c * 2)
        """
        # get offset from att_cache
        offset = att_cache.shape[4] if att_cache is not None else 0
        z = self.rand_noise[:, :, offset:offset+mu.size(2)] * temperature
        z = z.to(mu.dtype)
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        # cosine scheduling
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        x, new_cnn_cache, new_att_cache = self.solve_euler_chunk(
            x=z,
            t_span=t_span,
            mu=mu,
            spks=spks,
            cond=cond,
            att_cache=att_cache,
            cnn_cache=cnn_cache,
        )
        return x, new_cnn_cache, new_att_cache
