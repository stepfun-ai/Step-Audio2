# Copyright (c) 2024 Antgroup Inc (authors: Zhoubofan, hexisyztem@icloud.com)
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script is used to export the streaming token2wav model to onnx.
python3 tools/export_onnx_streaming_token2wav.py
"""

from __future__ import print_function

import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import os
import sys
import onnxruntime
import random
import torch
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml

import sys
import os
# add ../ to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_args():
    parser = argparse.ArgumentParser(description='export your model for deployment')
    parser.add_argument('--model_dir',
                        type=str,
                        default='Step-Audio-2-mini/token2wav',
                        help='local path')
    parser.add_argument('--onnx_model',
                        type=str,
                        default='flow.decoder.estimator.chunk.fp32.static_batch.onnx',
                        help='onnx model name')
    args = parser.parse_args()
    print(args)
    return args

def get_dummy_input_chunk(batch_size, seq_len, prev_seq_len, out_channels, estimator, device):
    x = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    mu = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    t = torch.rand((batch_size), dtype=torch.float32, device=device)
    spks = torch.rand((batch_size, out_channels), dtype=torch.float32, device=device)
    cond = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)

    depth = len(estimator.blocks)
    num_heads = estimator.blocks[0].attn.num_heads
    head_dim = estimator.blocks[0].attn.head_dim
    cnn_channels = estimator.blocks[0].conv.in_channels + estimator.blocks[0].conv.out_channels

    cnn_cache = torch.rand((depth, batch_size, cnn_channels, 2), dtype=torch.float32, device=device)
    att_cache = torch.rand((depth, batch_size, num_heads, prev_seq_len, head_dim * 2), dtype=torch.float32, device=device)
    return x, mu, t, spks, cond, cnn_cache, att_cache

class DiTChunkWrapper(torch.nn.Module):
    def __init__(self, dit_model):
        super().__init__()
        self.dit_model = dit_model

    def forward(self, x, mu, t, spks, cond, cnn_cache, att_cache):
        return self.dit_model.forward_chunk(x, mu, t, spks, cond, cnn_cache, att_cache)


@torch.no_grad()
def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    with open(f"{args.model_dir}/flow.yaml", "r") as f:
        configs = load_hyperpyyaml(f)
        flow_model = configs['flow']

    device = torch.device('cuda')


    # 1. export flow decoder estimator for chunk processing
    flow_model.load_state_dict(torch.load(f"{args.model_dir}/flow.pt", map_location="cpu", weights_only=True), strict=True)
    estimator = flow_model.decoder.estimator
    estimator.eval()
    estimator.to(device)

    estimator_chunk_wrapper = DiTChunkWrapper(estimator)

    batch_size, seq_len, prev_seq_len = 2, 500, 100
    out_channels = flow_model.decoder.estimator.out_channels
    dummy_inputs = get_dummy_input_chunk(batch_size, seq_len, prev_seq_len, out_channels, estimator, device)
    (x, mu, t, spks, cond, cnn_cache, att_cache) = dummy_inputs

    torch.onnx.export(
        estimator_chunk_wrapper,
        dummy_inputs,
        f'{args.model_dir}/{args.onnx_model}',
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['x', 'mu', 't', 'spks', 'cond', 'cnn_cache', 'att_cache'],
        output_names=['output', 'new_cnn_cache', 'new_att_cache'],
        dynamic_axes={
            'x': {0: 'batch_size', 2: 'seq_len'},
            'mu': {0: 'batch_size', 2: 'seq_len'},
            'cond': {0: 'batch_size', 2: 'seq_len'},
            't': {0: 'batch_size'},
            'spks': {0: 'batch_size'},
            'cnn_cache': {1: 'batch_size'},
            'att_cache': {1: 'batch_size', 3: 'prev_seq_len'},
            'output': {0: 'batch_size', 2: 'seq_len'},
            'new_cnn_cache': {1: 'batch_size'},
            'new_att_cache': {1: 'batch_size', 3: 'total_seq_len'},
        }
    )

    # 2. test computation consistency
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider']
    estimator_onnx = onnxruntime.InferenceSession(f'{args.model_dir}/{args.onnx_model}',
                                                  sess_options=option, providers=providers)

    for _ in tqdm(range(10)):
        seq_len = random.randint(16, 512)
        prev_seq_len = random.randint(16, 1024)
        dummy_inputs = get_dummy_input_chunk(batch_size, seq_len, prev_seq_len, out_channels, estimator, device)
        (x, mu, t, spks, cond, cnn_cache, att_cache) = dummy_inputs

        output_pytorch, new_cnn_cache_pytorch, new_att_cache_pytorch = estimator_chunk_wrapper(*dummy_inputs)

        ort_inputs = {
            'x': x.cpu().numpy(),
            'mu': mu.cpu().numpy(),
            't': t.cpu().numpy(),
            'spks': spks.cpu().numpy(),
            'cond': cond.cpu().numpy(),
            'cnn_cache': cnn_cache.cpu().numpy(),
            'att_cache': att_cache.cpu().numpy(),
        }
        output_onnx, new_cnn_cache_onnx, new_att_cache_onnx = estimator_onnx.run(None, ort_inputs)

        torch.testing.assert_allclose(output_pytorch, torch.from_numpy(output_onnx).to(device), rtol=1e-2, atol=1e-4)
        torch.testing.assert_allclose(new_cnn_cache_pytorch, torch.from_numpy(new_cnn_cache_onnx).to(device), rtol=1e-2, atol=1e-4)
        torch.testing.assert_allclose(new_att_cache_pytorch, torch.from_numpy(new_att_cache_onnx).to(device), rtol=1e-2, atol=1e-4)

    logging.info('successfully export chunk-wise estimator')


if __name__ == "__main__":
    main()
