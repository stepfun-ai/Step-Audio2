# Copyright (c) 2025 Tsinghua Univ. (authors: Xingchen Song)
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
import time
from datetime import datetime

import s3tokenizer
import torch
from tqdm import tqdm

from flashcosyvoice.config import Config, SamplingParams
from flashcosyvoice.engine.llm_engine import LLMEngine
from flashcosyvoice.modules.flow import CausalMaskedDiffWithXvec
from flashcosyvoice.modules.hifigan import HiFTGenerator
from device_utils import get_device


class CosyVoice2(torch.nn.Module):
    def __init__(self, config: Config = None):
        super().__init__()
        self.config = Config() if config is None else config

        device = get_device()
        self.audio_tokenizer = s3tokenizer.load_model("speech_tokenizer_v2_25hz").to(device).eval()

        self.llm = LLMEngine(**self.config.__dict__)

        self.use_tqdm = torch.distributed.get_node_local_rank() == 0

        self.flow = CausalMaskedDiffWithXvec()
        if self.config.hf_config.fp16_flow:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            tqdm.write(f"[{timestamp}] - [INFO] - Casting flow to fp16")
            self.flow.half()
        self.flow.load_state_dict(torch.load(f"{self.config.model}/flow.pt", map_location="cpu", weights_only=True), strict=True)
        device = get_device()
        self.flow.to(device).eval()

        self.hift = HiFTGenerator()
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(f"{self.config.model}/hift.pt", map_location="cpu", weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        device = get_device()
        self.hift.to(device).eval()

    @torch.inference_mode()
    def forward(
        self, prompt_mels_for_llm: torch.Tensor, prompt_mels_lens_for_llm: torch.Tensor,
        prompt_text_tokens_for_llm: list[list[int]], text_tokens_for_llm: list[list[int]],
        prompt_mels_for_flow: torch.Tensor, prompt_mels_lens_for_flow: torch.Tensor,
        spk_emb_for_flow: torch.Tensor,
        sampling_params: SamplingParams | list[SamplingParams],
        batch_size_flow: int,
        only_llm: bool,
        **kwargs,  # for compatibility
    ):
        timing_stats = {}

        # Audio tokenization
        start_time = time.time()
        prompt_speech_tokens, prompt_speech_tokens_lens = self.audio_tokenizer.quantize(
            prompt_mels_for_llm.to(get_device()), prompt_mels_lens_for_llm.to(get_device())
        )
        timing_stats['audio_tokenization'] = time.time() - start_time

        batch_size = prompt_speech_tokens.shape[0]
        assert len(prompt_text_tokens_for_llm) == batch_size

        # Prepare LLM inputs
        start_time = time.time()
        valid_prompt_speech_tokens = []
        inputs = []
        for i in range(batch_size):
            speech_tokens_i = prompt_speech_tokens[i, :prompt_speech_tokens_lens[i].item()].tolist()
            valid_prompt_speech_tokens.append(speech_tokens_i)
            inputs.append([self.config.hf_config.speech_vocab_size] + prompt_text_tokens_for_llm[i] + text_tokens_for_llm[i] + [self.config.hf_config.speech_vocab_size + 1] + speech_tokens_i)
        timing_stats['prepare_llm_inputs'] = time.time() - start_time

        # LLM generation
        start_time = time.time()
        llm_outputs = self.llm.generate(inputs, sampling_params, use_tqdm=self.use_tqdm)
        timing_stats['llm_generation'] = time.time() - start_time

        results_dict = {
            "prompt_speech_tokens": valid_prompt_speech_tokens,
            "generated_speech_tokens": [o['token_ids'][:-1] for o in llm_outputs],
        }
        if only_llm:
            return results_dict, timing_stats

        # Prepare Flow inputs
        start_time = time.time()
        flow_inputs = []
        flow_inputs_lens = []
        for i, o in enumerate(llm_outputs):
            generated_speech_tokens = o['token_ids'][:-1]  # ignore last eos
            prompt_speech_tokens = valid_prompt_speech_tokens[i]
            flow_inputs.append(torch.tensor(prompt_speech_tokens + generated_speech_tokens))
            flow_inputs_lens.append(len(prompt_speech_tokens) + len(generated_speech_tokens))
        flow_inputs = torch.nn.utils.rnn.pad_sequence(flow_inputs, batch_first=True, padding_value=0)
        flow_inputs_lens = torch.tensor(flow_inputs_lens)
        timing_stats['prepare_flow_inputs'] = time.time() - start_time

        # Flow generation and HiFi-GAN generation (with batching)
        total_batch_size = flow_inputs.shape[0]
        generated_wavs = []
        flow_total_time = 0.0
        hifigan_total_time = 0.0

        # Process in batches according to batch_size_flow, batch_size_flow <= total_batch_size
        # NOTE(xcsong): When executing both LLM and Flow on the same GPU,
        #   Flow can easily fill up the SM and memory. Therefore, batch processing is required to avoid OOM.
        num_batches = (total_batch_size + batch_size_flow - 1) // batch_size_flow
        batch_iterator = range(0, total_batch_size, batch_size_flow)
        if self.use_tqdm:
            batch_iterator = tqdm(batch_iterator, desc="Generating wavs (Flow+HiFi-GAN)", leave=False, unit="batch",
                                  total=num_batches, dynamic_ncols=True, position=self.config.rank + 1)

        for start_idx in batch_iterator:
            end_idx = min(start_idx + batch_size_flow, total_batch_size)
            batch_flow_inputs = flow_inputs[start_idx:end_idx]
            batch_flow_inputs_lens = flow_inputs_lens[start_idx:end_idx]
            batch_prompt_mels = prompt_mels_for_flow[start_idx:end_idx]
            batch_prompt_mels_lens = prompt_mels_lens_for_flow[start_idx:end_idx]
            batch_spk_emb = spk_emb_for_flow[start_idx:end_idx]

            # Flow generation for this batch
            flow_start_time = time.time()
            device = get_device()
            if device.type == "mps":
                # MPS has limited autocast support, disable for compatibility
                batch_generated_mels, batch_generated_mels_lens = self.flow(
                    batch_flow_inputs.to(get_device()), batch_flow_inputs_lens.to(get_device()),
                    batch_prompt_mels.to(get_device()), batch_prompt_mels_lens.to(get_device()), batch_spk_emb.to(get_device()),
                    streaming=False, finalize=True
                )
            elif device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16 if self.config.hf_config.fp16_flow else torch.float32):
                    batch_generated_mels, batch_generated_mels_lens = self.flow(
                        batch_flow_inputs.to(get_device()), batch_flow_inputs_lens.to(get_device()),
                        batch_prompt_mels.to(get_device()), batch_prompt_mels_lens.to(get_device()), batch_spk_emb.to(get_device()),
                        streaming=False, finalize=True
                    )
            else:
                # CPU fallback - no autocast
                batch_generated_mels, batch_generated_mels_lens = self.flow(
                    batch_flow_inputs.to(get_device()), batch_flow_inputs_lens.to(get_device()),
                    batch_prompt_mels.to(get_device()), batch_prompt_mels_lens.to(get_device()), batch_spk_emb.to(get_device()),
                    streaming=False, finalize=True
                )
            flow_total_time += time.time() - flow_start_time

            # HiFi-GAN generation for this batch
            hifigan_start_time = time.time()
            batch_size_current = end_idx - start_idx
            for i in range(batch_size_current):
                mel = batch_generated_mels[i, :, batch_prompt_mels_lens[i].item():batch_generated_mels_lens[i].item()].unsqueeze(0)
                wav, _ = self.hift(speech_feat=mel)
                generated_wavs.append(wav)
            hifigan_total_time += time.time() - hifigan_start_time

        timing_stats['flow_generation'] = flow_total_time
        timing_stats['hifigan_generation'] = hifigan_total_time

        # Calculate total time and batch statistics
        timing_stats['model.forward_total'] = sum(timing_stats.values())
        timing_stats['batch_size'] = len(generated_wavs)
        timing_stats['batch_size_flow'] = batch_size_flow

        results_dict['generated_wavs'] = generated_wavs
        return results_dict, timing_stats
