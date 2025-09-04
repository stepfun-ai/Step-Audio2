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
""" Example Usage: see README.md
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import onnxruntime
import s3tokenizer
import torch
import torch.distributed as dist
from device_utils import get_device
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from flashcosyvoice.config import Config, CosyVoice2LLMConfig, SamplingParams
from flashcosyvoice.cosyvoice2 import CosyVoice2
from flashcosyvoice.utils.audio import mel_spectrogram


def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def save_file_async(
    wav, prompt_speech_tokens, generated_speech_tokens,
    info, timing_stats
):
    """Save audio asynchronously."""
    try:
        os.makedirs(os.path.dirname(info['wav']), exist_ok=True)
        if wav is not None:
            wav = wav.cpu()
            torchaudio.save(info['wav'], wav, 24000)
            duration = wav.shape[-1] / 24000.0
            rtf = ((timing_stats['dataloader_time'] + timing_stats['model_inference_time']) / timing_stats['batch_size']) / duration
            timing_stats['rtf'] = rtf
        else:
            duration = 0.0
        info['timing_stats'] = timing_stats
        info['prompt_speech_tokens'] = prompt_speech_tokens
        info['generated_speech_tokens'] = generated_speech_tokens
        with open(f"{info['wav'].replace('.wav', '.json')}", "w") as f:
            json.dump(info, f, ensure_ascii=False, indent=4)
        return duration
    except Exception as e:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        tqdm.write(f"[{timestamp}] - [ERROR] - Error saving audio {info.get('key', 'unknown')}: {e}")
        return 0.0


class AudioDataset(Dataset):

    def __init__(self, text_norm, text_tokenizer, data_list, model_config: Config):
        self.datas = []
        self.text_norm = text_norm
        self.model_config = model_config

        """Example data_list:
        ```
        {"key": "uttid_1", "prompt_text": "你好，我是小明。", "text": "你好，我是小红。", "prompt_wav": "/mnt/data/audio/00000000.wav", "wav": "/mnt/data/audio_synthetic/uttid_1.wav"}
        {"key": "uttid_2", "prompt_text": "你好，我是小红。", "text": "你好，我是小明。", "prompt_wav": "/mnt/data/audio/00000001.wav", "wav": "/mnt/data/audio_synthetic/uttid_2.wav"}
        ```
        Note:
            - `key` is the key of this sample.
            - `prompt_text` is the text used for prompt.
            - `text` is the text used for generating real audio.
            - `prompt_wav` is the audio used for prompt.
            - `wav` is the path to the generated audio to be saved (we highly recommend to pre-define the save path before running the script).
        """
        missing = 0
        with open(data_list, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_lines = len(lines)
            if torch.distributed.get_node_local_rank() == 0:
                iterator = tqdm(lines, desc='Loading data')
            else:
                iterator = lines
            for line in iterator:
                data = json.loads(line.strip())
                valid = True
                for k in ['key', 'prompt_text', 'text', 'prompt_wav']:
                    if k not in data:
                        valid = False
                        break
                    if data[k] is None:
                        valid = False
                        break
                if not os.path.exists(data['prompt_wav']):
                    valid = False
                if valid:
                    self.datas.append(data)
                else:
                    missing += 1
        if torch.distributed.get_node_local_rank() == 0:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            tqdm.write(f'[{timestamp}] - [INFO] - Loaded {total_lines} lines, found {missing} missing lines, total valid lines == {len(self.datas)}.')

        self.text_tokenizer = text_tokenizer

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.spk_model = onnxruntime.InferenceSession(f"{self.model_config.model}/campplus.onnx", sess_options=option,
                                                      providers=["CPUExecutionProvider"])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]

        try:
            # 1. feature for s3tokenizer
            audio = s3tokenizer.load_audio(data['prompt_wav'], sr=16000)  # [T]
            log_mel = s3tokenizer.log_mel_spectrogram(audio)  # [num_mels, T]

            # 2. feature for speaker embedding
            spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
            spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
            spk_emb = self.spk_model.run(
                None, {self.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()}
            )[0].flatten().tolist()

            # 3. feature for flow
            audio, sample_rate = torchaudio.load(data['prompt_wav'], backend='soundfile')
            audio = audio.mean(dim=0, keepdim=True)  # [1, T]
            if sample_rate != 24000:
                audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)(audio)
            mel = mel_spectrogram(audio).transpose(1, 2).squeeze(0)  # [T, num_mels]
            mel_len = mel.shape[0]

            # 4. feature for llm
            if self.text_norm is not None:
                prompt_texts = [i["text"] for i in json.loads(self.text_norm.do_voicegen_frd(data['prompt_text'].strip()))["sentences"]]
                prompt_text = ''.join(prompt_texts)
                texts = [i["text"] for i in json.loads(self.text_norm.do_voicegen_frd(data['text'].strip()))["sentences"]]
                text = ''.join(texts)
            else:
                prompt_text = data['prompt_text']
                text = data['text']
            prompt_text_ids = self.text_tokenizer.encode(prompt_text)
            prompt_text_ids = [i + self.model_config.hf_config.speech_vocab_size + 2 for i in prompt_text_ids]
            text_ids = self.text_tokenizer.encode(text)
            text_ids = [i + self.model_config.hf_config.speech_vocab_size + 2 for i in text_ids]
            item = {
                "prompt_text_tokens": prompt_text_ids, "text_tokens": text_ids,
                "spk_emb": spk_emb, "mel": mel, "mel_len": mel_len, "log_mel": log_mel, "info": data,
                "min_tokens": len(text_ids) * self.model_config.min_token_text_ratio,
                "max_tokens": len(text_ids) * self.model_config.max_token_text_ratio,
            }
        except Exception as e:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            tqdm.write(f"[{timestamp}] - [WARNING] - Error processing data item {data.get('key', idx)}: {e}")
            return None
        return item


def collate_fn(batch):
    prompt_mels_for_llm = [item["log_mel"] for item in batch if item is not None]
    prompt_mels_for_llm, prompt_mels_lens_for_llm = s3tokenizer.padding(prompt_mels_for_llm)  # [B, num_mels=128, T]
    prompt_text_tokens_for_llm = [item["prompt_text_tokens"] for item in batch if item is not None]
    text_tokens_for_llm = [item["text_tokens"] for item in batch if item is not None]
    prompt_mels_for_flow = [item["mel"] for item in batch if item is not None]
    prompt_mels_for_flow = torch.nn.utils.rnn.pad_sequence(prompt_mels_for_flow, batch_first=True, padding_value=0)  # [B, T', num_mels=80]
    prompt_mels_lens_for_flow = [item["mel_len"] for item in batch if item is not None]
    prompt_mels_lens_for_flow = torch.tensor(prompt_mels_lens_for_flow)
    spk_emb_for_flow = [item["spk_emb"] for item in batch if item is not None]
    spk_emb_for_flow = torch.tensor(spk_emb_for_flow)
    sampling_params = [SamplingParams(min_tokens=item["min_tokens"], max_tokens=item["max_tokens"], use_ras=True) for item in batch if item is not None]
    infos = [item["info"] for item in batch if item is not None]
    return {
        "prompt_mels_for_llm": prompt_mels_for_llm,
        "prompt_mels_lens_for_llm": prompt_mels_lens_for_llm,
        "prompt_text_tokens_for_llm": prompt_text_tokens_for_llm,
        "text_tokens_for_llm": text_tokens_for_llm,
        "prompt_mels_for_flow": prompt_mels_for_flow,
        "prompt_mels_lens_for_flow": prompt_mels_lens_for_flow,
        "spk_emb_for_flow": spk_emb_for_flow,
        "sampling_params": sampling_params,
        "infos": infos,
    }


def init_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
    tqdm.write(f'[{timestamp}] - [INFO] - Inference on multiple gpus, this gpu {local_rank}, rank {rank}, world_size {world_size}')
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return world_size, local_rank, rank


def get_args():
    parser = argparse.ArgumentParser(description='FlashCosyVoice')
    parser.add_argument('--model_path',
                        required=True,
                        type=str,
                        help='model path')
    parser.add_argument('--data_list',
                        required=True,
                        type=str,
                        help='data list')
    parser.add_argument('--batch_size_dataloader',
                        required=True,
                        type=int,
                        help='batch size (per-device) for dataloading')
    parser.add_argument('--batch_size_flow',
                        required=True,
                        type=int,
                        help='batch size (per-device) for flow-matching')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='workers for dataloader')
    parser.add_argument('--prefetch',
                        type=int,
                        default=5,
                        help='prefetch for dataloader')
    parser.add_argument('--enable_tn',
                        action='store_true',
                        help='enable text normalization')
    parser.add_argument('--only_llm',
                        action='store_true',
                        help='only generate speech tokens from llm')
    parser.add_argument('--fp16_flow',
                        action='store_true',
                        help='enable fp16 flow')
    parser.add_argument('--seed',
                        type=int,
                        default=1986,
                        help='random seed for generation')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.enable_tn:
        # Check python version, if == 3.10, use ttsfrd
        if sys.version_info.major == 3 and sys.version_info.minor == 10:
            # Check if ttsfrd is installed
            try:
                import ttsfrd
                from cosyvoice_ttsfrd import get_resource_path
            except ImportError as e:
                raise ImportError("ttsfrd is not installed, please install it first, see `https://github.com/xingchensong/CosyVoice-ttsfrd` for installation guide.") from e
            text_norm = ttsfrd.TtsFrontendEngine()
            text_norm.initialize(get_resource_path())
            text_norm.set_lang_type('pinyinvg')
        else:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            tqdm.write(f"[{timestamp}] - [WARNING] - Only python 3.10 is supported for ttsfrd, see `https://github.com/xingchensong/CosyVoice-ttsfrd` for more info. Setting enable_tn to False...")
            # TODO: maybe we should use wetext if python version is not 3.10?
            args.enable_tn = False
            text_norm = None
    else:
        text_norm = None

    device = get_device()
    assert (device.type in ["cuda", "mps"]), "Either CUDA or MPS must be available"
    world_size, local_rank, rank = init_distributed()
    config = Config(model=args.model_path, enforce_eager=True, tensor_parallel_size=1,
                    max_num_seqs=args.batch_size_dataloader,
                    hf_config=CosyVoice2LLMConfig(fp16_flow=args.fp16_flow), rank=local_rank)
    model = CosyVoice2(config)

    set_all_random_seed(args.seed)

    dataset = AudioDataset(text_norm, model.llm.tokenizer, args.data_list, config)
    sampler = DistributedSampler(dataset,
                                 num_replicas=world_size,
                                 rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_dataloader, num_workers=args.num_workers, pin_memory=True,
                            sampler=sampler, shuffle=False, prefetch_factor=args.prefetch, collate_fn=collate_fn)
    total_steps = len(dataset)

    if local_rank == 0:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        tqdm.write(f"[{timestamp}] - [INFO] - {args}")
        progress_bar = tqdm(total=total_steps, desc="Processing samples", unit="wav",
                            position=0, leave=True, dynamic_ncols=True)

    cpu_counts = os.cpu_count()
    executor = ThreadPoolExecutor(max_workers=min(args.batch_size_dataloader, cpu_counts // 8))
    pending_futures = []
    dataloader_iter = iter(dataloader)
    succeed_duration = 0.01  # avoid division by zero
    start_time = time.time()
    estimated_total_wavs = 0
    succeed_wavs = 0
    failed_wavs = 0
    last_print_time = start_time

    while True:
        try:
            dataloader_start = time.time()
            batch = next(dataloader_iter)
            dataloader_time = time.time() - dataloader_start

            if len(batch['infos']) == 0:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                tqdm.write(f"[{timestamp}] - [WARNING] - rank {rank} of {world_size}: No valid batch found, skipping this batch...")
                continue

            model_start = time.time()
            results_dict, timing_stats = model(**batch, batch_size_flow=args.batch_size_flow,
                                               only_llm=args.only_llm)
            model_time = time.time() - model_start

            estimated_total_wavs += len(results_dict['generated_wavs'])

            timing_stats['dataloader_time'] = dataloader_time
            timing_stats['model_inference_time'] = model_time

            if args.only_llm:
                results_dict['generated_wavs'] = [None] * len(results_dict['prompt_speech_tokens'])

            for i in range(len(results_dict['generated_wavs'])):
                future = executor.submit(
                    save_file_async, results_dict['generated_wavs'][i],
                    results_dict['prompt_speech_tokens'][i],
                    results_dict['generated_speech_tokens'][i],
                    batch['infos'][i].copy(), timing_stats.copy()
                )
                pending_futures.append(future)

            completed_futures = []
            for future in pending_futures:
                if future.done():
                    try:
                        duration = future.result()
                        succeed_duration += duration
                        succeed_wavs += 1
                    except Exception as e:
                        failed_wavs += 1
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                        tqdm.write(f"[{timestamp}] - [ERROR] - rank {rank} of {world_size}: Error in async save task: {e}")
                    completed_futures.append(future)

            for future in completed_futures:
                pending_futures.remove(future)

            if local_rank == 0:
                update_n = world_size * len(batch["prompt_text_tokens_for_llm"])
                if progress_bar.n + update_n > progress_bar.total:
                    progress_bar.update(progress_bar.total - progress_bar.n)
                else:
                    progress_bar.update(update_n)

                current_time = time.time()
                if current_time - last_print_time >= 120 and not args.only_llm:
                    elapsed_time = current_time - start_time
                    avg_duration = succeed_duration / succeed_wavs if succeed_wavs > 0 else 0
                    estimated_total_duration = avg_duration * estimated_total_wavs
                    current_rtf = elapsed_time / estimated_total_duration if estimated_total_duration > 0.01 else 0
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                    tqdm.write(f"[{timestamp}] - [INFO] - rank {rank} of {world_size}: Estimated total wavs: {estimated_total_wavs} ({estimated_total_wavs - succeed_wavs} pending to save), Succeed wavs: {succeed_wavs}, Failed wavs: {failed_wavs}, Estimated total duration: {estimated_total_duration:.2f}s ({estimated_total_duration / 3600:.2f} h), Estimated RTF: {current_rtf:.5f}, Elapsed time: {elapsed_time:.2f}s")  # noqa
                    last_print_time = current_time
        except StopIteration:
            break
        except Exception as e:
            failed_wavs += 1
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            tqdm.write(f"[{timestamp}] - [ERROR] - rank {rank} of {world_size}: Error in main loop: {e}")
            continue

    total_time = time.time() - start_time

    if local_rank == 0:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        tqdm.write(f"[{timestamp}] - [INFO] - Waiting for {len(pending_futures)} pending save tasks to complete...")

    for future in pending_futures:
        try:
            duration = future.result(timeout=60)
            succeed_duration += duration
            succeed_wavs += 1
        except Exception as e:
            failed_wavs += 1
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            tqdm.write(f"[{timestamp}] - [ERROR] - rank {rank} of {world_size}: Error in final async save task: {e}")
    executor.shutdown(wait=True)

    if local_rank == 0:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        tqdm.write(f"[{timestamp}] - [INFO] - All async save tasks completed.")
        progress_bar.close()

    if not args.only_llm:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        tqdm.write(f"[{timestamp}] - [INFO] - rank {rank} of {world_size}: Final Report - Succeed wavs: {succeed_wavs}, Failed wavs: {failed_wavs}, Total duration: {succeed_duration:.2f}s ({succeed_duration / 3600:.2f} h), RTF: {total_time / succeed_duration:.5f}")  # noqa

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
