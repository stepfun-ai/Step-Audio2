import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

import torch
import torch.distributed as dist

from flashcosyvoice.config import Config
from flashcosyvoice.engine.sequence import Sequence
from flashcosyvoice.modules.qwen2 import Qwen2ForCausalLM
from flashcosyvoice.modules.sampler import RasSampler, Sampler
from flashcosyvoice.utils.context import (get_context, reset_context,
                                          set_context)
from flashcosyvoice.utils.loader import load_model
from device_utils import get_device


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # TODO(xcsong): support tp > 1
        if self.world_size > 1:
            dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        device = get_device()
        if device.type == "cuda":
            torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device(device)
        self.model = Qwen2ForCausalLM(hf_config)
        load_model(self.model, config.model, hf_config)
        self.sampler = Sampler()
        self.ras_sampler = RasSampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager and device.type == "cuda":
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="flashcosyvoice", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="flashcosyvoice")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        device = get_device()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if self.world_size > 1:
            dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        device = get_device()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        elif device.type == "mps":
            torch.mps.empty_cache()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        device = get_device()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        device = get_device()
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info()
            used = total - free
            peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
            current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        else:
            # MPS and CPU fallback - use reasonable defaults
            total = 8 * 1024 * 1024 * 1024  # 8GB
            used = 1 * 1024 * 1024 * 1024   # 1GB
            free = total - used
            peak = used
            current = used
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0, "try to **increase** gpu_memory_utilization"
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        device = get_device()
        if device.type == "cuda":
            block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        else:
            block_tables = torch.tensor(block_tables, dtype=torch.int32).to(device)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        device = get_device()
        if device.type == "cuda":
            input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        else:
            input_ids = torch.tensor(input_ids, dtype=torch.int64).to(device)
            positions = torch.tensor(positions, dtype=torch.int64).to(device)
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32).to(device)
            cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32).to(device)
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32).to(device)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        device = get_device()
        if device.type == "cuda":
            input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        else:
            input_ids = torch.tensor(input_ids, dtype=torch.int64).to(device)
            positions = torch.tensor(positions, dtype=torch.int64).to(device)
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32).to(device)
            context_lens = torch.tensor(context_lens, dtype=torch.int32).to(device)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        top_ks = []
        win_sizes = []
        tau_rs = []
        top_ps = []
        min_tokens_list = []
        use_ras_list = []

        for seq in seqs:
            temperatures.append(seq.temperature)
            top_ks.append(seq.top_k)
            win_sizes.append(seq.win_size)
            tau_rs.append(seq.tau_r)
            top_ps.append(seq.top_p)
            min_tokens_list.append(seq.min_tokens)
            use_ras_list.append(seq.use_ras)

        device = get_device()
        if device.type == "cuda":
            temperatures_tensor = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        else:
            temperatures_tensor = torch.tensor(temperatures, dtype=torch.float32).to(device)
        # check all items equal
        assert all(item == temperatures[0] for item in temperatures)
        assert all(item == top_ks[0] for item in top_ks)
        assert all(item == win_sizes[0] for item in win_sizes)
        assert all(item == tau_rs[0] for item in tau_rs)
        assert all(item == top_ps[0] for item in top_ps)
        assert all(item == use_ras_list[0] for item in use_ras_list)

        return {
            'temperatures': temperatures_tensor,
            'top_k': top_ks[0],
            'win_size': win_sizes[0],
            'tau_r': tau_rs[0],
            'top_p': top_ps[0],
            'eos_token': self.config.eos,
            'min_tokens': min_tokens_list,
            'use_ras': use_ras_list[0]
        }

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        if self.rank == 0 or self.world_size == 1:
            sample_params = self.prepare_sample(seqs)
            logits = self.run_model(input_ids, positions, is_prefill)

            if sample_params['use_ras']:
                # Prepare decoded tokens list for RasSampler
                decoded_tokens_list = [seq.completion_token_ids for seq in seqs]
                # Pass all parameters as lists to RasSampler
                token_ids = self.ras_sampler(
                    logits,
                    decoded_tokens_list,
                    win_size=sample_params['win_size'],
                    tau_r=sample_params['tau_r'],
                    top_p=sample_params['top_p'],
                    top_k=sample_params['top_k'],
                    eos_token=sample_params['eos_token'],
                    min_tokens=sample_params['min_tokens']
                ).tolist()
            else:
                # Use the default sampler with list form of top_ks
                token_ids = self.sampler(logits, sample_params['temperatures'], sample_params['top_k']).tolist()
        else:
            logits = self.run_model(input_ids, positions, is_prefill)
            token_ids = None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
