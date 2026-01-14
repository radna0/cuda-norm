from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SGLangTeacherOutput:
    # Hidden states for the prompt tokens (prefill), shape: [seq_len, hidden_size].
    hidden_states: "torch.Tensor"
    # The token ids used (flattened), shape: [seq_len].
    input_ids: "torch.Tensor"


class SGLangInprocTeacher:
    """
    In-process (no IPC) SGLang teacher-forward for GPT-OSS that returns GPU tensors.

    Why this exists:
      - `sglang.Engine(..., return_hidden_states=True)` currently serializes hidden
        states to CPU Python lists (via .cpu().tolist()), which is not usable for
        DFlash training at long contexts.
      - This helper instantiates SGLang's ModelRunner directly and runs a single
        EXTEND (prefill) forward pass, returning `LogitsProcessorOutput.hidden_states`
        as a GPU torch.Tensor.

    Notes / assumptions:
      - Designed for tp_size=1, pp_size=1 (single GPU) initially.
      - Allocates KV slots from ModelRunner pools and frees them after the call.
      - Returns the *post-norm* hidden states, consistent with SGLang's default
        hidden-state export. (If you need "before norm", we can flip the flag.)
    """

    def __init__(
        self,
        *,
        model_path: str,
        attention_backend: str,
        context_length: int,
        layers_to_capture: Optional[list[int]] = None,
        dtype: str = "bfloat16",
        gpu_id: int = 0,
        nccl_port: int = 23456,
        mem_fraction_static: float = 0.80,
        enable_return_hidden_states: bool = True,
        return_hidden_states_before_norm: bool = False,
    ) -> None:
        import torch

        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.server_args import ServerArgs

        server_args = ServerArgs(
            model_path=str(model_path),
            tokenizer_path=str(model_path),
            tp_size=1,
            pp_size=1,
            device="cuda",
            dtype=str(dtype),
            attention_backend=str(attention_backend),
            context_length=int(context_length),
            max_running_requests=1,
            # Conservative default; forward needs KV cache space for prefill.
            max_total_tokens=int(max(4096, min(context_length * 2, 65536))),
            disable_cuda_graph=True,
            allow_auto_truncate=True,
            enable_return_hidden_states=bool(enable_return_hidden_states),
        )

        model_config = ModelConfig.from_server_args(
            server_args,
            model_path=str(model_path),
            model_revision=None,
            is_draft_model=False,
        )

        self._server_args = server_args
        self._model_config = model_config
        self._gpu_id = int(gpu_id)
        self._nccl_port = int(nccl_port)
        self._mem_fraction_static = float(mem_fraction_static)
        self._return_hidden_states_before_norm = bool(return_hidden_states_before_norm)
        self._layers_to_capture = list(layers_to_capture) if layers_to_capture else None

        # ModelRunner initializes distributed state internally (even for tp=1).
        self._runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=self._mem_fraction_static,
            gpu_id=self._gpu_id,
            tp_rank=0,
            tp_size=1,
            moe_ep_rank=0,
            moe_ep_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=self._nccl_port,
            server_args=server_args,
            dp_rank=None,
            is_draft_worker=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
        )

        # Sanity: we expect CUDA.
        assert str(self._runner.device) == "cuda"
        assert torch.cuda.is_available()

        # Optional: capture per-layer features (aux hidden states) for DFlash conditioning.
        # For GPT-OSS, this is implemented by `GptOssModel.layers_to_capture` and the
        # `GptOssForCausalLM.capture_aux_hidden_states` switch.
        if self._layers_to_capture:
            try:
                if hasattr(self._runner.model, "capture_aux_hidden_states"):
                    self._runner.model.capture_aux_hidden_states = True  # type: ignore[attr-defined]
                inner = getattr(self._runner.model, "model", None)
                if inner is not None and hasattr(inner, "layers_to_capture"):
                    inner.layers_to_capture = list(self._layers_to_capture)  # type: ignore[attr-defined]
            except Exception:
                # If anything goes wrong, keep default (final hidden only).
                pass

    @property
    def device(self) -> str:
        return str(self._runner.device)

    @property
    def hidden_size(self) -> int:
        return int(self._model_config.hidden_size)

    def close(self) -> None:
        # No Engine subprocesses. Best-effort destroy process group to avoid warnings.
        try:
            import torch

            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass

    def embed_tokens(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        """
        Return embeddings for token ids using the teacher's embedding table.

        Accepts 1D [N] or 2D [1, N]. Returns 2D [N, hidden_size] on CUDA.
        """
        import torch

        flat = self._flatten_input_ids(input_ids).to(device=self._runner.device, dtype=torch.long)
        embed = getattr(getattr(self._runner.model, "model", None), "embed_tokens", None)
        if embed is None:
            raise RuntimeError("Teacher model has no embed_tokens")
        out = embed(flat)
        if out.device.type != "cuda":
            raise RuntimeError(f"Expected CUDA embeddings, got {out.device}")
        if out.ndim != 2:
            out = out.view(flat.numel(), -1)
        return out

    def lm_head_logits(self, hidden_states: "torch.Tensor") -> "torch.Tensor":
        """
        Project hidden states to vocab logits using the teacher's lm_head.

        hidden_states: [N, hidden_size] (CUDA)
        returns: [N, vocab_size] (CUDA)
        """
        import torch

        if hidden_states.device.type != "cuda":
            raise RuntimeError("hidden_states must be on CUDA")
        if hidden_states.ndim != 2:
            hidden_states = hidden_states.view(hidden_states.shape[0], -1)

        # SGLang's ParallelLMHead intentionally has no forward() (it raises),
        # but exposes `weight` to be used by the logits processor / sampler.
        head = getattr(self._runner.model, "lm_head", None)
        if head is None or not hasattr(head, "weight"):
            raise RuntimeError("Teacher model has no lm_head.weight")

        w = head.weight  # [vocab_padded, hidden]
        logits = torch.matmul(hidden_states.to(w.dtype), w.T)

        # Trim to original vocab size (ignore padding vocab).
        vocab_size = int(getattr(self._model_config, "vocab_size", logits.shape[-1]))
        if logits.shape[-1] > vocab_size:
            logits = logits[:, :vocab_size]

        if logits.device.type != "cuda":
            raise RuntimeError(f"Expected CUDA logits, got {logits.device}")
        return logits

    @staticmethod
    def _make_sampling_info(*, device: str, vocab_size: int, batch_size: int):
        import torch

        from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

        temperatures = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
        top_ps = torch.ones((batch_size,), dtype=torch.float32, device=device)
        top_ks = torch.ones((batch_size,), dtype=torch.int32, device=device)
        min_ps = torch.zeros((batch_size,), dtype=torch.float32, device=device)
        return SamplingBatchInfo(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            is_all_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=int(vocab_size),
            grammars=None,
            vocab_mask=None,
            apply_mask_func=None,
            penalizer_orchestrator=None,
            acc_linear_penalties=None,
            has_custom_logit_processor=False,
            custom_params=None,
            custom_logit_processor=None,
            sampling_seed=None,
            device=str(device),
            logit_bias=None,
        )

    @staticmethod
    def _flatten_input_ids(input_ids: "torch.Tensor") -> "torch.Tensor":
        import torch

        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor")
        if input_ids.ndim == 2:
            if int(input_ids.shape[0]) != 1:
                raise ValueError("SGLangInprocTeacher currently supports batch_size=1 only")
            return input_ids[0].contiguous()
        if input_ids.ndim == 1:
            return input_ids.contiguous()
        raise ValueError(f"Unexpected input_ids shape: {tuple(input_ids.shape)}")

    def prefill_hidden_states(
        self,
        input_ids: "torch.Tensor",
        *,
        return_hidden_states_before_norm: Optional[bool] = None,
    ) -> SGLangTeacherOutput:
        """
        Run a single SGLang EXTEND forward and return hidden states as a GPU tensor.
        """
        import torch

        from sglang.srt.managers.schedule_batch import ModelWorkerBatch
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )

        runner = self._runner
        device = runner.device
        flat = self._flatten_input_ids(input_ids).to(device=device, dtype=torch.long)
        seq_len = int(flat.numel())
        if seq_len <= 0:
            raise ValueError("Empty input_ids")

        # Allocate one request slot.
        req_pool_idx = runner.req_to_token_pool.alloc(1)[0]
        req_pool_indices = torch.tensor([req_pool_idx], device=device, dtype=torch.int64)

        # Allocate KV slots for all tokens in the prompt.
        out_cache_loc = runner.token_to_kv_pool_allocator.alloc(seq_len)
        if out_cache_loc is None:
            raise RuntimeError("token_to_kv_pool_allocator.alloc failed (OOM / pool exhausted)")

        # Populate req_to_token mapping for this request.
        # req_to_token is int32; out_cache_loc is int64.
        runner.req_to_token_pool.req_to_token[req_pool_idx, :seq_len] = out_cache_loc.to(torch.int32)

        seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int64)
        seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int64)
        orig_seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)

        sampling_info = self._make_sampling_info(
            device=device,
            vocab_size=int(self._model_config.vocab_size),
            batch_size=1,
        )

        capture_hidden = CaptureHiddenMode.FULL
        rhbn = (
            self._return_hidden_states_before_norm
            if return_hidden_states_before_norm is None
            else bool(return_hidden_states_before_norm)
        )

        batch = ModelWorkerBatch(
            forward_mode=ForwardMode.EXTEND,
            input_ids=flat,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            orig_seq_lens=orig_seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=int(seq_len),
            return_logprob=False,
            top_logprobs_nums=[0],
            token_ids_logprobs=[None],
            global_num_tokens=None,
            global_num_tokens_for_logprob=None,
            is_extend_in_batch=True,
            can_run_dp_cuda_graph=False,
            tbo_split_seq_index=None,
            global_forward_mode=None,
            extend_num_tokens=int(seq_len),
            extend_seq_lens=[int(seq_len)],  # extend lengths
            extend_prefix_lens=[0],  # prefix lengths
            extend_logprob_start_lens=[0],
            multimodal_inputs=[None],
            encoder_cached=None,
            encoder_lens=None,
            encoder_lens_cpu=None,
            encoder_out_cache_loc=None,
            lora_ids=[None],
            sampling_info=sampling_info,
            input_embeds=None,
            token_type_ids=None,
            spec_algorithm=None,
            spec_info=None,
            hicache_consumer_index=-1,
            capture_hidden_mode=capture_hidden,
            extend_input_logprob_token_ids=None,
            is_prefill_only=True,
            dimensions=None,
            dllm_block_offsets=None,
            dllm_config=None,
            reqs=None,
            has_grammar=False,
            mamba_track_indices=None,
            mamba_track_mask=None,
            mamba_track_seqlens=None,
            return_hidden_states_before_norm=bool(rhbn),
        )

        forward_batch = ForwardBatch.init_new(batch, runner)
        out = runner.forward(forward_batch).logits_output
        hidden = out.hidden_states
        if hidden is None:
            raise RuntimeError("SGLang did not return hidden_states; enable_return_hidden_states may be off")
        if hidden.device.type != "cuda":
            raise RuntimeError(f"Expected CUDA hidden states, got {hidden.device}")

        # hidden is [seq_len, hidden_size] for this single request
        # Free allocations so we can call repeatedly.
        runner.token_to_kv_pool_allocator.free(out_cache_loc)
        runner.req_to_token_pool.free(req_pool_idx)

        return SGLangTeacherOutput(hidden_states=hidden, input_ids=flat)
