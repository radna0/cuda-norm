from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _lazy_torch() -> Any:
    import torch

    return torch


def sample_logits(logits, temperature: float = 0.0):
    torch = _lazy_torch()
    if temperature is None or float(temperature) <= 0:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits / float(temperature), dim=-1)
    return torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(logits.shape[:-1])


@dataclass
class SpecDecodeStats:
    acceptance_lengths: list[int]
    total_steps: int


@_lazy_torch().inference_mode()
def dflash_spec_generate(
    *,
    draft_model,
    target_model,
    tokenizer,
    input_ids,
    max_new_tokens: int,
    block_size: int,
    temperature: float = 0.0,
    stop_token_ids: list[int] | None = None,
    mask_token: str = "<|MASK|>",
) -> tuple[Any, SpecDecodeStats]:
    torch = _lazy_torch()

    # Prefer an existing token id (no model resize) for broad runtime compatibility
    # (e.g., SGLang/TRTLLM teachers). Fall back to adding a real mask token only if
    # there's no pad token available.
    if tokenizer.mask_token_id is not None:
        mask_token_id = int(tokenizer.mask_token_id)
    elif tokenizer.pad_token_id is not None:
        mask_token_id = int(tokenizer.pad_token_id)
    else:
        tokenizer.add_special_tokens({"mask_token": mask_token})
        target_model.resize_token_embeddings(len(tokenizer))
        mask_token_id = int(tokenizer.mask_token_id)
    num_input_tokens = int(input_ids.shape[1])
    max_length = num_input_tokens + int(max_new_tokens)

    output_ids = torch.full(
        (input_ids.shape[0], max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=target_model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=target_model.device).unsqueeze(0)

    from transformers import DynamicCache

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # Prefill target (ctx only)
    out = target_model(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    # First token always from target.
    first = sample_logits(out.logits[:, -1, :], temperature=temperature)
    output_ids[:, num_input_tokens : num_input_tokens + 1] = first.unsqueeze(1)

    target_hidden = draft_model.extract_context_feature(out.hidden_states)

    acceptance_lengths: list[int] = []
    start = num_input_tokens
    steps = 0
    while start < max_length:
        steps += 1
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]

        noise_embedding = target_model.model.embed_tokens(block_output_ids)
        # If we don't have a dedicated mask token, treat mask_token_id as a
        # synthetic mask and replace its embedding with a learned vector.
        try:
            if hasattr(draft_model, "mask_embedding"):
                m = (block_output_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]
                if m.numel() > 0:
                    noise_embedding[0, m, :] = draft_model.mask_embedding.to(noise_embedding.dtype)[None, :]
        except Exception:
            pass
        # Draft predicts tokens 1..block_size-1
        draft_out = draft_model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
            past_key_values=past_key_values_draft,
            use_cache=True,
        )
        draft_logits = target_model.lm_head(draft_out[:, -block_size + 1 :, :])
        block_output_ids[:, 1:] = sample_logits(draft_logits, temperature=temperature)
        past_key_values_draft.crop(start)

        out = target_model(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        posterior = sample_logits(out.logits, temperature=temperature)

        # Accept prefix of the proposed block.
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(int(acceptance_length) + 1)
        start += int(acceptance_length) + 1
        past_key_values_target.crop(start)
        target_hidden = draft_model.extract_context_feature(out.hidden_states)[:, : acceptance_length + 1, :]

        if stop_token_ids:
            stop = torch.tensor(stop_token_ids, device=output_ids.device)
            if torch.isin(output_ids[:, num_input_tokens:start], stop).any():
                break

    # Do not "filter out" mask_token_id (it may be a real token like pad);
    # just return the prefix we actually filled.
    final_len = min(int(start), int(max_length))
    out_ids = output_ids[:, :final_len]
    if stop_token_ids:
        stop = torch.tensor(stop_token_ids, device=out_ids.device)
        stop_idx = torch.isin(out_ids[0][num_input_tokens:], stop).nonzero(as_tuple=True)[0]
        if stop_idx.numel() > 0:
            out_ids = out_ids[:, : num_input_tokens + stop_idx[0] + 1]

    return out_ids, SpecDecodeStats(acceptance_lengths=acceptance_lengths, total_steps=steps)
