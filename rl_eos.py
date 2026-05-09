from __future__ import annotations

from typing import Iterable, Optional


DEFAULT_EOS_TOKEN_NAMES = ("<|im_end|>", "<|endoftext|>")


def as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _iter_token_names(eos_token_names: Optional[Iterable[str]]):
    if eos_token_names is None:
        return list(DEFAULT_EOS_TOKEN_NAMES)
    if isinstance(eos_token_names, str):
        return [eos_token_names]
    return [str(token) for token in eos_token_names if token]


def resolve_eos_token_ids(tokenizer, eos_token_names=None):
    eos_ids = []
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    for token in _iter_token_names(eos_token_names):
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            token_id = None
        if isinstance(token_id, int) and token_id >= 0:
            eos_ids.append(int(token_id))
    return sorted(set(eos_ids))


def resolve_eos_token_strings(tokenizer, eos_token_names=None):
    tokens = []
    if getattr(tokenizer, "eos_token", None):
        tokens.append(tokenizer.eos_token)
    tokens.extend(_iter_token_names(eos_token_names))
    seen = set()
    result = []
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return result


def first_eos_index(token_ids, eos_token_ids):
    eos_set = {int(token_id) for token_id in eos_token_ids if token_id is not None}
    for idx, token_id in enumerate(token_ids):
        if int(token_id) in eos_set:
            return idx
    return None


def truncate_text_at_first_eos(text, eos_token_strings, include_eos=False):
    cut = None
    cut_token = ""
    for token in eos_token_strings:
        if not token:
            continue
        idx = text.find(token)
        if idx >= 0 and (cut is None or idx < cut):
            cut = idx
            cut_token = token
    if cut is None:
        return text
    end = cut + len(cut_token) if include_eos else cut
    return text[:end]


def eos_response_metadata(
    token_ids,
    *,
    prompt_length=0,
    eos_token_ids=(),
    pad_token_ids=(),
    mask_token_ids=(),
    text=None,
    eos_token_strings=(),
    precomputed_eos_then_continues=None,
):
    token_ids = [int(token_id) for token_id in token_ids]
    eos_set = {int(token_id) for token_id in eos_token_ids if token_id is not None}
    pad_set = {int(token_id) for token_id in pad_token_ids if token_id is not None}
    mask_set = {int(token_id) for token_id in mask_token_ids if token_id is not None}
    first_idx = first_eos_index(token_ids, eos_set)

    if first_idx is None:
        valid_response_length = len([x for x in token_ids if x not in pad_set])
        first_global = None
        eos_first = False
        tail_ids = []
    else:
        valid_response_length = len([x for x in token_ids[:first_idx] if x not in pad_set])
        first_global = int(prompt_length) + first_idx
        eos_first = first_idx == 0
        tail_ids = token_ids[first_idx + 1 :]

    tail_content = [
        token_id
        for token_id in tail_ids
        if token_id not in pad_set and token_id not in mask_set and token_id not in eos_set
    ]
    eos_then_continues = bool(tail_content)
    if precomputed_eos_then_continues is not None:
        eos_then_continues = bool(eos_then_continues or precomputed_eos_then_continues)

    raw_response_length = len([x for x in token_ids if x not in pad_set])
    truncated_response = None
    training_response = None
    if text is not None:
        truncated_response = truncate_text_at_first_eos(
            text, eos_token_strings, include_eos=False
        )
        training_response = truncate_text_at_first_eos(
            text, eos_token_strings, include_eos=True
        )

    return {
        "first_eos_index": first_idx,
        "first_eos_global_index": first_global,
        "valid_response_length": int(valid_response_length),
        "raw_response_length": int(raw_response_length),
        "eos_then_continues": bool(eos_then_continues),
        "eos_first": bool(eos_first),
        "missing_eos": bool(first_idx is None),
        "post_eos_content_token_count": int(len(tail_content)),
        "truncated_response": truncated_response,
        "training_response": training_response,
    }


def pad_after_first_eos(token_ids, eos_token_ids, pad_token_id):
    first_idx = first_eos_index(token_ids, eos_token_ids)
    if first_idx is None or pad_token_id is None:
        return list(token_ids), first_idx
    result = list(token_ids)
    result[first_idx + 1 :] = [int(pad_token_id)] * max(len(result) - first_idx - 1, 0)
    return result, first_idx

