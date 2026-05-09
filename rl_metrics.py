import math
from statistics import median

from rl_eos import first_eos_index, truncate_text_at_first_eos


def repeated_ngram_rate_from_tokens(tokens, n=4):
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - len(set(ngrams)) / max(len(ngrams), 1)


def repeated_line_ratio(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def _valid_generated_indices(token_ids, eos_token_ids, pad_token_ids):
    pad_set = {int(x) for x in pad_token_ids if x is not None}
    first_eos = first_eos_index(token_ids, eos_token_ids)
    end = len(token_ids) if first_eos is None else first_eos
    valid_indices = [
        i for i, token_id in enumerate(token_ids[:end]) if int(token_id) not in pad_set
    ]
    return valid_indices, first_eos


def _text_before_first_eos(text, eos_token_strings):
    return truncate_text_at_first_eos(text, eos_token_strings, include_eos=False)


def compute_rollout_metrics(
    token_ids,
    step_map,
    text,
    eos_token_ids=(),
    pad_token_ids=(),
    eos_token_strings=(),
    max_gen_length=None,
    tail_tokens=256,
    tail_repetition_threshold=0.3,
):
    valid_indices, first_eos = _valid_generated_indices(
        token_ids, eos_token_ids, pad_token_ids
    )
    valid_tokens = [int(token_ids[i]) for i in valid_indices]
    valid_steps = [
        int(step_map[i])
        for i in valid_indices
        if i < len(step_map) and int(step_map[i]) > 0
    ]
    denoise_steps = len(set(valid_steps))
    response_len = len(valid_tokens)
    tpf = response_len / denoise_steps if denoise_steps else 0.0

    first_eos_seen = first_eos is not None
    pad_set = {int(x) for x in pad_token_ids if x is not None}
    eos_set = {int(x) for x in eos_token_ids if x is not None}
    eos_then_continues = False
    if first_eos_seen:
        eos_then_continues = any(
            int(token_id) not in pad_set and int(token_id) not in eos_set
            for token_id in token_ids[first_eos + 1 :]
        )

    overlong = False
    if max_gen_length is not None:
        overlong = (not first_eos_seen) and response_len >= int(max_gen_length)

    tail = valid_tokens[-tail_tokens:]
    rep4 = repeated_ngram_rate_from_tokens(valid_tokens, n=4)
    rep4_tail = repeated_ngram_rate_from_tokens(tail, n=4)
    text_for_lines = _text_before_first_eos(text, eos_token_strings)
    line_rep = repeated_line_ratio(text_for_lines)

    metrics = {
        "response_len": float(response_len),
        "overlong": float(overlong),
        "eos_then_continues": float(eos_then_continues),
        "eos_first": float(first_eos == 0),
        "missing_eos": float(not first_eos_seen),
        "first_eos_index": float(first_eos) if first_eos is not None else 0.0,
        "tpf": float(tpf),
        "decoded_tokens_per_step": float(tpf),
        "denoise_steps": float(denoise_steps),
        "rep4": float(rep4),
        "rep4_tail": float(rep4_tail),
        "repeated_line_ratio": float(line_rep),
        "tail_rep_gt_threshold": float(rep4_tail > tail_repetition_threshold),
    }
    _validate_rollout_metrics(metrics)
    return metrics


def _percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * pct
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(ordered[lo])
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def _as_metric_list(items):
    metrics = []
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get("rollout_metrics")
        if isinstance(value, dict):
            metrics.append(value)
        elif isinstance(value, list):
            metrics.extend(x for x in value if isinstance(x, dict))
    return metrics


def aggregate_rollout_metrics(items):
    rows = _as_metric_list(items)
    if not rows:
        return {}

    def col(name):
        return [float(row.get(name, 0.0)) for row in rows]

    response_lens = col("response_len")
    tpf = col("tpf")
    denoise_steps = col("denoise_steps")
    decoded_tokens_per_step = col("decoded_tokens_per_step")
    rep4 = col("rep4")
    rep4_tail = col("rep4_tail")
    line_rep = col("repeated_line_ratio")
    first_eos_indices = [
        float(row.get("first_eos_index", 0.0))
        for row in rows
        if float(row.get("missing_eos", 0.0)) <= 0.0
    ]

    metrics = {
        "rollout/tpf_mean": _mean(tpf),
        "rollout/tpf_median": float(median(tpf)) if tpf else 0.0,
        "rollout/tpf_p10": _percentile(tpf, 0.10),
        "rollout/tpf_p90": _percentile(tpf, 0.90),
        "rollout/decoded_tokens_per_step_mean": _mean(decoded_tokens_per_step),
        "rollout/denoise_steps_mean": _mean(denoise_steps),
        "rollout/response_len_mean": _mean(response_lens),
        "rollout/response_len_median": float(median(response_lens))
        if response_lens
        else 0.0,
        "rollout/response_len_p90": _percentile(response_lens, 0.90),
        "rollout/response_len_p95": _percentile(response_lens, 0.95),
        "rollout/response_len_max": max(response_lens) if response_lens else 0.0,
        "rollout/overlong_rate": _mean(col("overlong")),
        "rollout/eos_then_continues_rate": _mean(col("eos_then_continues")),
        "rollout/eos_first_rate": _mean(col("eos_first")),
        "rollout/missing_eos_rate": _mean(col("missing_eos")),
        "rollout/mean_first_eos_index": _mean(first_eos_indices),
        "rollout/rep4_mean": _mean(rep4),
        "rollout/rep4_tail_mean": _mean(rep4_tail),
        "rollout/repeated_line_ratio": _mean(line_rep),
        "rollout/tail_rep_gt_0.3": _mean(col("tail_rep_gt_threshold")),
    }
    for key, value in metrics.items():
        if key.endswith("_rate") or "rep" in key or key.endswith("_gt_0.3"):
            metrics[key] = min(max(float(value), 0.0), 1.0)
        else:
            metrics[key] = float(value) if math.isfinite(float(value)) else 0.0
    return metrics


def _validate_rollout_metrics(metrics):
    for key in ("rep4", "rep4_tail", "repeated_line_ratio", "tail_rep_gt_threshold"):
        value = float(metrics.get(key, 0.0))
        if value < -1e-6 or value > 1.0 + 1e-6:
            raise ValueError(f"{key} out of range: {value}")
    tpf = float(metrics.get("tpf", 0.0))
    if tpf < 0.0 or not math.isfinite(tpf):
        raise ValueError(f"invalid TPF: {tpf}")


def format_metrics(metrics):
    return "  ".join(f"{key}: {value}" for key, value in metrics.items())
