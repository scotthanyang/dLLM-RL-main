from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from jinja2 import Template
from omegaconf import ListConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_eos import as_bool, resolve_eos_token_ids
from dream import DreamTokenizer
from dream.modeling_dream import DreamModel
from dream.generation_utils_block import DreamGenerationConfig
from dream_rl_rollout import sample_tokens, top_k_logits, top_p_logits


SCRIPT_DIR = Path(__file__).resolve().parent
DLM_RL_DIR = SCRIPT_DIR.parent


MATH_PROMPT = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
You need to put your final answer in \\boxed{}. This is the problem:
{{problem}}<|im_end|>
<|im_start|>assistant
"""

CODE_FUNCTION_PROMPT = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{{problem}}
Place your code within a single Python code block ```python ```. Do not include more than one code block. <|im_end|>
<|im_start|>assistant
"""

CODE_STDIO_PROMPT = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
This is the problem:
{{problem}}
You should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>
<|im_start|>assistant
"""


class TraceWriter:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.output_path.open("w", encoding="utf-8")

    def write(self, text: str = "") -> None:
        print(text, flush=True)
        self._fh.write(text + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


def first_scalar(value):
    if isinstance(value, ListConfig):
        if len(value) != 1:
            raise ValueError(f"Expected a single value, got {list(value)}")
        return value[0]
    return value


def copy_eval_config_to_rollout(config) -> None:
    fields = [
        "steps",
        "max_gen_length",
        "temperature",
        "top_p",
        "top_k",
        "remasking_strategy",
        "dynamic_threshold",
        "target",
        "block_size",
        "use_cache",
        "further_horizon",
        "pad_target_penalty",
        "alg_temp",
    ]
    for field in fields:
        value = OmegaConf.select(config, f"evaluation.{field}", default=None)
        if value is not None:
            setattr(config.rollout, field, first_scalar(value))


def apply_trace_decode_config(config) -> str:
    source = str(
        OmegaConf.select(config, "trace.decode_config_source", default="evaluation")
    ).lower()
    if source == "evaluation":
        copy_eval_config_to_rollout(config)
    elif source == "rollout":
        pass
    else:
        raise ValueError(
            f"Unsupported trace.decode_config_source={source!r}; expected 'evaluation' or 'rollout'"
        )
    return source


def resolve_checkpoint_path(config) -> str:
    override = OmegaConf.select(config, "trace.checkpoint_path", default=None)
    if override:
        return str(override)

    if int(config.experiment.current_epoch) == 1:
        return str(config.model.pretrained_model)
    return str(DLM_RL_DIR / config.experiment.project / "ckpt" / config.model.optimized_name)


def resolve_prompt_dataset(config) -> tuple[str, str, str]:
    source = str(
        OmegaConf.select(config, "trace.prompt_dataset_source", default="evaluation")
    ).lower()
    if source in {"evaluation", "eval"}:
        dataset = str(config.evaluation.eval_dataset)
        data_type = str(config.evaluation.data_type)
        return dataset, data_type, "evaluation"
    if source in {"rollout", "train", "training"}:
        dataset = OmegaConf.select(config, "dataset.train_dataset", default=None)
        if dataset is None:
            raise ValueError("trace.prompt_dataset_source=rollout requires dataset.train_dataset")
        data_type = OmegaConf.select(
            config,
            "dataset.data_type",
            default=OmegaConf.select(config, "evaluation.data_type", default="math"),
        )
        return str(dataset), str(data_type), "rollout"
    raise ValueError(
        f"Unsupported trace.prompt_dataset_source={source!r}; expected 'evaluation' or 'rollout'"
    )


def load_trace_prompt(config) -> tuple[dict, str, int, str, str]:
    dataset, data_type, dataset_source = resolve_prompt_dataset(config)
    data_path = DLM_RL_DIR / "data" / f"{dataset}.json"
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    example_idx = int(OmegaConf.select(config, "trace.example_idx", default=0))
    if example_idx < 0 or example_idx >= len(data):
        raise IndexError(f"trace.example_idx={example_idx} is outside dataset size {len(data)}")

    item = data[example_idx]
    if data_type == "code":
        if item.get("test_method") == "stdio":
            prompt_template = CODE_STDIO_PROMPT
        else:
            prompt_template = CODE_FUNCTION_PROMPT + item.get("prefix", "")
    else:
        prompt_template = MATH_PROMPT
    return item, Template(prompt_template).render(problem=item["question"]), example_idx, dataset, dataset_source


def apply_sampling_filters(logits, temperature: float, top_p: Optional[float], top_k: Optional[int]):
    logits = logits.float()
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, int(top_k))
    return logits


def top_predictions(logits, temperature: float, top_p: Optional[float], top_k: Optional[int]):
    filtered = apply_sampling_filters(logits, temperature, top_p, top_k)
    probs = F.softmax(filtered, dim=-1)
    confidence, token_ids = probs.max(dim=-1)
    return token_ids, confidence


def visible_token(tokenizer, token_id: int, mask_id: int, pad_id: int, width: int) -> str:
    if int(token_id) == int(mask_id):
        text = "<MASK>"
    elif int(token_id) == int(pad_id):
        text = "<PAD>"
    else:
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        text = text.replace("\n", "\\n").replace("\t", "\\t")
        if text == "":
            text = "<EMPTY>"
    if len(text) > width:
        text = text[: max(1, width - 1)] + "…"
    return text


def format_cell(text: str, width: int) -> str:
    return text.center(width)


def emit_step_trace(
    writer: TraceWriter,
    *,
    tokenizer,
    x: torch.Tensor,
    logits: torch.Tensor,
    logits_global_start: int,
    selected_global: torch.Tensor,
    prompt_len: int,
    current_block_start: int,
    current_block_end: int,
    block_length: int,
    max_length: int,
    mask_id: int,
    pad_id: int,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
    global_step: int,
    block_idx: int,
    block_step: int,
    phase: str,
    columns_per_row: int,
    token_width: int,
) -> None:
    display_start = current_block_start
    display_end = min(current_block_end + block_length, max_length, logits_global_start + logits.shape[1])
    local_start = display_start - logits_global_start
    local_end = display_end - logits_global_start
    if local_start < 0 or local_end > logits.shape[1]:
        raise ValueError(
            f"Trace window [{display_start}, {display_end}) is outside logits range "
            f"[{logits_global_start}, {logits_global_start + logits.shape[1]})"
        )

    pred_ids, confidences = top_predictions(
        logits[:, local_start:local_end],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    pred_ids = pred_ids[0].detach().cpu().tolist()
    confidences = confidences[0].detach().float().cpu().tolist()
    current_ids = x[0, display_start:display_end].detach().cpu().tolist()
    selected = selected_global[0, display_start:display_end].detach().cpu().tolist()

    decoded_count = int((x[0, current_block_start:current_block_end] != mask_id).sum().item())
    writer.write(
        f"\n=== decode step {global_step} | block {block_idx} | block_step {block_step} | {phase} ==="
    )
    writer.write(
        f"current generated block: {current_block_start - prompt_len}..{current_block_end - prompt_len - 1} | "
        f"printed generated window: {display_start - prompt_len}..{display_end - prompt_len - 1} | "
        f"decoded in current block before update: {decoded_count}/{block_length}"
    )

    for chunk_start in range(0, len(pred_ids), columns_per_row):
        chunk_end = min(chunk_start + columns_per_row, len(pred_ids))
        rel_positions = [
            str(display_start + offset - prompt_len)
            for offset in range(chunk_start, chunk_end)
        ]
        states = []
        top_tokens = []
        confs = []
        current_tokens = []
        for offset in range(chunk_start, chunk_end):
            is_decoded = int(current_ids[offset]) != int(mask_id)
            is_selected = bool(selected[offset])
            if is_selected and not is_decoded:
                state = "SELECT"
            elif is_decoded:
                state = "DECODED"
            else:
                state = "MASK"
            states.append(state)
            top_tokens.append(visible_token(tokenizer, pred_ids[offset], mask_id, pad_id, token_width))
            confs.append(f"{confidences[offset]:.3f}")
            current_tokens.append(visible_token(tokenizer, current_ids[offset], mask_id, pad_id, token_width))

        writer.write("pos   " + " ".join(format_cell(v, token_width) for v in rel_positions))
        writer.write("state " + " ".join(format_cell(v, token_width) for v in states))
        writer.write("top   " + " ".join(format_cell(v, token_width) for v in top_tokens))
        writer.write("conf  " + " ".join(format_cell(v, token_width) for v in confs))
        writer.write("cur   " + " ".join(format_cell(v, token_width) for v in current_tokens))


@torch.no_grad()
def traced_dream_sample(
    model,
    tokenizer,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor],
    generation_config: DreamGenerationConfig,
    writer: TraceWriter,
    *,
    block_length: int,
    use_cache: bool,
    further_horizon: Optional[int],
    mask_token_id: int,
    eos_token_ids: tuple[int, ...],
    pad_token_id: int,
    pad_target_penalty: float,
    unmask_threshold: Optional[float],
    remasking_strategy: str,
    enforce_eos_stop: bool,
    force_after_first_eos_to_eos: bool,
    suppress_eos_at_first_token: bool,
    columns_per_row: int,
    token_width: int,
    max_blocks: Optional[int],
    max_steps_per_block: Optional[int],
) -> torch.LongTensor:
    max_length = int(generation_config.max_length)
    steps = int(generation_config.steps)
    temperature = float(generation_config.temperature)
    top_p = generation_config.top_p
    top_k = generation_config.top_k
    tar = generation_config.tar
    alg_temp = generation_config.alg_temp
    cgws = further_horizon

    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
    prompt_len = input_ids.shape[1]
    gen_length = max_length - input_ids.shape[1]
    batch_size = input_ids.shape[0]
    if batch_size != 1:
        raise ValueError("dream_decode_trace.py traces exactly one prompt at a time")
    if gen_length % block_length != 0:
        raise ValueError(f"gen_length ({gen_length}) must be divisible by block_length ({block_length})")

    num_blocks = gen_length // block_length
    if max_blocks is not None:
        num_blocks = min(num_blocks, int(max_blocks))

    base, rem = divmod(steps, gen_length // block_length)
    steps_per_block = [base + (1 if i < rem else 0) for i in range(gen_length // block_length)]
    timesteps = [
        torch.linspace(1, generation_config.eps, spb + 1, device=x.device)
        for spb in steps_per_block
    ]

    first_eos_indices = torch.full((batch_size,), -1, dtype=torch.long, device=x.device)
    eos_then_continues = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

    if attention_mask is not None and torch.any(attention_mask == 0.0):
        attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
        attention_mask = torch.where(
            attention_mask,
            torch.tensor(0.0, device=attention_mask.device),
            torch.tensor(float("-inf"), device=attention_mask.device),
        )
    else:
        tok_idx = None
        attention_mask = "full"

    def attention_is_full() -> bool:
        return isinstance(attention_mask, str) and attention_mask == "full"

    def tokens_in(values, token_ids):
        result = torch.zeros_like(values, dtype=torch.bool)
        for token_id in token_ids:
            result |= values == int(token_id)
        return result

    def suppress_first_token_eos(logits_tensor, global_position):
        if not suppress_eos_at_first_token or global_position != prompt_len:
            return logits_tensor
        logits_tensor = logits_tensor.clone()
        for token_id in eos_token_ids:
            logits_tensor[:, global_position, int(token_id)] = torch.finfo(logits_tensor.dtype).min
        return logits_tensor

    def apply_eos_stop():
        if not enforce_eos_stop or not eos_token_ids:
            return
        gen = x[:, prompt_len:]
        eos_map = tokens_in(gen, eos_token_ids)
        has_eos = eos_map.any(dim=1)
        if not has_eos.any():
            return

        first_positions = torch.argmax(eos_map.long(), dim=1)
        for row in torch.nonzero(has_eos, as_tuple=False).flatten().tolist():
            first_pos = int(first_positions[row].item())
            previous = int(first_eos_indices[row].item())
            if previous < 0 or first_pos < previous:
                first_eos_indices[row] = first_pos

            tail = gen[row, first_pos + 1 :]
            if tail.numel() == 0:
                continue
            tail_content = (
                (tail != pad_token_id)
                & (tail != mask_token_id)
                & ~tokens_in(tail, eos_token_ids)
            )
            if tail_content.any():
                eos_then_continues[row] = True
            fill_token_id = int(gen[row, first_pos].item()) if force_after_first_eos_to_eos else int(pad_token_id)
            tail.fill_(fill_token_id)

    global_step = 0
    for num_block in range(num_blocks):
        if enforce_eos_stop and bool((first_eos_indices >= 0).all().item()):
            break

        current_block_start = prompt_len + num_block * block_length
        current_block_end = current_block_start + block_length
        window_end = min(current_block_end + cgws, max_length) if cgws is not None else max_length
        window_slice = slice(current_block_start, window_end)

        if use_cache:
            model_output = model(x, attention_mask, tok_idx, use_cache=True)
            past_key_values = model_output.past_key_values
            new_past_key_values = []
            for i in range(len(past_key_values)):
                new_past_key_values.append(())
                for j in range(len(past_key_values[i])):
                    new_past_key_values[i] += (past_key_values[i][j][:, :current_block_start, :],)
            past_key_values = new_past_key_values
        else:
            model_output = model(x, attention_mask, tok_idx, use_cache=False)
            past_key_values = None

        logits = model_output.logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        logits = suppress_first_token_eos(logits, current_block_start)
        _, x0_all = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)

        selected_global = torch.zeros_like(x, dtype=torch.bool)
        active_rows = first_eos_indices < 0 if enforce_eos_stop else torch.ones(batch_size, dtype=torch.bool, device=x.device)
        selected_global[active_rows, current_block_start] = True
        emit_step_trace(
            writer,
            tokenizer=tokenizer,
            x=x,
            logits=logits,
            logits_global_start=0,
            selected_global=selected_global,
            prompt_len=prompt_len,
            current_block_start=current_block_start,
            current_block_end=current_block_end,
            block_length=block_length,
            max_length=max_length,
            mask_id=mask_token_id,
            pad_id=pad_token_id,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            global_step=global_step,
            block_idx=num_block,
            block_step=0,
            phase="initial block token",
            columns_per_row=columns_per_row,
            token_width=token_width,
        )
        global_step += 1

        x[active_rows, current_block_start] = x0_all[active_rows, current_block_start]
        apply_eos_stop()

        spb = steps_per_block[num_block]
        block_step = 1
        while True:
            if max_steps_per_block is not None and block_step > int(max_steps_per_block):
                writer.write(f"Stopping trace in block {num_block}: trace.max_steps_per_block reached")
                break

            if cgws is not None:
                mask_index = x[:, window_slice] == mask_token_id
                logits_global_start = current_block_start
            else:
                mask_index = x[:, current_block_start:] == mask_token_id
                logits_global_start = current_block_start

            if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                break

            if not attention_is_full():
                if cgws is not None:
                    current_attention_mask = attention_mask[:, :, window_slice, :window_end]
                else:
                    current_attention_mask = attention_mask[:, :, current_block_start:, :]
            else:
                current_attention_mask = attention_mask

            if use_cache:
                if cgws is not None:
                    model_output = model(
                        x[:, window_slice],
                        current_attention_mask,
                        tok_idx[:, window_slice] if tok_idx is not None else None,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                else:
                    model_output = model(
                        x[:, current_block_start:],
                        current_attention_mask,
                        tok_idx[:, current_block_start:] if tok_idx is not None else None,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                logits = model_output.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            else:
                model_output = model(x, attention_mask, tok_idx, use_cache=False)
                logits = model_output.logits[:, current_block_start:]
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            mask_index[:, block_length:] = False
            mask_logits = logits[mask_index]
            if suppress_eos_at_first_token and eos_token_ids:
                global_positions = (
                    torch.arange(current_block_start, window_end, device=x.device)
                    if cgws is not None
                    else torch.arange(current_block_start, max_length, device=x.device)
                )
                first_token_mask = mask_index & (global_positions.unsqueeze(0) == prompt_len)
                flat_first_token_mask = first_token_mask[mask_index]
                if flat_first_token_mask.any():
                    mask_logits = mask_logits.clone()
                    first_rows = torch.nonzero(flat_first_token_mask, as_tuple=False).flatten()
                    for token_id in eos_token_ids:
                        mask_logits[first_rows, int(token_id)] = torch.finfo(mask_logits.dtype).min

            target, x0 = sample_tokens(
                mask_logits,
                temperature,
                top_p=top_p,
                top_k=top_k,
                tar=tar,
            )
            raw_top1 = torch.argmax(mask_logits, dim=-1)

            pad_mask_flat = x0 == pad_token_id
            if pad_mask_flat.any():
                target = target.clone()
                target[pad_mask_flat] = target[pad_mask_flat] / pad_target_penalty

            xwin = x[:, window_slice] if cgws is not None else x[:, current_block_start:]
            full_target = torch.full_like(xwin, -torch.inf, dtype=logits.dtype, device=x.device).float()
            full_target[mask_index] = target
            full_target[:, block_length:] = -torch.inf

            selected_map = torch.zeros_like(xwin, dtype=torch.bool)
            if remasking_strategy in {"AR", "AR_top1"}:
                has_mask = mask_index.any(dim=-1)
                if has_mask.any():
                    leftmost_idx = torch.argmax(mask_index.long(), dim=-1)
                    selected_rows = torch.nonzero(has_mask, as_tuple=False).squeeze(-1)
                    selected_map[selected_rows, leftmost_idx[selected_rows]] = True

                selected_map &= mask_index
                x_candidates = torch.full_like(xwin, mask_token_id, dtype=torch.long)
                x_candidates[mask_index] = x0

                if remasking_strategy == "AR_top1":
                    raw_top1_candidates = torch.full_like(xwin, -1, dtype=torch.long)
                    raw_top1_candidates[mask_index] = raw_top1
                    forced_is_top1 = torch.zeros(xwin.shape[0], dtype=torch.bool, device=x.device)
                    if has_mask.any():
                        forced_is_top1[selected_rows] = (
                            x_candidates[selected_rows, leftmost_idx[selected_rows]]
                            == raw_top1_candidates[selected_rows, leftmost_idx[selected_rows]]
                        )

                    threshold_map = torch.zeros_like(xwin, dtype=torch.bool)
                    threshold_map[mask_index] = target >= unmask_threshold
                    threshold_map &= forced_is_top1.unsqueeze(1)
                    threshold_map &= ~selected_map
                    selected_map |= threshold_map
            elif unmask_threshold is None:
                num_mask_token = mask_index.sum() / mask_index.shape[0]
                t = timesteps[num_block][block_step]
                s = timesteps[num_block][block_step + 1]
                number_transfer_tokens = int(num_mask_token * (1 - s / t)) if block_step < spb - 1 else int(num_mask_token)
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_target, number_transfer_tokens)
                    else:
                        sample_scores = F.softmax(full_target / alg_temp, dim=-1)
                        transfer_index = torch.multinomial(sample_scores, num_samples=number_transfer_tokens)
                    row_indices = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(transfer_index)
                    selected_map[row_indices, transfer_index] = True
                x_candidates = torch.full_like(xwin, mask_token_id, dtype=torch.long)
                x_candidates[mask_index] = x0
            else:
                selected_map[mask_index] = target >= unmask_threshold
                no_sel = ~selected_map.any(dim=-1)
                no_sel = no_sel & mask_index.any(dim=-1)
                if no_sel.any():
                    masked_scores = full_target.masked_fill(~mask_index, float("-inf"))
                    best_idx = torch.argmax(masked_scores, dim=-1)
                    selected_rows = torch.nonzero(no_sel, as_tuple=False).squeeze(-1)
                    selected_map[selected_rows, best_idx[selected_rows]] = True
                selected_map &= mask_index
                x_candidates = torch.full_like(xwin, mask_token_id, dtype=torch.long)
                x_candidates[mask_index] = x0

            selected_global = torch.zeros_like(x, dtype=torch.bool)
            if cgws is not None:
                selected_global[:, window_slice] = selected_map
            else:
                selected_global[:, current_block_start:] = selected_map

            emit_step_trace(
                writer,
                tokenizer=tokenizer,
                x=x,
                logits=logits,
                logits_global_start=logits_global_start,
                selected_global=selected_global,
                prompt_len=prompt_len,
                current_block_start=current_block_start,
                current_block_end=current_block_end,
                block_length=block_length,
                max_length=max_length,
                mask_id=mask_token_id,
                pad_id=pad_token_id,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                global_step=global_step,
                block_idx=num_block,
                block_step=block_step,
                phase=remasking_strategy,
                columns_per_row=columns_per_row,
                token_width=token_width,
            )
            global_step += 1

            xwin[selected_map] = x_candidates[selected_map]
            apply_eos_stop()

            block_step += 1
            if block_step >= spb:
                # Match the normal sampler's final-step behavior without
                # walking beyond the precomputed timestep array.
                if (x[:, current_block_start:current_block_end] == mask_token_id).sum() != 0:
                    writer.write(f"Stopping trace in block {num_block}: configured steps_per_block exhausted")
                break

        block_all_pad = torch.all(x[:, current_block_start:current_block_end] == pad_token_id)
        if block_all_pad:
            if current_block_end < x.size(1):
                x[:, current_block_end:] = pad_token_id
            break

    return x


def main() -> None:
    config = get_config()
    if str(config.model.model_base) != "dream":
        raise ValueError("dream_decode_trace.py only supports model.model_base=dream")

    decode_config_source = apply_trace_decode_config(config)

    seed = int(OmegaConf.select(config, "trace.seed", default=10086))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    item, prompt, example_idx, prompt_dataset, prompt_dataset_source = load_trace_prompt(config)
    checkpoint_path = resolve_checkpoint_path(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = DreamTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = DreamModel.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()

    pad_id = int(model.config.pad_token_id)
    mask_id = int(model.config.mask_token_id)
    eos_token_names = OmegaConf.select(
        config,
        "rollout.eos_token_names",
        default=["<|im_end|>", "<|endoftext|>"],
    )
    eos_ids = tuple(resolve_eos_token_ids(tokenizer, eos_token_names))
    if not eos_ids:
        fallback = tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_ids = (int(fallback),)

    enc = tokenizer(
        [prompt],
        padding=True,
        return_tensors="pt",
        padding_side="left",
    )
    prompt_ids = enc["input_ids"].to(device)
    attn_mask = prompt_ids.ne(pad_id).to(device)

    if as_bool(config.rollout.use_cache, default=True) is False:
        config.rollout.further_horizon = None

    block_length = int(first_scalar(config.rollout.block_size))
    max_gen_length = int(config.rollout.max_gen_length)
    if max_gen_length % block_length != 0:
        raise ValueError(
            f"rollout.max_gen_length={max_gen_length} must be divisible by block_size={block_length}"
        )

    generation_config = DreamGenerationConfig(
        output_history=False,
        return_dict_in_generate=False,
        max_length=max_gen_length + prompt_ids.shape[1],
        steps=int(config.rollout.steps),
        temperature=float(config.rollout.temperature),
        top_p=float(config.rollout.top_p) if config.rollout.top_p is not None else None,
        top_k=int(config.rollout.top_k) if config.rollout.top_k is not None else None,
        tar=str(config.rollout.target),
        alg_temp=config.rollout.alg_temp,
    )

    remasking_strategy = str(first_scalar(config.rollout.remasking_strategy))
    unmask_threshold = None if remasking_strategy == "low_confidence_static" else float(config.rollout.dynamic_threshold)

    project_name = str(config.experiment.project)
    output_path = OmegaConf.select(config, "trace.output_path", default=None)
    if output_path is None:
        output_path = (
            DLM_RL_DIR
            / project_name
            / "temp_data"
            / f"decode_trace_{prompt_dataset}_example{example_idx}.txt"
        )
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = DLM_RL_DIR / output_path

    columns_per_row = int(OmegaConf.select(config, "trace.columns_per_row", default=8))
    token_width = int(OmegaConf.select(config, "trace.token_width", default=14))
    max_blocks = OmegaConf.select(config, "trace.max_blocks", default=None)
    max_steps_per_block = OmegaConf.select(config, "trace.max_steps_per_block", default=None)
    max_blocks = None if max_blocks in (None, "", "null") else int(max_blocks)
    max_steps_per_block = None if max_steps_per_block in (None, "", "null") else int(max_steps_per_block)

    with TraceWriter(Path(output_path)) as writer:
        writer.write("Dream decode trace")
        writer.write(f"checkpoint: {checkpoint_path}")
        writer.write(f"dataset: {prompt_dataset}")
        writer.write(f"dataset source: {prompt_dataset_source}")
        writer.write(f"example_idx: {example_idx}")
        writer.write(f"question: {item.get('question', '')}")
        writer.write(f"decode config source: {decode_config_source}")
        writer.write(
            "decode config: "
            f"steps={config.rollout.steps}, max_gen_length={config.rollout.max_gen_length}, "
            f"temperature={config.rollout.temperature}, top_p={config.rollout.top_p}, top_k={config.rollout.top_k}, "
            f"remasking_strategy={remasking_strategy}, dynamic_threshold={config.rollout.dynamic_threshold}, "
            f"block_size={block_length}, further_horizon={config.rollout.further_horizon}, use_cache={config.rollout.use_cache}"
        )

        sequences = traced_dream_sample(
            model,
            tokenizer,
            prompt_ids,
            attention_mask=attn_mask,
            generation_config=generation_config,
            writer=writer,
            block_length=block_length,
            use_cache=as_bool(config.rollout.use_cache, default=True),
            further_horizon=(
                None
                if config.rollout.further_horizon is None
                else int(config.rollout.further_horizon)
            ),
            mask_token_id=mask_id,
            eos_token_ids=eos_ids,
            pad_token_id=pad_id,
            pad_target_penalty=float(config.rollout.pad_target_penalty),
            unmask_threshold=unmask_threshold,
            remasking_strategy=remasking_strategy,
            enforce_eos_stop=as_bool(
                OmegaConf.select(config, "rollout.enforce_eos_stop", default=True),
                default=True,
            ),
            force_after_first_eos_to_eos=as_bool(
                OmegaConf.select(config, "rollout.force_after_first_eos_to_eos", default=True),
                default=True,
            ),
            suppress_eos_at_first_token=as_bool(
                OmegaConf.select(config, "rollout.suppress_eos_at_first_token", default=False),
                default=False,
            ),
            columns_per_row=columns_per_row,
            token_width=token_width,
            max_blocks=max_blocks,
            max_steps_per_block=max_steps_per_block,
        )

        generated_ids = sequences[0, prompt_ids.shape[1] :].detach().cpu().tolist()
        final_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        writer.write("\n=== final generated response ===")
        writer.write(final_text)
        writer.write(f"\ntrace saved to: {Path(output_path)}")


if __name__ == "__main__":
    main()
