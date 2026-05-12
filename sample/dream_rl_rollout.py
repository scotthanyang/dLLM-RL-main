from __future__ import annotations
import math, json, os, time
import sys
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from jinja2 import Template
import torch
from termcolor import cprint
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModel
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_metrics import compute_rollout_metrics
from rl_eos import (
    as_bool,
    eos_response_metadata,
    resolve_eos_token_ids,
    resolve_eos_token_strings,
)


from dream import DreamTokenizer
from dream.modeling_dream import DreamModel
from dream.generation_utils_block import DreamGenerationMixin
import types
from dream.generation_utils_block import DreamGenerationConfig


from transformers.utils import ModelOutput
from typing import Any, Dict, Optional, Tuple, Union
import torch.distributions as dists
from dataclasses import dataclass
from torch.nn import functional as F
import torch


from omegaconf import DictConfig, ListConfig, OmegaConf
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, tar=None):

    logits = logits.float()

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    
    dist = dists.Categorical(logits=logits)
    x0 = dist.sample()
    probs = dist.probs

    if temperature > 0:
        target = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        target, x0 = probs.max(dim=-1)
    
    if tar == "confidence":
        return target, x0
    
    if tar == "margin_confidence":
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0]
        
        top2_probs = sorted_probs[:, 1]
        # Calculate confidence as top1 - top2
        target = top1_probs - top2_probs 
    
    if tar == "neg_entropy":
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        target = torch.sum(probs * log_probs, dim=-1)
    
    return target, x0



@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None
    first_eos_indices: Optional[torch.LongTensor] = None
    eos_then_continues: Optional[torch.BoolTensor] = None


@torch.no_grad()
def _sample(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor],
    generation_config: DreamGenerationConfig,
    block_length: Optional[int] = 32,
    use_cache: bool = False,
    further_horizon: int = 128,
    mask_token_id: int = 151666,
    eos_token_id: int = 151645,
    eos_token_ids: Optional[Tuple[int, ...]] = None,
    pad_token_id: int = 151643,
    pad_target_penalty: float = 1.0,
    unmask_threshold: float = 0.9,
    remasking_strategy: Union[str, ListConfig] = "low_confidence_dynamic",
    enforce_eos_stop: bool = True,
    force_after_first_eos_to_eos: bool = True,
    suppress_eos_at_first_token: bool = False,
) -> Union[DreamModelOutput, torch.LongTensor]:
    # init values
    
    output_history = generation_config.output_history
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    steps = generation_config.steps
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k
    tar = generation_config.tar
    alg_temp = generation_config.alg_temp
    cgws = further_horizon
    if isinstance(remasking_strategy, ListConfig):
        if len(remasking_strategy) != 1:
            raise ValueError(
                "Dream rollout currently expects a single remasking_strategy value"
            )
        remasking_strategy = remasking_strategy[0]
    remasking_strategy = str(remasking_strategy)
    eos_ids = tuple(
        sorted(
            {
                int(token_id)
                for token_id in (eos_token_ids or (eos_token_id,))
                if token_id is not None and int(token_id) >= 0
            }
        )
    )


    histories = [] if (return_dict_in_generate and output_history) else None

    # pad input_ids to max_length
    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
    prompt_len = input_ids.shape[1]
    gen_length = max_length - input_ids.shape[1]
    batch_size = input_ids.shape[0]
    first_eos_indices = torch.full(
        (batch_size,), -1, dtype=torch.long, device=input_ids.device
    )
    eos_then_continues = torch.zeros(
        batch_size, dtype=torch.bool, device=input_ids.device
    )
    
    # Handle block configuration
    if block_length is None:
        block_length = gen_length  # Default: single block (original behavior)
    
    assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
    num_blocks = gen_length // block_length
    
    base, rem = divmod(steps, num_blocks)
    steps_per_block = [base + (1 if i < rem else 0) for i in range(num_blocks)]
    timesteps = [
        torch.linspace(1, generation_config.eps, spb + 1, device=x.device)
        for spb in steps_per_block
    ]

    if attention_mask is not None and torch.any(attention_mask == 0.0):
        # we do not mask the [MASK] tokens so value = 1.0
        attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        # attention_mask is of shape [B, N]
        # broadcast to [B, 1, N, N]
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
        attention_mask = torch.where(attention_mask, torch.tensor(0.0, device=attention_mask.device), torch.tensor(float("-inf"), device=attention_mask.device))
    else:
        tok_idx = None
        attention_mask = "full"
    


    # Initialize cache for the prompt
    past_key_values = None

    def tokens_in(values, token_ids):
        result = torch.zeros_like(values, dtype=torch.bool)
        for token_id in token_ids:
            result |= values == int(token_id)
        return result

    def suppress_first_token_eos(logits_tensor, global_position):
        if not suppress_eos_at_first_token or global_position != prompt_len:
            return logits_tensor
        if not eos_ids:
            return logits_tensor
        logits_tensor = logits_tensor.clone()
        for token_id in eos_ids:
            logits_tensor[:, global_position, int(token_id)] = torch.finfo(logits_tensor.dtype).min
        return logits_tensor

    def apply_eos_stop():
        if not enforce_eos_stop or not eos_ids:
            return
        gen = x[:, prompt_len:]
        eos_map = tokens_in(gen, eos_ids)
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
            tail_content = (tail != pad_token_id) & (tail != mask_token_id) & ~tokens_in(tail, eos_ids)
            if tail_content.any():
                eos_then_continues[row] = True
            if force_after_first_eos_to_eos:
                fill_token_id = int(gen[row, first_pos].item())
            else:
                fill_token_id = int(pad_token_id)
            tail.fill_(fill_token_id)

    # Process each block
    for num_block in range(num_blocks):
        if enforce_eos_stop and bool((first_eos_indices >= 0).all().item()):
            break
        
        current_block_start = input_ids.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        if cgws is not None:
            window_end  = max_length if cgws is None else min(current_block_end + cgws, max_length)
            window_slice = slice(current_block_start, window_end)

        # update cache
        if use_cache:
            model_output = model(x, attention_mask, tok_idx, use_cache=True)
            past_key_values = model_output.past_key_values
            # Extract only previous block cache
            new_past_key_values = []
            for i in range(len(past_key_values)):
                new_past_key_values.append(())
                for j in range(len(past_key_values[i])):
                    new_past_key_values[i] += (past_key_values[i][j][:, :current_block_start, :],)
            past_key_values = new_past_key_values

        else:
            model_output = model(x, attention_mask, tok_idx, use_cache=False)
        
        logits = model_output.logits
        logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
        logits = suppress_first_token_eos(logits, current_block_start)
        _, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        active_rows = first_eos_indices < 0 if enforce_eos_stop else torch.ones(batch_size, dtype=torch.bool, device=x.device)
        x[active_rows, current_block_start] = x0[active_rows, current_block_start]
        apply_eos_stop()
        if histories is not None:
            histories.append(x.clone().cpu())
        
        
        
        spb = steps_per_block[num_block]
        i = 1
        while True:
            
            
            if cgws is not None:
                mask_index = (x[:, window_slice] == mask_token_id)
            else:
                mask_index = (x[:, current_block_start:] == mask_token_id)
            
            
            # Prepare attention mask for cached generation
            if attention_mask != "full":
                # Adjust attention mask for current position
                if cgws is not None:
                    current_attention_mask = attention_mask[:, :, window_slice, :window_end]
                else:
                    current_attention_mask = attention_mask[:, :, current_block_start:, :]
            else:
                current_attention_mask = attention_mask
            
            if use_cache:
                if cgws is not None:
                    model_output = model(x[:, window_slice], current_attention_mask, 
                                    tok_idx[:, window_slice] if tok_idx is not None else None, 
                                    past_key_values=past_key_values, use_cache=True)
                else:
                    model_output = model(x[:, current_block_start:], current_attention_mask,
                                    tok_idx[:, current_block_start:] if tok_idx is not None else None, 
                                    past_key_values=past_key_values, use_cache=True)
                logits = model_output.logits
                logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            else:
                model_output = model(x, attention_mask, tok_idx, use_cache=False)
                logits = model_output.logits
                logits = logits[:, current_block_start:]
                logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            
            if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                break
            
            
            mask_index[:, block_length:] = False
            mask_logits = logits[mask_index]
            if suppress_eos_at_first_token and eos_ids:
                if cgws is not None:
                    global_positions = torch.arange(current_block_start, window_end, device=x.device)
                else:
                    global_positions = torch.arange(current_block_start, max_length, device=x.device)
                first_token_mask = mask_index & (global_positions.unsqueeze(0) == prompt_len)
                flat_first_token_mask = first_token_mask[mask_index]
                if flat_first_token_mask.any():
                    mask_logits = mask_logits.clone()
                    first_rows = torch.nonzero(flat_first_token_mask, as_tuple=False).flatten()
                    for token_id in eos_ids:
                        mask_logits[first_rows, int(token_id)] = torch.finfo(mask_logits.dtype).min
            target, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, tar=tar)
            raw_top1 = torch.argmax(mask_logits, dim=-1)

            # —— pad token penalty ——
            _pad_target_divisor = pad_target_penalty
            _pad_mask_flat = (x0 == pad_token_id)  
            if _pad_mask_flat.any():
                target = target.clone()
                target[_pad_mask_flat] = target[_pad_mask_flat] / _pad_target_divisor

            if cgws is not None:
                full_target = torch.full_like(x[:, window_slice], -torch.inf, device=model.device, dtype=logits.dtype)
            else:
                full_target = torch.full_like(x[:, current_block_start:], -torch.inf, device=model.device, dtype=logits.dtype)
            full_target = full_target.float()
            full_target[mask_index] = target
            full_target[:, block_length:] = -torch.inf

            if remasking_strategy in {"AR", "AR_top1"}:
                if cgws is not None:
                    xwin = x[:, window_slice]
                else:
                    xwin = x[:, current_block_start:]

                selected_map = torch.zeros_like(xwin, dtype=torch.bool)
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

                    forced_is_top1 = torch.zeros(xwin.shape[0], dtype=torch.bool, device=xwin.device)
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

                xwin[selected_map] = x_candidates[selected_map]
            elif unmask_threshold is None:

                num_mask_token = mask_index.sum() / mask_index.shape[0]
                t = timesteps[num_block][i]
                s = timesteps[num_block][i + 1]
                number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < spb - 1 else int(num_mask_token)
                
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_target, number_transfer_tokens)
                    else:
                        full_target = full_target / alg_temp
                        full_target = F.softmax(full_target, dim=-1)
                        transfer_index = torch.multinomial(full_target, num_samples=number_transfer_tokens)
                    
                    if cgws is not None:
                        x_ = torch.zeros_like(x[:, window_slice], device=model.device, dtype=torch.long) + mask_token_id
                    else:
                        x_ = torch.zeros_like(x[:, current_block_start:], device=model.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    row_indices = torch.arange(x.size(0), device=model.device).unsqueeze(1).expand_as(transfer_index)

                    
                    
                    if cgws is not None:
                        x[:, window_slice][row_indices,transfer_index] = x_[row_indices,transfer_index]
                    else:
                        x[:, current_block_start:][row_indices,transfer_index] = x_[row_indices,transfer_index]
                    
                
                    
            else:
                if cgws is not None:
                    xwin = x[:, window_slice]
                else:
                    xwin = x[:, current_block_start:]
                
                selected_map = torch.zeros_like(xwin, dtype=torch.bool)
                selected_map[mask_index] = (target >= unmask_threshold)
                no_sel = ~selected_map.any(dim=-1)  # [B]
                no_sel = no_sel & mask_index.any(dim=-1)

                if no_sel.any():
                    masked_scores = full_target.masked_fill(~mask_index, float("-inf"))
                    best_idx = torch.argmax(masked_scores, dim=-1)
                    selected_rows = torch.nonzero(no_sel, as_tuple=False).squeeze(-1)
                    selected_map[selected_rows, best_idx[selected_rows]] = True

                selected_map &= mask_index
                x_candidates = torch.full_like(xwin, mask_token_id, dtype=torch.long)
                x_candidates[mask_index] = x0

                xwin[selected_map] = x_candidates[selected_map]

                
            
            apply_eos_stop()
            if histories is not None:
                histories.append(x.clone().cpu())

            i += 1

            if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                break
        
        block_all_pad = torch.all(
            x[:, current_block_start:current_block_end] == pad_token_id
        )
        if block_all_pad:
            if current_block_end < x.size(1):
                x[:, current_block_end:] = pad_token_id
            if histories is not None:
                histories.append(x.clone().cpu())
            break

    
    if return_dict_in_generate:
        return DreamModelOutput(
            sequences=x,
            history=histories,
            first_eos_indices=first_eos_indices,
            eos_then_continues=eos_then_continues,
        )
    else:
        return x










import random 
def random_select(data_list, random_k):
    data_list = random.sample(data_list, random_k)
    return data_list


# obtain prompt
def get_prompt(data_i):
    return Template(system_prompts).render(problem = data_i["question"])



def extract_final_boxed_answer(s: str):
    tag = r'\boxed{'
    start = s.rfind(tag)          # last \boxed{
    if start == -1:
        return "Can not extract the answer!"

    i = start + len(tag)
    depth = 1                    # we are already inside one '{'
    buf = []

    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:       # matching '}' for the opening \boxed{
                break
        buf.append(ch)
        i += 1

    return ''.join(buf) if depth == 0 else "Can not extract the answer!"




def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "We can not extract the code in the output. "
    return code_output



def denoise_step_map(history, mask_id: int, sample_idx: int = 0):
    L = history[0].shape[1]              
    step_map = torch.zeros(L, dtype=torch.long)
    prev = torch.full((L,), mask_id, dtype=torch.long)

    for t, snap in enumerate(history, start=0): 
        cur = snap[sample_idx]         
        changed = (prev == mask_id) & (cur != mask_id)
        step_map[changed] = t
        prev = cur
        if (step_map == 0).sum() == 0:      
            break
    return step_map



from tqdm import tqdm




def worker(pretrained_model, rank, prompts, orig_idx, seq_dict, step_dict, token_dict, eos_cont_dict, prompt_len_dict, batch_size, config):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # load model once
    model_gpu = (DreamModel.from_pretrained(pretrained_model,
                                  trust_remote_code=True,
                                  torch_dtype=torch.bfloat16)
                 .to(device)
                 .eval())
    model_gpu.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model_gpu)
    model_gpu._sample = types.MethodType(DreamGenerationMixin._sample, model_gpu)   
    tokenizer_gpu = DreamTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

    pad_id = model_gpu.config.pad_token_id
    mask_id = model_gpu.config.mask_token_id
    eos_token_names = OmegaConf.select(
        config, "rollout.eos_token_names", default=["<|im_end|>", "<|endoftext|>"]
    )
    eos_ids = resolve_eos_token_ids(tokenizer_gpu, eos_token_names)
    eos_id = eos_ids[0] if eos_ids else tokenizer_gpu.convert_tokens_to_ids("<|im_end|>")
    enforce_eos_stop = as_bool(
        OmegaConf.select(config, "rollout.enforce_eos_stop", default=True), default=True
    )
    suppress_eos_at_first_token = as_bool(
        OmegaConf.select(config, "rollout.suppress_eos_at_first_token", default=False),
        default=False,
    )
    force_after_first_eos_to_eos = as_bool(
        OmegaConf.select(config, "rollout.force_after_first_eos_to_eos", default=True),
        default=True,
    )

    # process in chunks of `batch_size`
    for start in tqdm(range(0, len(prompts), batch_size),
                      desc=f"GPU {rank}", position=rank, leave=True):
        batch_prompts = prompts[start:start+batch_size]
        batch_idxs    = orig_idx[start:start+batch_size]

        # tokenize & move to GPU
        enc = tokenizer_gpu(batch_prompts,
                            padding=True, #truncation=True,
                            return_tensors="pt", padding_side="left")
        prompt_ids = enc["input_ids"].to(device)

        attn_mask = prompt_ids.ne(pad_id)
        #attn_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
        attn_mask = attn_mask.to(device=model_gpu.device)

        if config.rollout.use_cache == False:
            config.rollout.further_horizon = None

        generation_config = DreamGenerationConfig(
            output_history=True,       
            return_dict_in_generate=True, 
            max_length=config.rollout.max_gen_length + prompt_ids.shape[1],      
            steps=config.rollout.steps,              
            temperature=config.rollout.temperature,         
            top_p=config.rollout.top_p,              
            top_k=config.rollout.top_k,        
            tar=config.rollout.target,       
            alg_temp=config.rollout.alg_temp,  
        )

        remasking_strategy = config.rollout.remasking_strategy
        if isinstance(remasking_strategy, ListConfig):
            if len(remasking_strategy) != 1:
                raise ValueError(
                    "Dream rollout currently expects a single remasking_strategy value"
                )
            remasking_strategy = remasking_strategy[0]
        remasking_strategy = str(remasking_strategy)

        if remasking_strategy == "low_confidence_static":
            unmask_threshold = None
        else:
            unmask_threshold = config.rollout.dynamic_threshold
        
        generation_ids = _sample(
            model_gpu,
            prompt_ids,
            attention_mask=attn_mask,
            generation_config=generation_config,
            block_length=config.rollout.block_size,
            use_cache=config.rollout.use_cache,
            further_horizon=config.rollout.further_horizon,
            mask_token_id = mask_id,
            eos_token_id = eos_id,
            eos_token_ids = tuple(eos_ids),
            pad_token_id = pad_id,
            pad_target_penalty = config.rollout.pad_target_penalty,
            unmask_threshold = unmask_threshold,
            remasking_strategy = remasking_strategy,
            enforce_eos_stop = enforce_eos_stop,
            force_after_first_eos_to_eos = force_after_first_eos_to_eos,
            suppress_eos_at_first_token = suppress_eos_at_first_token,
        )
        generation_ids.sequences = generation_ids.sequences.cpu()
        first_eos_indices = (
            generation_ids.first_eos_indices.cpu().tolist()
            if generation_ids.first_eos_indices is not None
            else [-1] * len(batch_idxs)
        )
        eos_then_continues = (
            generation_ids.eos_then_continues.cpu().tolist()
            if generation_ids.eos_then_continues is not None
            else [False] * len(batch_idxs)
        )
        torch.cuda.empty_cache()

        # decode
        seq_ids = generation_ids.sequences[:, prompt_ids.shape[1]:].tolist()
        texts   = tokenizer_gpu.batch_decode(
            seq_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)

        # compute and store step maps
        for i, idx in enumerate(batch_idxs):
            # extract step map for sample i in this batch
            m = denoise_step_map(generation_ids.history, mask_id=mask_id, sample_idx=i)
            step_map = m[prompt_ids.shape[1]:].tolist()
            seq_dict[idx]  = texts[i]
            step_dict[idx] = step_map
            token_dict[idx] = seq_ids[i]
            eos_cont_dict[idx] = bool(eos_then_continues[i])
            prompt_len_dict[idx] = int(prompt_ids.shape[1])

        # free unused GPU cache
        torch.cuda.empty_cache()

def get_data_chunk(data, num_node, node_idx):
    total = len(data)
    chunk_size = (total + num_node - 1) // num_node  # 向上取整
    start_idx = node_idx * chunk_size
    end_idx = min((node_idx + 1) * chunk_size, total)
    return data[start_idx:end_idx]


if __name__ == "__main__":

    config = get_config()

    mp.set_start_method("spawn", force=True)

    
    
    system_prompts = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nYou need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|im_end|>\n<|im_start|>assistant\n'''
    
    project_name = config.experiment.project

    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name

    code_task = False
    if config.experiment.function == "train":
        dataset = config.dataset.train_dataset
        k_sample = config.rollout.num_response_per_task
        batch_size = config.rollout.batch_size

        if config.dataset.data_type == "code":
            code_task = True
            system_prompts_function = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{{problem}}\nPlace your code within a single Python code block ```python ```. Do not include more than one code block. <|im_end|>\n<|im_start|>assistant\n'''
            system_prompts_stdio = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nThis is the problem:\n{{problem}}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>\n<|im_start|>assistant\n'''


        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset
        
    elif config.experiment.function == "evaluation":
        dataset = config.evaluation.eval_dataset
        if config.evaluation.data_type == "code":
            code_task = True
            system_prompts_function = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{{problem}}\nPlace your code within a single Python code block ```python ```. Do not include more than one code block. <|im_end|>\n<|im_start|>assistant\n'''
            system_prompts_stdio = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nThis is the problem:\n{{problem}}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>\n<|im_start|>assistant\n'''
        
        k_sample = config.evaluation.num_response_per_task
        batch_size = config.evaluation.batch_size

        config.rollout.steps = config.evaluation.steps
        config.rollout.max_gen_length = config.evaluation.max_gen_length
        config.rollout.temperature = config.evaluation.temperature
        config.rollout.top_p = config.evaluation.top_p
        config.rollout.top_k = config.evaluation.top_k
        config.rollout.remasking_strategy = config.evaluation.remasking_strategy
        config.rollout.dynamic_threshold = config.evaluation.dynamic_threshold
        config.rollout.target = config.evaluation.target
        config.rollout.block_size = config.evaluation.block_size
        config.rollout.use_cache = config.evaluation.use_cache
        config.rollout.further_horizon = config.evaluation.further_horizon
        config.rollout.pad_target_penalty = config.evaluation.pad_target_penalty

        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset

    

    with open("../data/" + dataset + ".json", 'r') as f:
        data = json.load(f)
    #data = [data[i] for i in range(50)]
    for source_idx, item in enumerate(data):
        item["source_idx"] = source_idx

    num_node = config.experiment.num_node
    node_index = config.experiment.node_index
    fixed_task_indices = None
    if config.experiment.function == "train":
        fixed_task_indices = OmegaConf.select(
            config, "rollout.fixed_task_indices", default=None
        )
        if fixed_task_indices is not None:
            fixed_task_index_set = {int(x) for x in fixed_task_indices}
            data = [
                item
                for item in data
                if int(item.get("source_idx", -1)) in fixed_task_index_set
            ]

    if num_node > 1 and fixed_task_indices is None:
        if config.experiment.function == "train":
            random.shuffle(data)
        data = get_data_chunk(data, num_node, node_index)
    
    if config.experiment.function == "train":
        if fixed_task_indices is None:
            random_select_num = config.rollout.num_task_per_step
            random_select_num = int(random_select_num / num_node)
            random_select_num = min(random_select_num, len(data))
            data = random_select(data, random_select_num)
    num = len(data)

    tokenizer = DreamTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

    




    # initialization
    generation_prompts = []
    prefix_list = []
    index_list = []
    for i in range(num):
        # preprocess
        if code_task:
            if data[i]["test_method"] == "stdio":
                system_prompts = system_prompts_stdio
                prefix_list = prefix_list + [None] * k_sample
            else:
                system_prompts = system_prompts_function + data[i]["prefix"]
                prefix_list = prefix_list + [data[i]["prefix"]] * k_sample
        generation_prompts = generation_prompts + [get_prompt(data[i])] * k_sample
        index_list = index_list + [i] * k_sample
        data[i]["full_output"] = []
        data[i]["raw_full_output"] = []
        data[i]["truncated_response"] = []
        data[i]["training_response"] = []
        data[i]["step_map"] = []
        data[i]["first_eos_index"] = []
        data[i]["first_eos_global_index"] = []
        data[i]["valid_response_length"] = []
        data[i]["eos_then_continues"] = []
        data[i]["eos_first"] = []
        data[i]["missing_eos"] = []
        data[i]["extracted_output"] = []
        data[i]["response_length"] = []
        data[i]["prompt"] = get_prompt(data[i])

    

    # --------------------------- 1. shuffle --------------------------
    cprint("start generation...", "green")

    all_prompts = generation_prompts
    N = len(all_prompts)

    shuffled_idx     = list(range(N))
    random.shuffle(shuffled_idx)
    shuffled_prompts = [all_prompts[i] for i in shuffled_idx]

    # --------------------- 2. split to each GPU ----------------------
    n_gpu = torch.cuda.device_count()
    if n_gpu < 1:
        raise RuntimeError("need at least 1 CUDA GPU for inference")

    def split_even(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]

    prompt_chunks = split_even(shuffled_prompts, n_gpu)
    idx_chunks    = split_even(shuffled_idx,     n_gpu)

    

    # ------------------- 4. launch all workers -----------------------
    manager    = mp.Manager()
    seq_dict   = manager.dict()   # {shuffled_pos: text}
    step_dict  = manager.dict()   # {shuffled_pos: step_map}
    token_dict = manager.dict()   # {shuffled_pos: generated token ids}
    eos_cont_dict = manager.dict() # {shuffled_pos: post-EOS content before truncation}
    prompt_len_dict = manager.dict() # {shuffled_pos: padded prompt length}
    procs = []

    for rk in range(n_gpu):
        p = mp.Process(target=worker,
                    args=(pretrained_model, rk,
                            prompt_chunks[rk],
                            idx_chunks[rk],
                            seq_dict,
                            step_dict,
                            token_dict,
                            eos_cont_dict,
                            prompt_len_dict,
                            batch_size,
                            config))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # ------------------- 5. restore original order -------------------
    restored_outputs    = [seq_dict[i]  for i in range(N)]
    restored_step_maps  = [step_dict[i] for i in range(N)]
    restored_token_ids   = [token_dict[i] for i in range(N)]
    restored_eos_continues = [eos_cont_dict.get(i, False) for i in range(N)]
    restored_prompt_lens = [prompt_len_dict.get(i, 0) for i in range(N)]

    cprint("generation job done!", "green")






    import re

    def get_token_lengths(strings, tokenizer):
        pad_token = tokenizer.pad_token

        escaped = re.escape(pad_token)
        pattern = rf"(?:{escaped})+"
        remove_pattern = escaped

        collapse_re = re.compile(pattern)

        lengths = []
        for s in strings:
            s_clean = collapse_re.sub(lambda _: pad_token if isinstance(pad_token, str) else '', s)
            s_clean = re.sub(remove_pattern, '', s_clean)
            lengths.append(len(tokenizer.encode(s_clean, add_special_tokens=False)))
        return lengths

    eos_token_names = OmegaConf.select(
        config, "rollout.eos_token_names", default=["<|im_end|>", "<|endoftext|>"]
    )
    eos_token_ids = set(resolve_eos_token_ids(tokenizer, eos_token_names))
    pad_token_ids = {x for x in (tokenizer.pad_token_id,) if x is not None}
    mask_token_ids = {x for x in (getattr(tokenizer, "mask_token_id", None),) if x is not None}
    eos_token_strings = resolve_eos_token_strings(tokenizer, eos_token_names)

    eos_metadata = [
        eos_response_metadata(
            restored_token_ids[i],
            prompt_length=restored_prompt_lens[i],
            eos_token_ids=eos_token_ids,
            pad_token_ids=pad_token_ids,
            mask_token_ids=mask_token_ids,
            text=restored_outputs[i],
            eos_token_strings=eos_token_strings,
            precomputed_eos_then_continues=restored_eos_continues[i],
        )
        for i in range(N)
    ]
    truncated_outputs = [
        item["truncated_response"] if item["truncated_response"] is not None else restored_outputs[i]
        for i, item in enumerate(eos_metadata)
    ]
    train_on_forced_eos_tail = as_bool(
        OmegaConf.select(config, "rollout.force_after_first_eos_to_eos", default=True),
        default=True,
    )
    training_outputs = []
    for i, item in enumerate(eos_metadata):
        if train_on_forced_eos_tail:
            training_outputs.append(restored_outputs[i])
        elif item["training_response"] is not None:
            training_outputs.append(item["training_response"])
        else:
            training_outputs.append(truncated_outputs[i])
    response_length = [int(item["valid_response_length"]) for item in eos_metadata]
    mean_response_length = sum(response_length) / len(response_length)
    rollout_metrics = [
        compute_rollout_metrics(
            token_ids=restored_token_ids[i],
            step_map=restored_step_maps[i],
            text=truncated_outputs[i],
            eos_token_ids=eos_token_ids,
            pad_token_ids=pad_token_ids,
            eos_token_strings=eos_token_strings,
            max_gen_length=config.rollout.max_gen_length,
        )
        for i in range(N)
    ]
    for i, item in enumerate(eos_metadata):
        rollout_metrics[i]["eos_then_continues"] = float(item["eos_then_continues"])
        rollout_metrics[i]["eos_first"] = float(item["eos_first"])
        rollout_metrics[i]["missing_eos"] = float(item["missing_eos"])
        rollout_metrics[i]["first_eos_index"] = (
            float(item["first_eos_index"]) if item["first_eos_index"] is not None else 0.0
        )



    # process generated codes
    i = 0
    for full_output in truncated_outputs:
        if code_task:
            if data[int(i/k_sample)]["test_method"] == "function":
                extracted_output = extract_code(prefix_list[i] + full_output)
            elif data[int(i/k_sample)]["test_method"] == "stdio":
                extracted_output = extract_code(full_output)
        else:
            extracted_output = extract_final_boxed_answer(full_output)
        index_i = index_list[i]
        data[index_i]["full_output"].append(full_output)
        data[index_i]["raw_full_output"].append(restored_outputs[i])
        data[index_i]["truncated_response"].append(truncated_outputs[i])
        data[index_i]["training_response"].append(training_outputs[i])
        data[index_i]["step_map"].append(restored_step_maps[i])
        data[index_i]["first_eos_index"].append(eos_metadata[i]["first_eos_index"])
        data[index_i]["first_eos_global_index"].append(eos_metadata[i]["first_eos_global_index"])
        data[index_i]["valid_response_length"].append(eos_metadata[i]["valid_response_length"])
        data[index_i]["eos_then_continues"].append(eos_metadata[i]["eos_then_continues"])
        data[index_i]["eos_first"].append(eos_metadata[i]["eos_first"])
        data[index_i]["missing_eos"].append(eos_metadata[i]["missing_eos"])
        data[index_i]["extracted_output"].append(extracted_output)
        data[index_i]["response_length"].append(response_length[i])
        data[index_i].setdefault("rollout_metrics", []).append(rollout_metrics[i])
        i += 1

    # output the data
    if num_node > 1:
        output_file_name = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"
    else:
        output_file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
