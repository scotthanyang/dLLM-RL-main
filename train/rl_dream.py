import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Optional, Union
import torch.nn.functional as F
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from termcolor import cprint

from transformers import AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed



from models import DreamTokenizer, DreamModel
from train.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

SYSTEM_PROMPT_LEN = 28

from train.utils import get_config, flatten_omega_conf, AverageMeter
from rl_metrics import format_metrics
from rl_eos import as_bool, resolve_eos_token_ids

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")







class TrainDataset(Dataset):
    def __init__(self, inputs, labels, pmasks, reward):
        self.inputs   = inputs
        self.labels   = labels
        self.pmasks   = pmasks
        self.reward   = reward
        L_minus1      = inputs.shape[1] - 1
        self.logp_old_tok = torch.full(
            (len(inputs), L_minus1), 
            float('-inf')
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            idx,                         
            self.inputs[idx],
            self.labels[idx],
            self.pmasks[idx],
            self.reward[idx],
        )


def build_optimizer_update_schedule(
    num_micro_batches: int,
    target_optimizer_updates: Optional[int],
):
    if target_optimizer_updates is None:
        return []
    if target_optimizer_updates <= 0:
        raise ValueError(
            "training.target_optimizer_updates_per_epoch must be a positive integer"
        )
    if num_micro_batches <= 0:
        return []

    effective_updates = min(target_optimizer_updates, num_micro_batches)
    base_group_size = num_micro_batches // effective_updates
    larger_group_count = num_micro_batches % effective_updates

    schedule = []
    for update_idx in range(effective_updates):
        extra = 1 if update_idx < larger_group_count else 0
        schedule.append(base_group_size + extra)
    return schedule


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()
    target_optimizer_updates_per_epoch = OmegaConf.select(
        config,
        "training.target_optimizer_updates_per_epoch",
        default=None,
    )
    if target_optimizer_updates_per_epoch is not None:
        target_optimizer_updates_per_epoch = int(target_optimizer_updates_per_epoch)
        if target_optimizer_updates_per_epoch <= 0:
            raise ValueError(
                "training.target_optimizer_updates_per_epoch must be a positive integer"
            )

    project_name = config.experiment.project
    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "./" + project_name + "/ckpt/" + config.model.optimized_name

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.project) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=None,
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.project,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint", None)

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.project, exist_ok=True)
        config_path = Path(config.experiment.project) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    tokenizer = DreamTokenizer.from_pretrained(pretrained_model)
    uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=config.training.max_prompt_len,
                                       max_gen_length=config.training.max_gen_length,
                                       ignore_id=-100)


    model = DreamModel.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16)
    model = model.to(accelerator.device)

    

    mask_id = model.config.mask_token_id
    pad_id = model.config.pad_token_id
    eos_token_names = OmegaConf.select(
        config, "rollout.eos_token_names", default=["<|im_end|>", "<|endoftext|>"]
    )
    eos_token_ids = resolve_eos_token_ids(tokenizer, eos_token_names)
    ignore_after_first_eos = as_bool(
        OmegaConf.select(config, "training.ignore_after_first_eos", default=False),
        default=False,
    )
    force_after_first_eos_to_eos = as_bool(
        OmegaConf.select(config, "training.force_after_first_eos_to_eos", default=True),
        default=True,
    )
    supervise_first_eos = as_bool(
        OmegaConf.select(config, "training.supervise_first_eos", default=True),
        default=True,
    )



    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")


    

    def collapse_k_unique(lst, k: int):
        if k <= 0:
            raise ValueError("k must be > 0")
        uniq = sorted(set(lst))

        mapping = {}
        n = len(uniq)
        for idx, val in enumerate(uniq):
            group = idx // k
            end_idx = min((group + 1) * k - 1, n - 1)
            rep = uniq[end_idx]
            mapping[val] = rep
        return [mapping[x] for x in lst]

    


    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    
    @torch.no_grad()
    def prepare_inputs_and_labels_for_text(
        prompt, response, step_map, reward, eps=1e-3, mask_id=mask_id
    ):
        input_ids_lm, labels_lm, start_pos, drop_num = uni_prompting((prompt, response))
        
        B, L = input_ids_lm.shape
        max_gen_len = config.training.max_gen_length
        if max_gen_len + start_pos < L:
            L_after = start_pos + max_gen_len
        else:
            L_after = L
        input_ids_lm = input_ids_lm[:, :L_after]
        labels_lm = labels_lm[:, :L_after]
        loss_active_mask_lm = input_ids_lm.ne(pad_id)
        loss_active_mask_lm[:, :start_pos] = False

        if (ignore_after_first_eos or force_after_first_eos_to_eos) and eos_token_ids:
            eos_set = {int(token_id) for token_id in eos_token_ids}
            for b in range(input_ids_lm.shape[0]):
                response_ids = input_ids_lm[b, start_pos:]
                first_eos = None
                for offset, token_id in enumerate(response_ids.tolist()):
                    if int(token_id) in eos_set:
                        first_eos = offset
                        break
                if first_eos is None:
                    continue

                eos_pos = start_pos + first_eos
                inactive_start = eos_pos + 1 if supervise_first_eos else eos_pos
                if force_after_first_eos_to_eos:
                    if inactive_start < input_ids_lm.shape[1]:
                        tail_active = loss_active_mask_lm[b, inactive_start:]
                        eos_fill_id = int(input_ids_lm[b, eos_pos].item())
                        tail_input_ids = input_ids_lm[b, inactive_start:]
                        tail_labels = labels_lm[b, inactive_start:]
                        tail_input_ids[tail_active] = eos_fill_id
                        tail_labels[tail_active] = eos_fill_id
                    continue

                if ignore_after_first_eos:
                    if inactive_start < input_ids_lm.shape[1]:
                        input_ids_lm[b, inactive_start:] = pad_id
                        labels_lm[b, inactive_start:] = -100
                        loss_active_mask_lm[b, inactive_start:] = False
                    if not supervise_first_eos:
                        labels_lm[b, eos_pos] = -100
                        loss_active_mask_lm[b, eos_pos] = False
    
        
        lower = config.training.lower_p
        upper = config.training.upper_p


        if config.training.method == "TraceRL":
            noisy_list, label_list, pmask_list, reward_list = [], [], [], []

            device = input_ids_lm.device
            B, L   = input_ids_lm.shape

            for b in range(B):
                
                order_list = list(step_map[b])
                order_list = collapse_k_unique(order_list, config.training.shrink)
                order = torch.as_tensor(order_list, device=device)
                order_full = torch.full((L_after,), -1, device=device)
                order_full[start_pos:] = order[: L_after - start_pos]
                loss_active_mask_b = loss_active_mask_lm[b].to(device=device)
                valid_order_values = order_full[start_pos:][loss_active_mask_b[start_pos:]]
                if valid_order_values.numel() == 0:
                    continue
                uniq_steps = torch.unique(valid_order_values, sorted=True)

                base_ids = input_ids_lm[b]

                if config.training.post_num is not None:
                    pad_mask_b = (base_ids == pad_id)
                    pad_mask_b[:start_pos] = False
                    keep_first_pad_b = pad_mask_b & (torch.cumsum(pad_mask_b.int(), dim=0) <= config.training.post_num)
                    tail_pad_b       = pad_mask_b & ~keep_first_pad_b
                else:
                    keep_first_pad_b = torch.zeros(L, dtype=torch.bool, device=device)
                    tail_pad_b       = torch.zeros(L, dtype=torch.bool, device=device)

                for step_val in uniq_steps:
                    tgt_mask = (order_full == step_val)
                    pmask_this = tgt_mask & ~tail_pad_b & loss_active_mask_b

                    if not pmask_this.any():
                        continue

                    noisy_ids = base_ids.clone()
                    mask_pos  = (order_full >= step_val) & loss_active_mask_b
                    noisy_ids[mask_pos] = mask_id

                    noisy_list.append(noisy_ids)
                    label_list.append(labels_lm[b])
                    pmask_list.append(pmask_this)
                    reward_list.append(reward[b])

            noisy_batch = torch.stack(noisy_list)
            labels_lm   = torch.stack(label_list)
            p_mask      = torch.stack(pmask_list)



        
        


            
        elif config.training.method == "random_masking":
            m = config.training.mask_times_per_sample
            B, L = input_ids_lm.shape
            device = input_ids_lm.device

            noisy_list, label_list, pmask_list, reward_list = [], [], [], []
            for b in range(B):
                base_ids  = input_ids_lm[b]
                label_ids = labels_lm[b]
                rwd       = reward[b]
                loss_active_mask_b = loss_active_mask_lm[b].to(device=device)

                if config.training.post_num is not None:
                    pad_mask_b = (base_ids == pad_id)
                    pad_mask_b[:start_pos] = False    
                    keep_first_pad_b = pad_mask_b & (torch.cumsum(pad_mask_b.int(), dim=0) <= config.training.post_num)
                    tail_pad_b       = pad_mask_b & ~keep_first_pad_b
                else:
                    keep_first_pad_b = torch.zeros(L, dtype=torch.bool, device=device)
                    tail_pad_b       = torch.zeros(L, dtype=torch.bool, device=device)

                for _ in range(m):
                    t = (upper - lower) * torch.rand(1, device=device) + lower
                    rand_mask = torch.rand(L, device=device) < t
                    rand_mask[:start_pos] = False
                    rand_mask = rand_mask & ~tail_pad_b & loss_active_mask_b

                    if not rand_mask.any():
                        continue

                    noisy_ids = base_ids.clone()
                    noisy_ids[rand_mask]  = mask_id
                    noisy_ids[tail_pad_b] = mask_id  

                    noisy_list.append(noisy_ids)
                    label_list.append(label_ids)
                    pmask_list.append(rand_mask)
                    reward_list.append(rwd)

            noisy_batch = torch.stack(noisy_list)
            labels_lm   = torch.stack(label_list)
            p_mask      = torch.stack(pmask_list)
        




        elif config.training.method == "coupled":
            m      = config.training.mask_times_per_sample
            B, L   = input_ids_lm.shape
            device = input_ids_lm.device

            noisy_list, label_list, pmask_list, reward_list = [], [], [], []
            for b in range(B):
                base_ids  = input_ids_lm[b]
                label_ids = labels_lm[b]
                rwd       = reward[b]
                loss_active_mask_b = loss_active_mask_lm[b].to(device=device)

                if config.training.post_num is not None:
                    pad_mask_b = (base_ids == pad_id)
                    pad_mask_b[:start_pos] = False
                    keep_first_pad_b = pad_mask_b & (torch.cumsum(pad_mask_b.int(), dim=0) <= config.training.post_num)
                    tail_pad_b       = pad_mask_b & ~keep_first_pad_b
                else:
                    keep_first_pad_b = torch.zeros(L, dtype=torch.bool, device=device)
                    tail_pad_b       = torch.zeros(L, dtype=torch.bool, device=device)

                for _ in range(m):
                    t = (upper - lower) * torch.rand(1, device=device) + lower
                    rand_mask = torch.rand(L, device=device) < t
                    rand_mask[:start_pos] = False

                    comp_mask = torch.zeros(L, device=device, dtype=torch.bool)
                    comp_mask[start_pos:] = ~rand_mask[start_pos:]

                    rand_mask  = rand_mask  & ~tail_pad_b & loss_active_mask_b
                    comp_mask  = comp_mask  & ~tail_pad_b & loss_active_mask_b

                    if rand_mask.any():
                        noisy_rand = base_ids.clone()
                        noisy_rand[rand_mask] = mask_id
                        noisy_rand[tail_pad_b] = mask_id
                        noisy_list.append(noisy_rand)
                        label_list.append(label_ids)
                        pmask_list.append(rand_mask)
                        reward_list.append(rwd)

                    if comp_mask.any():
                        noisy_comp = base_ids.clone()
                        noisy_comp[comp_mask] = mask_id
                        noisy_comp[tail_pad_b] = mask_id
                        noisy_list.append(noisy_comp)
                        label_list.append(label_ids)
                        pmask_list.append(comp_mask)
                        reward_list.append(rwd)

            noisy_batch = torch.stack(noisy_list)
            labels_lm   = torch.stack(label_list)
            p_mask      = torch.stack(pmask_list)
        

        valid_rows = p_mask.any(dim=1)
        noisy_batch = noisy_batch[valid_rows]
        labels_lm   = labels_lm[valid_rows]
        p_mask      = p_mask[valid_rows]
        keep_idx = torch.where(valid_rows)[0].tolist()
        reward_list = [reward_list[i] for i in keep_idx]

            
        
        return noisy_batch, labels_lm, p_mask, reward_list, start_pos, drop_num
    

 



    def make_attention_mask(input_ids, pad_id, pos):
        B, T = input_ids.shape
        device = input_ids.device
        dtype = input_ids.dtype
        idx = torch.arange(T, device=device)
        keep = ~input_ids.ne(pad_id) & (idx[None, :] <= pos)  # shape (B, T)
        # Allocate bias of shape (B,1,1,T)
        bias = torch.zeros(B, 1, 1, T, device=device)
        bias.masked_fill_(keep[:, None, None, :], float("-inf"))
        return bias
    
    
    @torch.no_grad()
    def compute_logp_old_tok_parallel(
            accelerator,
            dataset,
            train_dataloader_lm,
            start_pos, pad_id,
            batch_size):

        model.eval()

        dl = train_dataloader_lm

        for batch in dl:
            ids        = batch["ids"]                       # (b,)
            input_ids  = batch["input_ids"].to(accelerator.device)
            labels     = batch["labels"].to(accelerator.device)
            p_mask_lm  = batch["p_mask_lm"].to(accelerator.device)

            attn = make_attention_mask(input_ids, pad_id, start_pos)

            logits = model(input_ids, attention_mask=attn, is_causal=False).logits
            logp   = torch.log_softmax(logits[:, :-1, :], dim=-1)
            shift_labels = labels[:, 1:]
            shift_mask = p_mask_lm[:, 1:].bool()
            valid_shift_labels = (
                (shift_labels >= 0)
                & (shift_labels < logp.shape[-1])
                & (shift_labels != pad_id)
            )
            bad_active = shift_mask & ~valid_shift_labels
            if bad_active.any():
                raise ValueError("active old-logprob label must be a valid vocab id")
            safe_shift_labels = shift_labels.masked_fill(~valid_shift_labels, 0)
            tok_lp = logp.gather(-1, safe_shift_labels.unsqueeze(-1)).squeeze(-1)
            if shift_mask.any() and (~torch.isfinite(tok_lp[shift_mask])).any():
                raise ValueError("active old logprobs must be finite")
            tok_lp = tok_lp.masked_fill(~shift_mask, 0.0)

            dataset.logp_old_tok[ids] = tok_lp.float().cpu()

        accelerator.wait_for_everyone()

        model.train()


        

    def simple_collate(batch):
        idx, inp, lbl, msk, rwd = zip(*batch)
        return {
            "ids":        torch.tensor(idx),        # (b,)
            "input_ids":  torch.stack(inp),
            "labels":     torch.stack(lbl),
            "p_mask_lm":  torch.stack(msk),
            "reward":     rwd,
        }

    

    ##################################
    #       Preprocess data          #
    #################################
    logger.info("Preprocessing Data")


    with open("./" + project_name + "/temp_data/" + config.dataset.optimization_data + ".json", 'r') as f:
        dataset_load = json.load(f)
    #dataset_load = dataset_load[:2000]
    prompt_list = []
    response_list = []
    step_map_list = []
    reward_list = []
    for x in dataset_load:
        prompt_list.append(x["prompt"])
        response_list.append(x["response"])
        step_map_list.append(x["step_map"])
        reward_list.append(x["reward"])
    input_ids, labels, p_mask_lm, rewards, start_pos, drop_num = prepare_inputs_and_labels_for_text(prompt_list, response_list, step_map_list, reward_list)
    dataset_lm = TrainDataset(input_ids, labels, p_mask_lm, rewards)









    



    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    


    fixed_gradient_accumulation_steps = int(config.training.gradient_accumulation_steps)
    num_train_epochs = config.training.num_train_epochs

    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        sampler=None,
        collate_fn=simple_collate,
        num_workers=0
    )


    model, optimizer, train_dataloader_lm = accelerator.prepare(
        model, optimizer, train_dataloader_lm
    )

    num_micro_batches_per_epoch = len(train_dataloader_lm)
    if (
        target_optimizer_updates_per_epoch is not None
        and num_micro_batches_per_epoch == 0
    ):
        raise ValueError(
            "training.target_optimizer_updates_per_epoch was set, but the "
            "training dataloader is empty"
        )
    optimizer_update_schedule = build_optimizer_update_schedule(
        num_micro_batches=num_micro_batches_per_epoch,
        target_optimizer_updates=target_optimizer_updates_per_epoch,
    )
    if optimizer_update_schedule:
        avg_gradient_accumulation_steps = (
            sum(optimizer_update_schedule) / len(optimizer_update_schedule)
        )
        total_batch_size_lm = (
            config.training.batch_size_lm
            * accelerator.num_processes
            * avg_gradient_accumulation_steps
        )
        num_update_steps_per_epoch = len(optimizer_update_schedule)
    else:
        avg_gradient_accumulation_steps = float(fixed_gradient_accumulation_steps)
        total_batch_size_lm = (
            config.training.batch_size_lm
            * accelerator.num_processes
            * fixed_gradient_accumulation_steps
        )
        num_update_steps_per_epoch = math.ceil(len(dataset_lm) / total_batch_size_lm)
    max_train_steps = num_update_steps_per_epoch * num_train_epochs + 1

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )



    

    #################################
    #             Inference         #
    #################################
    logger.info("***** Running inference *****")

    compute_logp_old_tok_parallel(
        accelerator,
        dataset_lm,
        train_dataloader_lm,
        start_pos=start_pos,
        pad_id=pad_id,
        batch_size=config.training.batch_size_lm,
    )



    








    #################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    
    logger.info(f"  Num response = {len(dataset_load)}")
    logger.info(f"  Num sample dropped = {drop_num}")
    logger.info(f"  Num training data = {input_ids.shape[0]}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Optimizer updates per epoch = {num_update_steps_per_epoch}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    if optimizer_update_schedule:
        logger.info(
            "  Avg total train batch size (w. parallel, distributed & accumulation) = "
            f"{total_batch_size_lm:.3f}"
        )
        if target_optimizer_updates_per_epoch != len(optimizer_update_schedule):
            logger.info(
                "  Target optimizer updates per epoch = "
                f"{target_optimizer_updates_per_epoch} (clamped to "
                f"{len(optimizer_update_schedule)} because only "
                f"{num_micro_batches_per_epoch} microbatches are available)"
            )
        else:
            logger.info(
                "  Target optimizer updates per epoch = "
                f"{target_optimizer_updates_per_epoch}"
            )
        logger.info(
            "  Adaptive accumulation microbatches/update (min/max/avg) = "
            f"{min(optimizer_update_schedule)} / "
            f"{max(optimizer_update_schedule)} / "
            f"{avg_gradient_accumulation_steps:.3f}"
        )
    else:
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}"
        )
        logger.info(f"  Gradient Accumulation steps = {fixed_gradient_accumulation_steps}")

    first_epoch = 0
    data_time_m = AverageMeter()
    end = time.time()

    train_metric_totals = {}
    train_metric_counts = {}
    train_metric_extrema = {}
    grad_metric_totals = {}
    grad_metric_count = 0

    def add_metric(name, value, weight=1.0):
        value = float(value)
        weight = float(weight)
        if not math.isfinite(value) or weight <= 0:
            return
        train_metric_totals[name] = train_metric_totals.get(name, 0.0) + value * weight
        train_metric_counts[name] = train_metric_counts.get(name, 0.0) + weight

    def add_extreme(name, value, op):
        value = float(value)
        if not math.isfinite(value):
            return
        if name not in train_metric_extrema:
            train_metric_extrema[name] = value
        elif op == "max":
            train_metric_extrema[name] = max(train_metric_extrema[name], value)
        elif op == "min":
            train_metric_extrema[name] = min(train_metric_extrema[name], value)

    def accumulate_forward_metrics(metrics):
        active = float(metrics.get("metrics/active_tokens", 0.0))
        if active <= 0:
            raise ValueError("active token count must be positive for train metrics")
        token_weighted = [
            "metrics/entropy_mean",
            "metrics/ratio_mean",
            "metrics/clipped_ratio_mean",
            "metrics/clip_fraction",
            "metrics/kl_mean",
        ]
        for key in token_weighted:
            add_metric(key, metrics.get(key, 0.0), active)
        add_metric("metrics/ratio_std", metrics.get("metrics/ratio_std", 0.0), active)
        add_metric("metrics/kl_loss", metrics.get("metrics/kl_loss", 0.0), 1.0)
        add_metric("metrics/kl_sum_per_sample", metrics.get("metrics/kl_sum_per_sample", 0.0), 1.0)
        add_metric("metrics/loss", metrics.get("metrics/loss", 0.0), 1.0)
        add_metric("metrics/policy_loss", metrics.get("metrics/policy_loss", 0.0), 1.0)
        add_metric("metrics/active_tokens", active, 1.0)
        add_metric("metrics/ratio_mean_adv_pos", metrics.get("metrics/ratio_mean_adv_pos", 0.0), metrics.get("metrics/active_tokens_adv_pos", 0.0))
        add_metric("metrics/ratio_mean_adv_neg", metrics.get("metrics/ratio_mean_adv_neg", 0.0), metrics.get("metrics/active_tokens_adv_neg", 0.0))
        add_metric("metrics/clip_fraction_adv_pos", metrics.get("metrics/clip_fraction_adv_pos", 0.0), metrics.get("metrics/active_tokens_adv_pos", 0.0))
        add_metric("metrics/clip_fraction_adv_neg", metrics.get("metrics/clip_fraction_adv_neg", 0.0), metrics.get("metrics/active_tokens_adv_neg", 0.0))
        add_metric("metrics/active_tokens_adv_pos", metrics.get("metrics/active_tokens_adv_pos", 0.0), 1.0)
        add_metric("metrics/active_tokens_adv_neg", metrics.get("metrics/active_tokens_adv_neg", 0.0), 1.0)
        add_extreme("metrics/ratio_max", metrics.get("metrics/ratio_max", 0.0), "max")
        add_extreme("metrics/ratio_min", metrics.get("metrics/ratio_min", 0.0), "min")

    def global_grad_norm(parameters):
        total = 0.0
        for param in parameters:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            total += float(torch.sum(grad.float() * grad.float()).cpu())
        return math.sqrt(total)

    def accumulate_grad_metrics(before, after):
        nonlocal grad_metric_count
        grad_metric_totals["metrics/grad_norm"] = grad_metric_totals.get("metrics/grad_norm", 0.0) + float(before)
        grad_metric_totals["metrics/grad_norm_after_clip"] = grad_metric_totals.get("metrics/grad_norm_after_clip", 0.0) + float(after)
        grad_metric_count += 1

    def finalize_train_metrics():
        def reduce_sum(value):
            tensor = torch.tensor(float(value), device=accelerator.device)
            return float(accelerator.reduce(tensor, reduction="sum").detach().cpu())

        def gather_extreme(value, op):
            fill = -float("inf") if op == "max" else float("inf")
            tensor = torch.tensor(float(value) if value is not None else fill, device=accelerator.device)
            gathered = accelerator.gather_for_metrics(tensor)
            return float(gathered.max().cpu()) if op == "max" else float(gathered.min().cpu())

        metrics = {}
        for key, total in train_metric_totals.items():
            total = reduce_sum(total)
            count = reduce_sum(train_metric_counts.get(key, 0.0))
            if count > 0:
                metrics[key] = total / count
        for key, value in train_metric_extrema.items():
            op = "max" if key.endswith("_max") else "min"
            metrics[key] = gather_extreme(value, op)
        grad_count = reduce_sum(grad_metric_count)
        if grad_count:
            for key, total in grad_metric_totals.items():
                metrics[key] = reduce_sum(total) / grad_count
        clip_fraction = metrics.get("metrics/clip_fraction")
        if clip_fraction is not None and not (0.0 <= clip_fraction <= 1.0):
            raise ValueError(f"clip fraction out of range: {clip_fraction}")
        metrics["metrics/kl_estimator"] = "k3" if config.training.use_kl_estimator_k3 else "k1"
        return metrics

    def write_train_metrics(metrics):
        if not accelerator.is_main_process:
            return
        metrics_path = (
            Path(config.experiment.project)
            / "temp_data"
            / f"train_metrics-step{config.experiment.current_epoch}.json"
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        if config.experiment.current_epoch == 1:
            result_pretrained_model = config.model.pretrained_model
        else:
            result_pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name
        outputs_name = "rl-" + result_pretrained_model.replace("/", ".") + "-" + config.dataset.train_dataset
        outputs_result_name = Path(config.experiment.project) / "results" / f"results-{outputs_name}.txt"
        outputs_result_name.parent.mkdir(parents=True, exist_ok=True)
        output_text = (
            f"train step: {config.experiment.current_epoch}  "
            f"training diagnostics  {format_metrics(metrics)}"
        )
        cprint("\n\n\n" + output_text, color="green")
        with outputs_result_name.open("a", encoding="utf-8") as f:
            f.write(output_text + "\n")
        if getattr(accelerator, "trackers", None):
            numeric_metrics = {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
            accelerator.log(numeric_metrics, step=int(config.experiment.current_epoch))

    # TEMP_NAN_DEBUG: print-only diagnostics for tracing the first non-finite tensor.
    # Remove after the NaN collapse source is identified.
    temp_nan_debug_max_steps = int(os.environ.get("RL_TEMP_NAN_DEBUG_MAX_STEPS", "12"))

    def temp_nan_debug_should_print(debug_context, tensors=None):
        if debug_context is None:
            return False
        if int(debug_context.get("micro_step", 0)) <= temp_nan_debug_max_steps:
            return True
        if tensors is None:
            return False
        for tensor in tensors:
            if torch.is_tensor(tensor) and not torch.isfinite(tensor.detach()).all().item():
                return True
        return False

    def temp_nan_debug_tensor_stats(name, tensor, mask=None):
        with torch.no_grad():
            if not torch.is_tensor(tensor):
                tensor = torch.as_tensor(tensor, device=accelerator.device)
            values = tensor.detach()
            if mask is not None:
                values = values[mask]
            values = values.reshape(-1)
            count = int(values.numel())
            if count == 0:
                return f"{name}: count=0"

            values_f = values.float()
            finite = torch.isfinite(values_f)
            finite_values = values_f[finite]
            nan_count = int(torch.isnan(values_f).sum().item())
            posinf_count = int(torch.isposinf(values_f).sum().item())
            neginf_count = int(torch.isneginf(values_f).sum().item())
            if finite_values.numel():
                min_val = float(finite_values.min().item())
                max_val = float(finite_values.max().item())
                mean_val = float(finite_values.mean().item())
            else:
                min_val = max_val = mean_val = float("nan")
            return (
                f"{name}: count={count} all_finite={bool(finite.all().item())} "
                f"nan={nan_count} posinf={posinf_count} neginf={neginf_count} "
                f"min={min_val} max={max_val} mean={mean_val}"
            )

    def temp_nan_debug_print(debug_context, stage, lines):
        if debug_context is None:
            return
        prefix = (
            "TEMP_NAN_DEBUG "
            f"epoch={debug_context.get('epoch')} "
            f"micro_step={debug_context.get('micro_step')} "
            f"rank={accelerator.process_index} "
            f"stage={stage}"
        )
        print(prefix)
        for line in lines:
            print(f"TEMP_NAN_DEBUG {line}")

    def temp_nan_debug_grad_stats(debug_context, stage):
        if debug_context is None:
            return
        total_params = 0
        params_with_grad = 0
        params_with_bad_grad = 0
        bad_entries = 0
        first_bad = []
        for name, param in model.named_parameters():
            total_params += 1
            if param.grad is None:
                continue
            params_with_grad += 1
            grad = param.grad.detach()
            finite = torch.isfinite(grad)
            if not finite.all().item():
                params_with_bad_grad += 1
                bad_count = int((~finite).sum().item())
                bad_entries += bad_count
                if len(first_bad) < 5:
                    first_bad.append(f"{name}: bad={bad_count}/{grad.numel()}")
        temp_nan_debug_print(
            debug_context,
            stage,
            [
                f"grad total_params={total_params} params_with_grad={params_with_grad} "
                f"params_with_bad_grad={params_with_bad_grad} bad_grad_entries={bad_entries}",
                *first_bad,
            ],
        )

    def temp_nan_debug_param_stats(debug_context, stage):
        if debug_context is None:
            return
        total_params = 0
        params_with_bad_value = 0
        bad_entries = 0
        first_bad = []
        for name, param in model.named_parameters():
            total_params += 1
            value = param.detach()
            finite = torch.isfinite(value)
            if not finite.all().item():
                params_with_bad_value += 1
                bad_count = int((~finite).sum().item())
                bad_entries += bad_count
                if len(first_bad) < 5:
                    first_bad.append(f"{name}: bad={bad_count}/{value.numel()}")
        temp_nan_debug_print(
            debug_context,
            stage,
            [
                f"param total_params={total_params} "
                f"params_with_bad_value={params_with_bad_value} "
                f"bad_param_entries={bad_entries}",
                *first_bad,
            ],
        )

    def temp_nan_debug_input_stats(debug_context, input_ids, labels, p_mask_lm, attn_mask, start_pos):
        if debug_context is None:
            return
        def get_vocab_size():
            for candidate in (model, getattr(model, "module", None)):
                if candidate is None:
                    continue
                cfg = getattr(candidate, "config", None)
                if isinstance(cfg, dict) and cfg.get("vocab_size") is not None:
                    return int(cfg["vocab_size"])
                value = getattr(cfg, "vocab_size", None)
                if value is not None:
                    return int(value)
            value = getattr(tokenizer, "vocab_size", None)
            if value is not None:
                return int(value)
            try:
                return int(len(tokenizer))
            except TypeError:
                return None

        B, T = input_ids.shape
        idx = torch.arange(T, device=input_ids.device)
        active_per_sample = p_mask_lm.bool().sum(dim=1)
        pad_per_sample = (input_ids == pad_id).sum(dim=1)
        prompt_pad_per_sample = ((input_ids == pad_id) & (idx[None, :] <= start_pos)).sum(dim=1)
        response_pad_per_sample = ((input_ids == pad_id) & (idx[None, :] > start_pos)).sum(dim=1)
        eos_positions = []
        eos_set = {int(token_id) for token_id in eos_token_ids}
        for b in range(B):
            first_eos = None
            for pos, token_id in enumerate(input_ids[b, start_pos:].tolist(), start=start_pos):
                if int(token_id) in eos_set:
                    first_eos = pos
                    break
            eos_positions.append(-1 if first_eos is None else first_eos)

        active_labels = labels[p_mask_lm.bool()]
        vocab_size = get_vocab_size()
        bad_input = input_ids < 0
        if vocab_size is not None:
            bad_input = bad_input | (input_ids >= vocab_size)
        bad_input_count = int(bad_input.sum().item())
        if active_labels.numel():
            bad_active_labels = (active_labels < 0) | (active_labels == pad_id)
            if vocab_size is not None:
                bad_active_labels = bad_active_labels | (active_labels >= vocab_size)
            bad_active_label_count = int(bad_active_labels.sum().item())
        else:
            bad_active_label_count = 0

        attn_finite = torch.isfinite(attn_mask).all().item() if torch.is_tensor(attn_mask) else True
        all_masked_rows = 0
        allowed_key_min = allowed_key_max = None
        if torch.is_tensor(attn_mask):
            allowed = torch.isfinite(attn_mask) & (attn_mask == 0)
            if allowed.dim() == 4:
                allowed_counts = allowed.sum(dim=-1)
                all_masked_rows = int((allowed_counts == 0).sum().item())
                allowed_key_min = int(allowed_counts.min().item())
                allowed_key_max = int(allowed_counts.max().item())
            elif allowed.dim() == 2:
                allowed_counts = allowed.sum(dim=-1)
                all_masked_rows = int((allowed_counts == 0).sum().item())
                allowed_key_min = int(allowed_counts.min().item())
                allowed_key_max = int(allowed_counts.max().item())

        temp_nan_debug_print(
            debug_context,
            "pre_forward_inputs",
            [
                f"input_shape={tuple(input_ids.shape)} start_pos={start_pos}",
                f"vocab_size={vocab_size}",
                f"bad_input_id_count={bad_input_count}",
                f"bad_active_label_count={bad_active_label_count}",
                f"pad_per_sample={pad_per_sample.detach().cpu().tolist()}",
                f"prompt_pad_per_sample={prompt_pad_per_sample.detach().cpu().tolist()}",
                f"response_pad_per_sample={response_pad_per_sample.detach().cpu().tolist()}",
                f"first_eos_global_per_sample={eos_positions}",
                f"active_p_mask_per_sample={active_per_sample.detach().cpu().tolist()}",
                f"attention_mask_is_tensor={torch.is_tensor(attn_mask)} finite={bool(attn_finite)} "
                f"all_masked_rows={all_masked_rows} allowed_key_min={allowed_key_min} allowed_key_max={allowed_key_max}",
            ],
        )

    def forward_process(
        input_ids, labels, p_mask_lm,
        start_pos,
        adv,
        logp_old_tok,
        debug_context=None):

        adv = torch.as_tensor(adv, device=input_ids.device).detach()

        attn_mask = make_attention_mask(input_ids, pad_id, start_pos)
        if temp_nan_debug_should_print(debug_context, [input_ids.float(), labels.float(), p_mask_lm.float(), attn_mask]):
            temp_nan_debug_input_stats(debug_context, input_ids, labels, p_mask_lm, attn_mask, start_pos)
        logits = model(input_ids, attention_mask=attn_mask, is_causal=False).logits
        if temp_nan_debug_should_print(debug_context, [logits]):
            temp_nan_debug_print(
                debug_context,
                "post_forward_logits",
                [temp_nan_debug_tensor_stats("logits", logits)],
            )
            if not torch.isfinite(logits.detach()).all().item():
                temp_nan_debug_param_stats(debug_context, "params_after_nonfinite_logits")

        B, T, V = logits.shape

        shift_logits = logits[:, :-1, :]          # (B, T-1, V)
        shift_labels = labels[:, 1:]              # (B, T-1)
        shift_mask   = p_mask_lm[:, 1:].bool()
        valid_shift_labels = (
            (shift_labels >= 0)
            & (shift_labels < V)
            & (shift_labels != pad_id)
        )
        bad_active_labels = shift_mask & ~valid_shift_labels
        if bad_active_labels.any():
            raise ValueError("shift_mask active position has invalid shift_label")
        active_mask = shift_mask & valid_shift_labels
        safe_shift_labels = shift_labels.masked_fill(~valid_shift_labels, 0)

        active_count = active_mask.sum()
        inactive_mask = ~active_mask
        active_labels = shift_labels[active_mask]
        debug_pre_tensors = [logits, adv, logp_old_tok]
        if temp_nan_debug_should_print(debug_context, debug_pre_tensors):
            temp_nan_debug_print(
                debug_context,
                "pre_log_softmax",
                [
                    temp_nan_debug_tensor_stats("logits", logits),
                    temp_nan_debug_tensor_stats("adv", adv),
                    temp_nan_debug_tensor_stats("shift_mask", shift_mask.float()),
                    temp_nan_debug_tensor_stats("active_labels", active_labels.float()),
                    temp_nan_debug_tensor_stats("logp_old_tok_active", logp_old_tok, active_mask),
                    temp_nan_debug_tensor_stats("logp_old_tok_inactive", logp_old_tok, inactive_mask),
                    f"active_token_count={int(active_count.item())}",
                    f"active_shift_labels_eq_-100={int((active_labels == -100).sum().item())}",
                    f"active_shift_labels_lt_0={int((active_labels < 0).sum().item())}",
                    f"active_shift_labels_ge_vocab={int((active_labels >= V).sum().item())}",
                    f"active_shift_labels_eq_pad={int((active_labels == pad_id).sum().item())}",
                    f"any_logp_old_nonfinite_active={bool((~torch.isfinite(logp_old_tok[active_mask])).any().item()) if active_count.item() else False}",
                    f"any_adv_nonfinite={bool((~torch.isfinite(torch.as_tensor(adv, device=input_ids.device).float())).any().item())}",
                ],
            )

        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        logp_new_tok = log_probs.gather(-1, safe_shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

        log_ratio_tok = logp_new_tok - logp_old_tok
        log_ratio_tok = log_ratio_tok.masked_fill(~active_mask, 0.0)
        if temp_nan_debug_should_print(debug_context, [logp_new_tok, logp_old_tok, log_ratio_tok]):
            temp_nan_debug_print(
                debug_context,
                "pre_ratio_exp",
                [
                    temp_nan_debug_tensor_stats("logp_new_tok_active", logp_new_tok, active_mask),
                    temp_nan_debug_tensor_stats("logp_new_tok_inactive", logp_new_tok, inactive_mask),
                    temp_nan_debug_tensor_stats("logp_old_tok_active", logp_old_tok, active_mask),
                    temp_nan_debug_tensor_stats("logp_old_tok_inactive", logp_old_tok, inactive_mask),
                    temp_nan_debug_tensor_stats("log_ratio_tok_active", log_ratio_tok, active_mask),
                    temp_nan_debug_tensor_stats("log_ratio_tok_inactive", log_ratio_tok, inactive_mask),
                ],
            )

        ratio   = torch.exp(log_ratio_tok)          # (B, T-1)
        clipped = torch.clamp(ratio, 1 - config.training.eps, 1 + config.training.eps)            # (B, T-1)
        adv_tok = adv.unsqueeze(1)                                # (B, 1)
        if temp_nan_debug_should_print(debug_context, [ratio, clipped]):
            temp_nan_debug_print(
                debug_context,
                "post_ratio_exp",
                [
                    temp_nan_debug_tensor_stats("ratio_active", ratio, active_mask),
                    temp_nan_debug_tensor_stats("ratio_inactive", ratio, inactive_mask),
                    temp_nan_debug_tensor_stats("clipped_active", clipped, active_mask),
                    temp_nan_debug_tensor_stats("clipped_inactive", clipped, inactive_mask),
                ],
            )

        # token clip
        surrogate_tok = torch.min(ratio * adv_tok, clipped * adv_tok)  # (B, T-1)
        surrogate_tok = surrogate_tok * active_mask

        num_mask = torch.clamp(active_mask.sum(dim=1), min=1)
        surrogate_tok = surrogate_tok.sum(dim=1) / num_mask
        policy_loss = - (surrogate_tok.sum() / B)

        # KL penalty
        kl_tok = log_ratio_tok
        if config.training.use_kl_estimator_k3:
            kl_tok_for_loss = (-kl_tok).exp() - 1.0 + kl_tok
        else:
            kl_tok_for_loss = kl_tok
        if temp_nan_debug_should_print(debug_context, [kl_tok_for_loss]):
            temp_nan_debug_print(
                debug_context,
                "post_kl_estimator",
                [
                    f"kl_estimator={'k3' if config.training.use_kl_estimator_k3 else 'k1'}",
                    temp_nan_debug_tensor_stats("kl_tok_for_loss_active", kl_tok_for_loss, active_mask),
                    temp_nan_debug_tensor_stats("kl_tok_for_loss_inactive", kl_tok_for_loss, inactive_mask),
                ],
            )
        kl_loss = torch.tensor(0.0, device=policy_loss.device)
        if config.training.beta > 0:
            kl_seq = kl_tok_for_loss
            kl_seq = (kl_seq * active_mask).sum(dim=1)
            kl_loss = config.training.beta * kl_seq.sum() / B
            total_loss = policy_loss + kl_loss
        else:
            total_loss = policy_loss
        if temp_nan_debug_should_print(debug_context, [policy_loss, kl_loss, total_loss]):
            temp_nan_debug_print(
                debug_context,
                "losses",
                [
                    temp_nan_debug_tensor_stats("kl_seq", kl_seq if config.training.beta > 0 else torch.zeros(B, device=input_ids.device)),
                    temp_nan_debug_tensor_stats("policy_loss", policy_loss),
                    temp_nan_debug_tensor_stats("kl_loss", kl_loss),
                    temp_nan_debug_tensor_stats("total_loss", total_loss),
                ],
            )
        if active_count.item() <= 0:
            raise ValueError("active token count must be positive")

        active_ratio = ratio[active_mask].detach().float()
        active_clipped = clipped[active_mask].detach().float()
        active_log_probs = log_probs[active_mask].detach().float()
        active_entropy = -(active_log_probs.exp() * active_log_probs).sum(dim=-1)
        active_kl = kl_tok_for_loss[active_mask].detach().float()
        clip_mask = ((ratio < 1 - config.training.eps) | (ratio > 1 + config.training.eps)) & active_mask
        adv_pos_mask = active_mask & (adv_tok > 0)
        adv_neg_mask = active_mask & (adv_tok < 0)

        def masked_mean(values, mask):
            selected = values[mask]
            return selected.detach().float().mean() if selected.numel() else torch.tensor(0.0, device=values.device)

        def masked_fraction(mask, denom_mask):
            denom = denom_mask.sum()
            if denom.item() <= 0:
                return torch.tensor(0.0, device=mask.device)
            return mask.sum().float() / denom.float()

        ratio_std = active_ratio.std(unbiased=False) if active_ratio.numel() > 1 else torch.tensor(0.0, device=active_ratio.device)
        kl_seq_metric = (kl_tok_for_loss.detach().float() * active_mask.float()).sum(dim=1)
        metrics = {
            "metrics/active_tokens": active_count.detach().float(),
            "metrics/entropy_mean": active_entropy.mean(),
            "metrics/ratio_mean": active_ratio.mean(),
            "metrics/clipped_ratio_mean": active_clipped.mean(),
            "metrics/ratio_std": ratio_std,
            "metrics/ratio_max": active_ratio.max(),
            "metrics/ratio_min": active_ratio.min(),
            "metrics/clip_fraction": clip_mask.sum().detach().float() / active_count.detach().float(),
            "metrics/ratio_mean_adv_pos": masked_mean(ratio, adv_pos_mask),
            "metrics/ratio_mean_adv_neg": masked_mean(ratio, adv_neg_mask),
            "metrics/clip_fraction_adv_pos": masked_fraction(clip_mask & adv_pos_mask, adv_pos_mask),
            "metrics/clip_fraction_adv_neg": masked_fraction(clip_mask & adv_neg_mask, adv_neg_mask),
            "metrics/active_tokens_adv_pos": adv_pos_mask.sum().detach().float(),
            "metrics/active_tokens_adv_neg": adv_neg_mask.sum().detach().float(),
            "metrics/kl_mean": active_kl.mean(),
            "metrics/kl_sum_per_sample": kl_seq_metric.mean(),
            "metrics/kl_loss": kl_loss.detach().float(),
            "metrics/policy_loss": policy_loss.detach().float(),
            "metrics/loss": total_loss.detach().float(),
        }

        return total_loss, metrics



    from tqdm.auto import tqdm

    for epoch in range(first_epoch, num_train_epochs):
        
        model.train()
        update_idx = 0
        schedule_update_idx = 0
        micro_batches_in_current_update = 0
        
        progress_bar = tqdm(
            train_dataloader_lm,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,       
            leave=True                 
        )
        
        for step, batch in enumerate(progress_bar, start=1):
            
            # for loss calculation

            data_time_m.update(time.time() - end)

            input_ids = batch["input_ids"].to(accelerator.device)
            labels    = batch["labels"].to(accelerator.device)
            p_mask_lm = batch["p_mask_lm"].to(accelerator.device)
            old_lp = dataset_lm.logp_old_tok[batch["ids"].cpu()].to(accelerator.device)
            active_old_lp = p_mask_lm[:, 1:].bool()
            if active_old_lp.any() and (~torch.isfinite(old_lp[active_old_lp])).any().item():
                raise ValueError("active old logprobs must be finite before training")
            if torch.isneginf(old_lp[active_old_lp]).any().item():
                print(old_lp)
            
            reward = batch["reward"]
            debug_context = {
                "epoch": epoch + 1,
                "micro_step": step,
                "update_idx": update_idx,
            }

            loss_lm, forward_metrics = forward_process(
                    input_ids=input_ids,
                    labels=labels,
                    p_mask_lm=p_mask_lm,
                    start_pos=start_pos,
                    adv=reward,
                    logp_old_tok=old_lp,
                    debug_context=debug_context,
                )
            forward_metrics = {
                key: float(value.detach().float().cpu())
                if torch.is_tensor(value)
                else float(value)
                for key, value in forward_metrics.items()
            }
            accumulate_forward_metrics(forward_metrics)
            if optimizer_update_schedule:
                current_gradient_accumulation_steps = optimizer_update_schedule[schedule_update_idx]
            else:
                current_gradient_accumulation_steps = accelerator.gradient_accumulation_steps
            loss_lm = loss_lm / current_gradient_accumulation_steps
            if step <= 10:
                print(loss_lm)
            accelerator.backward(loss_lm)
            if temp_nan_debug_should_print(debug_context, [loss_lm]):
                temp_nan_debug_grad_stats(debug_context, "after_backward")

            if optimizer_update_schedule:
                micro_batches_in_current_update += 1
                should_step = (
                    micro_batches_in_current_update == current_gradient_accumulation_steps
                )
            else:
                should_step = (step + 1) % accelerator.gradient_accumulation_steps == 0

            if should_step:
                if temp_nan_debug_should_print(debug_context, [loss_lm]):
                    temp_nan_debug_grad_stats(debug_context, "before_clip")
                grad_norm_before = global_grad_norm(model.parameters())
                if config.training.max_grad_norm is not None:
                    clipped_norm = accelerator.clip_grad_norm_(model.parameters(),
                                                               config.training.max_grad_norm)
                    if clipped_norm is not None:
                        grad_norm_before = float(clipped_norm.detach().float().cpu()) if torch.is_tensor(clipped_norm) else float(clipped_norm)
                grad_norm_after = global_grad_norm(model.parameters())
                if temp_nan_debug_should_print(debug_context, [loss_lm, torch.tensor(grad_norm_before, device=accelerator.device), torch.tensor(grad_norm_after, device=accelerator.device)]):
                    temp_nan_debug_print(
                        debug_context,
                        "grad_norms",
                        [
                            f"grad_norm_before_clip={grad_norm_before}",
                            f"grad_norm_after_clip={grad_norm_after}",
                        ],
                    )
                    temp_nan_debug_grad_stats(debug_context, "after_clip")
                accumulate_grad_metrics(grad_norm_before, grad_norm_after)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_idx += 1
                if optimizer_update_schedule:
                    micro_batches_in_current_update = 0
                    schedule_update_idx += 1

                del input_ids, labels, p_mask_lm
                torch.cuda.empty_cache()

                


    accelerator.wait_for_everyone()
    write_train_metrics(finalize_train_metrics())
    temp_nan_debug_param_stats(
        {
            "epoch": num_train_epochs,
            "micro_step": "end",
            "update_idx": update_idx if "update_idx" in locals() else None,
        },
        "before_save_checkpoint",
    )

    # save checkpoint at the end of training
    save_checkpoint(model, tokenizer, config, accelerator, config.model.optimized_name)
    if config.experiment.current_epoch % config.experiment.save_every == 0:
        save_checkpoint(model, tokenizer, config, accelerator, f"epoch-{config.experiment.current_epoch}")

    accelerator.end_training()






def save_checkpoint(model, tokenizer, config, accelerator, name):
    output_dir = Path(config.experiment.project)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        ckpts = sorted(
            [d for d in output_dir.iterdir() if d.name.startswith("checkpoint")],
            key=lambda p: int(p.name.split("-")[1]),
        )
        if len(ckpts) >= checkpoints_total_limit:
            to_remove = ckpts[: len(ckpts) - checkpoints_total_limit + 1]
            logger.info(f"removing checkpoints: {', '.join(p.name for p in to_remove)}")
            for p in to_remove:
                shutil.rmtree(p, ignore_errors=True)

    save_base = output_dir / "ckpt"
    save_base.mkdir(exist_ok=True)

    model_to_save = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        model_to_save.save_pretrained(
            save_base / name,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(str(save_base / name))

        metadata = {
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model + tokenizer to {save_base / name}")
    















if __name__ == "__main__":
    main()
