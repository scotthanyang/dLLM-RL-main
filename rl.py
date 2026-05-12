import os
import sys
import json
import random
import subprocess
from termcolor import cprint
from rl_metrics import aggregate_rollout_metrics, format_metrics

from omegaconf import DictConfig, ListConfig, OmegaConf
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)

if __name__ == "__main__":
    config = get_config()

    start_from_scratch = config.experiment.start_from_scratch
    project_name = config.experiment.project
    model_base = config.model.model_base

    from omegaconf import MISSING
    if OmegaConf.select(config, "model.value_base_model", default=MISSING) is not MISSING:
        have_value_model = True
    else:
        have_value_model = False

    def begin_with(file_name):
        with open(file_name, "w") as f:
            f.write("")
    
    def init_value_model(i, cfg):
        project_name = cfg.experiment.project
        subprocess.run(
            f'python init_sdar_value_model.py '
            f'config=../configs/{project_name}.yaml '
            f'experiment.current_epoch={i} ',
            shell=True,
            cwd='train',
            check=True,
        )
    
    if start_from_scratch:
        os.makedirs(f"{project_name}/results", exist_ok=True)
        optimized_model = "../" + project_name + "/ckpt/" + config.model.optimized_name
        begin_with(f"{project_name}/results/results-rl-" + optimized_model.replace("/", ".") + "-" + config.dataset.train_dataset + ".txt")
        begin_with(f"{project_name}/results/results-eval-" + optimized_model.replace("/", ".") + "-" + config.dataset.train_dataset + ".txt")
        if have_value_model:
            init_value_model(1, config)
            optimized_value_model = "../" + project_name + "/ckpt/" + config.model.optimized_value_name
            begin_with(f"{project_name}/results/results-rl-" + optimized_value_model.replace("/", ".") + "-" + config.dataset.train_dataset + ".txt")

    def _format_cli_task_indices(task_indices):
        if task_indices is None:
            return "null"
        return "[" + ",".join(str(int(x)) for x in task_indices) + "]"

    def _get_optimization_data_path():
        return f"{project_name}/temp_data/{config.dataset.optimization_data}.json"

    def _load_train_task_ids():
        with open(f"data/{config.dataset.train_dataset}.json", "r") as f:
            data = json.load(f)
        return list(range(len(data)))

    def sample(i, type, block_size = None, top_k = None, remasking_strategy = None, task_indices = None):
        if model_base == "dream":
            script_name = "dream_rl_rollout.py"
        elif model_base == "llada" or model_base == "mmada":
            script_name = "llada_rl_rollout.py"
        elif model_base == "sdar":
            script_name = "sdar_rl_rollout.py"
        elif model_base == "trado":
            script_name = "trado_rl_rollout.py"
        fixed_task_arg = ""
        if task_indices is not None:
            fixed_task_arg = (
                f"rollout.fixed_task_indices='{_format_cli_task_indices(task_indices)}' "
            )
        subprocess.run(
            f'python {script_name} '
            f'config=../configs/{project_name}.yaml '
            f"experiment.function={type} "
            f"evaluation.block_size={block_size} "
            f"evaluation.top_k={top_k} "
            f"evaluation.remasking_strategy={remasking_strategy} "
            f"{fixed_task_arg}"
            f'experiment.current_epoch={i} ',
            shell=True,
            cwd='sample',
            check=True,
        )
    
    def reward(i, type, is_code_task, block_size = None, top_k = None, remasking_strategy = None, report_metrics = True):
        if is_code_task:
            script_name = "rl_code_reward.py"
        else:
            script_name = "rl_reward.py"
        subprocess.run(
            f'python {script_name} '
            f'config=../configs/{project_name}.yaml '
            f"experiment.function={type} "
            f"evaluation.block_size={block_size} "
            f"evaluation.top_k={top_k} "
            f"evaluation.remasking_strategy={remasking_strategy} "
            f"reward.report_metrics={str(report_metrics).lower()} "
            f'experiment.current_epoch={i} ',
            shell=True,
            cwd='reward',
            check=True,
        )
    
    
    def process_reward(i):
        cfg_i = f"config=../configs/{project_name}.yaml"
        ep    = f"experiment.current_epoch={i}"

        base = ["conda", "run", "-n", "CURE2", "--no-capture-output", "python", "-u"]

        subprocess.run(base + ["rl_process_divide_data.py", cfg_i, ep], cwd="reward", check=True)
        subprocess.run(base + ["llm_process_reward.py",    cfg_i, ep], cwd="sample", check=True)
        subprocess.run(base + ["rl_process_reward.py",     cfg_i, ep], cwd="reward", check=True)
    
    def execute(i, type):
        subprocess.run(
            f"python rl_execute.py "
            f"config=../configs/{project_name}.yaml "
            f"experiment.function={type} "
            f"experiment.current_epoch={i} ",
            shell=True,
            cwd='reward',
            check=True,
        )
            
    
    def train(i, target = None):
        if target is None:
            if model_base == "dream":
                script_name = "rl_dream.py"
            elif model_base == "llada":
                script_name = "rl_llada.py"
            elif model_base == "mmada":
                script_name = "rl_mmada.py"
            elif model_base == "sdar":
                script_name = "rl_sdar.py"
            elif model_base == "trado":
                script_name = "rl_trado.py"
        elif target == "policy":
            if model_base == "sdar":
                script_name = "train_sdar_policy.py"
            elif model_base == "trado":
                script_name = "train_trado_policy.py"
        elif target == "value":
            if model_base == "sdar":
                script_name = "train_sdar_value.py"
            elif model_base == "trado":
                script_name = "train_trado_value.py"
        subprocess.run(
            f'accelerate launch '
            f'--num_machines 1 '
            f'--machine_rank 0 '
            f'--main_process_ip 127.0.0.1 '
            f'--main_process_port 8888 '
            f'--config_file accelerate_configs/{config.experiment.deepspeed_file}.yaml '
            f'train/{script_name} '
            f'config=configs/{project_name}.yaml '
            f'experiment.current_epoch={i} ',
            shell=True,
            check=True,
        )

    def get_train_outputs_name(i):
        if i == 1:
            pretrained_model = config.model.pretrained_model
        else:
            pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name
        return "rl-" + pretrained_model.replace("/", ".") + "-" + config.dataset.train_dataset

    def write_dynamic_train_metrics(
        i,
        aggregate_stats,
        target_prompt_count,
        qualified_prompt_count,
        rounds,
        batch_stats,
        rollout_metrics,
    ):
        outputs_name = get_train_outputs_name(i)
        outputs_result_name = f"{project_name}/results/results-{outputs_name}.txt"
        os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)

        output_text = f"train step: {i}  "
        if config.model.model_base != "sdar" and config.model.model_base != "trado":
            output_text += (
                f"remasking_strategy: {config.rollout.remasking_strategy}  "
                f"block_size: {config.rollout.block_size}  "
            )
        else:
            output_text += (
                f"remasking_strategy: {config.rollout.remasking_strategy}  "
                f"top_k: {config.rollout.top_k}  "
            )
        output_text += (
            f"qualified prompts: {qualified_prompt_count}/{target_prompt_count}  "
            f"sampling rounds: {rounds}  "
            f"sampled prompts: {aggregate_stats['attempted_prompt_count']}  "
            f"filtered prompts: {aggregate_stats['attempted_prompt_count'] - qualified_prompt_count}  "
            f"accepted responses: {batch_stats['response_count']}  "
            f"acc: {batch_stats['acc']}  avg length: {batch_stats['avg_len']}  "
            f"raw reward: {batch_stats['raw_reward']}  "
            f"raw total reward: {batch_stats['raw_total_reward']}"
        )
        rollout_text = format_metrics(rollout_metrics)
        if rollout_text:
            output_text += "  " + rollout_text

        cprint("\n\n\n" + output_text, color="green")
        sys.stdout.flush()
        with open(outputs_result_name, "a") as f:
            f.write(output_text + "\n")

    def _summarize_train_records(records):
        response_count = len(records)
        correct_count = sum(int(bool(item.get("raw_reward", 0))) for item in records)
        raw_total_reward = float(
            sum(float(item.get("raw_reward", 0.0)) for item in records)
        )

        length_sum = 0.0
        for item in records:
            if item.get("valid_response_length") is not None:
                length_sum += float(item["valid_response_length"])
            elif item.get("response_length") is not None:
                length_sum += float(item["response_length"])
            else:
                length_sum += float(item.get("rollout_metrics", {}).get("response_len", 0.0))

        source_ids = {
            int(item["source_idx"])
            for item in records
            if item.get("source_idx") is not None
        }
        return {
            "prompt_count": len(source_ids),
            "response_count": response_count,
            "correct_count": correct_count,
            "response_length_sum": length_sum,
            "acc": correct_count / response_count if response_count else 0,
            "avg_len": length_sum / response_count if response_count else 0,
            "raw_reward": raw_total_reward / response_count if response_count else 0,
            "raw_total_reward": raw_total_reward,
        }

    def dynamic_train_rollout(i):
        target_prompt_count = int(config.rollout.num_task_per_step / config.experiment.num_node)
        target_prompt_count = max(target_prompt_count, 1)
        max_attempts = int(
            OmegaConf.select(config, "rollout.dynamic_sampling_max_attempts", default=0) or 0
        )

        all_task_ids = _load_train_task_ids()
        seed_base = OmegaConf.select(config, "rollout.rollout_seed", default=None)
        if seed_base is None:
            seed_base = OmegaConf.select(config, "training.seed", default=0)
        rng = random.Random(int(seed_base) + int(i))
        rng.shuffle(all_task_ids)

        optimization_path = _get_optimization_data_path()
        stats_path = f"{project_name}/temp_data/reward_stats-train-step{i}.json"
        accumulated_data = []
        accepted_task_id_set = set()
        tried_task_id_set = set()
        qualified_prompt_count = 0
        rounds = 0
        aggregate_stats = {
            "attempted_prompt_count": 0,
            "attempted_response_count": 0,
            "correct_count": 0,
            "response_length_sum": 0.0,
        }

        while qualified_prompt_count < target_prompt_count:
            rounds += 1
            if max_attempts and rounds > max_attempts:
                raise RuntimeError(
                    f"dynamic sampling reached rollout.dynamic_sampling_max_attempts={max_attempts} "
                    f"with {qualified_prompt_count}/{target_prompt_count} qualified prompts"
                )

            remaining_prompt_count = target_prompt_count - qualified_prompt_count
            candidate_task_ids = [
                task_id for task_id in all_task_ids if task_id not in tried_task_id_set
            ][:remaining_prompt_count]
            if not candidate_task_ids:
                raise RuntimeError(
                    "dynamic sampling exhausted the train dataset after "
                    f"{rounds - 1} rounds: accepted "
                    f"{qualified_prompt_count}/{target_prompt_count} prompts"
                )

            tried_task_id_set.update(candidate_task_ids)
            cprint(
                f"[dynamic_sampling] step={i} round={rounds} requesting "
                f"{len(candidate_task_ids)} new prompts "
                f"(accepted={qualified_prompt_count}/{target_prompt_count})",
                "cyan",
            )
            sys.stdout.flush()

            sample(i, "train", task_indices=candidate_task_ids)
            if is_code_task:
                execute(i, "train")
            reward(i, "train", is_code_task, report_metrics=False)

            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            with open(optimization_path, "r", encoding="utf-8") as f:
                accepted_data = json.load(f)

            for key in ("attempted_prompt_count", "attempted_response_count", "correct_count"):
                aggregate_stats[key] += int(stats.get(key, 0))
            aggregate_stats["response_length_sum"] += float(stats.get("response_length_sum", 0.0))
            round_attempted = int(stats.get("attempted_response_count", 0))
            round_acc = (
                int(stats.get("correct_count", 0)) / round_attempted
                if round_attempted
                else 0
            )
            round_avg_len = (
                float(stats.get("response_length_sum", 0.0)) / round_attempted
                if round_attempted
                else 0
            )

            records_by_source = {}
            for item in accepted_data:
                source_idx = item.get("source_idx")
                if source_idx is None:
                    raise KeyError(
                        "Dynamic task sampling requires reward records to include 'source_idx'."
                    )
                records_by_source.setdefault(int(source_idx), []).append(item)

            round_accepted = []
            for source_idx in candidate_task_ids:
                if source_idx in accepted_task_id_set:
                    continue
                records = records_by_source.get(int(source_idx), [])
                if not records:
                    continue
                accepted_task_id_set.add(int(source_idx))
                round_accepted.append(int(source_idx))
                accumulated_data.extend(records)

            qualified_prompt_count = len(accepted_task_id_set)

            cprint(
                f"[dynamic_sampling] step={i} round={rounds} accepted "
                f"{len(round_accepted)}/{len(candidate_task_ids)} prompts; "
                f"total={qualified_prompt_count}/{target_prompt_count}; "
                f"round acc: {round_acc}  round avg length: {round_avg_len}",
                "cyan",
            )
            sys.stdout.flush()

        batch_stats = _summarize_train_records(accumulated_data)
        rollout_metrics = aggregate_rollout_metrics(accumulated_data)
        os.makedirs(os.path.dirname(optimization_path), exist_ok=True)
        with open(optimization_path, "w", encoding="utf-8") as f:
            json.dump(accumulated_data, f, indent=2, ensure_ascii=False)
        dynamic_stats = {
            "current_epoch": int(i),
            "function": "train",
            "dynamic_sampling": True,
            "target_prompt_count": target_prompt_count,
            "qualified_prompt_count": qualified_prompt_count,
            "sampling_rounds": rounds,
            "attempted_prompt_count": aggregate_stats["attempted_prompt_count"],
            "attempted_response_count": aggregate_stats["attempted_response_count"],
            "attempted_correct_count": aggregate_stats["correct_count"],
            "attempted_response_length_sum": aggregate_stats["response_length_sum"],
            "final_prompt_count": batch_stats["prompt_count"],
            "final_response_count": batch_stats["response_count"],
            "final_correct_count": batch_stats["correct_count"],
            "final_response_length_sum": batch_stats["response_length_sum"],
            "raw_reward": batch_stats["raw_reward"],
            "raw_total_reward": batch_stats["raw_total_reward"],
            "rollout_metrics": rollout_metrics,
        }
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(dynamic_stats, f, indent=2)

        write_dynamic_train_metrics(
            i,
            aggregate_stats,
            target_prompt_count,
            qualified_prompt_count,
            rounds,
            batch_stats,
            rollout_metrics,
        )
    
    if config.dataset.data_type == "code":
        is_code_task = True
    else:
        is_code_task = False
    
    if OmegaConf.select(config, "model.process_reward_model", default=MISSING) is not MISSING and config.model.process_reward_model is not None:
        is_process_reward = True
    else:
        is_process_reward = False

    i = config.experiment.current_epoch

    while i <= config.experiment.total_step:
        
        
        dynamic_sampling_default = OmegaConf.select(
            config, "rollout.dynamic_task_sampling_enable", default=False
        )
        dynamic_sampling = parse_bool(
            OmegaConf.select(
                config, "rollout.dynamic_sampling", default=dynamic_sampling_default
            )
        )
        if dynamic_sampling and model_base != "dream":
            raise ValueError("rollout.dynamic_sampling is currently implemented for Dream RL only.")

        if is_process_reward or is_code_task or not dynamic_sampling:
            sample(i, "train")
            if is_code_task:
                execute(i, "train")
            if is_process_reward:
                process_reward(i)
            else:
                reward(i, "train", is_code_task)
        else:
            dynamic_train_rollout(i)

        if have_value_model:
            train(i, target = "value")
            train(i, target = "policy")
        else:
            train(i, target = None)

        if i % config.experiment.eval_every == 0:
            if model_base == "sdar":
                remasking_strategy_list = config.evaluation.remasking_strategy
                top_k_list = config.evaluation.top_k
                block_size = config.evaluation.block_size
                for j in range(len(remasking_strategy_list)):
                    remasking_strategy = remasking_strategy_list[j]
                    top_k = top_k_list[j]
                    sample(i, "evaluation", block_size = block_size, top_k = top_k, remasking_strategy = remasking_strategy)
                    if is_code_task:
                        execute(i, "evaluation")
                    reward(i, "evaluation", is_code_task, block_size = block_size, top_k = top_k, remasking_strategy = remasking_strategy)
            else:
                block_size_list = config.evaluation.block_size
                remasking_strategy_list = config.evaluation.remasking_strategy
                if OmegaConf.select(config, "evaluation.top_k", default=MISSING) is not MISSING:
                    top_k = config.evaluation.top_k
                else:
                    top_k = None
                for j in range(len(remasking_strategy_list)):
                    remasking_strategy = remasking_strategy_list[j]
                    if model_base == "dream":
                        block_size = block_size_list[j]
                    elif model_base == "llada" or model_base == "mmada":
                        block_size = config.evaluation.block_size
                    sample(i, "evaluation", block_size = block_size, top_k = top_k, remasking_strategy = remasking_strategy)
                    if is_code_task:
                        execute(i, "evaluation")
                    reward(i, "evaluation", is_code_task, block_size = block_size, top_k = top_k, remasking_strategy = remasking_strategy)

        i += 1
