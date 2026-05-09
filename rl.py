import os
import sys
import json
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
    
    def sample(i, type, block_size = None, top_k = None, remasking_strategy = None):
        if model_base == "dream":
            script_name = "dream_rl_rollout.py"
        elif model_base == "llada" or model_base == "mmada":
            script_name = "llada_rl_rollout.py"
        elif model_base == "sdar":
            script_name = "sdar_rl_rollout.py"
        elif model_base == "trado":
            script_name = "trado_rl_rollout.py"
        subprocess.run(
            f'python {script_name} '
            f'config=../configs/{project_name}.yaml '
            f"experiment.function={type} "
            f"evaluation.block_size={block_size} "
            f"evaluation.top_k={top_k} "
            f"evaluation.remasking_strategy={remasking_strategy} "
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
        attempts,
        raw_reward,
        raw_total_reward,
        rollout_metrics,
    ):
        outputs_name = get_train_outputs_name(i)
        outputs_result_name = f"{project_name}/results/results-{outputs_name}.txt"
        os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)

        attempted_response_count = aggregate_stats["attempted_response_count"]
        acc = (
            aggregate_stats["correct_count"] / attempted_response_count
            if attempted_response_count
            else 0
        )
        avg_len = (
            aggregate_stats["response_length_sum"] / attempted_response_count
            if attempted_response_count
            else 0
        )

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
            f"sampling attempts: {attempts}  "
            f"acc: {acc}  avg length: {avg_len}  "
            f"raw reward: {raw_reward}  raw total reward: {raw_total_reward}"
        )
        rollout_text = format_metrics(rollout_metrics)
        if rollout_text:
            output_text += "  " + rollout_text

        cprint("\n\n\n" + output_text, color="green")
        with open(outputs_result_name, "a") as f:
            f.write(output_text + "\n")

    def dynamic_train_rollout(i):
        target_prompt_count = int(config.rollout.num_task_per_step / config.experiment.num_node)
        target_prompt_count = max(target_prompt_count, 1)
        responses_per_prompt = int(config.rollout.num_response_per_task)
        target_response_count = target_prompt_count * responses_per_prompt
        max_attempts = int(
            OmegaConf.select(config, "rollout.dynamic_sampling_max_attempts", default=0) or 0
        )

        optimization_path = f"{project_name}/temp_data/{config.dataset.optimization_data}.json"
        stats_path = f"{project_name}/temp_data/reward_stats-train-step{i}.json"
        accumulated_data = []
        qualified_prompt_count = 0
        attempts = 0
        aggregate_stats = {
            "attempted_prompt_count": 0,
            "attempted_response_count": 0,
            "correct_count": 0,
            "response_length_sum": 0.0,
        }

        while qualified_prompt_count < target_prompt_count:
            attempts += 1
            if max_attempts and attempts > max_attempts:
                raise RuntimeError(
                    f"dynamic sampling reached rollout.dynamic_sampling_max_attempts={max_attempts} "
                    f"with {qualified_prompt_count}/{target_prompt_count} qualified prompts"
                )

            sample(i, "train")
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

            accepted_prompt_count = int(stats.get("qualified_prompt_count", 0))
            remaining_prompt_count = target_prompt_count - qualified_prompt_count
            prompts_to_add = min(accepted_prompt_count, remaining_prompt_count)
            responses_to_add = prompts_to_add * responses_per_prompt
            if responses_to_add:
                accumulated_data.extend(accepted_data[:responses_to_add])
                qualified_prompt_count += prompts_to_add

            cprint(
                f"dynamic sampling attempt {attempts}: "
                f"{qualified_prompt_count}/{target_prompt_count} qualified prompts",
                "yellow",
            )

        accumulated_data = accumulated_data[:target_response_count]
        raw_total_reward = float(
            sum(float(item.get("raw_reward", 0.0)) for item in accumulated_data)
        )
        raw_reward = raw_total_reward / len(accumulated_data) if accumulated_data else 0
        rollout_metrics = aggregate_rollout_metrics(accumulated_data)
        os.makedirs(os.path.dirname(optimization_path), exist_ok=True)
        with open(optimization_path, "w", encoding="utf-8") as f:
            json.dump(accumulated_data, f, indent=2, ensure_ascii=False)

        write_dynamic_train_metrics(
            i,
            aggregate_stats,
            target_prompt_count,
            qualified_prompt_count,
            attempts,
            raw_reward,
            raw_total_reward,
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
        
        
        dynamic_sampling = parse_bool(
            OmegaConf.select(config, "rollout.dynamic_sampling", default=True)
        )

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



