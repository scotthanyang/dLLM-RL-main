import json
import os
import sys
import math_utils
import nest_asyncio
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
import asyncio
from termcolor import cprint
from omegaconf import MISSING
from omegaconf import DictConfig, ListConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_metrics import aggregate_rollout_metrics, format_metrics
from rl_eos import as_bool, truncate_text_at_first_eos


def extract_final_boxed_answer(s: str):
    tag = r'\boxed{'
    start = s.rfind(tag)
    if start == -1:
        return "Can not extract the answer!"

    i = start + len(tag)
    depth = 1
    buf = []

    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1

    return ''.join(buf) if depth == 0 else "Can not extract the answer!"


def extract_code(full_output):
    import re

    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return "We can not extract the code in the output. "


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

    project_name = config.experiment.project
    report_metrics = parse_bool(OmegaConf.select(config, "reward.report_metrics", default=True))
    truncate_at_first_eos = as_bool(
        OmegaConf.select(config, "reward.truncate_at_first_eos", default=True),
        default=True,
    )
    eos_first_invalid = as_bool(
        OmegaConf.select(config, "reward.eos_first_invalid", default=True),
        default=True,
    )
    eos_after_continue_invalid = as_bool(
        OmegaConf.select(config, "rollout.eos_after_continue_invalid", default=True),
        default=True,
    )
    eos_token_names = OmegaConf.select(
        config, "rollout.eos_token_names", default=["<|im_end|>", "<|endoftext|>"]
    )
    
    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name
    

    if config.experiment.function == "train":
        shrink = config.training.shrink
        dataset = config.dataset.train_dataset
        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset
        
    elif config.experiment.function == "evaluation":
        dataset = config.evaluation.eval_dataset
        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset
    
    

    
    file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"

    with open(file_name, 'r') as f:
        data = json.load(f)

    eos_token_strings = [token for token in eos_token_names if token]

    def get_list_value(row, key, idx, default=None):
        values = row.get(key)
        if isinstance(values, list) and idx < len(values):
            return values[idx]
        return default

    def reward_visible_response(row, idx):
        if truncate_at_first_eos and isinstance(get_list_value(row, "truncated_response", idx), str):
            return get_list_value(row, "truncated_response", idx)
        text = get_list_value(row, "full_output", idx, "")
        if truncate_at_first_eos:
            text = truncate_text_at_first_eos(text, eos_token_strings, include_eos=False)
        return text

    def reward_extracted_output(row, idx):
        text = reward_visible_response(row, idx)
        if row.get("test_method") == "function" and "prefix" in row:
            return extract_code(row["prefix"] + text)
        if row.get("test_method") == "stdio":
            return extract_code(text)
        return extract_final_boxed_answer(text)


    index_list = []
    extracted_output_list = []
    ground_truth_list = []
    response_length_list = []
    for i in range(len(data)):
        data[i]["correctness"] = []
        response_length_list = response_length_list + data[i]["response_length"]
        index_list = index_list + [i] * len(data[i]["extracted_output"])
        if truncate_at_first_eos:
            safe_outputs = [
                reward_extracted_output(data[i], j)
                for j in range(len(data[i]["extracted_output"]))
            ]
            data[i]["reward_extracted_output"] = safe_outputs
            extracted_output_list = extracted_output_list + safe_outputs
        else:
            extracted_output_list = extracted_output_list + data[i]["extracted_output"]
        ground_truth_list = ground_truth_list + [data[i]["ground_truth_answer"]] * len(data[i]["extracted_output"])

    nest_asyncio.apply()

    async def get_correctness():
        executor = ThreadPoolExecutor(max_workers=64)
        tasks = []
        for i in range(len(index_list)):
            tasks.append(math_utils.is_equal(extracted_output_list[i], ground_truth_list[i], executor))
        results = await asyncio.gather(*tasks)
        return results

    correctness_list = asyncio.run(get_correctness())
    for i in range(len(index_list)):
        index_i = index_list[i]
        data[index_i]["correctness"].append(correctness_list[i])

    eos_first_invalid_count = 0
    eos_then_continue_invalid_count = 0


    def z_score_normalize(lst):
        mean = sum(lst) / len(lst)
        std = (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5
        if std == 0:
            return [0 for x in lst]
        return [(x - mean) / std for x in lst]






    def set_last_t(lst: list, t: int) -> None:
        new_lst = lst.copy()
        new_val = max(lst) + 1
        new_lst[-t:] = [new_val] * t
        return new_lst



    group_filter_min_success = float(
        OmegaConf.select(config, "reward.group_filter_min_success", default=0.2)
    )
    group_filter_max_success = float(
        OmegaConf.select(config, "reward.group_filter_max_success", default=0.8)
    )

    final_data = []
    qualified_prompt_count = 0
    corrected_correctness_list = []
    for i in range(len(data)):
        correctness = data[i]["correctness"]
        lengths = data[i]["response_length"]

        for j in range(len(lengths)):
            if OmegaConf.select(config, "rollout.max_gen_length", default=MISSING) is not MISSING and lengths[j] >= config.rollout.max_gen_length - 5:
                correctness[j] = False
            if OmegaConf.select(config, "rollout.max_token", default=MISSING) is not MISSING and lengths[j] >= config.rollout.max_token - 5:
                correctness[j] = False
            if eos_first_invalid and bool(get_list_value(data[i], "eos_first", j, False)):
                correctness[j] = False
                eos_first_invalid_count += 1
            if eos_after_continue_invalid and bool(get_list_value(data[i], "eos_then_continues", j, False)):
                correctness[j] = False
                eos_then_continue_invalid_count += 1
        corrected_correctness_list.extend(correctness)
        
        rewards = z_score_normalize(correctness)
        #rewards = [int(b) for b in correctness]

        data[i]["rewards"] = rewards
        
        if config.experiment.function == "train":

            proportion = sum(correctness) / len(correctness)
            if (
                proportion > group_filter_max_success
                or proportion < group_filter_min_success
            ):
                continue
            qualified_prompt_count += 1

            for j in range(len(rewards)):
                #if rewards[j] == 0:
                #    continue
                data_i = {}
                data_i["prompt"] = data[i]["prompt"]
                data_i["reward"] = rewards[j]
                data_i["raw_reward"] = int(bool(correctness[j]))
                data_i["response"] = get_list_value(
                    data[i], "training_response", j, data[i]["full_output"][j]
                )
                data_i["truncated_response"] = reward_visible_response(data[i], j)
                data_i["raw_full_output"] = get_list_value(
                    data[i], "raw_full_output", j, data[i]["full_output"][j]
                )
                data_i["step_map"] = data[i]["step_map"][j]
                for key in (
                    "first_eos_index",
                    "first_eos_global_index",
                    "valid_response_length",
                    "eos_then_continues",
                    "eos_first",
                    "missing_eos",
                ):
                    value = get_list_value(data[i], key, j, None)
                    if value is not None:
                        data_i[key] = value
                if "rollout_metrics" in data[i]:
                    data_i["rollout_metrics"] = data[i]["rollout_metrics"][j]
                final_data.append(data_i)
        
        if config.experiment.function == "evaluation":
            data[i]["step_map"] = []


    if config.experiment.function == "train":
        with open("../" + project_name + "/temp_data/" + config.dataset.optimization_data + ".json", "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

    raw_total_reward = float(sum(float(item.get("raw_reward", 0.0)) for item in final_data))
    raw_reward = raw_total_reward / len(final_data) if final_data else 0

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    stats_file_name = (
        "../"
        + project_name
        + f"/temp_data/reward_stats-{config.experiment.function}-step{config.experiment.current_epoch}.json"
    )
    rollout_stats = aggregate_rollout_metrics(final_data)
    stats = {
        "current_epoch": int(config.experiment.current_epoch),
        "function": config.experiment.function,
        "attempted_prompt_count": len(data),
        "attempted_response_count": len(corrected_correctness_list),
        "qualified_prompt_count": qualified_prompt_count,
        "qualified_response_count": len(final_data),
        "correct_count": int(sum(bool(x) for x in corrected_correctness_list)),
        "response_length_sum": float(sum(response_length_list)),
        "raw_reward": raw_reward,
        "raw_total_reward": raw_total_reward,
        "eos_first_invalid_count": eos_first_invalid_count,
        "eos_then_continue_invalid_count": eos_then_continue_invalid_count,
        "rollout_metrics": rollout_stats,
    }
    os.makedirs(os.path.dirname(stats_file_name), exist_ok=True)
    with open(stats_file_name, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if not report_metrics:
        raise SystemExit(0)


    outputs_result_name = "../" + project_name + "/results/results-" + outputs_name + ".txt"
    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
    with open(outputs_result_name, "a") as f:
        # Save + print
        def save_and_print(text):
            cprint("\n\n\n" + text, color="green")
            f.write(text + "\n")
        
        acc = stats["correct_count"] / stats["attempted_response_count"] if stats["attempted_response_count"] else 0
        avg_len = stats["response_length_sum"] / stats["attempted_response_count"] if stats["attempted_response_count"] else 0

        output_text = f"train step: {config.experiment.current_epoch}  "
        
        if config.experiment.function == "train":
            rollout_text = format_metrics(stats.get("rollout_metrics", {}))
            if config.model.model_base != "sdar" and config.model.model_base != "trado":
                output_text = output_text + f"remasking_strategy: {config.rollout.remasking_strategy}  block_size: {config.rollout.block_size}  acc: {acc}  avg length: {avg_len}  raw reward: {stats['raw_reward']}  raw total reward: {stats['raw_total_reward']}"
            else:
                output_text = output_text + f"remasking_strategy: {config.rollout.remasking_strategy}  top_k: {config.rollout.top_k}  acc: {acc}  avg length: {avg_len}  raw reward: {stats['raw_reward']}  raw total reward: {stats['raw_total_reward']}"
            if rollout_text:
                output_text = output_text + "  " + rollout_text
        else:
            if config.model.model_base != "sdar" and config.model.model_base != "trado":
                output_text = output_text + f"remasking_strategy: {config.evaluation.remasking_strategy}  block_size: {config.evaluation.block_size}  acc: {acc}  avg length: {avg_len}"
            else:
                output_text = output_text + f"remasking_strategy: {config.evaluation.remasking_strategy}  top_k: {config.evaluation.top_k}  acc: {acc}  avg length: {avg_len}"
        save_and_print(output_text)
