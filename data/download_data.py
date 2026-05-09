import argparse
from huggingface_hub import hf_hub_download
import shutil
import json


def extract_gsm8k_answer(answer: str) -> str:
    """Extract the final answer from the GSM8K rationale string."""
    return answer.split("####")[-1].strip()


parser = argparse.ArgumentParser(description="Download a dataset from HF hub")
parser.add_argument(
    "--dataset",
    choices=[
        "PrimeIntellect",
        "MATH_train",
        "demon_openr1math",
        "MATH500",
        "GSM8K",
        "GSM8K_train",
        "AIME2024",
        "LiveBench",
        "LiveCodeBench",
        "MBPP",
        "HumanEval",
    ],
    required=True,
    help="Which dataset to download",
)
args = parser.parse_args()
dataset = args.dataset


if dataset in {"MATH_train", "PrimeIntellect", "demon_openr1math"}:
    repo_dataset = dataset
    split = "train"
    output_name = dataset
elif dataset == "GSM8K_train":
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="train")
    records = []
    for item in ds:
        records.append(
            {
                "question": item["question"],
                "ground_truth_answer": extract_gsm8k_answer(item["answer"]),
                "subject": "GSM8K",
            }
        )
    with open("./GSM8K_train.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(records)} GSM8K train examples to ./GSM8K_train.json")
    raise SystemExit(0)
else:
    repo_dataset = dataset
    split = "test"
    output_name = dataset


cached_path = hf_hub_download(
    repo_id=f"Gen-Verse/{repo_dataset}",
    repo_type="dataset",
    filename=f"{split}/{repo_dataset}.json",
)
shutil.copy(cached_path, f"./{output_name}.json")