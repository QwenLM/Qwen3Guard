import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
import os

def load_input_data(dataset_path):
    """
    Load input data from either a local JSONL file or a Hugging Face dataset,
    and filter to keep only items where type == "thinking+response".
    
    Args:
        input_file (str): 
            - Local path: e.g., "./data/input.jsonl"
            - HF dataset: e.g., "user/dataset" or "user/dataset:split"
    
    Returns:
        List[dict]: List of data items with type == "thinking+response"
    """
    try:
        ds = load_dataset(dataset_path, split="thinking")
        # Convert to list of dicts and filter
        filtered_data = [
            example for example in ds 
        ]
        return filtered_data
    except Exception as e:
        raise ValueError(f"Failed to load input from Hugging Face dataset '{dataset_path}': {e}")

def main(model_path, input_file, output_file):
    # 加载模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()

    result = []

    input_data = load_input_data(input_file)
    # print(len(input_data))
    for data in tqdm(input_data, desc="Processing", ncols=80):
        # chat template
        texts = tokenizer.apply_chat_template(data["message"], tokenize=False, add_generation_prompt=False)
        model_inputs = tokenizer([texts], return_tensors="pt").to(model.device)

        # get guard result
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        content = tokenizer.decode(output_ids, skip_special_tokens=True)

        data["guard_predict"] = content
        result.append(data)

    # save result
    with open(output_file, 'w', encoding='utf-8') as w:
        for item in result:
            w.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Caculate F1
    i = j = m = n = 0  # i: total unsafe (label=1), j: correctly predicted unsafe, m: total predicted unsafe, n: true positives
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            label = 1 if data["label"] == "Unsafe" else 0
            pred = data.get("guard_predict", "")

            is_pred_unsafe = "Safety: Unsafe" in pred or "Safety: Controversial" in pred

            if label == 1:
                i += 1
                if is_pred_unsafe:
                    j += 1

            if is_pred_unsafe:
                m += 1
                if label == 1:
                    n += 1

    recall = j / i if i > 0 else 0.0
    precision = n / m if m > 0 else 0.0
    f1 = 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 0.0

    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run guard model inference and evaluate performance.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the guard model")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path of evaluation dataset")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file with predictions and metrics")

    args = parser.parse_args()
    main(args.model_path, args.input_file, args.output_file)