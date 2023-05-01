"""
batch_selfinstruct_generate.py

run:
python -m generate_instruction generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="text-davinci-003" \
"""
import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool
import openai
import numpy as np
import tqdm
from rouge_score import rouge_scorer
import utils

import fire
import nltk
from nltk.tokenize import word_tokenize
nltk.download("punkt")

openai.api_key = "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"
def limit_tokens(prompt, max_tokens):
    tokens = prompt.split()
    if len(tokens) > max_tokens:
        return " ".join(tokens[:max_tokens])
    return prompt

def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("./prompt.txt").read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
        prompt += f"###\n"
        prompt += f"{idx + 2}. Instruction:"
    return prompt


def post_process_gpt3_response(num_prompt_instructions, response):
    if response is None:
        return []
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response["text"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []

    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue

        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)

        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()

            # filter out too short or too long instructions
            if not is_valid_instruction(inst):
                continue
            # filter based on keywords that are not suitable for language models.
            instructions.append({"instruction": inst, "input": input, "output": output})

    return instructions

def is_valid_instruction(inst):
    # filter out too short or too long instructions
    if len(inst.split()) <= 3 or len(inst.split()) > 150:
        return False

    # filter based on keywords that are not suitable for language models
    blacklist = [
        "image", "images", "graph", "graphs", "picture", "pictures", "file", "files",
        "map", "maps", "draw", "plot", "go to", "video", "audio", "music",
        "flowchart", "diagram",
    ]
    if any(find_word_in_string(word, inst) for word in blacklist):
        return False

    # filter out instructions starting with "Write a program"
    if inst.startswith("Write a program"):
        return False

    # filter those starting with punctuation
    if inst[0] in string.punctuation:
        return False

    # filter those starting with non-english character
    if not inst[0].isascii():
        return False

    return True

def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data(
    output_dir="./output",
    seed_tasks_path="./seed_tasks.jsonl",
    num_instructions_to_generate=100,
    model_name="text-davinci-003",
    num_prompt_instructions=3,
    request_batch_size=5,
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
):
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)  # Add this line
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    os.makedirs(output_dir, exist_ok=True)

    request_idx = 0

    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, "regen.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [word_tokenize(inst) for inst in all_instructions]

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1
        batch_inputs = []
        for _ in range(request_batch_size):
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            prompt = encode_prompt(prompt_instructions)
            prompt = limit_tokens(prompt, max_tokens=4096)  # Limit the tokens to 4096
            batch_inputs.append(prompt)

        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=3072,
            top_p=top_p,
            stop=["\n20", "20.", "20."],
        )

        request_start = time.time()
        results = utils.openai_completion(
            prompts=batch_inputs,
            model_name=model_name,
            batch_size=request_batch_size,
            decoding_args=decoding_args,
            logit_bias={"50256": -100},
        )
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        for result in results:
            new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
            instruction_data += new_instructions

        total = len(instruction_data)
        top_n_similar = 3
        keep = 0
        for instruction_data_entry in instruction_data:
            new_instruction_tokens = word_tokenize(instruction_data_entry["instruction"])
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]

            if max(rouge_scores) > 0.7:
                continue
            else:
                keep += 1
                most_similar_indices = sorted(range(len(rouge_scores)), key=lambda i: rouge_scores[i], reverse=True)[:top_n_similar]
                most_similar_instructions = [all_instructions[i] for i in most_similar_indices]
                instruction_data_entry["most_similar_instructions"] = most_similar_instructions
                instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
                machine_instruction_data.append(instruction_data_entry)
                all_instructions.append(instruction_data_entry["instruction"])
                all_instruction_tokens.append(new_instruction_tokens)
                progress_bar.update(1)


        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")

        utils.jdump(machine_instruction_data, os.path.join(output_dir, "regen.json"))

def main(task, **kwargs):
    globals()[task](**kwargs)

if __name__ == "__main__":
    fire.Fire(main)
