import os
import json
import codecs
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import torch
from argparse import ArgumentParser

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def inference_parser():
    argparser = ArgumentParser()
    argparser.add_argument("--data_dir", type=str, default="../hackthon_data")
    argparser.add_argument("--data_name", type=str, default="open_assistant")
    argparser.add_argument("--eval_json", default='sampled_prompts_nobreaks.jsonl')
    argparser.add_argument("--model_name", type=str, default="for saving outputs")
    argparser.add_argument("--model_name_or_path", type=str, default="bigscience/bloom-560m")

    return argparser.parse_args()
    

def print_generation(model, tokenizer, prompt=None):

    if prompt is None:
        prompt = 'What is your purpose?\n\n'
    # Alpaca template 
    full_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Response:
    """.format(prompt)

    inputs = tokenizer(full_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to('cuda')
    model = model.to('cuda')
    with torch.no_grad():
        generation_output = model.generate(input_ids=input_ids,
                                           return_dict_in_generate=True,
                                           output_scores=True,
                                           max_new_tokens=256,
                                           temperature=1,
                                           num_beams=4,
                                           top_p=1,
                                           top_k=50, 
                                           no_repeat_ngram_size=6,
                                           repetition_penalty=1.1
                                        )

    full_generation = tokenizer.decode(generation_output.sequences[0])
    response = full_generation.split("### Response:")[1].strip()

    print("*"*89)
    print("Prompt: \n", prompt)
    print("Response: \n", response)
    print("\n")
    return response

def read_jsonl_file(file_path):
    result = []
    with codecs.open(file_path, 'r', 'utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            result.append(json_obj)
    return result


if __name__ == "__main__":
    args = inference_parser()

    print('cuda available:', torch.cuda.is_available())

    
    if 'llama' in args.model_name_or_path:
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    elif 'baichuan' in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    data_path = Path(args.data_dir) / 'open_assistant' / args.eval_json
    output_path = Path(args.data_dir) / 'open_assistant' / 'output' / args.model_name / args.eval_json
    json_objects = read_jsonl_file(data_path)
    
    if output_path.exists():
        output_path.unlink()
    for record in json_objects:
        record['response'] = print_generation(model, tokenizer, prompt=record['prompt'])
        with open(output_path, 'a', encoding='utf8') as outfile:
            json.dump(record, outfile, ensure_ascii=False)
            outfile.write('\n')