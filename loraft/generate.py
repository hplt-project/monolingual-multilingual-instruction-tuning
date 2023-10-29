import os
import sys
import json
import warnings
import fire
import torch
import transformers
from tqdm import tqdm
from peft import PeftModel
from transformers import GenerationConfig
from transformers import AutoModelForCausalLM #, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, LlamaTokenizer
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

lora_base_map = {"baichuan-13b":"baichuan-inc/Baichuan-13B-Base",
                 "baichuan-7b":"baichuan-inc/baichuan-7B",
                 "baichuan-2-7b":"baichuan-inc/Baichuan2-7B-Base",
                 "bloom-560m":"bigscience/bloom-560m",
                 "bloom-1b1":"bigscience/bloom-1b1",
                 "bloom-1b7":"bigscience/bloom-1b7",
                 "bloom-3b":"bigscience/bloom-3b",
                 "bloom-7b1":"bigscience/bloom-7b1",
                 "llama-7b":"decapoda-research/llama-7b-hf",
                 "llama-13b":"decapoda-research/llama-13b-hf",
                 "llama-30b":"decapoda-research/llama-30b-hf",
                 "ollama-3b":"openlm-research/open_llama_3b",
                 "ollama-7b":"openlm-research/open_llama_7b",
                 "ollama-13b":"openlm-research/open_llama_13b",
                 "pythia-70m":"EleutherAI/pythia-70m-deduped",
                 "pythia-160m":"EleutherAI/pythia-160m-deduped",
                 "pythia-410m":"EleutherAI/pythia-410m-deduped",
                 "pythia-1b":"EleutherAI/pythia-1b-deduped",
                 "pythia-1b4":"EleutherAI/pythia-1.4b-deduped",
                 "pythia-2b8":"EleutherAI/pythia-2.8b-deduped",
                 "pythia-6b9":"EleutherAI/pythia-6.9b-deduped",
                 "pythia-12b":"EleutherAI/pythia-12b-deduped"}

def main(
    load_8bit: bool = True,
    base_model: str = "",
    lora_weights: str = "",
    test_file: str = "",
    save_file: str = "",
    prompt_template: str = "alpaca",  # The prompt template to use, will default to alpaca.
):

    if lora_weights == "":
        assert base_model
        print("\n\n******WARNING: LoRA module is not specified. Loading only the base model for inference.******\n\n", flush=True)
    if lora_weights[-1] == "/":
        lora_weights = lora_weights[:-1]

    if not base_model:
        for suffix in lora_base_map:
            if lora_weights.endswith(suffix):
                base_model = lora_base_map[suffix]
                continue
        assert base_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:  # noqa: E722
        pass

    if test_file:
        test_lang = test_file.split(".jsonl")[0].split("_")[-1]
    if not save_file:
        save_file = "data/test-" + test_lang + "_decoded_by_" + lora_weights.split("/")[-1] + ".jsonl"
    if os.path.isfile(save_file):
        print("Test file's corresponding output exists, skipping now.", flush=True)
        print("Test: {}, Lora: {}".format(test_file, lora_weights))
        print("Save file: {}".format(save_file))
        return

    prompter = Prompter(prompt_template)

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    except:
        if "llama" in base_model.lower():
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
        else:
            raise NotImplementedError

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        if lora_weights != "":
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        if lora_weights != "":
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else: # CPU
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        if lora_weights != "":
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )

    if not load_8bit:
        model.half()  # seems to fix bugs for some.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    if device == "cuda":
        print("Using " + str(torch.cuda.device_count())  + " GPU devices", flush=True)

    def read_data(filename):
        data = []
        with open(filename) as f:
            for line in f:
                line = json.loads(line.strip())
                data.append({"instruction": line["prompt"], "input": None})
        return data


    def evaluate(
        instruction,
        input=None,
        temperature=1,
        top_p=1,
        top_k=50,
        num_beams=4, # perhaps can experiment with this
        max_new_tokens=256,
        no_repeat_ngram_size=6,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return prompter.get_response(output)

    # testing prompts
    if test_file:
        test_lang = test_file.split(".jsonl")[0].split("_")[-1]
        assert len(test_lang) == 2
        data = read_data(test_file)

        write_data = []
        for d in tqdm(data):
            instruction = d["instruction"]
            input = d["input"]
            response = evaluate(instruction, input).split("\n### ")[0].strip()
            d["response"] = response
            write_data.append(d)
        if not save_file:
            save_file = "data/test-" + test_lang + "_decoded_by_" + lora_weights.split("/")[-1] + ".jsonl"
        with open(save_file, "w") as out_f:
            for d in write_data:
                out_f.write(json.dumps(d) + "\n")
            print("Saved {}".format(save_file), flush=True)
    else:
        print("No test file provided, will test on a few pre-defined example questions.", flush=True)
        for instruction in [
            "What are the even numbers between 1 and 13?",
            "Please briefly introduce the animal alpaca.",
            "What is the meaning of life?",
        ]:
            print("Instruction:", instruction.strip())
            print("Response:", evaluate(instruction).split("\n### ")[0].strip())
            print()

        for instruction, input in zip(["Please write a sentence using the given word.",
                                       "Can you repeat the following phrase 3 times?"
                                      ],
                                      ["vicuna",
                                       "Scotland!"
                                      ]):
            print("Instruction:", instruction.strip())
            print("Input:", input.strip())
            print("Response:", evaluate(instruction, input).split("\n### ")[0].strip())
            print()


if __name__ == "__main__":
    try:
        fire.Fire(main)
    except Exception as e:
        print(e)
