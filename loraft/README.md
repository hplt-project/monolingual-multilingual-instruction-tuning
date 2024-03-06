## Low-Rank Adaptation

### Preparation
You might need to install the packages in the `requirements.txt` file. It is suggested that you use an environment manager, e.g. Conda.

`pip install -r requirements.txt`

Model weights are/will be available on Hugging Face [here](https://huggingface.co/collections/HPLT/instruction-tuning-65dba9834e23db813d863951).

### Training

We provide an example script to train BLOOM-7B1 below. It is worth noting that you might need to change `WORLD_SIZE` depending on your GPU configuration as well as the `--lora_target_modules` option depending on where you would like to apply LoRA. The name also varies by the base model you use. The training data should be fed via `--data_path`. You can prepare your own training set, download from this repository, or directly load via Hugging Face dataset: `pinzhenchen/alpaca-cleaned-${LANG}`, where `${LANG}` should be a two digit language code--see [here](https://huggingface.co/collections/HPLT/instruction-tuning-65dba9834e23db813d863951)

```

export CUDA_VISIBLE_DEVICES= # the GPUs you wish to use

LANG= # the language to train. Please check our `training-data` folder for available languages.
# You can also directly provide a training set to --data_path instead of specifying a language.
BASE_MODEL=bigscience/bloom-7b1 # the base model, as per hugging face naming style

OUTPUT_DIR=# directory to save LoRA checkpoints
mkdir -p ${OUTPUT_DIR}

WORLD_SIZE=4 torchrun --nproc_per_node=4 --master_port 12344 finetune.py \
    --base_model ${BASE_MODEL} \
    --data_path ./data/alpaca_data_cleaned.${LANG}.json \
    --output_dir ./${OUTPUT_DIR} \
    --batch_size 128 \
    --micro_batch_size 2 \
    --num_epochs 5 \
    --learning_rate 3e-4 \
    --cutoff_len 512 \
    --val_set_size 1000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '["query_key_value"]' \
    --train_on_inputs \
    --group_by_length
```

### Inference

We provide an example script to perform inference. You need to provide a test file and LoRA weights--you can download it from our Hugging Face [repository](https://huggingface.co/collections/HPLT/instruction-tuning-65dba9834e23db813d863951) or train your own. Note that the inference code does not do batching so it's not efficient.
```
export CUDA_VISIBLE_DEVICES=$1

TEST_FILE=test_en.jsonl # the path to the test file. Check out our `test-data` folder.

LORA= # this should be the directory containing the LoRA checkpoints/weights.

python generate.py \
    --lora_weights ${LORA} \
    --test_file ${TESTFILE}
```
