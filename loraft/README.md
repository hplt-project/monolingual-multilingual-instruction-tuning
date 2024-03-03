## Low-Rank Adaptation

### Preparation
You might need to install the packages in the `requirements.txt` file. It is suggested that you use an environment manager, e.g. `conda`.

### Training

We provide an example script to train BLOOM-7B1 below. It is worth noting that you might need to change `WORLD_SIZE` depending on your GPU configuration as well as the `--lora_target_modules` option depending on where you would like to apply LoRA. The name also varies by the base model you use.

```

export CUDA_VISIBLE_DEVICES= # the GPUs you wish to use

LANG= # the language to train. Please check our `training-data` folder for available languages.
BASE_MODEL=bigscience/bloom-7b1 # the base model, as per hugging face naming style

OUTPUT=lora-bloom-7b1 # you can change this
OUTPUT_DIR=# directory to save checkpoints

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

We provide an example script to perform inference below.
```
export CUDA_VISIBLE_DEVICES=$1

TEST_FILE=test_en.jsonl # the path to the test file. Check out our `test-data` folder.

LORA= # this should be the output directory where you saved LoRA checkpoints from training.

python generate.py \
    --lora_weights ${LORA} \
    --test_file ${TESTFILE}
```