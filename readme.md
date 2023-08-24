# LLM Science

## Basic Setting
This is a repo for [https://www.kaggle.com/competitions/kaggle-llm-science-exam/](https://www.kaggle.com/competitions/kaggle-llm-science-exam/)

We plan to use [https://huggingface.co/meta-llama/Llama-2-13b-hf](meta-llama/Llama-2-13b-hf) as our backbone LLM.


`data/`: The data folder holds a variety of data that can be used to instruct-tuning the LLMs.

`utils/`: The utils folder holds a variety of tool py files related to data processing and others.

`example.py`: demo

### available datalist: 
- [https://huggingface.co/datasets/Sangeetha/Kaggle-LLM-Science-Exam](https://huggingface.co/datasets/Sangeetha/Kaggle-LLM-Science-Exam)
- [https://huggingface.co/datasets/derek-thomas/ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA)
- [https://huggingface.co/datasets/openbookqa](https://huggingface.co/datasets/openbookqa)
## Training

The training process can use [https://github.com/hiyouga/LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning) repo, and make the following modifications:
> 1. move our own instruction dataset to `/data`.
> 2. modified `data/dataset_info.json`` to add our own instruction dataset.
> 3. run the training script.

The training script:

```bash
torchrun --nproc_per_node=8 --master_port==1134 src/train_bash.py \
    --stage sft \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --do_train \
    --dataset our_own_science_dataset \
    --template llama2 \
    --finetuning_type full \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 2.0 \
    --plot_loss \
    --bf16 \
    --deepspeed deepspeed.json
```
where deepspeed.json is:
```json
{
  "bf16": {
    "enabled": "auto"
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
        "lr": "auto",
        "weight_decay": "auto",
        "torch_adam": true,
        "adam_w_mode": true
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
        "warmup_min_lr": "auto",
        "warmup_max_lr": "auto",
        "warmup_num_steps": "auto",
        "total_num_steps": "auto"
    }
  },
  "zero_optimization": {
      "stage": 2,
      "allgather_partitions": true,
      "allgather_bucket_size": 2e8,
      "reduce_scatter": true,
      "reduce_bucket_size": "auto",
      "overlap_comm": true,
      "contiguous_gradients": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 5,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

The format of the instruct-tuning for llama2 is the same as for alpaca, which is the following format:
```json
{
    "instruction": "Describe a time when you had to make a difficult decision.",
    "input": "",
    "output": "As an AI assistant, I do not have my own personal experiences or physical existence. However, I am programmed to make decisions based on algorithms, logic, and data. There are times when I encounter a situation where making a decision is difficult because there may be conflicting information or multiple possible outcomes. In such cases, I use my ability to analyze and weigh the risks and benefits of each option and come to a conclusion that is most likely to achieve the desired result."
},
```
🪛 First edition dataprompt:
```json
{
    "instruction": "The following is a multiple choice question, answer to the best of your capabilities.",
    "input": "Question: {The question of the science QA}\nA:{}\nB:{}\nC:{}\nD:{}\nE:{}",
    "output": "{Ground Truth}"
},
```

-----------------------------------------------------------------

## Experiments:

| Models | Dates | MAP@3 (test) |  Des  |
|--------| ------|-------|-------|
| LLAMA2-13B | 2023-8-23 | 69.92 | use 5-shot to test the performance of vallia LLAMA2-13B



| Models | Dates | MAP@3 (private) |  Des  |
|--------| ------|-------|-------|
| LLAMA2-13B | 2023-8-23 | xx.xx | use 5-shot to test the performance of vallia LLAMA2-13B

