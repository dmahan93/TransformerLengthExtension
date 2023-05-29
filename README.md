# Transformer Length Extension

This is a training pipeline for length extension of transformer models.

Currently only Transformer XL is supported. To start:
```bash
python CreateXlDataset.py
deepspeed run_clm.py --model_name_or_path=EleutherAI/pythia-410m-deduped --per_device_train_batch_size=8 --num_train_epochs 4 --save_strategy=epoch --output_dir=pythia-410m-xl --report_to "wandb" --dataset_name dmayhem93/toolformer-v0-postprocessed --tokenizer_name EleutherAI/pythia-410m-deduped --block_size 1024 --gradient_accumulation_steps 1 --do_train  --logging_strategy=epoch --bf16 --overwrite_output_dir --adam_beta1=0.9 --adam_beta2=0.95 --weight_decay=2e-02 --learning_rate=1e-05 --warmup_steps=100 --per_device_eval_batch_size=1 --cache_dir="hf_cache" --gradient_checkpointing=True --deepspeed ds_config_pythia_410m.json
```

Note: I haven't actually trained anything yet.