# Works on A100 80G single GPU
/Library/Frameworks/Python.framework/Versions/3.8/bin/python3 finetune_lora.py \
--model_name_or_path='beomi/KoAlpaca-Polyglot-12.8B' \
--train_file='.data.json' \
--num_train_epochs=3 \
--block_size=2048 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=64 \
--torch_dtype=float16 \
--fp16 \
--output_dir='output' \
--do_train \
--optim='adafactor' \
--learning_rate='2e-5' \
--logging_strategy='steps' \
--logging_first_step \
--run_name='polyglot-12.8b-gme-v1-singlegpu-qlora' \
--low_cpu_mem_usage
