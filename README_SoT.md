# (Optional) fresh environment
# conda create -y -n iclr26_SoT python=3.10 && conda activate iclr26_SoT
pip install "torch>=2.2" "transformers>=4.43" "accelerate>=0.31" "trl>=0.9.6" "datasets>=2.20" "vllm>=0.5.4" math-verify

# Run full fine-tuning (paths are relative and anonymized)
python train_iclr.py \
  --model_name Qwen/Qwen2.5-1.5B \
  --tokenizer_name Qwen/Qwen2.5-1.5B \
  --train_dataset_path data/train.jsonl \
  --valid_dataset_path data/valid.jsonl \
  --output_dir outputs/iclr_run \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-6 \
  --num_train_epochs 10 \
  --seq_length 1024 \
  --use_validation_split False


# Run evals on math or gsm8k 

python eval_math_gsm8k.py \
  --eval_mode checkpoints \
  --checkpoint_dir ckpts/full_checkpoints \
  --checkpoints checkpoint-100 checkpoint-200 \
  --dataset data/test.jsonl \
  --output_dir outputs/eval_run \
  --output_suffix fullckpt_eval \
  --shot_mode zero_shot \
  --tensor_parallel_size 1


# Run evals on countdown

python eval_countdown.py \
  --eval_mode checkpoints \
  --checkpoint_dir ckpts/full_checkpoints \
  --checkpoints checkpoint-100 checkpoint-200 \
  --dataset data/test.jsonl \
  --output_dir outputs/eval_run \
  --output_suffix fullckpt_eval \
  --shot_mode zero_shot \
  --tensor_parallel_size 1

