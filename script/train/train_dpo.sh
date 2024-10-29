export PYTHONPATH="./:$PYTHONPATH"
torchrun --master_port=49344  --nproc_per_node=8 ppllava/train/train_dpo.py --cfg-path config/dpo/dpo_ppllava_vicuna7b.yaml