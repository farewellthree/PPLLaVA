export PYTHONPATH="./:$PYTHONPATH"
torchrun --master_port=49344  --nproc_per_node=8 ppllava/train/train_hf.py --cfg-path config/vicuna/ppllava_vicuna7b_image_video.yaml