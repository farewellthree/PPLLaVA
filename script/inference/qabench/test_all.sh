export PYTHONPATH="./:$PYTHONPATH"
torchrun --nproc_per_node=8 --master_port=33009 stllm/test/qabench/qabench_infer_mp.py \
    --cfg-path config/vicuna/ppllava_vicuna7b_image_video.yaml \
    --ckpt-path /Path/to/ppllava_vicuna7b_image_video \
    --output_dir test_output/qabench/ppllava_vicuna7b_image_video \
    --num-frames 32 \
    --dataset ActivityNet \

#dataset option: ActivityNet  MSRVTT   MSVD
