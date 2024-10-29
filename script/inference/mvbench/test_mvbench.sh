export PYTHONPATH="./:$PYTHONPATH"
torchrun --nproc_per_node=8 --master_port=33999 ppllava/test/mvbench/mv_bench_infer_mp.py \
    --cfg-path config/vicuna/ppllava_vicuna7b_image_video.yaml\
    --ckpt-path /Path/to/ppllava_vicuna7b_image_video \
    --output_dir test_output/mvbench/ \
    --output_name ppllava_vicuna7b_image_video \
    --num-frames 32 \
    --ask_simple \
    --system_llm \
    
    

