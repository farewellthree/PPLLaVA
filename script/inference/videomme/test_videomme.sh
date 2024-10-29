export PYTHONPATH="./:$PYTHONPATH"
torchrun --nproc_per_node=8 --master_port=43099 ppllava/test/video_mme/videomme_infer.py \
    --cfg-path config/vicuna/ppllava_vicuna7b_image_video_multiimage.yaml \
    --ckpt-path /Path/to/ppllava_vicuna7b_image_video_multiimage \
    --output_dir test_output/videomme/ \
    --output_name ppllava_vicuna7b_image_video_multiimage \
    --num-frames 0 \      
    --use_subtitles \
#--num-frames=0 means more frames for long video
    
    
    

