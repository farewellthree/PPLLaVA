export PYTHONPATH="./:$PYTHONPATH"
python ppllava/test/gpt_evaluation/evaluate_activitynet_qa.py \
    --pred_path test_output/qabench/ppllava_vicuna7b_image_video/MSRVTT.json \
    --output_dir test_output/qabench/ppllava_vicuna7b_image_video/score_msrvttQA \
    --output_json score_msrvttQA.json \
    --api_key openai_api_key \
    --num_tasks 3