## 1. Prepare the Pretrained Weights
Although some weights can be downloaded dynamically at runtime, it is recommended to pre-download them for speeding up each run.

### LLaVA-Next-hf 
```
git clone https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf
```
### CLIP-L14-336
```
git clone https://huggingface.co/openai/clip-vit-large-patch14-336
```
the path of LLaVA and CLIP Weight can be modified in ```llama_model``` and ```clip_weight``` in each config

## 2. Training 
#### Data
Annotations for all training datasets involved can be found [here](https://huggingface.co/datasets/farewellthree/ppllava_json).

For kinetics, ssv2, next_qa, clevrer_qa and clevrer_mc, please follow the [VideoChat2 instructions](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/DATA.md) to prepare the videos.

For llava_v1.5_300k, please follow the [LLaVA data instructions](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/DATA.md) to prepare the images.

Videos from LLava-Hound-300k can be found [here](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/tree/main/train_300k).

Multiple images from M4-Instruction can be found [here](https://huggingface.co/datasets/lmms-lab/M4-Instruct-Data/tree/main).

#### Run SFT
Please first modify the path in [train script](script/train/train_sft.sh) for the desired config from [config folder](config), then run
```
bash script/train/train_sft.sh
```

#### Run DPO
Please first modify the path in [train script](script/train/train_dpo.sh) for the desired config from [config folder](config), then run
```
bash script/train/train_dpo.sh
```

## 3. Inference
#### VideoMME 
Please first modify the checkpoint path in [test script](script/inference/videomme/test_videomme.sh) and annotation path in [ppllava/test/video_mme/utils.py](ppllava/test/video_mme/utils.py), then run 
```
bash script/inference/videomme/test_videomme.sh
```

#### MVBench 
Please first modify the checkpoint path in [test script](script/inference/mvbench/test_mvbench.sh) and annotation path in [ppllava/test/mvbench/utils.py](ppllava/test/mvbench/utils.py), then run 
```
bash script/inference/mvbench/test_mvbench.sh
```

#### VcgBench 
All evaluation scripts can be found [here](script/inference/vcgbench).
To achieve the result files for all evaluation items, please first modify the checkpoint path in [test script](script/inference/vcgbench/test_all.sh) and annotation path in [ppllava/test/vcgbench/utils.py](ppllava/test/vcgbench/utils.py), then run 
```
bash script/inference/vcgbench/test_all.sh
```
and then execute the corresponding evaluation script to perform benchmarking, for example, for temporal score, run:
```
bash script/inference/vcgbench/score_temporal.sh
```

#### VideoQABench
All testing procedures are identical to VCGbench， where all evaluation scripts are [here](script/inference/qabench).
For instance, to evaluate the result on MSVD, we first modify the checkpoint path and annotation path, then run
```
bash script/inference/qabench/test_all.sh
```
and run
```
bash script/inference/qabench/score_msvd.sh
