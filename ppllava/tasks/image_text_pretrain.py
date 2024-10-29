"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import numpy as np
from collections import OrderedDict
from ppllava.common.registry import registry
from ppllava.tasks.base_task import BaseTask
from ppllava.datasets.datasets.instruction_data import available_corpus, train_transform
from ppllava.datasets.datasets.image_video_itdatasets import ITImgTrainDataset, ITVidTrainDataset,\
    LLaVAITVidTrainDataset, PTVidTrainDataset, PTImgTrainDataset
from ppllava.datasets.datasets.interleave_sft_dataset import Interleave_sft_dataset
from .util import get_sim_matrix

@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        all_text_embed = []
        all_vid_embed = []
        for data in data_loader:
            data.pop('dict')
            result = model(return_loss=False, **data)
            all_text_embed.append(result.text_embeds.cpu())
            all_vid_embed.append(result.image_embeds.cpu())

        torch.cuda.empty_cache()
        video_features = torch.cat(all_vid_embed, dim=0).cuda()
        text_features = torch.cat(all_text_embed, dim=0).cuda()

        #if model.wdim=='cls':
        #    similarity = text_features.cpu().numpy() @ video_features.cpu().numpy().T
        #else:
        #    retrieve_logits = torch.einsum('ad,bvd->abv', [text_features, video_features])
        #    tv_softmax = torch.softmax(retrieve_logits*100, dim=-1) 
        #    similarity = torch.sum(retrieve_logits * tv_softmax, dim=-1)
        #    similarity = similarity.cpu().numpy()
        
        similarity = get_sim_matrix(model, text_features, video_features)

        sx = np.sort(-similarity)
        d = np.diag(-similarity)
        ind = np.where((sx - d[:, None]) == 0)[1]

        metric_list = ['R1', 'R5', 'R10', 'MdR', 'MnR']
        metrics = OrderedDict()
        for metric in metric_list:
            if metric == 'R1':
                metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
            elif metric == 'R5':
                metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
            elif metric == 'R10':
                metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
            elif metric == 'MdR':
                metrics['MdR'] = np.median(ind) + 1
            elif metric == 'MnR':
                metrics['MnR'] = np.mean(ind) + 1
        
        print (metrics)
        return metrics
        
@registry.register_task("video_text_it")
class VideoTextItTask(ImageTextPretrainTask):
    def __init__(self):
        super().__init__()

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """
        datasets = dict()
        datasets_config = cfg.datasets_cfg
        assert len(datasets_config) > 0, "At least one dataset has to be specified."
        simple = cfg.model_cfg.get('qformer_text_input',False)
        llava = ('llava_vid' in cfg.model_cfg.get('arch'))
        for name in datasets_config:
            dataset_config = datasets_config[name]
            dataset_info = available_corpus[name]
            if llava:
                dataset_cls = LLaVAITVidTrainDataset
                dataset_config['llama_model'] = cfg.model_cfg.get('llama_model')
                dataset_config['no_grid'] = (cfg.model_cfg.get('arch') == 'llava_vid_nogrid')
                dataset_config['clip_model'] = cfg.model_cfg.get('clip_weight', None)
            else:
                dataset_cls = ITImgTrainDataset if get_media_type(dataset_info)=="image" else ITVidTrainDataset

            datasets[name] = {'train': dataset_cls(ann_file=dataset_info, simple=simple,
                        transform=train_transform, **dataset_config)}

        return datasets

@registry.register_task("video_text_ft")
class VideoTextFtTask(ImageTextPretrainTask):
    def __init__(self):
        super().__init__()

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """
        datasets = dict()
        datasets_config = cfg.datasets_cfg
        assert len(datasets_config) > 0, "At least one dataset has to be specified."
  
        for name in datasets_config:
            dataset_config = datasets_config[name]
            dataset_info = available_corpus[name]
            
            dataset_cls = PTImgTrainDataset if get_media_type(dataset_info)=="image" else PTVidTrainDataset

            phrase = 'eval' if name=='msrvtt' else 'train'
            datasets[name] = {phrase: dataset_cls(ann_file=dataset_info, model_config=cfg.model_cfg,
                        transform=train_transform, **dataset_config)}

        return datasets

@registry.register_task("interleave_sft")
class InterLeaveTask(ImageTextPretrainTask):
    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """
        datasets = dict()
        datasets_config = cfg.datasets_cfg
        assert len(datasets_config) > 0, "At least one dataset has to be specified."
        
        interleave_datasets = datasets_config.pop('interleave_datasets')
        all_dataset = []
        for name in interleave_datasets:
            dataset_config = interleave_datasets[name]
            dataset_info = available_corpus[name]
            dataset_config['label_file'] = dataset_info[0]
            dataset_config['data_root'] = dataset_info[1]
            dataset_config['media_type'] = get_media_type(dataset_info)
            all_dataset.append(dataset_config)

        datasets['dataset'] = {'train': Interleave_sft_dataset(all_dataset, cfg.model_cfg, **datasets_config)}
        return datasets


def get_media_type(dataset_info):
    if len(dataset_info) == 3 and dataset_info[2] == "video":
        return "video"
    elif len(dataset_info) == 3 and dataset_info[2] == "multi-image":
        return "multi-image"
    else:
        return "image"
