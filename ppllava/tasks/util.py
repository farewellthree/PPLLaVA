
import torch
import numpy as np

def get_sim_matrix(model, t_feat_list, v_feat_list, mini_batch=32, T=None):
    batch_t_feat = torch.split(t_feat_list, mini_batch)
    batch_v_feat = torch.split(v_feat_list, mini_batch)
    sim_matrix = []
    with torch.no_grad():
        for idx1, t_feat in enumerate(batch_t_feat):
            # logger.info('batch_list_t [{}] / [{}]'.format(idx1, len(batch_list_t)))
            each_row = []
            for idx2, v_feat in enumerate(batch_v_feat):
                if model.wdim=='cls':
                    b1b2_logits = t_feat.cpu().numpy() @ v_feat.cpu().numpy().T
                else:
                    retrieve_logits = torch.einsum('ad,bvd->abv', [t_feat, v_feat])
                    tv_softmax = torch.softmax(retrieve_logits*100, dim=-1) 
                    similarity = torch.sum(retrieve_logits * tv_softmax, dim=-1)
                    b1b2_logits = similarity.cpu().numpy()
                
                b1b2_logits = b1b2_logits
                each_row.append(b1b2_logits)
            each_row = np.concatenate(tuple(each_row), axis=-1)
            sim_matrix.append(each_row)
    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
    return sim_matrix