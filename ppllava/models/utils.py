import re
import os
import numpy as np

import torch
import torch.nn.functional as F
from safetensors import safe_open

def RandomMaskingGenerator(num_patches, mask_ratio, batch, device='0'):
    num_mask = int(mask_ratio * num_patches)

    mask_list = []
    for _ in range(batch):
        mask = np.hstack([
            np.zeros(num_patches - num_mask),
            np.ones(num_mask),
        ])
        np.random.shuffle(mask)
        mask_list.append(mask)
    mask = torch.Tensor(mask_list).to(f'cuda:{device}' if isinstance(device, int) else device, non_blocking=True).to(f'cuda:{torch.bool}' if isinstance(torch.bool, int) else torch.bool)
    return mask

def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return  torch.tensor(sinusoid_table,dtype=torch.float, requires_grad=False).unsqueeze(0) 

def get_size(input_size, output_size=None, kernel=None, stride=None):
    if input_size is None:
        raise ValueError("input_size must not be None")
    
    I_D, I_H, I_W = input_size
    
    if kernel is None and stride is None:
        if output_size is None:
            raise ValueError("output_size must not be None when both kernel and stride are None")
        O_D, O_H, O_W = output_size
        stride_D = I_D // O_D
        stride_H = I_H // O_H
        stride_W = I_W // O_W
        kernel_D = I_D - (O_D - 1) * stride_D
        kernel_H = I_H - (O_H - 1) * stride_H
        kernel_W = I_W - (O_W - 1) * stride_W
        return (O_D, O_H, O_W), (kernel_D, kernel_H, kernel_W), (stride_D, stride_H, stride_W)

    if output_size is None and kernel is not None and stride is not None:
        kernel_D, kernel_H, kernel_W = kernel
        stride_D, stride_H, stride_W = stride
        O_D = (I_D - kernel_D) // stride_D + 1
        O_H = (I_H - kernel_H) // stride_H + 1
        O_W = (I_W - kernel_W) // stride_W + 1
        return (O_D, O_H, O_W), kernel, stride
    
    elif kernel is None and output_size is not None and stride is not None:
        O_D, O_H, O_W = output_size
        stride_D, stride_H, stride_W = stride
        kernel_D = I_D - (O_D - 1) * stride_D
        kernel_H = I_H - (O_H - 1) * stride_H
        kernel_W = I_W - (O_W - 1) * stride_W
        return output_size, (kernel_D, kernel_H, kernel_W), stride
    
    elif stride is None and output_size is not None and kernel is not None:
        O_D, O_H, O_W = output_size
        kernel_D, kernel_H, kernel_W = kernel
        stride_D = (I_D - kernel_D) // (O_D - 1)
        stride_H = (I_H - kernel_H) // (O_H - 1)
        stride_W = (I_W - kernel_W) // (O_W - 1)
        return output_size, kernel, (stride_D, stride_H, stride_W)

    else:
        raise ValueError("Invalid combination of parameters. One of output_size, kernel, or stride must be None.")

def weighted_adaptive_avg_pool3d_loop(input, output_size, weights=None, temperature=0.01):
    N, C, D, H, W = input.shape
    out_D, out_H, out_W = output_size

    if weights is None:
        weights = torch.ones((N, D, W, H))
    # 计算每个维度的步长和卷积核大小
    stride_D = D // out_D
    stride_H = H // out_H
    stride_W = W // out_W

    kernel_D = D - (out_D - 1) * stride_D
    kernel_H = H - (out_H - 1) * stride_H
    kernel_W = W - (out_W - 1) * stride_W

    # 初始化输出张量
    output = torch.zeros((N, C, out_D, out_H, out_W), device=f'cuda:{input.device}' if isinstance(input.device, int) else input.device, dtype=input.dtype)

    for n in range(N):
        for i in range(out_D):
            for j in range(out_H):
                for k in range(out_W):
                    start_D = i * stride_D
                    start_H = j * stride_H
                    start_W = k * stride_W

                    end_D = start_D + kernel_D
                    end_H = start_H + kernel_H
                    end_W = start_W + kernel_W

                    region = input[n, :, start_D:end_D, start_H:end_H, start_W:end_W]  # 取出整个通道维度
                    weight_region = weights[n, start_D:end_D, start_H:end_H, start_W:end_W]

                    # 应用温度缩放的 softmax 操作
                    weight_region = F.softmax(weight_region.reshape(-1) / temperature, dim=0).reshape(weight_region.shape)

                    # 计算加权平均值
                    weighted_sum = (region * weight_region.unsqueeze(0)).sum(dim=(1, 2, 3))
                    output[n, :, i, j, k] = weighted_sum

    return output

def weighted_adaptive_avg_pool3d_unfold(input, output_size=None, kernel=None, stride=None, weights=None, temperature=0.01):
    N, C, D, H, W = input.shape

    output_size, kernel, stride = get_size((D, H, W), output_size, kernel, stride)
    out_D, out_H, out_W = output_size
    stride_D, stride_H, stride_W = stride
    kernel_D, kernel_H, kernel_W = kernel

    if weights is None:
        weights = torch.ones((N, D, W, H)).to(f'cuda:{input.device}' if isinstance(input.device, int) else input.device, input.dtype)

    # 展开输入张量和权重张量
    input_unf = input.unfold(2, kernel_D, stride_D).unfold(3, kernel_H, stride_H).unfold(4, kernel_W, stride_W)
    weights_unf = weights.unfold(1, kernel_D, stride_D).unfold(2, kernel_H, stride_H).unfold(3, kernel_W, stride_W)

    # 调整形状以便于计算
    input_unf = input_unf.contiguous().view(N, C, out_D, out_H, out_W, -1)
    weights_unf = weights_unf.contiguous().view(N, out_D, out_H, out_W, -1)

    # 应用温度缩放的 softmax 操作
    weights_unf = F.softmax(weights_unf / temperature, dim=-1)

    # 计算加权平均值
    weighted_sum = (input_unf * weights_unf.unsqueeze(1)).sum(dim=-1)

    return weighted_sum


def get_state_dict(path, prefix='(model|non_lora_trainable)'):
    pattern = re.compile(r'^(model|non_lora_trainable).*?(\.safetensors|\.bin)$')
    matching_files = [filename for filename in os.listdir(path) if pattern.match(filename)]
    model_state_dict = {}
    for model_path in matching_files:
        if model_path.endswith('safetensors'):
            with safe_open(os.path.join(path,model_path), framework="pt", device='cpu') as f:
                for k in f.keys():
                    model_state_dict[k] = f.get_tensor(k)
        elif model_path.endswith('bin'):
            partial_state_dict = torch.load(os.path.join(path,model_path), map_location=torch.device('cpu'))
            model_state_dict.update(partial_state_dict)
    return model_state_dict 

