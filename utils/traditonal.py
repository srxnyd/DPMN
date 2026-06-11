
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np


def total_variation(image):
    """
    计算图像的总变差
    Args:
        image: 输入图像张量，形状为 (batch_size, channels, height, width)

    Returns:
        total_variation: 总变差张量
    """

    tensor_size = image.size()
    B = tensor_size[0]
    w = tensor_size[1]
    h = tensor_size[2]
    # 计算水平方向上的梯度
    h_gradient = torch.zeros_like(image)
    h_gradient[:, :, :h - 1] = torch.abs(image[ :, :, :-1] - image[ :, :, 1:])
    h_gradient[:,:,h-1:h] = image[:,:, h-1:h]


    # 计算垂直方向上的梯度
    v_gradient = torch.zeros_like(image)
    v_gradient[:, :w-1, :] = torch.abs(image[ :, :-1, :] - image[ :, 1:, :])
    v_gradient[:, w-1:w, :]= image[:,w-1:w,:]


    '''#计算光谱维的差分
    b_gradient = torch.zeros_like(image)
    b_gradient[:b-1,:,:] = torch.abs(image[ :-1, :, :] - image[ 1:, :, :])
    b_gradient[b-1:b,:,:] = image[b-1:b,:,:]'''


    # 求和得到总变差
    tv = low_rank_tensor1(h_gradient) +  low_rank_tensor1(v_gradient)

    return tv


def low_rank_tensor(image):
    """
    计算低秩重构损失
    Args:
        image (torch.Tensor): 输入图像张量，形状为 (batch_size, channels, height, width)

    Returns:
        torch.Tensor: 低秩重构损失值
    """
    channels, height, width = image.shape
    #flatten_image = image.view(batch_size * channels, height , width)
   # nuclear_norm = tensor_nuclear_norm(flatten_image)

    # 将图像转换为二维矩阵形式
    flattened_image = image.view(channels, height * width)

    # 使用SVD进行低秩重构
    U, S, V = torch.svd(flattened_image)

    # 选择前K个奇异值（rank）

    # 计算低秩重构损失
    lrr_loss = torch.sum(S)
    return lrr_loss


def low_rank_tensor1(tensor, tol=0.01):
    """
    使用SVD近似张量并动态选择低秩
    Args:
        tensor: 输入张量
        tol: 低秩近似的阈值

    Returns:
        low_rank_tensor: 低秩近似的张量
    """
    # 对张量进行奇异值分解
    tensor_size = tensor.size()
    batch_size = tensor_size[0]
    w = tensor_size[1]
    h = tensor_size[2]
    tensor_2d = tensor.view(batch_size, -1)
    u, s, v = torch.svd(tensor_2d)

    # 根据阈值动态选择低秩
    rank = torch.sum(s > tol * torch.max(s))
    lrr_loss = torch.sum(s[rank:] ** 2)

    return lrr_loss










