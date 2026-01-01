"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import cv2
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from os.path import join as ospj
from os.path import dirname as ospd
import torch.nn.functional as F

import torch

from utils.utils_loc_metrics import BoxEvaluator
from utils.utils_loc_metrics import MaskEvaluator
from utils.utils_loc_metrics import configure_metadata
from utils.utils import t2n
from captum.attr import ShapleyValueSampling
from nets.tail_model import *
import functools
from nets.sota_wsol.vitol import generate_cam
from nets.sota_wsol.explainability.ViT_explanation_generator import LRP, Baselines
import matplotlib.pyplot as plt

from captum.attr import LayerGradientXActivation, LayerActivation, LayerGradCam

import time

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224



def normalize_scoremap(cam):
    """
    Args:
        cam: 类激活图，numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam


def min_max_normalize(tensor):
    """
    对输入张量进行最大最小归一化，使其值在0到1之间。
    
    参数:
    tensor (torch.Tensor): 输入的张量。
    
    返回:
    torch.Tensor: 归一化后的张量。
    """
    # 找到张量中的最小值和最大值
    tensor_min = torch.min(tensor)
    tensor_max = torch.max(tensor)

    # 如果最大值等于最小值，说明张量中所有元素相同，归一化后张量全为0
    if tensor_min == tensor_max:
        return torch.zeros_like(tensor)

    # 进行最大最小归一化
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)

    return normalized_tensor


def upsample_and_sum(feature_maps):
    """
    对特征图进行上采样并相加
    """
    # 找到最大分辨率
    max_resolution = max([f.shape[-2:] for f in feature_maps])

    upsampled_maps = []
    for f in feature_maps:
        # 对每个特征图进行双线性上采样至最大分辨率
        upsampled = F.interpolate(f, size=max_resolution, mode='bilinear', align_corners=True)
        mean = torch.mean(upsampled, dim=1)
        upsampled_maps.append(mean)

    sum_featuremaps = torch.sum(torch.stack(upsampled_maps), dim=0)
    return sum_featuremaps


def upsample_and_mul(feature_maps, cam=None):
    if cam != None:
        feature_maps.append(cam)
    """
    对特征图进行上采样并相加
    """
    # 找到最大分辨率

    max_resolution = max([f.shape[-2:] for f in feature_maps])

    upsampled_maps = []
    for f in feature_maps:
        # 对每个特征图进行双线性上采样至最大分辨率
        upsampled = F.interpolate(f, size=max_resolution, mode='bilinear', align_corners=True)
        mean = torch.mean(upsampled, dim=1)
        upsampled_maps.append(mean)

    product_tensor = torch.ones_like(upsampled_maps[0])
    for i in upsampled_maps:
        product_tensor *= i
    return product_tensor

def upsample_and_mul2(feature_maps):
    """
    对特征图进行上采样并相加
    """
    # 找到最大分辨率

    max_resolution = max([f.shape[-2:] for f in feature_maps])

    upsampled_maps = []
    for f in feature_maps:
        # 对每个特征图进行双线性上采样至最大分辨率
        upsampled = F.interpolate(f.unsqueeze(1), size=max_resolution, mode='bilinear', align_corners=True)
        upsampled_maps.append(upsampled.squeeze(1))

    product_tensor = torch.ones_like(upsampled_maps[0])
    for i in upsampled_maps:
        product_tensor *= i
    return product_tensor


def upsample_and_mul_and_add(feature_maps):


    feature_map_7x7 = None
    feature_map_14x14 = None
    feature_map_28x28 = None
    # 找到最大分辨率
    max_resolution = max([f.shape[-2:] for f in feature_maps])

    upsampled_maps = []
    for f in feature_maps:
        # 对每个特征图进行双线性上采样至最大分辨率
        _, _, H, W = f.shape
        if H == 7 and W == 7:
            feature_map_7x7 = F.interpolate(f, size=max_resolution, mode='bilinear', align_corners=True)
            feature_map_7x7 = torch.mean(feature_map_7x7, dim=1)
        elif H == 14 and W == 14:
            feature_map_14x14 = F.interpolate(f, size=max_resolution, mode='bilinear', align_corners=True)
            feature_map_14x14 = torch.mean(feature_map_14x14, dim=1)
        elif H == 28 and W == 28:
            feature_map_28x28 = F.interpolate(f, size=max_resolution, mode='bilinear', align_corners=True)
            feature_map_28x28 = torch.mean(feature_map_28x28, dim=1)

    product_tensor = torch.maximum(min_max_normalize(feature_map_7x7 * feature_map_14x14),
                                   min_max_normalize(feature_map_7x7 * feature_map_28x28))
    return product_tensor


class forward_fn(nn.Module):
    def __init__(self, model, target_layer):
        super(forward_fn, self).__init__()
        self.model = model
        self.target_layer = target_layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        return self.model(x, feature_path='high_resoluation')


class HR_TailModel(nn.Module):
    def __init__(self, model):
        super(HR_TailModel, self).__init__()
        self.model = model
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, feature):
        x = self.avgpool(feature)
        x = torch.flatten(x,1)
        logit = self.model.high_fc(x)

        return logit
def refine_cam_with_scg(gt_cam_map, sc_maps, fg_th=0.1, bg_th=0.05):
    B, H, W = gt_cam_map.shape
    refined_cams = []

    for i in range(B):
        cam_map_cls = gt_cam_map[i].detach().cpu().numpy()
        sc_map_cls = 0

        for sc_map in sc_maps:
            sc_map = sc_map[i].detach().squeeze().cpu().numpy()
            wh_sc = sc_map.shape[0]
            h_sc, w_sc = int(np.sqrt(wh_sc)), int(np.sqrt(wh_sc))

            cam_map_cls_resized = cv2.resize(cam_map_cls, dsize=(w_sc, h_sc))
            cam_map_cls_vector = cam_map_cls_resized.reshape(-1)

            # Positive regions
            cam_map_cls_id = np.arange(wh_sc).astype(np.int)
            cam_map_cls_th_ind_pos = cam_map_cls_id[cam_map_cls_vector >= fg_th]
            sc_map_sel_pos = sc_map[:, cam_map_cls_th_ind_pos]
            sc_map_sel_pos = (sc_map_sel_pos - np.min(sc_map_sel_pos, axis=0, keepdims=True)) / (
                    np.max(sc_map_sel_pos, axis=0, keepdims=True) - np.min(sc_map_sel_pos, axis=0,
                                                                           keepdims=True) + 1e-10)

            if sc_map_sel_pos.shape[1] > 0:
                sc_map_sel_pos = np.sum(sc_map_sel_pos, axis=1).reshape(h_sc, w_sc)
                sc_map_sel_pos = (sc_map_sel_pos - np.min(sc_map_sel_pos)) / (
                        np.max(sc_map_sel_pos) - np.min(sc_map_sel_pos) + 1e-10)
            else:
                sc_map_sel_pos = 0

            # Negative regions
            cam_map_cls_th_ind_neg = cam_map_cls_id[cam_map_cls_vector <= bg_th]
            sc_map_sel_neg = sc_map[:, cam_map_cls_th_ind_neg]
            sc_map_sel_neg = (sc_map_sel_neg - np.min(sc_map_sel_neg, axis=0, keepdims=True)) / (
                    np.max(sc_map_sel_neg, axis=0, keepdims=True) - np.min(sc_map_sel_neg, axis=0,
                                                                           keepdims=True) + 1e-10)

            if sc_map_sel_neg.shape[1] > 0:
                sc_map_sel_neg = np.sum(sc_map_sel_neg, axis=1).reshape(h_sc, w_sc)
                sc_map_sel_neg = (sc_map_sel_neg - np.min(sc_map_sel_neg)) / (
                        np.max(sc_map_sel_neg) - np.min(sc_map_sel_neg) + 1e-10)
            else:
                sc_map_sel_neg = 0

            # Refinement step
            sc_map_cls_i = sc_map_sel_pos - sc_map_sel_neg
            sc_map_cls_i = sc_map_cls_i * (sc_map_cls_i >= 0)
            # sc_map_cls_i = (sc_map_cls_i - np.min(sc_map_cls_i)) / (
            #             np.max(sc_map_cls_i) - np.min(sc_map_cls_i) + 1e-10)
            sc_map_cls_i = cv2.resize(sc_map_cls_i, dsize=(W, H))

            sc_map_cls = np.maximum(sc_map_cls, sc_map_cls_i)

        refined_cams.append(sc_map_cls)

    # 转换为numpy
    refined_cams = np.array(refined_cams, dtype=np.float64)

    return refined_cams


class LayerCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor, feature_path='both')

        # Use auxiliary output (aux_logit)
        if isinstance(output, tuple):
            output = output[2]

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[:, target_class]
        class_score = class_score.sum()
        class_score.backward()

        gradients = self.gradients
        activations = self.activations

        # Directly multiply gradients with activations
        gradients = F.relu(gradients)
        cam = gradients * activations

        cam = F.relu(cam)

        return cam
    def generate_class_agnostic_cam(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor, feature_path='high_resoluation')

        # Use auxiliary output (aux_logit)
        if isinstance(output, tuple):
            output = output[2]

        class_agnostic_cam = self.activations
        return class_agnostic_cam



class CAMComputer(object):
    def __init__(self, model, loader, metadata_root, mask_root,
                 iou_threshold_list, dataset_name, split,
                 multi_contour_eval, epoch_step, epoch, Epoch,
                 cam_curve_interval=.001, log_folder=None, backbone=None, norm_method='ivr', percentile=0.3,
                 args=None):
        """
        :param model: 模型
        :param loader: dataloader
        :param metadata_root: localization file path
        :param mask_root: mask localization file path
        :param iou_threshold_list: 阈值列表
        :param dataset_name: 数据集
        :param split: train or val or test
        :param multi_contour_eval: 是否一个对应多个标签
        :param cam_curve_interval:
        :param log_folder: 保存的路径
        """
        self.model = model
        self.model.eval()
        self.loader = loader
        self.split = split
        self.log_folder = log_folder
        self.epoch = epoch
        self.epoch_step = epoch_step
        self.Epoch = Epoch
        self.backbone = backbone
        self.norm_method = norm_method
        self.percentile = percentile

        if args != None:
            self.args = args

        metadata = configure_metadata(metadata_root)
        cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))

        self.evaluator = {"OpenImages": MaskEvaluator,
                          "CUB": BoxEvaluator,
                          "ILSVRC": BoxEvaluator,
                          "PN2": BoxEvaluator,
                          "C45V2": BoxEvaluator,
                          }[dataset_name](metadata=metadata,
                                          dataset_name=dataset_name,
                                          split=split,
                                          cam_threshold_list=cam_threshold_list,
                                          iou_threshold_list=iou_threshold_list,
                                          mask_root=mask_root,
                                          multi_contour_eval=multi_contour_eval)

    
    def compute_and_evaluate_cams2(self):
        if self.epoch == None or self.Epoch == None:
            pbar = tqdm(total=self.epoch_step, postfix=dict, mininterval=0.3)
        else:
            pbar = tqdm(total=self.epoch_step, desc=f'Epoch {self.epoch + 1}/{self.Epoch}', postfix=dict,
                        mininterval=0.3)

        for images, targets, image_ids in self.loader:
            image_size = images.shape[2:]
            images = images.cuda()

            feature_maps = []
            outputs = self.model(images, labels=targets, return_cam=True, n_layers=self.args.num_cct, attention_type = self.args.attention_type)

            if len(outputs) == 3:
                cls_attentions, patch_attn, conv_cams  = outputs
                B, c1, w1, h1 = cls_attentions.shape
                # print(patch_attn.shape, conv_cams.shape)
                device = cls_attentions.device
                
                patch_attn = torch.sum(patch_attn, dim=0) # 16, 196. 196
                if self.args.patch_attn_refine:
                    # 在layer的维度进行压缩
                    
                    cls_attentions = torch.matmul(patch_attn.unsqueeze(1), cls_attentions.view(B, c1, -1, 1)).reshape(B, c1, w1, h1) #16，1，196，196 * 16，11，196，1 -> 16，11，196，1 ->16，11，14. 14
                    labels = targets.view(B, 1, 1, 1).expand(-1, 1, w1, h1).to(device)
                    cls_attentions = torch.gather(cls_attentions, 1, labels).squeeze(1)
                    feature_maps.append(cls_attentions)
                    # print('cls-refine')
                else:
                    labels = targets.view(B, 1, 1, 1).expand(-1, 1, w1, h1).to(device)
                    cls_attentions = torch.gather(cls_attentions, 1, labels).squeeze(1)
                    feature_maps.append(cls_attentions)
                if self.args.conv_attn_refine:
                    conv_cams_upsampled = F.interpolate(conv_cams.unsqueeze(1), size=(w1, h1), mode='bilinear', align_corners=True)
                    conv_cams_upsampled = torch.matmul(patch_attn.unsqueeze(1), conv_cams_upsampled.view(B, 1, -1, 1)).reshape(B, 1, w1, h1).squeeze(1)
                    feature_maps.append(conv_cams_upsampled)
                    # print('conv-refine')
                else:
                    feature_maps.append(conv_cams)

                fused = upsample_and_mul2(feature_maps)
                cams = t2n(fused)

            else:
                raise ValueError
            
            for cam, image_id in zip(cams, image_ids):
                cam_resized = cv2.resize(cam, image_size,
                                         interpolation=cv2.INTER_CUBIC)
                cam_normalized = self.normalize_scoremap(cam_resized)
                if self.split in ('val', 'test'):
                    cam_path = ospj(self.log_folder, 'scoremaps', image_id)
                    if not os.path.exists(ospd(cam_path)):
                        os.makedirs(ospd(cam_path))
                    np.save(ospj(cam_path), cam_normalized)
                self.evaluator.accumulate(cam_normalized, image_id)
            pbar.update(1)
        return self.evaluator.compute()
        
    def compute_and_evaluate_cams2_fps(self):
        if self.epoch == None or self.Epoch == None:
            pbar = tqdm(total=self.epoch_step, postfix=dict, mininterval=0.3)
        else:
            pbar = tqdm(total=self.epoch_step, desc=f'Epoch {self.epoch + 1}/{self.Epoch}', postfix=dict,
                        mininterval=0.3)

        # 记录总推理时间和推理次数
        total_inference_time = 0
        num_inferences = 0

        for images, targets, image_ids in self.loader:
            start_time = time.time()  # 开始计时

            image_size = images.shape[2:]
            images = images.cuda()

            feature_maps = []
            outputs = self.model(images, labels=targets, return_cam=True, attention_type=self.args.attention_type)

            if len(outputs) == 3:
                cls_attentions, patch_attn, conv_cams = outputs
                B, c1, w1, h1 = cls_attentions.shape
                feature_maps.append(conv_cams)
                if self.args.patch_attn_refine:
                    device = cls_attentions.device

                    # 在 layer 的维度进行压缩
                    patch_attn = torch.sum(patch_attn, dim=0)  # 16, 196, 196
                    cls_attentions = torch.matmul(
                        patch_attn.unsqueeze(1), cls_attentions.view(B, c1, -1, 1)
                    ).reshape(B, c1, w1, h1)  # 计算 refined cls_attentions
                    labels = targets.view(B, 1, 1, 1).expand(-1, 1, w1, h1).to(device)
                    cls_attentions = torch.gather(cls_attentions, 1, labels).squeeze(1)
                    feature_maps.append(cls_attentions)

                    fused = upsample_and_mul2(feature_maps)
                    cams = t2n(fused)
                else:
                    device = cls_attentions.device

                    labels = targets.view(B, 1, 1, 1).expand(-1, 1, w1, h1).to(device)
                    cls_attentions = torch.gather(cls_attentions, 1, labels).squeeze(1)
                    feature_maps.append(cls_attentions)

                    fused = upsample_and_mul2(feature_maps)
                    cams = t2n(fused)

            end_time = time.time()  # 结束计时

            # 计算单次推理时间
            inference_time = end_time - start_time
            total_inference_time += inference_time
            num_inferences += 1

            # 输出单次推理时间（可选）
            print(f"Single inference time: {inference_time:.4f} seconds")

        # 计算平均 FPS
        average_fps = num_inferences / total_inference_time if total_inference_time > 0 else 0
        print(f"Average FPS: {average_fps:.2f}")

        return average_fps
                

    def normalize_scoremap(self, cam):
        """
            Args:
                cam: numpy.ndarray(size=(H, W), dtype=np.float)
            Returns:
                numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
                If input array is constant, a zero-array is returned.
            """
        if np.isnan(cam).any():
            return np.zeros_like(cam)
        if cam.min() == cam.max():
            return np.zeros_like(cam)
        if self.norm_method == 'minmax':
            cam -= cam.min()
            cam /= cam.max()
        elif self.norm_method == 'max':
            cam = np.maximum(0, cam)
            cam /= cam.max()
        elif self.norm_method == 'pas':
            cam -= cam.min()
            cam_copy = cam.flatten()
            cam_copy.sort()
            maxx = cam_copy[int(cam_copy.size * 0.9)]
            cam /= maxx
            cam = np.minimum(1, cam)
        elif self.norm_method == 'ivr':
            cam_copy = cam.flatten()
            cam_copy.sort()
            minn = cam_copy[int(cam_copy.size * self.percentile)]
            cam -= minn
            cam = np.maximum(0, cam)
            cam /= cam.max()
        else:
            print('Norm not defined')
        return cam
