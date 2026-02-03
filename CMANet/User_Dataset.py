import os
from random import sample
import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class MSCNN_Dataset(Dataset):
    def __init__(
            self,
            optical_image_list,
            sar_image_list,
            label_list2,
            label_list3,
            normalize=True
    ):
        """
        参数初始化，获取样本路径以及标签等信息
        parameters:
            optical_image_list: 光学样本list
            sar_image_list: 雷达样本list
            label_list2: 第二层次标签
            label_list3: 第三层次标签
        """
        super(MSCNN_Dataset, self).__init__()
        self.optical_image_list = optical_image_list
        self.sar_image_list = sar_image_list
        self.label_list2 = label_list2
        self.label_list3 = label_list3
        self.normalize = normalize

    def __getitem__(self, index):
        optical_image = tiff.imread(self.optical_image_list[index]).astype(np.float32)
        sar_image = tiff.imread(self.sar_image_list[index]).astype(np.float32)
        label2 = self.label_list2[index]
        label3 = self.label_list3[index]
        if self.normalize:
            optical_image = self.normalize_image(optical_image)
            sar_image = self.normalize_image(sar_image)
        optical_image = torch.from_numpy(optical_image)
        sar_image = torch.from_numpy(sar_image)
        sample = {
            'optical_image': optical_image,
            'sar_image': sar_image,
            'label2': label2,
            'label3': label3
        }
        return sample  # 组合成字典类型输出

    def __len__(self):
        return len(self.optical_image_list)

    def normalize_image(self, image):
        image_min = np.min(image)
        image_max = np.max(image)
        normalized_image = (image - image_min) / (image_max - image_min)
        return normalized_image


class CMANet_Dataset(Dataset):
    def __init__(
            self,
            optical_image_list,
            sar_image_list,
            label_list,
            normalize=True
    ):
        """
        参数初始化，获取样本路径以及标签等信息
        parameters:
            optical_image_list: 光学样本list
            sar_image_list: 雷达样本list
            label_list2: 第二层次标签
            label_list3: 第三层次标签
        """
        super(CMANet_Dataset, self).__init__()
        self.optical_image_list = optical_image_list
        self.sar_image_list = sar_image_list
        self.label_list = label_list
        self.normalize = normalize

    def __getitem__(self, index):
        optical_image = tiff.imread(self.optical_image_list[index]).astype(np.float32)
        # optical_image = (optical_image-optical_image.min())/(optical_image.max()-optical_image.min())  # 归一化
        sar_image = tiff.imread(self.sar_image_list[index]).astype(np.float32)
        # sar_image = (sar_image - sar_image.min()) / (sar_image.max() - sar_image.min())
        label = self.label_list[index]
        if self.normalize:
            # Normalize optical_image and sar_image here
            optical_image = self.normalize_image(optical_image)
            sar_image = self.normalize_image(sar_image)
        optical_image = torch.from_numpy(optical_image)
        sar_image = torch.from_numpy(sar_image)
        sample = {
            'optical_image': optical_image,
            'sar_image': sar_image,
            'label': label
        }
        return sample  # 组合成字典类型输出

    def __len__(self):
        return len(self.optical_image_list)  # 让程序知道这个数据集多大

    # def normalize_image(self, image):  # Normalize image to [0, 1]
    #     image_min = np.min(image)
    #     image_max = np.max(image)
    #     normalized_image = (image - image_min) / (image_max - image_min)
    #     return normalized_image
    def normalize_image(self, image, per_channel=False):  # Normalize image to [0, 1]
        if per_channel:  # 针对每个通道独立归一化
            normalized_image = ((image - image.mean(axis=(1, 2), keepdims=True)) /
                                (image.std(axis=(1, 2), keepdims=True) + 1e-8))
        else:  # 整体归一化
            normalized_image = (image - image.min()) / (image.max() - image.min())
        return normalized_image


class ResNet_VGG_forresnet(Dataset):
    def __init__(
            self,
            optical_image_list,
            label_list2,
            label_list3
    ):
        super(ResNet_VGG_forresnet, self).__init__()
        self.optical_image_list = optical_image_list
        self.label_list2 = label_list2
        self.label_list3 = label_list3

    def __getitem__(self, index):
        optical_image = tiff.imread(self.optical_image_list[index]).astype(np.float32)
        label2 = self.label_list2[index]
        label3 = self.label_list3[index]
        optical_image = torch.from_numpy(optical_image)

        sample = {
            'optical_image': optical_image,
            'label2': label2,
            'label3': label3
        }
        return sample

    def __len__(self):
        return len(self.optical_image_list)


class ResNet_VGG_forvgg(Dataset):
    def __init__(
            self,
            optical_image_list,
            label_list2,
            label_list3
    ):
        super(ResNet_VGG_forvgg, self).__init__()
        self.optical_image_list = optical_image_list
        self.label_list2 = label_list2
        self.label_list3 = label_list3

    def __getitem__(self, index):
        optical_image = tiff.imread(self.optical_image_list[index]).astype(np.float32)
        label2 = self.label_list2[index]
        label3 = self.label_list3[index]
        optical_image = torch.from_numpy(optical_image)
        # optical_image = optical_image[0:12, :, :] # 光学
        # optical_image = optical_image[12:16, :, :] # 雷达
        sample = {
            'optical_image': optical_image,
            'label2': label2,
            'label3': label3
        }
        return sample

    def __len__(self):
        return len(self.optical_image_list)


class BaseC_Dataset(Dataset):
    def __init__(
            self,
            optical_image_list,
            label_list2,
            label_list3
    ):
        super(BaseC_Dataset, self).__init__()
        self.optical_image_list = optical_image_list
        self.label_list2 = label_list2
        self.label_list3 = label_list3

    def __getitem__(self, index):
        optical_image = tiff.imread(self.optical_image_list[index]).astype(np.float32)
        label2 = self.label_list2[index]
        label3 = self.label_list3[index]
        optical_image = torch.from_numpy(optical_image)
        # optical_image = optical_image[0:12, :, :] # 光学
        # optical_image = optical_image[12:16, :, :] # 雷达
        sample = {
            'image': optical_image,
            'label2': label2,
            'label3': label3
        }
        return sample

    def __len__(self):
        return len(self.optical_image_list)
