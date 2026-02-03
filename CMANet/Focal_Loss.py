import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class  # 类别数
        self.alpha = alpha  # 正负样本权重
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average  # 默认情况下，损失按每个批次中的每个损失进行平均

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)  # 将权重alpha设置为1
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)  # 将alpha改成self.num_class行1列
            self.alpha = self.alpha / self.alpha.sum()  # alpha所占的比例（权重）
        elif isinstance(self.alpha, float):  # 应该是针对2分类
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        epsilon = 1e-8
        # logit = F.softmax(input, dim=1) + epsilon  # 在softmax输出上加上小常数
        logit = F.softmax(input, dim=1)
        if torch.isnan(logit).any():
            print("NaN detected in logit after softmax")
        if torch.isinf(logit).any():
            print("Inf detected in logit after softmax")

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)

        alpha = self.alpha
        if alpha.device != input.device:  # 保证计算式所在的显卡相同
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()  # 创建独热编码
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth, 1.0 - self.smooth)

        # 检查logit和one_hot_key的极端值
        if torch.isnan(logit).any():
            print("NaN detected in logit")
        if torch.isinf(logit).any():
            print("Inf detected in logit")
        if torch.isnan(one_hot_key).any():
            print("NaN detected in one_hot_key")
        if torch.isinf(one_hot_key).any():
            print("Inf detected in one_hot_key")

        pt = (one_hot_key * logit).sum(1) + epsilon

        # 检查pt的极端值
        if torch.isnan(pt).any():
            print("NaN detected in pt")
            print("logit:", logit)
            print("one_hot_key:", one_hot_key)
        if torch.isinf(pt).any():
            print("Inf detected in pt")

        logpt = pt.log()

        if torch.isnan(logpt).any():
            print("NaN detected in logpt")
        if torch.isinf(logpt).any():
            print("Inf detected in logpt")

        gamma = self.gamma
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        # 检查和处理损失中的极端值
        if torch.isnan(loss).any():
            print("NaN detected in loss before reduction")
        if torch.isinf(loss).any():
            print("Inf detected in loss before reduction")

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()

        # 再次检查和处理损失中的极端值
        if torch.isnan(loss).any():
            print("NaN detected in loss after reduction")
        if torch.isinf(loss).any():
            print("Inf detected in loss after reduction")

        return loss
