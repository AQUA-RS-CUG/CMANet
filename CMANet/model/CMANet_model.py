import torch
import torch.nn as nn
import torch.nn.functional as F


def check_for_nan(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN or Inf detected in {name}")


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)  # 通道注意力
        x = x * ca
        sa = torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)
        sa = self.spatial_attention(sa)
        x = x * sa
        return x


# 深度可分离卷积模块 --- 设置 padding = (kernel_size - 1) // 2 以保证输入数据的高度和宽度在每次卷积操作后不变
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DepthwiseSeparableConv, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2  # 根据kernel_size自动计算padding值，以保证输入数据的高度和宽度在每次卷积操作后不变
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DWFE(nn.Module):
    def __init__(self, in_channel1, num_class):
        super(DWFE, self).__init__()
        block1_channels = 32
        block2_channels = 64
        block3_channels = 64
        block4_channels = 96
        block_channels = block1_channels + block2_channels + block3_channels + block4_channels
        # 使用ModuleList管理多个卷积块，每个卷积块可以设置不同的channel数
        self.feature_ops = nn.ModuleList([
            nn.Sequential(
                DepthwiseSeparableConv(in_channel1, block1_channels, kernel_size=3, stride=1, padding=1),  # Block 1
                nn.BatchNorm2d(block1_channels),
                nn.ReLU(inplace=True),
                CBAMBlock(block1_channels)
            ),
            nn.Sequential(
                DepthwiseSeparableConv(block1_channels, block2_channels, kernel_size=3, stride=1, padding=1),  # Block 2
                nn.BatchNorm2d(block2_channels),
                nn.ReLU(inplace=True),
                CBAMBlock(block2_channels)
            ),
            nn.Sequential(
                DepthwiseSeparableConv(block2_channels, block3_channels, kernel_size=3, stride=2, padding=1),  # Block 3
                nn.BatchNorm2d(block3_channels),
                nn.ReLU(inplace=True),
                CBAMBlock(block3_channels)
            ),
            nn.Sequential(
                DepthwiseSeparableConv(block3_channels, block4_channels, kernel_size=5, stride=2, padding=2),  # Block 3
                nn.BatchNorm2d(block4_channels),
                nn.ReLU(inplace=True),
                CBAMBlock(block4_channels)
            )
        ])
        # 调整拼接特征的通道数，使其与初始输入特征的通道数匹配
        self.concat_adjust = nn.Sequential(
            nn.Conv2d(block_channels, in_channel1, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channel1),
            nn.ReLU()
        )
        # 调整初始输入特征的通道数，使其与拼接特征的通道数匹配
        self.adjust_conv = nn.Sequential(
            nn.Conv2d(in_channel1, block_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(block_channels),
            nn.ReLU()
        )

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()  # (64, 12, 16, 16)
        check_for_nan(x, "DWFE Input x check nan")
        x_init = x  # 存储初始特征x
        block_outputs = []  # 存储每个Block的输出
        for op in self.feature_ops:
            x = op(x)  # 最后x的shape为(batch_size, block4_channels, 16, 16)
            block_outputs.append(x)  # 存储每个Block的输出
        target_size = (height, width)
        upsampled_outputs = []
        for output in block_outputs:
            upsampled_output = F.interpolate(output, size=target_size, mode='bilinear', align_corners=False)
            upsampled_outputs.append(upsampled_output)
        # x_concat = torch.cat(block_outputs, dim=1)  # 在通道维度(dim=1)上拼接所有Block的输出 x_concat --> (batch_size, 256, 16, 16)
        x_concat = torch.cat(upsampled_outputs, dim=1)  # 在通道维度上拼接所有Block的输出 x_concat --> (batch_size, 256, 16, 16)
        x_concat_adjusted = self.concat_adjust(x_concat)  # 调整拼接通道维度 x_concat_adjusted --> (batch_size, 12, 16, 16)
        x_output = x_concat_adjusted + x_init  # 输出 x_output2 --> (batch_size, 12, 16, 16)
        return x_concat, x_concat_adjusted


# 位置编码模块
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding2D, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        # 使用 sin 和 cos 来生成位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用 cos
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0).to(x.device)  # 根据输入的 seq_len 截取位置编码
        return x


# 取时间步长模块
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(d_model, 1)

    def forward(self, x):
        attention_scores = self.attention(x)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)
        x = torch.matmul(attention_weights.transpose(1, 2), x)  # (batch_size, 1, d_model)
        return x.squeeze(1)  # (batch_size, d_model)


class ComputeQKV(nn.Module):
    def __init__(self, in_channel1, in_channel2, hidden_dim, num_heads, dropout=0.1):
        super(ComputeQKV, self).__init__()
        self.in_channel2 = in_channel2
        self.num_heads = num_heads
        self.height = 16
        self.width = 16
        self.seq_len = 5
        self.hidden_dim = hidden_dim
        self.d_model = in_channel2 * 8 * 8  # 256
        # SAR的query，Optical的key和value
        self.query_proj_sar = nn.Linear(in_channel2, hidden_dim)
        self.key_proj_opt = nn.Linear(in_channel1, hidden_dim)
        self.value_proj_opt = nn.Linear(in_channel1, hidden_dim)
        # 卷积层用于空间特征提取，避免丢失空间结构
        self.conv_sar1 = nn.Sequential(
            nn.Conv2d(in_channel2 * self.seq_len, in_channel2 * self.seq_len, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channel2 * self.seq_len),
            nn.ReLU()
        )
        self.conv_sar2 = nn.Sequential(
            nn.Conv2d(in_channel2, in_channel2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channel2),
            nn.ReLU()
        )
        self.conv_opt = nn.Sequential(
            nn.Conv2d(in_channel1, in_channel1, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm2d(in_channel1),
            nn.ReLU()
        )
        # 添加相对位置编码
        self.relative_position_bias = nn.Parameter(torch.zeros(4 * 4, self.hidden_dim))
        # Position Encoding
        self.positional_encoding = PositionalEncoding2D(d_model=self.d_model, max_seq_len=1024)
        # 取时间步长
        self.attention_pooling = AttentionPooling(self.d_model)
        # 多头注意力机制
        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=num_heads,
                                                         dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Sigmoid()
        self.attn_norm = nn.Linear(hidden_dim, in_channel1)

    def forward(self, x_sar, x_opt):
        check_for_nan(x_sar, "ComputeQKV Input x check nan")
        batch_size_sar, seq_len, channels_sar, height, width = x_sar.size()
        x_sar = x_sar.view(batch_size_sar, channels_sar * seq_len, height, width)  # x --> (batch_size, 5*4, 16, 16)
        x_sar = self.conv_sar1(x_sar)  # x --> (batch_size, 5*4, 8, 8)
        x_sar = x_sar.view(batch_size_sar, seq_len, channels_sar, 8, 8)  # x --> (batch_size, 5, 4, 8, 8)
        x_sar = x_sar.view(batch_size_sar, seq_len, channels_sar * 8 * 8)  # x --> (batch_size, 5, 4*8*8)
        x_sar = self.positional_encoding(x_sar)  # 添加位置编码 x --> (batch_size, 5, 256)
        x_sar = self.attention_pooling(x_sar)  # 取时间步长 x --> (batch_size, 256)
        x_sar = x_sar.view(batch_size_sar, channels_sar, 8 * 8)  # 调整 x --> (batch_size, 4, 64)
        x_sar = x_sar.view(batch_size_sar, channels_sar, 8, 8)  # x --> (batch_size, 4, 8, 8)
        x_sar = self.conv_sar2(x_sar)  # x --> (batch_size, 4, 4, 4)
        x_sar = x_sar.view(batch_size_sar, channels_sar, 4 * 4)  # 调整 x --> (batch_size, 4, 16)

        x_sar = x_sar.permute(0, 2, 1)  # 调整 x --> (batch_size, H*W, 4)
        q_sar1 = self.query_proj_sar(x_sar)  # q_sar1 (batch_size, H*W, hidden_dim)
        q_sar1 = q_sar1.permute(1, 0, 2)  # q_sar1 --> (H*W, batch_size, hidden_dim)
        batch_size_opt, channel_opt, _, _ = x_opt.size()  # x_opt (batch_size, 12, 16, 16)
        x_opt = self.conv_opt(x_opt)  # x_opt (batch_size, 12, 4, 4)
        x_opt = x_opt.view(batch_size_opt, channel_opt, -1)  # 展平为 (batch_size, 12, H*W)
        x_opt = x_opt.permute(0, 2, 1)  # (batch_size, H*W, 12)
        k_opt = self.key_proj_opt(x_opt)
        v_opt = self.value_proj_opt(x_opt)
        # 添加相对位置编码
        q_sar2 = q_sar1.permute(1, 0, 2)  # q_sar2 (batch_size, H*W, hidden_dim)
        q_sar2 = q_sar2 + self.relative_position_bias
        k_opt = k_opt + self.relative_position_bias
        # 调整 q_sar, k_opt, v_opt 由 (batch_size, H*W, hidden_dim) --> (H*W, batch_size, hidden_dim)
        q_sar3 = q_sar2.permute(1, 0, 2)
        k_opt = k_opt.permute(1, 0, 2)
        v_opt = v_opt.permute(1, 0, 2)
        # 归一化处理
        q_sar3 = self.norm(q_sar3)
        k_opt = self.norm(k_opt)
        v_opt = self.norm(v_opt)
        # 门控控制
        q_sar3 = self.gate(q_sar3) * q_sar3  # q_sar3 (H*W, batch_size, hidden_dim)
        q_sar4 = q_sar3.permute(1, 2, 0)  # q_sar4 (batch_size, hidden_dim, H*W)
        q_sar4 = q_sar4.view(batch_size_sar, self.hidden_dim, 4, 4)  # q_sar4 (batch_size, hidden_dim, 4, 4)
        # 添加相对位置编码后的交叉注意力计算
        attn_output, attn_weights = self.multihead_attention(q_sar3, k_opt, v_opt)
        attn_output = attn_output.permute(1, 0, 2)  # attn_output --> (batch_size, H*W, hidden_dim)
        batch_size_attn, hw, hidden_dim = attn_output.size()
        attn_output = attn_output.view(batch_size_attn, 4, 4, hidden_dim)  # attn_output --> (batch_size, 4, 4, hidden_dim)
        attn_output = self.attn_norm(attn_output)  # attn_output --> (batch_size, 4, 4, 12)
        attn_output = attn_output.permute(0, 3, 1, 2)  # attn_output --> (batch_size, 12, 4, 4)

        return q_sar4, attn_output


# 堆叠多个TransFE模块
class MultiLayer(nn.Module):
    def __init__(self, in_channel1, in_channel2, hidden_dim, num_heads, dropout=0.1):
        super(MultiLayer, self).__init__()

        # 定义三个ComputeQKV模块
        self.ComputeQKV1 = ComputeQKV(in_channel1, in_channel2, hidden_dim, num_heads, dropout)
        self.ComputeQKV2 = ComputeQKV(in_channel1, in_channel2, hidden_dim, num_heads, dropout)
        self.ComputeQKV3 = ComputeQKV(in_channel1, in_channel2, hidden_dim, num_heads, dropout)

    def forward(self, x_sar, x_opt):
        # 第一个ComputeQKV模块输入：光学特征x_opt和雷达时序特征x_sar，输出融合后的特征
        out1, attn_weights1 = self.ComputeQKV1(x_sar, x_opt)
        # print("第1个ComputeQKV模块输出: ", out1.shape)
        # 第二个ComputeQKV模块输入：雷达时序特征x_sar和第一个TransFE模块的输出out1
        out2, attn_weights2 = self.ComputeQKV2(x_sar, out1)
        # print("第2个ComputeQKV模块输出: ", out2.shape)
        # 第三个ComputeQKV模块输入：雷达时序特征x_sar和第二个TransFE模块的输出out2
        out3, attn_weights3 = self.ComputeQKV3(x_sar, out2)
        # print("第3个ComputeQKV模块输出: ", out3.shape)
        # 最终输出out3和所有注意力权重
        return out3, [attn_weights1, attn_weights2, attn_weights3]


class CMANet(nn.Module):
    def __init__(self, in_channel1, in_channel2, num_class, hidden_dim, num_heads, dropout):
        super(CMANet, self).__init__()
        self.dwfe = DWFE(in_channel1, num_class)
        self.qkv = ComputeQKV(in_channel1, in_channel2, hidden_dim, num_heads, dropout)
        self.multiLayer = MultiLayer(in_channel1, in_channel2, hidden_dim, num_heads, dropout)
        self.layernorm = nn.LayerNorm(in_channel1)
        self.ffn = nn.Sequential(
            nn.Linear(in_channel1, in_channel1 * 4),
            nn.ReLU(),
            nn.Linear(in_channel1 * 4, in_channel1)
        )
        self.dropout = nn.Dropout(dropout)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))  # Global pooling for feature compression
        self.fc = nn.Linear(in_channel1, num_class)

    def forward(self, opt_input, sar_input):
        opt_fea, opt_output = self.dwfe(opt_input)  # opt_output (batch_size, 12, 16, 16)
        sar_fea, cross_fea = self.qkv(sar_input, opt_output)  # ComputeQKV (batch_size, 12, 4, 4)
        output_fea = cross_fea.permute(0, 2, 3, 1)  # output_fea (batch_size, 4, 4, 12)
        batch_size_fea, h, w, channel_fea = output_fea.size()
        output_fea = output_fea.view(batch_size_fea, h * w, channel_fea)  # output_fea (batch_size, H*W, 12)
        # Normalize and apply FFN to the cross-attention output
        output_fea = self.layernorm(output_fea)
        output_fea = self.ffn(output_fea)
        output_fea = self.dropout(output_fea)
        output_fea = output_fea.permute(0, 2, 1).contiguous().view(output_fea.size(0), -1, 4, 4)
        pooled_fea = self.pooling(output_fea)  # (batch_size, hidden_dim, 1, 1)
        pooled_fea = pooled_fea.view(pooled_fea.size(0), -1)  # Flatten to (batch_size, hidden_dim)
        output_joint = self.fc(pooled_fea)  # output (batch_size, class)

        # return opt_fea, sar_fea, cross_fea
        return output_joint


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CMANet(in_channel1=12, in_channel2=4, num_class=5, hidden_dim=1024, num_heads=8, dropout=0.1).cuda()
    Optical = torch.randn(64, 12, 16, 16)
    Sar = torch.randn(64, 5, 4, 16, 16)

    Output_opt_fea, Output_sar_fea, Output_fea = model(Optical.cuda(), Sar.cuda())
    print("SVTN 输出 Output_opt_fea 的 shape 为 {}".format(Output_opt_fea.shape))
    print("SVTN 输出 Output_sar_fea 的 shape 为 {}".format(Output_sar_fea.shape))
    print("SVTN 输出 Output_fea 的 shape 为 {}".format(Output_fea.shape))
