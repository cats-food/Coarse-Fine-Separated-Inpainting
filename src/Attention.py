import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import tshow


class GatedAttention(nn.Module):
    def __init__(self, inchannel, patch_size=1, propagate_size=3, stride=1):
        super(GatedAttention, self).__init__()
        self.gate = Gate()
        self.att = Att(patch_size, propagate_size, stride)
        self.combiner = nn.Conv2d(inchannel * 2, inchannel, kernel_size=1)  # k1s1p0的卷积，文中称为pixel-wise convolution

    def forward(self, feature_in, imgs, masks):  #
        gate_scores = self.gate(imgs, masks)
        feature_out = self.att(feature_in,  gate_scores)  # 输入图和mask一起进入att
        feature_out = torch.cat([feature_out, feature_in], dim=1)  # 最开始的输入和att的输出拼接
        feature_out = self.combiner(feature_out)  # 拼接以后再来一次k1s1p0的卷积，文中称为pixel-wise convolution
        return feature_out


class Att(nn.Module):
    def __init__(self, patch_size=3, propagate_size=3, stride=1):
        super(Att, self).__init__()
        self.patch_size = patch_size
        self.propagate_size = propagate_size
        self.stride = stride
        self.prop_kernels = None

    def forward(self, inputs, gate_scores):
        bz, c, h, w = inputs.size()  # batchsize, 通道数， 高，宽
        conv_kernels_all = inputs.view(bz, c, w * h, 1, 1)  # view和reshape一样其实
        conv_kernels_all = conv_kernels_all.permute(0, 2, 1, 3, 4)  # 改变维度顺序，成了 bz * wh * c * 1 * 1 的tensor
        output_tensor = []
        for i in range(bz):
            feature_map = inputs[i:i + 1]  # 被卷积的特征图，尺寸为 1*c*w*h。索引[i:i+1]的作用是，设a是4维张量，a[i:i+1]得到的仍然是4维（保留维度），而a[i]得到的是3维（降一维），它们的数值是一样的。
            # feature_map = feature_map / torch.sqrt(torch.sum(feature_map ** 2, 1)+ 0.0000001) # 不需要norm，因为对后面的channel-wise softmax来说是常数

            conv_kernels = conv_kernels_all[i] + 0.0000001  # wh*c*1*1  加很小的数是防止后面求sqrt在0处不可导
            norm_factor = torch.sqrt(torch.sum(conv_kernels ** 2, [1, 2, 3], keepdim=True))  # wh*1*1*1
            conv_kernels_norm = conv_kernels / norm_factor  # wh*c*1*1，可理解为wh个 c*1*1的卷积核，相当于每个像素都被抽出来当一个卷积核

            attention_scores = F.conv2d(feature_map, conv_kernels_norm, padding=self.patch_size // 2)  # 对应论文(4)式，输出尺寸1*wh*w*h，代表每个像素和其余所有像素的余弦相似度 (wh个 c*1*1的卷积核 卷 1*c*w*h的特征图得到的结果)
            #conv_result = F.avg_pool2d(conv_result, 3, 1, padding=1)   # 1*wh*w*h 在余弦相似度图上求3*3的平均池化，对应论文5式，可是这里为什么要乘以9很奇怪？？？

            attention_scores = attention_scores + gate_scores[i:i + 1] # higher score means higher contribution

            # softmax
            #attention_scores = F.softmax(1e8*conv_result, dim=1)  # 在通道维度上做softmax，得到文中的socre'，尺寸为 1*wh*w*h
            attention_scores = F.softmax(attention_scores, dim=1)  # 在通道维度上做softmax，得到文中的socre'，尺寸为 1*wh*w*h

            #feature_map = F.conv_transpose2d(attention_scores, conv_kernels, stride=1, padding=self.patch_size // 2)  # 1*c*w*h 转置卷积 重构
            #                                 1*wh*w*h          wh*c*1*1
            feature_map = F.conv_transpose2d(attention_scores, conv_kernels, stride=1, padding=self.patch_size // 2)
            output_tensor.append(feature_map)

        return torch.cat(output_tensor, dim=0)  # bz*c*w*h 和輸入一樣


class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()
        self.pi=torch.tensor([3.1415926],device='cuda')
        self.gate=nn.Sequential(  # input: img + mask(1:hole, 0:src)
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=7), # 256
            nn.ReLU(),
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5,stride=2), #128
            nn.ReLU(),
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5), #128
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2),  # 64
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3),  # 64
            nn.ReLU(),
        )
        self.init_weights()

    def forward(self, x, m):
        x = torch.cat([x,m],dim=1) # input: img + mask
        s = self.gate(x)  # 0~inf [b,1,256,256]

        ## tan plan
        # [0~n]->[-inf~inf], higher s leads to higher contribution, so holes should be 0 (feed 1-m in model.py line281)
        s = torch.tan(self.pi * (torch.tanh(s) - 0.5))

        bz, c, h, w = s.size()
        s = s.view(bz, w * h, 1, 1) #[b,256**2,1,1]
        return s

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)
        self.apply(init_func)

