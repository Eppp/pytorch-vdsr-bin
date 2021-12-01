import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
"""
pytorch-vdsr-recurrence-bin 使用二值网络binaryconv3x3代替了原本的Conv_Relu
"""
# 神经网络结构块
class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        # stride步长为1 padding填充为1 bias不添加偏置参数作为可学习参数
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # 对从上层网络Conv2d中传递下来的tensor直接进行修改，inplace变量替换能够节省运算内存，不用多存储其他变量
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))

class BConv_ReLU_Block(nn.Module):
    def __init__(self):
        super(BConv_ReLU_Block, self).__init__()
        self.conv3x3 = binaryconv3x3()
        self.norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


def binaryconv3x3(in_planes=64, out_planes=64, stride=1):
    # 3x3 convolution with padding
    return HardBinaryConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

# 二值网络结构块
class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        # number_of_weights = 64 * 64 * 3 * 3
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        # torch.rand: Returns a tensor filled with random numbers from a uniform distribution on the interval
        # 将所有的weights设置为可训练的参数 使参数可以优化
        self.weights = nn.Parameter(torch.rand((self.number_of_weights, 1)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights), dim=3, keepdim=True), dim=2, keepdim=True), dim=1, keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        # 将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
        return y

# 主要网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.residual_layer = self.make_layer(BConv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # Conv2d中参数的初始化 normal高斯
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 3 3 64  最后一次3 3 1
                # print(m.kernel_size[0], m.kernel_size[1], m.out_channels)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        # Sequential一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out

if __name__ == '__main__':
    modeltest = Net()
    print(modeltest)