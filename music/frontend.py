import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _calc_same_pad(i, k, s, d):
    return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    Adopted from: https://github.com/human-analysis/MUXConv/blob/master/conv2d_helpers.py
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        pad_h = _calc_same_pad(ih, kh, self.stride[0], self.dilation[0])
        pad_w = _calc_same_pad(iw, kw, self.stride[1], self.dilation[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class BlockChoi(nn.Module):
    """     
    See Choi et al. 2016 https://arxiv.org/abs/1609.04243
    
    Differences from paper:
     - ELU
     - Pool before normalize
     - flag for max/avg --> False/True 
    """
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 pool_size,
                 avgpool_flag):
        super(BlockChoi, self).__init__()
        
        self.conv = Conv2dSame(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=kernel_size)
        
        self.bn = nn.BatchNorm2d(out_channels)
        
        self.activation = nn.ELU() #nn.ELU() 
        
        self.pool = nn.AvgPool2d(pool_size) if avgpool_flag else nn.MaxPool2d(pool_size)
        
        self.dropout = nn.Dropout(0.1) #see paper
    
    def forward(self, inputs):
        x = self.conv(inputs)
        # pool before normalize (less computations)
        # https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/
        x = self.pool(x) 
        x = self.bn(x)
        x = self.activation(x)       
        x = self.dropout(x)
        return x

class Frontend_mine(nn.Module):
    """
    
    # see paper Choi et al. recurrent...
    
    Usage example:

    front_end_dict = {"list_out_channels":[64,128,128,256,256,646], 
                  "list_kernel_sizes":[(3,3),(3,3),(3,3),(3,3),(3,3),(3,3)],
                  "list_pool_sizes":  [(4,1),(2,2),(2,2),(2,2),(2,2),(2,2)], 
                  "list_avgpool_flags":[False,False,False,False,False,True]}

    conv_stack = Frontend_mine(front_end_dict)

    print(conv_stack)

    print(sum(p.numel() for p in conv_stack.parameters() if p.requires_grad)) # number of trainable parameters

    print(conv_stack(torch.rand((32,1,128,1000))).shape)
    """
    def __init__(self,front_end_dict,in_channels=1):
        super(Frontend_mine, self).__init__()
        
        #self.version = version
        self.depth = len(front_end_dict["list_out_channels"])
        self.spec_bn = nn.BatchNorm2d(1)

        # set class attributes in a for loop
        for i in range(self.depth):
            setattr(self, 
                    f"conv_block{i+1}", 
                    BlockChoi(in_channels if i==0 else front_end_dict["list_out_channels"][i-1],
                              front_end_dict["list_out_channels"][i],
                              front_end_dict["list_kernel_sizes"][i],
                              front_end_dict["list_pool_sizes"][i],
                              front_end_dict["list_avgpool_flags"][i]))
    
    def forward(self, inputs):
        
        #inputs shape = (Batch,Channel,Freq,Time)
        
        x = self.spec_bn(inputs) # 
        
        x = getattr(self,f"conv_block{1}")(x)

        for i in range(1,self.depth):
            x = getattr(self,f"conv_block{i+1}")(x)

        # squeeze the singleton frequency dimension
        if x.size()[2]==1: 
            x = x.squeeze(2)
        else:
            raise Exception("Insufficient pooling along the frequency axis, "+
                            "the required resulting size is 1.")    
            
        return x
