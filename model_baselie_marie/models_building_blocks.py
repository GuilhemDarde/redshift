import torch.nn as nn
import torch
import numpy as np
from collections import OrderedDict
import utils.config as config


                
class Inception(nn.Module):
    
    def __init__(self, in_channels, conv_block=None, archi = [64,48,48,48,64,64], in_dim = [64,64],all_linear = False, groups = 1,**kwargs):
        super(Inception, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )
        if conv_block is None:
            conv_block = BasicConv2d
            acti_func = "relu"
            if all_linear:
                acti_func="iden"

            
        self.archi = archi
        print("archi[0]",archi, "in_dim[0]",in_dim)
        
        self.block1 = nn.Sequential(OrderedDict([
            ('conv1x1_0', conv_block(in_channels, archi[0], kernel_size=1, in_dim = in_dim[0],acti_func =acti_func, groups = groups).to(device))
        ]))

        self.block2 = nn.Sequential(OrderedDict([
            ('conv1x1_1', conv_block(in_channels, archi[1], kernel_size=1, in_dim = in_dim[0],acti_func =acti_func, groups = groups).to(device)),
            ('conv3x3_1', conv_block(archi[1], archi[4], kernel_size=3, in_dim = in_dim[0],acti_func =acti_func, groups = groups).to(device) )
        ]))
        
        self.block3 = nn.Sequential(OrderedDict([
            ('conv1x1_2', conv_block(in_channels, archi[2], kernel_size=1, in_dim = in_dim[0],acti_func =acti_func, groups = groups).to(device)),
            ('conv5x5_2', conv_block(archi[2], archi[5], kernel_size=5, in_dim = in_dim[0],acti_func =acti_func, groups = groups).to(device) )
        ]))

        self.block4 = nn.Sequential(OrderedDict([
            ('conv1x1_3', conv_block(in_channels, archi[3], kernel_size=1, in_dim = in_dim[0],acti_func =acti_func, groups = groups).to(device)),
            ('padding_same_3', nn.ZeroPad2d( (0, compute_padding_same(in_dim[0],2,1), 0, compute_padding_same(in_dim[0],2,1) ) ) ),
            ('avgpool_2x2_stride_1_3', nn.AvgPool2d((2,2), stride=(1,1)) ) 
        ]))
        self.useless_avg_pool = nn.AvgPool2d((1,1), stride=(1,1)) 

        

    def forward(self, x):
        # outputs = [self.block1(x),self.block2(x),self.block3(x),self.block4(x)]
        return self.useless_avg_pool(torch.cat([self.block1(x),self.block2(x),self.block3(x),self.block4(x)], 1))

    
    def num_out_channels(self):
        return self.archi[0]+ self.archi[4]  + self.archi[5] +  self.archi[3]
    
  
class BasicConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, in_dim, kernel_size, stride = 1, apply_bn = False, bias = True,acti_func = "relu", apply_pad = True, **kwargs):
        super(BasicConv2d, self).__init__()
        self.apply_pad = apply_pad
        if self.apply_pad:
            padding = "same"

        else:
            padding = "valid"

        
        self.conv = nn.Conv2d(in_channels, out_channels, bias=bias, kernel_size = kernel_size, padding = padding , **kwargs)
        # torch.nn.init.xavier_uniform(self.conv .weight)
        if config.CONFIG["BATCH_NORM"]:
            self.BN = nn.BatchNorm2d(out_channels)
        nn.init.xavier_uniform_(self.conv.weight)
        self.conv.bias.data.fill_(0.0)

        if acti_func =="relu":
            self.acti_func = nn.ReLU()
        elif acti_func =="prelu":
            self.acti_func = nn.PReLU()
        elif acti_func == "tanh":
            self.acti_func = nn.Tanh()
        elif acti_func == "iden":
            self.acti_func = identity
        elif acti_func == "swish":
            self.acti_func = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        if config.CONFIG["BATCH_NORM"]:
            x = self.BN(x)
        return self.acti_func(x)
    
class BasicFC(nn.Module):
    
    def __init__(self, in_dim, out_dim, bias = True, activation = True, acti_func = "relu", **kwargs):
        super(BasicFC, self).__init__()

        #print("init BasicFC",int(in_dim),out_dim)
        self.fc  = nn.Linear(int(in_dim), out_dim, bias = bias)
        #print("self.fc computed")
        
        nn.init.xavier_uniform_(self.fc.weight)
        if bias:
            self.fc.bias.data.fill_(0.1)
        
        self.activation = activation

        if acti_func =="relu":
            self.acti_func = nn.ReLU()
        elif acti_func =="prelu":
            self.acti_func = nn.PReLU()
        elif acti_func == "sigmoid":
            self.acti_func = nn.Sigmoid()
        elif acti_func == "tanh":
            self.acti_func = nn.Tanh()
        elif acti_func == "iden":
            self.acti_func = identity
        elif acti_func == "swish":
            self.acti_func = nn.SiLU()

        # if self.activation:
        #     self.acti_func = nn.ReLU()


    def forward(self, x):
        #print('x in forward',x.shape)
        x = self.fc(x)
        if  self.activation:
            return self.acti_func(x)

        return x
    
    

def identity(x):
    return x

    
def compute_padding_same(in_dim,k_size,stride):
    return int(np.ceil(((stride-1)*in_dim-stride+k_size) / 2))



def before_inception_block(archi,n_channels_in,activations,in_dim=[64,64], apply_bn = False, kernel_size = 5, groups = 1):
    layer0 = BasicConv2d(n_channels_in,archi[0], kernel_size=kernel_size, in_dim = in_dim[0], acti_func=activations[0], groups = groups )
    layers = nn.Sequential(*([layer0] + [ BasicConv2d(archi[i-1],archi[i], kernel_size=kernel_size, in_dim = in_dim[0], acti_func=activations[i], groups = groups ) for i in range(1,len(archi))]))
    return layers

def inception_block(archi, in_channels, in_dims, start_with_pooling,add_dropout = False, do_rate = 0, groups = 1, inception_net = None,**kwargs):
    to_return = []
    if start_with_pooling:
        to_return.append(nn.AvgPool2d(2, stride=2))
    if inception_net is None:
        inception_net = Inception
    
    to_return.append(inception_net(in_channels,archi=archi, in_dim = in_dims, groups = groups,**kwargs))

    num_out_channels = to_return[-1].num_out_channels()
    
    if add_dropout:     
        to_return.append(nn.Dropout(p=do_rate))
    return nn.Sequential(*to_return), num_out_channels

