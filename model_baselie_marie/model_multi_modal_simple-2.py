import utils.config as config
import torch.nn as nn
import torch
import numpy as np
from collections import OrderedDict
from models.models_building_blocks import *

import torch.nn.init as init
from typing import List


def inception_block(archi, in_channels, in_dims, start_with_pooling,add_dropout = False, do_rate = 0, groups = 1, inception_net = None,**kwargs):
    to_return = []
    if start_with_pooling:
        to_return.append(nn.AvgPool2d(2, stride=2))

    
    to_return.append(
         BasicConv2d(in_channels,archi, kernel_size= 3, in_dim = in_dims, acti_func='relu', groups = groups )
    )
    # to_return.append(inception_net(in_channels,archi=archi, in_dim = in_dims, groups = groups,**kwargs))

    num_out_channels = archi
    
    if add_dropout:     
        to_return.append(nn.Dropout(p=do_rate))
    return nn.Sequential(*to_return), num_out_channels




class Model_multi_modal_simple(nn.Module):
    def __init__(self,
                in_dim,
                n_outputs,
                modalities,
                mags_input_size,
                parallel_before_inception_archi  =None ,
                parallel_inception_archi = None,
                inception_archi =None,
                parallel_pooling_before_inceptions =None,
                pooling_before_inceptions = None,
                convs_after_inception=[96,96,96],
                convs_after_inception_pad=[False,False,False],
                first_FC_dim = 1024,
                classification_FCs_archi_ = [1024],
                regression_FCs_archi_ = [512]):

        

        super(Model_multi_modal_simple, self).__init__()
        
        ## Default parameters

        if parallel_before_inception_archi is None:
            parallel_before_inception_archi  = [32,32] 
        if parallel_inception_archi is None:
            parallel_inception_archi = [[36,32,32,32,42,42],[36,32,32,32,42,42]]
        if inception_archi is None:
            inception_archi = [ [109,101,101,101,156,156],[109,101,101,101,156,156],[109,101,101,101,156,156],[109,101,101,101,156,156]]
        if parallel_pooling_before_inceptions is None:
            parallel_pooling_before_inceptions = [True, False]
        if pooling_before_inceptions is None:
            pooling_before_inceptions = [False, True, False, True]

        
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )
        self.in_dim = in_dim
        self.n_outputs = n_outputs
        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.modalities = modalities

        print("self.in_dim",self.in_dim)
        print("self.modalities",len(self.modalities))

        self.parrellel_block = Parellel_Inception(in_dim,modalities = self.modalities,before_inception_archi=parallel_before_inception_archi,inception_archi = parallel_inception_archi, pooling_before_inceptions=parallel_pooling_before_inceptions)
        

        inception_blocks = []
        in_c_inception = self.parrellel_block.out_n_channels
        inception_in_dims = [[e / 2 ** sum(pooling_before_inceptions[0:i+1]) for e  in self.parrellel_block.outs_dims] for i in range(len(inception_archi))]


        for i in range(len(inception_archi)):
            
            block , in_c_inception = inception_block(inception_archi[i], in_c_inception, inception_in_dims[i], pooling_before_inceptions[i])
            inception_blocks.append(block)
            

        self.inception_blocks = nn.Sequential(*(inception_blocks))
        out_c_inception = in_c_inception    
    

        feature_map_size_output = [e / 2 ** sum(pooling_before_inceptions) for e  in self.parrellel_block.outs_dims]

        self.convs_after_inceptions_output = [feature_map_size_output[0] - (2 * (i+1)) for i in range(len(convs_after_inception))]

        if config.CONFIG["ONE_CONV_AFTER_INCEPTIONS"]:
            self.convs_after_inceptions = nn.Sequential(OrderedDict([
                ('avgpool_4x4_stride_4', nn.AvgPool2d((4,4), stride=(4,4)) ),
                ('conv3x3_0', BasicConv2d(out_c_inception, convs_after_inception[0], kernel_size=3, in_dim = feature_map_size_output[0],acti_func = "relu", apply_pad = True)),
                ('avgpool_2x2_stride_2', nn.AvgPool2d((2,2), stride=(2,2)) )
            ]))

        else : 
            self.convs_after_inceptions = nn.Sequential(OrderedDict([
                ('conv3x3_0', BasicConv2d(out_c_inception, convs_after_inception[0], kernel_size=3, in_dim = feature_map_size_output[0],acti_func = "relu", apply_pad = convs_after_inception_pad[0])),
                ('conv3x3_1', BasicConv2d(convs_after_inception[0], convs_after_inception[1], kernel_size=3, in_dim = self.convs_after_inceptions_output[0],acti_func = "relu", apply_pad = convs_after_inception_pad[1])),
                ('conv3x3_2', BasicConv2d(convs_after_inception[1], convs_after_inception[2], kernel_size=3, in_dim = self.convs_after_inceptions_output[1],acti_func = "relu", apply_pad = convs_after_inception_pad[2])),
                ('avgpool_2x2_stride_1', nn.AvgPool2d((2,2), stride=(2,2)) )
            ]))





        ## devide by 2 for the avg pool
        feature_map_size_output_after_conv = [self.convs_after_inceptions_output[2] /2, self.convs_after_inceptions_output[2] /2]
        fc_in = feature_map_size_output_after_conv[0] * feature_map_size_output_after_conv[1] * convs_after_inception[len(convs_after_inception)-1] 
        fc_in += 1

        if mags_input_size is not None :
            output_size = 96
            #mags_fc_archi = [mags_input_size,128,256,256,output_size]
            #self.mags_FC = nn.Sequential(*[BasicFC(mags_fc_archi[jj], mags_fc_archi[jj+1]) for jj in range(len(mags_fc_archi)-1)])
            self.mags_FC = MLP(input_dim=mags_input_size, output_dim=output_size, hidden_dims=[64, 128, 256], use_bias=True)  # FIRST TEST: [64, 128]
            fc_in += output_size  
            print("mags have been added, fc_in=",fc_in)


        self.first_FC = BasicFC(int(fc_in),first_FC_dim, activation= True)
        #print('self.first_FC computed:', fc_in,'-->',first_FC_dim)


        classification_FCs = []
        classification_FCs_archi = classification_FCs_archi_.copy()
        classification_FCs_archi += [self.n_outputs]

        fc_in = first_FC_dim
        for i in range(len(classification_FCs_archi)):
            classification_FCs.append(BasicFC(fc_in,classification_FCs_archi[i], activation= i < len(classification_FCs_archi) -1))
            fc_in = classification_FCs_archi[i] 
            
        self.classification_FCs = nn.Sequential(*classification_FCs)



        regression_FCs = []
        regression_FCs_archi = regression_FCs_archi_.copy()
        regression_FCs_archi += [1] ## for regression output

        
        fc_in = first_FC_dim
        for i in range(len(regression_FCs_archi)):
            regression_FCs.append(BasicFC(fc_in,regression_FCs_archi[i], activation= i < len(regression_FCs_archi) -1))
            fc_in = regression_FCs_archi[i] 
            
        self.regression_FCs = nn.Sequential(*regression_FCs)

        if config.CONFIG["MODALITY_DROP_OUT"] is not None : 
            # self.modality_DO_layer = nn.Dropout(p=config.CONFIG["MODALITY_DROP_OUT"])
            self.total_number_modalities_bands = len([ ee  for e in self.modalities for ee in e])

    def modality_DO(self,X): # DROPOUT
        modality_factors = torch.ones(1, self.total_number_modalities_bands, int(X.shape[1]/self.total_number_modalities_bands)).to(self.device)
        factors = torch.ones(1,  self.total_number_modalities_bands, 1 ).to(self.device)
        
        if  self.training:
            if config.CONFIG["SPECEFIC_MODALITIES_SWITCH_OFF"]:
                p = np.random.uniform(0,1)
                modalities_to_switch_off = []
                if  (p <= config.CONFIG["MOD_DO_PROBA"]):
                    modalities_to_switch_off = config.CONFIG["SPECEFIC_MODALITIES_SWITCH_OFF"]
            else:
                n_m_to_switch_off = np.random.choice(range(round(config.CONFIG["MODALITY_DROP_OUT"] * len(self.modalities))))
                # n_m_to_switch_off = round(config.CONFIG["MODALITY_DROP_OUT"] * len(self.modalities)) 
                modalities_to_switch_off = np.random.choice(range(len(self.modalities)),n_m_to_switch_off, replace=False)
        else : 
            modalities_to_switch_off =  config.CONFIG["VAL_MODALITIES_TO_SWITCH_OFF"] 
        
        
        if modalities_to_switch_off is not None and len(modalities_to_switch_off) > 0:
            indexs_correspending_to_chosen_modalities = []
            for m in modalities_to_switch_off : 
                start  =sum([len(self.modalities[i] )for i in range(0,m)])
                end = start + len(self.modalities[m])
                indexs_correspending_to_chosen_modalities += list(range(start,end))
            factors *= self.total_number_modalities_bands/(self.total_number_modalities_bands - len(indexs_correspending_to_chosen_modalities) )
            factors[0,indexs_correspending_to_chosen_modalities,0] *= 0
        do_factors = factors


        modality_factors *= do_factors
        X = X * modality_factors.flatten(start_dim = 1)[:,:,None,None]
        return X


    def forward(self, X, ebv = None, return_latent_repr = False, mags = None):

 
        X = self.parrellel_block(X)
        if config.CONFIG["MODALITY_DROP_OUT"] is not None : 
            X = self.modality_DO(X)
        # X = self.inception_blocks(X)
        
        for i in range(len(self.inception_blocks)):
            X = self.inception_blocks[i](X)
            if i == config.CONFIG["CNN_INPUT_STAGE"]:
                latent_repr_cnn = X

        X = self.convs_after_inceptions(X)

        latent_repr = torch.flatten(X , start_dim=1)
        if mags is not None :
            mags_latent_space = self.mags_FC(mags)
            latent_repr = torch.cat((latent_repr,mags_latent_space), dim = 1)
        
        if ebv is not None:
            X = torch.cat((latent_repr,torch.unsqueeze(ebv,1)),dim = 1)
        
        X = self.first_FC(X)
        if return_latent_repr :
            if config.CONFIG["USE_CNN_ADV"]:
                return self.classification_FCs(X), self.regression_FCs(X), latent_repr_cnn

            else : 
                return self.classification_FCs(X), self.regression_FCs(X), latent_repr

        
        return self.classification_FCs(X), self.regression_FCs(X) 



class Parellel_Inception(nn.Module):
    def __init__(self,
        in_dim,
        modalities,
        before_inception_archi  = [32,32] ,
        inception_archi = [[36,32,32,32,42,42],[36,32,32,32,42,42]], 
        pooling_before_inceptions = [True, False]):
        super(Parellel_Inception, self).__init__()
        self.modalities = modalities
        self.parallel_blocks = []
        self.out_n_channels = 0
         

        before_inception_archi = [[ e * len(self.modalities[i]) for e in before_inception_archi] for i in range(len(self.modalities))]
        inception_in_dims = [[e / 2 ** sum(pooling_before_inceptions[0:i+1]) for e  in in_dim] for i in range(len(inception_archi))]
        if len(inception_in_dims) > 0:
            self.outs_dims = inception_in_dims[-1]
        else : 
            self.outs_dims = in_dim
             
       
        inception_archi = [(np.array(inception_archi) * len(self.modalities[i])).tolist() for i in range(len(self.modalities))]

        activations = ["prelu","tanh"]
        if config.CONFIG["ONE_CONV_AFTER_INCEPTIONS"]:
            print("HAAAAAACK")
            activations = ["tanh"]

        for i in range(len(self.modalities)):

            before_inception = before_inception_block(before_inception_archi[i], n_channels_in = len(self.modalities[i]), activations = activations )
            
            in_c_inception = before_inception_archi[i][-1] 
            
            inception_in_dims = [[e / 2 ** sum(pooling_before_inceptions[0:i+1]) for e  in in_dim] for i in range(len(inception_archi[i]))]
            

            inception_blocks = [before_inception]



            for j in range(len(inception_archi[i])):
             
                block , in_c_inception = inception_block(inception_archi[i][j], in_c_inception, inception_in_dims[j], pooling_before_inceptions[j])
                inception_blocks.append(block)
                
            self.out_n_channels += in_c_inception
            self.inception_blocks = nn.Sequential(*(inception_blocks))
            self.parallel_blocks.append(self.inception_blocks )
        
        self.parallel_blocks = nn.ModuleList(self.parallel_blocks)

        if config.CONFIG['USE_CROSS_FUSION']:
            self.fusion_conv = BasicConv2d(self.out_n_channels, int(self.out_n_channels/len(self.modalities)+1),self.outs_dims, kernel_size=3, apply_pad = True)
            self.out_n_channels =  int(self.out_n_channels/len(self.modalities)+1) *  (len(self.modalities)+1)


    def forward(self, X):
        if config.CONFIG['USE_CROSS_FUSION']:
            return self.forward_cross_fusion(X)
        else:
            return self.forward_concat(X)

    def forward_concat(self, X):
        res = []

        if config.CONFIG["USE_MODALITY_TRANSFORMERS"] and X.dim() == 5:
            for i in range(len(self.parallel_blocks)):
                res.append(self.parallel_blocks[i](X[:,i,:,:,:]))

        else:
            for i in range(len(self.parallel_blocks)):
                res.append(self.parallel_blocks[i](X[:,self.modalities[i],:,:]))
        return  torch.cat(res,dim=1)


    def forward_cross_fusion(self,X):
        outputs = [[None for i in range(len(self.modalities))] for j in range(len(self.modalities))]
        
        parallel_processed_inputs = []
        for i in range(len(self.modalities)):
            parallel_processed_inputs.append(self.parallel_blocks[i][0:2](X[:,self.modalities[i],:,:]))
            # print(parallel_processed_inputs[-1].shape)
            # print(parallel_processed_inputs[-1][0][0][0])

        # print("==========")
        for i in range(len(self.modalities)):
            for j in range(len(self.modalities)):
                # outputs[i][j] = self.indivd_inception_blocks[i](modalities[j])
                # outputs[i][j] = self.parallel_blocks[i](X[:,self.modalities[j],:,:])
                outputs[i][j] = self.parallel_blocks[i][2](parallel_processed_inputs[j])
                # print(outputs[i][j].shape)
                # print(outputs[i][j][0][0][0])

        # print("========== ========== ==========")
        # sys.exit()

        outputs2 = [None for i in range(len(self.modalities) + 1)] 
        outputs2[len(self.modalities)] = []

        for i in range(len(self.modalities)):
            outputs2[i] = self.fusion_conv(torch.cat(outputs[:][i],dim = 1))
            outputs2[len(self.modalities)].append(sum(outputs[i][:]))
        outputs2[len(self.modalities)] = self.fusion_conv(torch.cat(outputs2[len(self.modalities)],dim = 1))

        X =  torch.cat(outputs2,dim = 1)

        return X 
    



class Fast_Parellel_Inception(nn.Module):
    def __init__(self,
        in_dim,
        modalities,
        before_inception_archi  = [32,32] ,
        inception_archi = [[36,32,32,32,42,42],[36,32,32,32,42,42]], 
        pooling_before_inceptions = [True, False]):
        super(Fast_Parellel_Inception, self).__init__()
        self.modalities = modalities
        inception_blocks = []
        self.out_n_channels = 0

        self.flattend_modalities = [ee for e in self.modalities for ee in e]
        n_channels_in = len(self.flattend_modalities)
        

        before_inception_archi = [ e * len(self.flattend_modalities) for e in before_inception_archi] 
        self.before_inceptions = before_inception_block(before_inception_archi, n_channels_in, activations = ["prelu","tanh"], groups = len(self.modalities)) 
        
        inception_in_dims = [[e / 2 ** sum(pooling_before_inceptions[0:i+1]) for e  in in_dim] for i in range(len(inception_archi))]        
        in_c_inception = before_inception_archi[-1] 

        for i in range(len(inception_archi)):
            inception_archi[i] = [ e * len(self.flattend_modalities) for e in inception_archi[i] ]


        

        for i in range(len(inception_archi)):
            
            block , in_c_inception = inception_block(inception_archi[i], in_c_inception, inception_in_dims[i], pooling_before_inceptions[i], groups = len(self.modalities) )
            inception_blocks.append(block)
            

        self.inception_blocks = nn.Sequential(*(inception_blocks))


        self.outs_dims = inception_in_dims[-1]
        self.out_n_channels = in_c_inception
       

    def forward(self, X):
        X = self.before_inceptions(X[:,self.flattend_modalities,:,:])
        X = self.inception_blocks(X)
        return X 





class MLP(nn.Module):
    def __init__(self, input_dim: int = 6, output_dim: int = 96, 
                 hidden_dims: List[int] = [64, 128, 256], use_bias: bool = True):
        """
            Args:
            input_dim: Number of input features
            output_dim: Dimension of output vector
            hidden_dims: List of hidden layer dimensions
            use_bias: Whether to include bias terms
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.use_bias = use_bias
        
        # Build network
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            layer = nn.Linear(in_dim, out_dim, bias=use_bias)
            init.xavier_uniform_(layer.weight, gain=init.calculate_gain('relu'))
            if use_bias:
                init.zeros_(layer.bias)
            layers.append(layer)
            
            if i < len(layer_dims) - 2:  # No activation after last layer
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch_size, input_dim) -> (batch_size, output_dim)"""
        return self.network(x)
    
    def __repr__(self) -> str:
        return (f"MLP(input_dim={self.input_dim}, output_dim={self.output_dim}, "
                f"hidden_dims={self.hidden_dims}, use_bias={self.use_bias})")

# Example usage
"""
if __name__ == "__main__":
    # Instantiate
    model = MLP()
    print(model)  # Clean representation
    
    # Test forward pass
    x = torch.randn(16, 6)  # Batch of 16 samples
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")  # Should be (16, 96)
"""
    


