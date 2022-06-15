'''
@author: Yang Hu

network models, self created or loaded from [torchvision, vit-pytorch, etc.]
the list:
    https://github.com/pytorch/vision
    https://github.com/lucidrains/vit-pytorch
'''

'''
------------------ some basic functions for networks -------------------
'''

import os

from torch import nn
import torch
from torch.autograd.function import Function
from torchvision import models
from vit_pytorch.dino import Dino, get_module_device
from vit_pytorch.extractor import Extractor
from vit_pytorch.mae import MAE
from vit_pytorch.recorder import Recorder
from vit_pytorch.vit import ViT

from support.tools import Time
import torch.nn.functional as F


def store_net(apply_tumor_roi, store_dir, trained_net, 
              algorithm_name, optimizer, init_obj_dict={}):
    """
    store the trained models
    
    Args:
        ENV_task: a object which packaged all parames in specific ENV_TASK
        trained_net:
        algorithm_name:
        optimizer:
        init_obj_dict:
    """
    
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    
    tumor_roi_flag = '-TROI' if apply_tumor_roi == True else ''
    store_filename = 'checkpoint_' + trained_net.name + '-' + algorithm_name + '{}-'.format(tumor_roi_flag) + Time().date + '.pth'
    init_obj_dict.update({'state_dict': trained_net.state_dict(),
                          'optimizer': optimizer.state_dict()})
    
    store_filepath = os.path.join(store_dir, store_filename)
    torch.save(init_obj_dict, store_filepath)
    
    return store_filepath


def reload_net(model_net, model_filepath):
    """
    reload network models only for testing
    
    Args:
        model_net: an empty network need to reload
        model_filepath:
        
    Return: only the 'state_dict' of models
    """
    checkpoint = torch.load(model_filepath)
    model_net.load_state_dict(checkpoint['state_dict'], False)
    return model_net, checkpoint

'''
------------------- CNN (encoder) --------------------
'''

class BasicResNet18(nn.Module):
    
    def __init__(self, output_dim, imagenet_pretrained=True):
        super(BasicResNet18, self).__init__()
        """
        Args: 
            output_dim: number of classes
            imagenet_pretrained: use the weight with pre-trained on ImageNet
        """
        
        self.name = 'ResNet18'
        
        self.backbone = models.resnet18(pretrained=imagenet_pretrained)
        self.fc_id = nn.Identity()
        self.backbone.fc = self.fc_id
        
        self.fc = nn.Linear(in_features=512, out_features=output_dim, bias=True)
    
    def forward(self, X):
        x = self.backbone(X)
        output = self.fc(x)  
        return output
    
''' ------------------ gradient reversed networks (encoder) ------------------ '''
   
class ReverseGrad_Layer(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

    
class ReverseResNet18(nn.Module):
    
    def __init__(self, output_dim, imagenet_pretrained=True):
        super(ReverseResNet18, self).__init__()
        """
        Args: 
            output_dim: number of classes
            imagenet_pretrained: use the weight with pre-trained on ImageNet
        """
        
        self.name = 'ReResNet18'
        
        self.backbone = models.resnet18(pretrained=imagenet_pretrained)
        self.fc_id = nn.Identity()
        self.backbone.fc = self.fc_id
        
        self.fc = nn.Linear(in_features=512, out_features=output_dim, bias=True)
        
    def forward_ahd(self, X_pos):
        x_pos = self.backbone(X_pos)

        output_pos = self.fc(x_pos)
        return output_pos
    
    def forward_rev(self, X_neg, alpha):
        x_neg = self.backbone(X_neg)
        x_reversed = ReverseGrad_Layer.apply(x_neg, alpha)
        
        output_neg = self.fc(x_reversed)
        return output_neg
    
    def forward(self, X, ahead=True, alpha=0.1):
        '''
        ahead (ahd) means with normal gradient BP
        otherwise with reversed gradient BP
        '''
        if ahead == True:
            output = self.forward_ahd(X)
        else:
            output = self.forward_rev(X, alpha)
          
        return output
    

''' 
---------------- attention based feature aggregation networks -----------------
'''
   
class AttentionPool(nn.Module):
    
    def __init__(self, embedding_dim, output_dim):
        super(AttentionPool, self).__init__()
        
        self.name = 'AttPool'
        
        self.embedding_dim = embedding_dim
        self.att_layer_width = [256, 128]
        self.output_layer_width = 128
        self.att_dim = 1
        
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.embedding_dim, out_features=self.att_layer_width[0]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.att_layer_width[0], out_features=self.att_layer_width[1]),
            nn.Tanh()
            )
        
        self.bn = nn.BatchNorm1d(self.att_layer_width[1])
        self.attention = nn.Linear(in_features=self.att_layer_width[1],
                                   out_features=self.att_dim, bias=False)
        
#         self.softmax = nn.Softmax(dim=-1)
        self.fc_out = nn.Linear(in_features=self.output_layer_width,
                                out_features=output_dim, bias=False)
        
    def forward(self, X_e, bag_lens):
        """
        Args:
            X_e: Embedding of the tiles in slide X as the input
            bag_lens: 
        """
        X_e = self.encoder(X_e)
#         X_e = self.bn(X_e.transpose(-2, -1)).transpose(-2, -1)
        att = self.attention(X_e)
        att = att.transpose(-2, -1)
        ''' record the attention value (before softmax) '''
#         att_r = torch.squeeze(att, dim=1)
        att = F.softmax(att, dim=-1)
#         att = torch.sigmoid(att)
        mask = (torch.arange(att.shape[-1], device=att.device).expand(att.shape) < bag_lens.unsqueeze(1).unsqueeze(1)).byte()
        att = att * mask
        
        att_H = att.matmul(X_e)
        output = self.fc_out(att_H).squeeze(1)
        
        ''' record the attention value (after softmax) '''
        att_r = torch.squeeze(att, dim=1)
#         output_label = F.softmax(output, dim=1).argmax(dim=1)

        return output, att_r, att_H

    
class GatedAttentionPool(nn.Module):
    
    def __init__(self, embedding_dim, output_dim):
        super(GatedAttentionPool, self).__init__()
        
        self.name = 'GatedAttPool'
        
        self.embedding_dim = embedding_dim
        self.att_layer_width = [256, 128]
        self.output_layer_width = 128
        self.att_dim = 1
        
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.embedding_dim, out_features=self.att_layer_width[0]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.att_layer_width[0], out_features=self.att_layer_width[1]),
            )
        
        self.attention_U = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.att_layer_width[1], out_features=self.att_layer_width[1]),
            nn.Tanh()
            )
        self.attention_V = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.att_layer_width[1], out_features=self.att_layer_width[1]),
            nn.Sigmoid()
            )
        
        self.bn = nn.BatchNorm1d(self.att_layer_width[1])
        self.attention = nn.Linear(in_features=self.att_layer_width[1],
                                   out_features=self.att_dim, bias=False)
        
        self.fc_out = nn.Linear(in_features=self.output_layer_width,
                                out_features=output_dim, bias=False)
        
    def forward(self, X_e, bag_lens):
        """
        Args:
            X_e: Embedding of the tiles in slide X as the input
            bag_lens: 
        """
        X_e = self.encoder(X_e)
#         X_e = self.bn(X_e.transpose(-2, -1)).transpose(-2, -1)
        att_U = self.attention_U(X_e)
        att_V = self.attention_V(X_e)
        att = self.attention(att_V * att_U)
        att = att.transpose(-2, -1)
        ''' record the attention value (before softmax) '''
#         att_r = torch.squeeze(att, dim=1)
        att = F.softmax(att, dim=-1)
        
        mask = (torch.arange(att.shape[-1], device=att.device).expand(att.shape) < bag_lens.unsqueeze(1).unsqueeze(1)).byte()
        att = att * mask
        
        att_H = att.matmul(X_e)
        output = self.fc_out(att_H).squeeze(1)
        
        ''' record the attention value (after softmax) '''
        att_r = torch.squeeze(att, dim=1)
#         output_label = F.softmax(output, dim=1).argmax(dim=1)

        return output, att_r, att_H

if __name__ == '__main__':
    pass