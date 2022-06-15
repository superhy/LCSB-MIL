'''
@author: Yang Hu
'''

import csv
import pickle

from sklearn import metrics
from torch import nn, optim
import torch
from torch.nn.functional import softmax, softplus, relu
from torch.nn.modules.loss import L1Loss, NLLLoss, CrossEntropyLoss, \
    TripletMarginLoss
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

import numpy as np
from support.env import ENV
from support.tools import Time


#######################################################################
#------------- a list of self-designed losses (criterion) ------------#
#######################################################################
class NLL_Manual_Loss(nn.Module):

    def __init__(self):
        super(NLL_Manual_Loss, self).__init__()
        
    def forward(self, y_pred, y):
        y_pred = torch.clamp(y_pred, min=1e-5, max=1. - 1e-5)  # avoid the NAN error warning
        neg_log_likelihood = -1.0 * (y * torch.log(y_pred) + (1.0 - y) * torch.log(1.0 - y_pred))
        return neg_log_likelihood.mean()

class SCELoss(nn.Module):

    def __init__(self, alpha, beta, num_classes=2):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = nn.functional.one_hot(labels, self.num_classes).float().cuda()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

class CombinationLoss(nn.Module):
    
    def __init__(self, nb_losses, loss_lambda: list=[0.5, 0.5]):
        super(CombinationLoss, self).__init__()
        self.weights = []
        self.left_lambda = 1.0
        if loss_lambda != None:
            i = 0
            for _lambda in loss_lambda:
                self.weights.append(_lambda)
                self.left_lambda -= _lambda
                i += 1
            if i < nb_losses:
                self.weights.extend(list(max(self.left_lambda, 0) / (nb_losses - i) for j in range(nb_losses - i)))
#             para = nn.Parameter(torch.tensor(0.5), requires_grad=True)
#             para = torch.clamp(para, min=0.0, max=1.0)
#             self.weights.append(para)
#             self.weights.append(1.0 - para)
        else:
            for i in range(nb_losses):
                self.weights.append(1.0)
                
    def forward(self, _losses):
        '''
        Args:
            _losses: multiple computed losses
        '''
        comb_loss = self.weights[0] * _losses[0]
        for i in range(len(_losses) - 1):
            comb_loss = comb_loss + self.weights[i + 1] * _losses[i + 1]
            
        return comb_loss


'''
------------- call various loss functions ------------
'''

def l1_loss():
    return L1Loss().cuda()

def nll_loss():
    return NLLLoss().cuda()

def nll_manual_loss():
    return NLL_Manual_Loss().cuda()

def cel_loss():
    return CrossEntropyLoss().cuda()

def weighted_cel_loss(weight=0.5):
    w = torch.Tensor([1 - weight, weight])
    loss = CrossEntropyLoss(w).cuda()
    return loss

def sce_loss(a=0.8, b=0.2):
    return SCELoss(alpha=a, beta=b, num_classes=2).cuda()

def triplet_margin_loss():
    return TripletMarginLoss(margin=1.0, p=2).cuda()

def combination_loss(n_losses, loss_lambda=[0.5, 0.5]):
    return CombinationLoss(n_losses, loss_lambda).cuda()

''' ------------------ optimizers for all algorithms (models) ------------------ '''


def optimizer_sgd_basic(net, lr=1e-2):
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    return optimizer, scheduler


def optimizer_adam_basic(net, lr=1e-4, wd=1e-4):
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    return optimizer


def optimizer_rmsprop_basic(net, lr=1e-5):
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    return optimizer


def optimizer_adam_pretrained(net, lr=1e-4, wd=1e-4):
    output_params = list(map(id, net.fc.parameters()))
    feature_params = filter(lambda p: id(p) not in output_params, net.parameters())
    
    optimizer = optim.Adam([{'params': feature_params},
                           {'params': net.fc.parameters(), 'lr': lr * 1}],
                            lr=lr, weight_decay=wd)
    return optimizer


''' ------------------ dataloader ------------------ '''

def get_data_loader(dataset, batch_size, num_workers=4, sf=False, p_mem=True):
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=sf, pin_memory=p_mem)
    return data_loader


''' 
------------------ 
data transform with for loading batch data,
with / without data augmentation
------------------ 
'''

def get_transform():
    '''
    data transform with only image normalization
    '''
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.1, 0.1, 0.1]
    )
    transform_augs = transforms.Compose([
        transforms.Resize(size=(ENV.TRANSFORMS_RESIZE, ENV.TRANSFORMS_RESIZE)),
        transforms.ToTensor(),
        normalize
        ])
    return transform_augs

def get_data_arg_transform():
    '''
    data transform with slight data augumentation
    '''
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.1, 0.1, 0.1]
    )
#     _resize = (256, 256) if ENV.TRANSFORMS_RESIZE < 300 else (ENV.TRANSFORMS_RESIZE - 20, TRANSFORMS_RESIZE - 20)
    transform_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=ENV.TRANSFORMS_RESIZE, scale=(0.8, 1.0)),
#         transforms.CenterCrop(size=_resize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(degrees=15),
        transforms.Resize(size=(ENV.TRANSFORMS_RESIZE, ENV.TRANSFORMS_RESIZE)),
        transforms.ToTensor(),
        normalize
        ])
    return transform_augs


''' --------------- functions of training --------------- '''  
def train_epoch(net, train_loader, loss, optimizer, epoch_info: tuple=(-2, -2)):
    """
    trainer for tile-level encoder 
    
    Args:
        net:
        train_loader:
        loss:
        optimizer:
        epoch: the idx of running epoch (default: None (unknown))
    """
    net.train()
    epoch_loss_sum, epoch_acc_sum, batch_count, time = 0.0, 0.0, 0, Time()
    
    for X, y in train_loader:
        X = X.cuda()
        y = y.cuda()
        # feed forward
        y_pred = net(X)
        batch_loss = loss(y_pred, y)
        # BP
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # loss count
        epoch_loss_sum += batch_loss.cpu().item()
        epoch_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
        batch_count += 1
    
#     train_log = 'batch_loss-> %.6f, train acc-> %.4f, time: %s sec' % (epoch_loss_sum / batch_count, epoch_acc_sum / len(train_loader.dataset), str(time.elapsed()))
    train_log = 'epoch [%d/%d], batch_loss-> %.4f, train acc (on tiles)-> %.4f, time: %s sec' % (epoch_info[0] + 1, epoch_info[1],
                                                                                                 epoch_loss_sum / batch_count,
                                                                                                 epoch_acc_sum / len(train_loader.dataset),
                                                                                                 str(time.elapsed())[:-5])
    return train_log

def train_rev_epoch(net, train_neg_loader, loss, optimizer,
                         revg_grad_a=1e-4, epoch_info: tuple=(-2, -2)):
    """
    trainer for tile-level encoder, perform only negative gradient
    
    Args:
        net:
        train_pos_loader:
        loss:
        optimizer:
        revg_grad_a: alpha of reversed gradient
        epoch: the idx of running epoch (default: None (unknown))
        now_round: record the round id to determine the alpha for reversed gradient bp
    """
    net.train()
    epoch_loss_neg_sum, batch_count_neg, time = 0.0, 0, Time()
    
    # training on reversed gradient samples
    for X_neg, y_neg in train_neg_loader:
        X_neg = X_neg.cuda()
        y_neg = y_neg.cuda()
          
        y_pred_neg = net(X_neg, ahead=False, alpha=revg_grad_a)
        batch_loss_neg = loss(y_pred_neg, y_neg)
        
        optimizer.zero_grad()
        batch_loss_neg.backward()
        optimizer.step()
          
        epoch_loss_neg_sum += batch_loss_neg.cpu().item()
        batch_count_neg += 1
          
    train_log = 'epoch [%d/%d], batch_neg_loss-> %.4f, time: %s sec' % (epoch_info[0] + 1, epoch_info[1],
                                                                        epoch_loss_neg_sum / batch_count_neg,
                                                                        str(time.elapsed())[:-5])
      
    return train_log  

def train_agt_epoch(net, train_loader, loss, optimizer, epoch_info: tuple=(-2, -2)):
    """
    trainer for slide-level(WSI) feature aggregation and/or classification
    
    Args:
        net: diff with other training function, net need to input <mat_X, bag_dim>
        data_loader:
        loss:
        optimizer:
        epoch: the idx of running epoch (default: None (unknown))
    """
    net.train()
    epoch_loss_sum, epoch_acc_sum, batch_count, time = 0.0, 0.0, 0, Time()
    
    for mat_X, bag_dim, y in train_loader:
        mat_X = mat_X.cuda()
        bag_dim = bag_dim.cuda()
        y = y.cuda()
        # feed forward
        y_pred, _, _ = net(mat_X, bag_dim)
        batch_loss = loss(y_pred, y)
        # BP
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # loss count
        epoch_loss_sum += batch_loss.cpu().item()
        epoch_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
        batch_count += 1
    
#     train_log = 'batch_loss-> %.6f, train acc-> %.4f, time: %s sec' % (epoch_loss_sum / batch_count, epoch_acc_sum / len(train_loader.dataset), str(time.elapsed()))
    train_log = 'epoch [%d/%d], batch_loss-> %.4f, train acc-> %.4f, time: %s sec' % (epoch_info[0] + 1, epoch_info[1],
                                                                                      epoch_loss_sum / batch_count,
                                                                                      epoch_acc_sum / len(train_loader.dataset),
                                                                                      str(time.elapsed())[:-5])
    return train_log

def train_pt_agt_epoch(net, pretrain_loader, loss, optimizer, epoch_info: tuple=(-2, -2)):
    """
    pre-training the aggregator in MIL with a contrastive-style strategy, one epoch
    """
    net.train()
    epoch_loss_sum, batch_count, time = 0.0, 0, Time()
    
    for mat_X1, bag_dim1, mat_X2, bag_dim2, sim_y in pretrain_loader:
        mat_X1, mat_X2 = mat_X1.cuda(), mat_X2.cuda()
        bag_dim1, bag_dim2 = bag_dim1.cuda(), bag_dim2.cuda()
        sim_y = sim_y.cuda()
        # feed forward
        y_logits = net(mat_X1, bag_dim1, mat_X2, bag_dim2)
        batch_loss = loss(y_logits, sim_y)
        # BP
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # loss count
        epoch_loss_sum += batch_loss.cpu().item()
        batch_count += 1
        
    train_log = 'epoch [%d/%d], batch_loss-> %.4f, time: %s sec' % (epoch_info[0] + 1, epoch_info[1],
                                                                    epoch_loss_sum / batch_count,
                                                                    str(time.elapsed())[:-5])
    return train_log

    
''' -------------- function of evaluation -------------- '''
    
def regular_evaluation(y_scores, y_label):
    '''
    '''
    y_pred = np.array([1 if score > 0.5 else 0 for score in y_scores.tolist()])
    acc = metrics.balanced_accuracy_score(y_label, y_pred)
    fpr, tpr, threshold = metrics.roc_curve(y_label, y_scores)
    auc = metrics.auc(fpr, tpr)
        
    return acc, fpr, tpr, auc

def store_evaluation_roc(csv_path, roc_set):
    '''
    store the evaluation results as ROC as csv file
    '''
    acc, fpr, tpr, auc = roc_set
    with open(csv_path, 'w', newline='') as record_file:
        csv_writer = csv.writer(record_file)
        csv_writer.writerow(['acc', 'auc', 'fpr', 'tpr'])
        for i in range(len(fpr)):
            csv_writer.writerow([acc, auc, fpr[i], tpr[i]])
    print('write roc record: {}'.format(csv_path))
            
def load_evaluation_roc(csv_path):
    '''
    load the evaluation ROC from csv file
    '''
    with open(csv_path, 'r', newline='') as roc_file:
        print('load record from: {}'.format(csv_path))
        csv_reader = csv.reader(roc_file)
        acc, auc, fpr, tpr = 0.0, 0.0, [], []
        line = 0
        for record_line in csv_reader:
            if line == 0:
                line += 1
                continue
            if line == 1:
                acc = float(record_line[0])
                auc = float(record_line[1])
            fpr.append(float(record_line[2]))
            tpr.append(float(record_line[3]))
            line += 1
    return acc, auc, fpr, tpr

if __name__ == '__main__':
    pass





