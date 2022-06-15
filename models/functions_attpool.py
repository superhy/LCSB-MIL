'''
@author: Yang Hu
'''

import gc
import os
import warnings

import torch
from torch.nn.functional import softmax

from models import functions
from models.datasets import load_slides_tileslist, Simple_Tile_Dataset, \
    SlideMatrix_Dataset
from models.functions import train_agt_epoch, regular_evaluation, \
    store_evaluation_roc
from models.networks import BasicResNet18, GatedAttentionPool, AttentionPool, \
    reload_net, store_net
import numpy as np
from support.env import ENV, devices
from support.files import clear_dir
from support.metadata import query_task_label_dict_fromcsv
from support.tools import Time, normalization
from wsi.process import recovery_tiles_list_from_pkl


def store_slide_tilesmatrix(ENV_task, slide_id, slide_matrix, for_train=True, print_info=True):
    """
    store the tiles matrix for one slide
    
    Args:
        ENV_task: the task environment object
    """
    
    ''' prepare some parames '''
    _env_pretrain_slide_mat_train_dir = ENV_task.TASK_PRETRAIN_REPO_SLIDE_MATRIX_TRAIN_DIR
    _env_pretrain_slide_mat_test_dir = ENV_task.TASK_PRETRAIN_REPO_SLIDE_MATRIX_TEST_DIR
    
    slide_matrix_dir = _env_pretrain_slide_mat_train_dir if for_train == True else _env_pretrain_slide_mat_test_dir
    if not os.path.exists(slide_matrix_dir):
        os.makedirs(slide_matrix_dir)
    
    matrix_filename = slide_id + '-(tiles_encode)' + '.npy'
    matrix_path = os.path.join(slide_matrix_dir, matrix_filename)
    np.save(matrix_path, slide_matrix)
#         with open(matrix_path, 'wb') as f_matrix:
#             pickle.dump(slide_matrix_dict[slide_id], f_matrix)
    if print_info == True:
        print('Store slide tiles matrix at: {}'.format(matrix_path))
        
    return matrix_path

def recovery_slide_matrix_filesets(ENV_task, round_id=None, for_train=False):
    """
    get the history slide matrix filesets with different round_id
    
    Args:
        ENV_task: the task environment object
        
    Return:
        slide_matrix_file_sets: if round_id 
    """
    
    ''' prepare some parames '''
    _env_process_slide_tumor_tile_pkl_train_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TUMOR_TILE_PKL_TRAIN_DIR
    _env_process_slide_tumor_tile_pkl_test_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TUMOR_TILE_PKL_TEST_DIR
    _env_process_slide_tile_pkl_train_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TILE_PKL_TRAIN_DIR
    _env_process_slide_tile_pkl_test_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TILE_PKL_TEST_DIR
    _env_pretrain_slide_mat_train_dir = ENV_task.TASK_PRETRAIN_REPO_SLIDE_MATRIX_TRAIN_DIR
    _env_pretrain_slide_mat_test_dir = ENV_task.TASK_PRETRAIN_REPO_SLIDE_MATRIX_TEST_DIR
    _env_tile_size_dir = ENV_task.TILESIZE_DIR

    slide_matrix_file_sets = []
    
    slide_matrix_dir = _env_pretrain_slide_mat_train_dir if for_train == True else _env_pretrain_slide_mat_test_dir
    if round_id is not None:
        slide_matrix_dir = slide_matrix_dir.replace(str(_env_tile_size_dir), '{}_[{}]'.format(_env_tile_size_dir, round_id))
    
    if ENV.APPLY_TUMOR_ROI == True:
        pkl_dir = _env_process_slide_tumor_tile_pkl_train_dir if for_train == True else _env_process_slide_tumor_tile_pkl_test_dir
    else:
        pkl_dir = _env_process_slide_tile_pkl_train_dir if for_train == True else _env_process_slide_tile_pkl_test_dir
    
    for slide_matrix_file in os.listdir(slide_matrix_dir):
        slide_id = slide_matrix_file[:slide_matrix_file.find('-(tiles')]
        # get the tiles len from pkl tiles_list
        slide_tiles_len = len(recovery_tiles_list_from_pkl(os.path.join(pkl_dir, slide_id + '-(tiles{}_list).pkl'.format('_tumor' if ENV.APPLY_TUMOR_ROI else ''))))
        slide_matrix_file_sets.append((slide_id, slide_tiles_len, os.path.join(slide_matrix_dir, slide_matrix_file)))
        
    return slide_matrix_file_sets


''' ---------------- functions for prepare slide matrices ---------------- '''

def encode_tiles_4slides(tile_loader, en_net, slide_id='N/A', print_info=True):
    '''
    '''
    en_net.eval()
#     tile_loader.shuffle = False # WARNING: this is not work!
    
    slide_matrix = None
    with torch.no_grad():
        for i, tile_x in enumerate(tile_loader):
            tile_x = tile_x.cuda()
            encode_y = en_net(tile_x)
            ''' process the encoding into numpy and combine them into a numpy matrix '''
            encode_nd = encode_y.detach().cpu().numpy()
            if slide_matrix is None:
                slide_matrix = encode_nd
            else:
                slide_matrix = np.concatenate((slide_matrix, encode_nd), axis=0)
        
        if print_info == True:
            print('get the tiles encoding matrix for slide: {}'.format(slide_id))
            
    return slide_matrix   


def make_slides_embedding_matrices(ENV_task, batch_size_ontiles, tile_loader_num_workers, encoder_net, for_train=True, print_info=True):
    '''
    make the embedding matrices for all slides,
    which contains the encoding for each tile in the slide,
    these encoding is encoded by the pre-trained ResNet
    
    Args:
        ENV_task: the task environment object
    '''
    
    ''' step-1: prepare the tiles for slides, encode them and get the numpy matrices '''
    slide_tiles_dict = load_slides_tileslist(ENV_task, for_train=for_train)
    transform_augs = functions.get_transform() # use basic transforms
    
    max_bag_dim = max(len(slide_tiles_dict[slide_id]) for slide_id in slide_tiles_dict.keys())
    
    slide_matrix_file_sets = []
    for slide_id in slide_tiles_dict.keys():
        ''' make a tile list dataset for tile image encoding '''
        tiles_list = slide_tiles_dict[slide_id]
        bag_dim = len(tiles_list)
        
        tile_set = Simple_Tile_Dataset(tiles_list=tiles_list, transform=transform_augs)
        tile_loader = functions.get_data_loader(tile_set, batch_size=batch_size_ontiles,
                                                num_workers=tile_loader_num_workers, sf=False)
        slide_matrix = encode_tiles_4slides(tile_loader, encoder_net, slide_id, print_info)
        embedding_dim = slide_matrix.shape[-1]
        if bag_dim < max_bag_dim:
            padding_matrix = np.zeros((max_bag_dim - bag_dim, embedding_dim), dtype=np.float32)
            slide_matrix = np.concatenate((slide_matrix, padding_matrix), axis=0)
#             print(padding_matrix.shape, end=', ')
#         print(slide_matrix.shape)
        
        ''' store slide encoding matrix on the disk '''
        slide_matrix_path = store_slide_tilesmatrix(ENV_task, slide_id, slide_matrix, for_train, print_info)
        del tile_loader, slide_matrix
        gc.collect()
        
        slide_matrix_file_sets.append((slide_id, len(tiles_list), slide_matrix_path))
        
    return slide_matrix_file_sets


def check_load_slide_matrix_files(ENV_task, batch_size_ontiles, tile_loader_num_workers,
                                  encoder_net=None, force_refresh=False, for_train=True, print_info=True):
    """
    check the slide matrix files on disk
    If Ture: recovery them and return
    If False: make a new one on the disk, and return the slide matrix file set
    
    Args:
        ENV_task: the task environment object
        encoder_net: set own encoder_net for get the tiles' encoding
        force_refresh: 
        for_train:
    """
    
    ''' prepare some parameters '''
    _env_pretrain_slide_mat_train_dir = ENV_task.TASK_PRETRAIN_REPO_SLIDE_MATRIX_TRAIN_DIR
    _env_pretrain_slide_mat_test_dir = ENV_task.TASK_PRETRAIN_REPO_SLIDE_MATRIX_TEST_DIR
    
    slide_matrix_dir = _env_pretrain_slide_mat_train_dir if for_train else _env_pretrain_slide_mat_test_dir
    if os.path.exists(slide_matrix_dir) and len(os.listdir(slide_matrix_dir)) > 0 and force_refresh == False:
        slide_matrix_file_sets = recovery_slide_matrix_filesets(ENV_task, for_train=for_train)
    else:
        if os.path.exists(slide_matrix_dir):
            clear_dir([slide_matrix_dir])
        slide_matrix_file_sets = make_slides_embedding_matrices(ENV_task, batch_size_ontiles, tile_loader_num_workers,
                                                                encoder_net=encoder_net, for_train=for_train,
                                                                print_info=print_info)
        
    return slide_matrix_file_sets

''' 
functions for inquiry the results of attention pool 
'''
def query_slides_attscore(data_loader, trained_net, cutoff_padding=True, norm=False):
    """
    query and load the attentional score for each tile in the slide
    
    Args:
        data_loader: can be train_loader or test_loader depend what you want,
            all data_loader introduced should be set as: "shuffle = False"
        trained_net: 
        cutoff_padding: 
    """
    trained_net.eval()
    # load the slide sets from dataset in dataloader
    slide_matrix_file_sets = data_loader.dataset.slide_matrix_file_sets
    slide_matrix_sets = data_loader.dataset.slide_matrix_sets
    if slide_matrix_sets is None:
        max_bag_dim = np.load(slide_matrix_file_sets[0][2]).shape[0]
    else:
        max_bag_dim = slide_matrix_sets[0][2].shape[0]
    att_scores = torch.FloatTensor(len(data_loader.dataset), max_bag_dim).zero_()
    
#     data_loader.shuffle = False # very important
    batch_size = data_loader.batch_size
    with torch.no_grad():
        for i, (mat_X, bag_dim, _) in enumerate(data_loader):
            mat_X = mat_X.cuda()
            bag_dim = bag_dim.cuda()
            
            _, att, _ = trained_net(mat_X, bag_dim)
            att_scores[i * batch_size: i * batch_size + mat_X.size(0), :] = att.detach().clone()
    
    ''' produce a dict for slide --> att_scores pairs '''
    slide_attscores_dict = {}
    for i in range(len(slide_matrix_file_sets)):
        if slide_matrix_sets is not None:
            slide_set = slide_matrix_sets[i][:-1]
        else:
            slide_set = (slide_matrix_file_sets[i][0], slide_matrix_file_sets[i][1])
        slide_id, real_bag_dim = slide_set[0], slide_set[1]
        slide_att_score = att_scores[i].cpu().numpy()
        slide_att_score = slide_att_score[:real_bag_dim] if cutoff_padding == True else slide_att_score
        if norm == True:
            slide_att_score = normalization(slide_att_score)
        slide_attscores_dict[slide_id] = slide_att_score
    
    print('<-- load the tiles\' attention score for %d slides -->' % (len(slide_attscores_dict.keys())))
        
    return slide_attscores_dict


''' --------------------- class of attpool algorithm --------------------- '''

class AttPool_MIL():
    '''
    MIL with Attention Pooling based method
    '''
    def __init__(self, ENV_task, encoder=None, aggregator_name='GatedAttPool', 
                 model_filename=None, test_epoch=1, test_mode=False):
        
        if model_filename is None and test_mode is True:
            warnings.warn('no trained model for testing, please check!')
            return
        
        self.ENV_task = ENV_task
        ''' prepare some parames '''
        _env_task_name = self.ENV_task.TASK_NAME
        _env_loss_package = self.ENV_task.LOSS_PACKAGE

        self.apply_tumor_roi = self.ENV_task.APPLY_TUMOR_ROI
        self.model_store_dir = self.ENV_task.MODEL_STORE_DIR
        pt_prefix = 'pt_' if model_filename is not None and model_filename.find('PTAGT') != -1 else '' 
        self.alg_name = '{}{}Pool_{}{}_{}'.format(pt_prefix, 'g_' if aggregator_name=='GatedAttPool' else '',
                                                self.ENV_task.SLIDE_TYPE, self.ENV_task.FOLD_SUFFIX, _env_task_name)
        
        print('![Initial Stage] test mode: {}'.format(test_mode))
        print('Initializing the training/testing slide matrices...', end=', ')
        
        self.num_epoch = self.ENV_task.NUM_ATT_EPOCH
        self.num_least_epoch = self.ENV_task.NUM_INIT_S_EPOCH
        self.batch_size_ontiles = self.ENV_task.MINI_BATCH_TILE
        self.batch_size_onslides = self.ENV_task.MINI_BATCH_SLIDEMAT
        self.tile_loader_num_workers = self.ENV_task.TILE_DATALOADER_WORKER
        self.slidemat_loader_num_workers = self.ENV_task.SLIDEMAT_DATALOADER_WORKER
        self.last_eval_epochs = self.ENV_task.NUM_LAST_EVAL_EPOCHS
        self.overall_stop_loss = self.ENV_task.OVERALL_STOP_LOSS
        self.test_epoch = test_epoch
        self.record_points = self.ENV_task.ATTPOOL_RECORD_EPOCHS
        
        if encoder is None:
            self.encoder = BasicResNet18(output_dim=2)
        else:
            self.encoder = encoder
        self.encoder = self.encoder.cuda()
            
        init_time = Time()
        if test_mode is False:
            self.train_slidemat_file_sets = check_load_slide_matrix_files(self.ENV_task, self.batch_size_ontiles, self.tile_loader_num_workers, 
                                                                          encoder_net=self.encoder.backbone, for_train=True, 
                                                                          force_refresh=True, print_info=False)
        else:
            self.train_slidemat_file_sets = []
        self.test_slidemat_file_sets = check_load_slide_matrix_files(self.ENV_task, self.batch_size_ontiles, self.tile_loader_num_workers, 
                                                                     encoder_net=self.encoder.backbone, for_train=False if not self.ENV_task.DEBUG_MODE else True,
                                                                     force_refresh=True, print_info=False)   
        
        print('train slides: %d, test slides: %d, time: %s' % (len(self.train_slidemat_file_sets), 
                                                               len(self.test_slidemat_file_sets), 
                                                               str(init_time.elapsed())))
        
        if len(self.train_slidemat_file_sets) > 0:
            embedding_dim = np.load(self.train_slidemat_file_sets[0][2]).shape[-1]
        else:
            ''' for test mode '''
            embedding_dim = np.load(self.test_slidemat_file_sets[0][2]).shape[-1]
        
        if aggregator_name == 'GatedAttPool':
            self.aggregator = GatedAttentionPool(embedding_dim=embedding_dim, output_dim=2)
        elif aggregator_name == 'AttPool':
            self.aggregator = AttentionPool(embedding_dim=embedding_dim, output_dim=2)
        else:
            self.aggregator = GatedAttentionPool(embedding_dim=embedding_dim, output_dim=2)
        self.check_point = None
        if model_filename != None:
            if model_filename.find(self.aggregator.name) == -1:
                warnings.warn('Get a wrong network model name... auto initialize the brand new network!')
                pass
            else:
                model_filepath = os.path.join(self.model_store_dir, model_filename)
                print('Reload network model from: {}'.format(model_filepath))
                self.aggregator, self.check_point = reload_net(self.aggregator, model_filepath)
                if model_filename.find('PTAGT') != -1:
                    self.check_point = None
                
        self.aggregator = self.aggregator.cuda()
        print('On-standby: {} algorithm'.format(self.alg_name), end=', ')
        print('Network: {}, train on -> {}'.format(self.aggregator.name, devices))
        
        # make label
        self.label_dict = query_task_label_dict_fromcsv(self.ENV_task)
        
        if _env_loss_package[0] == 'wce':
            self.criterion = functions.weighted_cel_loss(_env_loss_package[1][0])
        else:
            self.criterion = functions.cel_loss()
        self.optimizer = functions.optimizer_adam_basic(self.aggregator, lr=1e-4)
        
        # some setup change for pre-trained aggregator
        if model_filename is not None and model_filename.find('PTAGT') != -1:
            self.num_least_epoch = 1
#             self.optimizer = functions.optimizer_adam_basic(self.aggregator, lr=4e-5)
#             self.overall_stop_loss = self.ENV_task.OVERALL_STOP_LOSS + 0.1
        
        if self.check_point != None:
            self.optimizer.load_state_dict(self.check_point['optimizer'])
            
        if len(self.train_slidemat_file_sets) > 0:
            self.train_slidemat_set = SlideMatrix_Dataset(self.train_slidemat_file_sets, self.label_dict)
        else:
            ''' for test mode '''
            self.train_slidemat_set = None
        self.test_slidemat_set = SlideMatrix_Dataset(self.test_slidemat_file_sets, self.label_dict)
        
    def slide_epoch(self, epoch, train_slidemat_loader, overall_epoch_stop):
        '''
        the move of one epoch of training on slide
        '''
        print('In training...', end='')
        if self.check_point != None:
            epoch += self.check_point['epoch']
            
        train_log = train_agt_epoch(self.aggregator, train_slidemat_loader, 
                                    self.criterion, self.optimizer,
                                    epoch_info=(epoch, self.num_epoch))
        print(train_log)
        
        attpool_current_loss = float(train_log[train_log.find('loss->') + 6: train_log.find(', train')])
        if epoch >= self.num_least_epoch - 1 and attpool_current_loss < self.overall_stop_loss:
            overall_epoch_stop = True
        return overall_epoch_stop
    
    def optimize(self):
        '''
        training the models in this algorithm
        '''
        if self.train_slidemat_set is None:
            warnings.warn('test mode, can not running training!')
            return        
        
        print('![Training Stage]')
        train_slidemat_loader = functions.get_data_loader(self.train_slidemat_set, self.batch_size_onslides,
                                                          num_workers=self.slidemat_loader_num_workers, sf=True)
        test_slidemat_loader = functions.get_data_loader(self.test_slidemat_set, self.batch_size_onslides,
                                                         num_workers=self.slidemat_loader_num_workers, sf=False)
        
        checkpoint_auc = 0.0 if self.check_point == None else self.check_point['auc']
        epoch = self.check_point['epoch'] if self.check_point != None else 0
        
        overall_epoch_stop = False
        queue_auc = []
        while epoch < self.num_epoch and overall_epoch_stop == False:
            overall_epoch_stop = self.slide_epoch(epoch, train_slidemat_loader, overall_epoch_stop)
            
            # evaluation
            if not self.test_epoch == None and epoch + 1 >= self.test_epoch:
                print('>>> In testing...', end='')
                test_log, test_loss, y_pred_scores, y_labels = self.predict(test_slidemat_loader)
                test_acc, _, _, test_auc = functions.regular_evaluation(y_pred_scores, y_labels)
                 
                queue_auc.append(test_auc)
                if len(queue_auc) > self.last_eval_epochs:
                    queue_auc.remove(queue_auc[0])
                if epoch in self.record_points or overall_epoch_stop == True:
                    checkpoint_auc = test_auc
                    self.record(epoch, checkpoint_auc)
                print('>>> on attpool -> test acc: %.4f, test auc: %.4f' % (test_acc, test_auc))
                
            epoch += 1
            
        ''' calculate the final performance '''
        print(queue_auc)
        final_avg_auc = np.average(queue_auc)
        print('### final evaluation on attpool -> average test auc: %.4f' % (final_avg_auc))
        
    def record(self, epoch, checkpoint_auc):
        '''
        store the trained models
        '''
        alg_store_name = self.alg_name + '_[{}]'.format(epoch + 1)
        init_obj_dict = {'epoch': epoch + 1,
                         'auc': checkpoint_auc}
        store_filepath = store_net(self.apply_tumor_roi, self.model_store_dir, self.aggregator, alg_store_name, self.optimizer, init_obj_dict)
        print('store the milestone point<{}>, '.format(store_filepath), end='')
        
    
    def predict(self, test_slidemat_loader):
        '''
        perform the prediction based on the trained model
        '''
        self.aggregator.eval()
        epoch_loss_sum, epoch_acc_sum, batch_count, time = 0.0, 0.0, 0, Time()
        
        y_pred_scores, y_labels = [], []
        with torch.no_grad():
            for mat_X, bag_dim, y in test_slidemat_loader:
                mat_X = mat_X.cuda()
                bag_dim = bag_dim.cuda()
                y = y.cuda()
                # feed forward
                y_pred, _, _ = self.aggregator(mat_X, bag_dim)
                batch_loss = self.criterion(y_pred, y)
                # loss count
                epoch_loss_sum += batch_loss.cpu().item()
                y_pred = softmax(y_pred, dim=-1)
                epoch_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                batch_count += 1
                
                y_pred_scores.extend(y_pred.detach().cpu().numpy()[:, -1].tolist())
                y_labels.extend(y.cpu().numpy().tolist())
        
        test_log = 'test_loss-> %.4f, time: %s sec' % (epoch_loss_sum / batch_count,
                                                       str(time.elapsed())[:-5])
        
        return test_log, epoch_loss_sum / batch_count, np.array(y_pred_scores), np.array(y_labels)
            
            
''' ---------------- running functions can be directly called ---------------- '''

''' training functions '''
def _run_train_attpool_resnet18(ENV_task, exist_model_name=None):
    encoder = BasicResNet18(output_dim=2)
    method = AttPool_MIL(ENV_task, encoder, 'AttPool', model_filename=exist_model_name)
    method.optimize()

def _run_train_gated_attpool_resnet18(ENV_task, exist_model_name=None):
    encoder = BasicResNet18(output_dim=2)
    method = AttPool_MIL(ENV_task, encoder, 'GatedAttPool', model_filename=exist_model_name)
    method.optimize()

''' testing functions '''
def _run_test_gated_attpool_resnet18(ENV_task, model_filename):
    method = AttPool_MIL(ENV_task=ENV_task, aggregator_name='GatedAttPool', 
                         model_filename=model_filename, test_mode=True)
    
    test_slidemat_loader = functions.get_data_loader(method.test_slidemat_set, method.batch_size_onslides, 
                                                     num_workers=method.slidemat_loader_num_workers, sf=False)
    _, _, y_scores, y_labels = method.predict(test_slidemat_loader)
    acc, fpr, tpr, auc = regular_evaluation(y_scores, y_labels)
    roc_pkl_path = os.path.join(ENV_task.PLOT_STORE_DIR, 'roc_{}-{}.csv'.format(method.alg_name, Time().date))
    store_evaluation_roc(csv_path=roc_pkl_path, roc_set=(acc, fpr, tpr, auc))
    print('Eval %s, Test, BACC: %.4f, AUC: %.4f, store the roc result at: %s' % (method.alg_name, acc, auc, roc_pkl_path) )


if __name__ == '__main__':
    pass