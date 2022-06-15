'''
@author: Yang Hu
'''
import collections
import gc
import json
import os
import random
import warnings

import torch
from torch.nn.functional import softmax

from models import functions_attpool, functions
from models.datasets import load_richtileslist_fromfile, SlideMatrix_Dataset, \
    AttK_MIL_Dataset
from models.functions import train_agt_epoch, train_epoch, regular_evaluation, \
    store_evaluation_roc
from models.networks import BasicResNet18, GatedAttentionPool, AttentionPool, \
    reload_net, store_net, ViT_Tiny, ViT_D6_H8
import numpy as np
from support.env import devices
from support.metadata import query_task_label_dict_fromcsv
from support.tools import Time


def safe_random_sample(pickpool, K):
    
    if len(pickpool) > K:
        return random.sample(pickpool, K)
    else:
        return pickpool


def filter_singlesldie_topattKtiles(tiles_all_list, slide_tileidxs_list, slide_attscores, K):
    """
    Args:
        tiles_all_list: tiles list with all tiles in (training / testing) set 
        slide_tileidxs_list: list contains the tiles' idx for the slide,
            each idx is corresponding with the original all tiles list
        slide_attscores: numpy slide_attscores with same sequence as in <slide_tileidxs_list>
        
    Return:
        attK_slide_tiles_list: list of top-K attentional tiles for the inputed slide
    """
    slide_tileidx_array = np.array(slide_tileidxs_list)
    
    order = np.argsort(slide_attscores)
    # from max to min
    slide_tileidx_sort_array = np.flipud(slide_tileidx_array[order])
    slide_K_tileidxs = slide_tileidx_sort_array.tolist()[:K]
    
    attK_slide_tiles_list = []
    for idx in slide_K_tileidxs:
        attK_slide_tiles_list.append(tiles_all_list[idx])
        
    return attK_slide_tiles_list
    

def filter_slides_posKtiles(slide_attscores_dict, slide_tileidxs_dict,
                            K, R_K=0, top_range=0.05, buf_range=[0.05, 0.5]):
    """
    pick K + R_K tiles (attention and supplementary tiles)
    all these tiles are assigned positive (pos) gradient in optimization
    
    Args:
        slide_attscores_dict: [key]: slide_id, [value]: numpy slide_attscores
        tileidx_slideid_dict:
        K:
        top_range: will random picked from top (example: 0.05) of all tiles in one slide
    """
    ''' check if the attscores have been cutoff, otherwise: throw the error and return '''
    if len(set(list(len(slide_attscores_dict[slide_id]) for slide_id in slide_attscores_dict.keys()))) == 1:
        warnings.warn('slide_attscores should be cutoff, please check!')
        return
    
    ''' for each slide: tileidx <-> attention_scores: (0, 1, 2, ...) <-> (float, float, float, ...) '''
    filter_trainK_slide_tileidx_dict = {}
    reusable_slide_pos_pickpool_dict = {}
    reusable_slide_rand_pickpool_dict = {}
    for slide_id in slide_tileidxs_dict.keys():
        slide_tileidx_array = np.array(slide_tileidxs_dict[slide_id])
        slide_attscores = slide_attscores_dict[slide_id]

        order = np.argsort(slide_attscores)
        # from max to min
        slide_tileidx_sort_array = np.flipud(slide_tileidx_array[order])
        idx_size = len(slide_tileidx_array)
        
        rand_K_tileidxs = []
        if top_range > 0.0:
            slide_pos_tileidx_pickpool = slide_tileidx_sort_array.tolist()[:round(idx_size * top_range)]
            slide_K_tileidxs = safe_random_sample(slide_pos_tileidx_pickpool, K)
            reusable_slide_pos_pickpool_dict[slide_id] = slide_pos_tileidx_pickpool
        else:
            slide_K_tileidxs = slide_tileidx_sort_array.tolist()[:K]
            reusable_slide_pos_pickpool_dict[slide_id] = slide_K_tileidxs
            
        if buf_range[1] - buf_range[0] > 0.0 and R_K > 0:
                slide_rand_tileidx_pickpool = slide_tileidx_sort_array.tolist()[round(idx_size * buf_range[0]): round(idx_size * buf_range[1])]
                rand_K_tileidxs = safe_random_sample(slide_rand_tileidx_pickpool, R_K)
                reusable_slide_rand_pickpool_dict[slide_id] = slide_rand_tileidx_pickpool
        
        filter_trainK_slide_tileidx_dict[slide_id] = slide_K_tileidxs + rand_K_tileidxs
        
    return filter_trainK_slide_tileidx_dict, reusable_slide_pos_pickpool_dict, reusable_slide_rand_pickpool_dict


def filter_slides_posKtiles_fromloader(slidemat_loader, attpool_net, slide_tileidxs_dict,
                                       K, R_K, top_range=0.05, buf_range=[0.05, 0.5]):
    """
    pick K + R_K tiles (attention and supplementary tiles) with slidemat DataLoader
    all these tiles are assigned positive (pos) gradient in optimization
    
    Args:
        slidemat_loader: 
        attpool_net:
        slide_tileidxs_dict:
        K:
        R_K: 
    """
    slide_attscores_dict = functions_attpool.query_slides_attscore(slidemat_loader, attpool_net,
                                                                   cutoff_padding=True, norm=True)
    filter_trainK_slide_tileidx_dict, reusable_slide_pos_pickpool_dict, reusable_slide_rand_pickpool_dict = filter_slides_posKtiles(slide_attscores_dict,
                                                                                                                                    slide_tileidxs_dict,
                                                                                                                                    K, R_K,
                                                                                                                                    top_range, buf_range)
            
    return filter_trainK_slide_tileidx_dict, reusable_slide_pos_pickpool_dict, reusable_slide_rand_pickpool_dict


def repick_filter_slides_Ktiles(reusable_slide_pickpool_dict, reusable_slide_rand_pickpool_dict, K, R_K):
    """
    re-pick another batch of K + R_K tiles (attention and supplementary tiles)
    all these tiles are assigned positive (pos) gradient in optimization
    """
    
    filter_sideK_slide_tileidx_dict = {}
    for slide_id in reusable_slide_pickpool_dict.keys():
        slide_tileidx_pickpool = reusable_slide_pickpool_dict[slide_id]
        slide_K_tileidxs = safe_random_sample(slide_tileidx_pickpool, K)
        
        filter_sideK_slide_tileidx_dict[slide_id] = slide_K_tileidxs
    
    if len(reusable_slide_rand_pickpool_dict.keys()) > 0:
        for slide_id in reusable_slide_rand_pickpool_dict.keys():
            slide_rand_tileidx_pickpool = reusable_slide_rand_pickpool_dict[slide_id]
            slide_rand_K_tileidxs = safe_random_sample(slide_rand_tileidx_pickpool, R_K)
            
            filter_sideK_slide_tileidx_dict[slide_id].extend(slide_rand_K_tileidxs)
        
    return filter_sideK_slide_tileidx_dict


def reload_modelpaths_dict(_env_records_dir, modelpaths_dict_filename):
    '''
    reload the dictionary of modelpaths
    '''
    modelpaths_dict_filepath = os.path.join(_env_records_dir, modelpaths_dict_filename)
    with open(modelpaths_dict_filepath, 'r') as json_f:
        modelpaths_dict = json.load(json_f)
        
    return modelpaths_dict


''' --------------------- class of lcsb (interactive mil) algorithm --------------------- '''


class LCSB_MIL():
    '''
    MIL with basic version of Self-Interactive learning algorithm
    '''
    def __init__(self, ENV_task, encoder=None, aggregator_name='GatedAttPool', pt_agt_name=None,
                 tile_data_augmentation=False, model_filename_tuple=(None, None), test_mode=False):
        '''
        Args:
        '''
        self.ENV_task = ENV_task
        ''' prepare some parames '''
        _env_task_name = self.ENV_task.TASK_NAME
        _env_records_dir = self.ENV_task.RECORDS_REPO_DIR
        _env_loss_package = self.ENV_task.LOSS_PACKAGE
        self.history_record_rounds = self.ENV_task.HIS_RECORD_ROUNDS
        
        self.apply_tumor_roi = self.ENV_task.APPLY_TUMOR_ROI
        self.model_store_dir = self.ENV_task.MODEL_STORE_DIR
        pt_prefix = 'pt_' if pt_agt_name is not None else ''
        self.alg_name = '{}{}LCSB_{}{}_{}'.format(pt_prefix, 'g_' if aggregator_name=='GatedAttPool' else '',
                                                self.ENV_task.SLIDE_TYPE, self.ENV_task.FOLD_SUFFIX, _env_task_name)
        
        continue_train = model_filename_tuple[0] != None and model_filename_tuple[1] != None
        if continue_train is False and test_mode is True:
            warnings.warn('no trained model for testing, please check!')
            return
        
        print('![Initial Stage], test mode: {}'.format(test_mode))
        
        self.num_round = self.ENV_task.NUM_ROUND
        self.pt_agt_name = pt_agt_name
        if test_mode is True:
            ''' in test mode, we don't need pre-trained aggregator '''
            self.pt_agt_name = None
        # if the aggregator has been pre-trained, do not need delay stop in the 1st round
        self.attpool_stop_maintains = self.ENV_task.ATTPOOL_STOP_MAINTAINS # if self.pt_agt_name == None else 2
        self.num_init_s_epoch = self.ENV_task.NUM_INIT_S_EPOCH if self.pt_agt_name == None else 1
        
        self.num_inround_s_epoch = self.ENV_task.NUM_INROUND_S_EPOCH
        self.num_inround_t_epoch = self.ENV_task.NUM_INROUND_T_EPOCH
        self.inround_t_refresh_pulse = self.ENV_task.POS_REFRESH_PULSE
        self.batch_size_ontiles = self.ENV_task.MINI_BATCH_TILE
        self.batch_size_onslides = self.ENV_task.MINI_BATCH_SLIDEMAT
#         self.model_filename_tuple = model_filename_tuple
        self.tile_loader_num_workers = self.ENV_task.TILE_DATALOADER_WORKER
        self.slidemat_loader_num_workers = self.ENV_task.SLIDEMAT_DATALOADER_WORKER
        self.last_eval_epochs = self.ENV_task.NUM_LAST_EVAL_EPOCHS
        self.reset_optimizer = self.ENV_task.RESET_OPTIMIZER
        self.attpool_stop_loss = self.ENV_task.ATTPOOL_STOP_LOSS
        self.overall_stop_loss = self.ENV_task.OVERALL_STOP_LOSS if self.pt_agt_name == None else self.ENV_task.OVERALL_STOP_LOSS + 0.05
        self.att_K = self.ENV_task.ATT_K
        self.rand_K = self.ENV_task.SUP_K
        self.top_range = self.ENV_task.TOP_RANGE_RATE
        self.buf_range = self.ENV_task.SUP_RANGE_RATE
        self.fold = self.ENV_task.FOLD_SUFFIX
        
        if encoder is None:
            self.encoder = BasicResNet18(output_dim=2)
        else:
            self.encoder = encoder
        self.encoder = self.encoder.cuda()
        
        if test_mode is False:
            ''' in test mode, we don't need the test slidemat sets '''
            print('Initializing the training slide matrices...', end=', ')        
            train_slidemat_init_time = Time()
            self.train_slidemat_file_sets = functions_attpool.check_load_slide_matrix_files(self.ENV_task,
                                                                                            batch_size_ontiles=self.batch_size_ontiles,
                                                                                            tile_loader_num_workers=self.tile_loader_num_workers,
                                                                                            encoder_net=self.encoder.backbone,
                                                                                            force_refresh=False if continue_train and test_mode is False else True,
                                                                                            for_train=True, print_info=False)
            print('slides: %d, time: %s' % (len(self.train_slidemat_file_sets), str(train_slidemat_init_time.elapsed())))
        else:
            self.train_slidemat_file_sets = []
            
        print('Initializing the testing slide matrices...', end=', ')
        test_slidemat_init_time = Time()
        self.test_slidemat_file_sets = functions_attpool.check_load_slide_matrix_files(self.ENV_task,
                                                                                       batch_size_ontiles=self.batch_size_ontiles,
                                                                                       tile_loader_num_workers=self.tile_loader_num_workers,
                                                                                       encoder_net=self.encoder.backbone,
                                                                                       force_refresh=False if continue_train and test_mode is False else True,
                                                                                       for_train=False if not self.ENV_task.DEBUG_MODE else True,
                                                                                       print_info=False)
        print('slides: %d, time: %s' % (len(self.test_slidemat_file_sets), str(test_slidemat_init_time.elapsed())))
        
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
        
        if self.pt_agt_name is not None and continue_train is False:
            '''use pre-trained weights for aggregator, not suitable for continue training '''
            self.aggregator, _ = reload_net(self.aggregator, os.path.join(ENV_task.MODEL_STORE_DIR, self.pt_agt_name))
            
        self.check_point_s, self.check_point_t = None, None
        if continue_train:
            if model_filename_tuple[0].find(self.aggregator.name) == -1 or model_filename_tuple[1].find(self.encoder.name) == -1:
                warnings.warn('Get the wrong network model name in tuple... auto initialize the brand new networks!')
                pass
            else:
                model_s_filepath = os.path.join(self.model_store_dir, model_filename_tuple[0])
                model_t_filepath = os.path.join(self.model_store_dir, model_filename_tuple[1])
                self.aggregator, self.check_point_s = reload_net(self.aggregator, model_s_filepath)
                self.encoder, self.check_point_t = reload_net(self.encoder, model_t_filepath)
                print('Reload network model from: {}//({}, {})'.format(self.model_store_dir, model_filename_tuple[0], model_filename_tuple[1]))
        self.aggregator = self.aggregator.cuda()
        print('On-standby: {} algorithm'.format(self.alg_name))
        print('Network: {} and {}, train on -> {}'.format(self.aggregator.name, self.encoder.name, devices))
        
        # make tiles data
        if test_mode is False:
            self.train_tiles_list, _, self.train_slide_tileidxs_dict = load_richtileslist_fromfile(self.ENV_task, for_train=True)
        else:
            ''' for test mode '''
            self.train_tiles_list, self.train_slide_tileidxs_dict = [], None
        self.test_tiles_list, _, self.test_slide_tileidxs_dict = load_richtileslist_fromfile(self.ENV_task,
                                                                                             for_train=False if not self.ENV_task.DEBUG_MODE else True)
        
        # make label
        self.label_dict = query_task_label_dict_fromcsv(self.ENV_task)
        
        if _env_loss_package[0] == 'wce':
            self.criterion_s = functions.weighted_cel_loss(_env_loss_package[1][0])
            self.criterion_t = functions.weighted_cel_loss(_env_loss_package[1][0])
        else:
            self.criterion_s = functions.cel_loss()
            self.criterion_t = functions.cel_loss()
        self.optimizer_s = functions.optimizer_adam_basic(self.aggregator, lr=1e-4) # if self.pt_agt_name == None else 1e-4)
        self.optimizer_t = functions.optimizer_adam_pretrained(self.encoder, lr=1e-4)
        
        # prepare the initial Dataset
        if len(self.train_slidemat_file_sets) > 0:
            self.train_slidemat_set = SlideMatrix_Dataset(self.train_slidemat_file_sets, self.label_dict)
        else:
            ''' for test mode '''
            self.train_slidemat_set = None
        self.test_slidemat_set = SlideMatrix_Dataset(self.test_slidemat_file_sets, self.label_dict)
        
        if tile_data_augmentation is True:
            self.train_transform_augs = functions.get_data_arg_transform()
        else:
            self.train_transform_augs = functions.get_transform()
            
        if len(self.train_tiles_list) > 0:
            self.train_attK_tiles_set = AttK_MIL_Dataset(tiles_list=self.train_tiles_list, label_dict=self.label_dict,
                                                         transform=self.train_transform_augs)
        else:
            ''' for test mode'''
            self.train_attK_tiles_set = []
        
        self.test_transform_augs = functions.get_transform()
            
        self.stored_modelpath_dict = {'checkpoint': {'attpool': None,
                                                     'resnet': None},
                                      'milestone': {'attpool': [],
                                                    'resnet': []}}
        
        self.modelpaths_log_path = os.path.join(_env_records_dir,
                                                'recorded_modelpaths_{}{}.json'.format(self.alg_name, '_TROI' if self.apply_tumor_roi == True else ''))
                
    def reset_optim(self):
        ''' check and reset the optimizer_s and optimizer_t '''
        if self.reset_optimizer == True:
            self.optimizer_s.state = collections.defaultdict(dict)
            self.optimizer_t.state = collections.defaultdict(dict)
    
    def reload_optim(self):
        ''' check and reload the optimizer_s and optimizer_t '''
        if self.check_point_s != None:
            self.optimizer_s.load_state_dict(self.check_point_s['optimizer'])
        if self.check_point_t != None:
            self.optimizer_t.load_state_dict(self.check_point_t['optimizer'])
                
        inround_s_epoch = self.check_point_s['epoch'] if self.check_point_s != None else 0
        inround_t_epoch = self.check_point_t['epoch'] if self.check_point_t != None else 0
        
        return inround_s_epoch, inround_t_epoch
    
    def slide_epoch(self, inround_s_epoch, train_slidemat_loader, acc_maintain_epoch, att_epoch_stop, overall_round_stop):
        '''
        the move of one epoch of training on slide
        '''
        train_log_s = train_agt_epoch(self.aggregator, train_slidemat_loader,
                                      self.criterion_s, self.optimizer_s,
                                      epoch_info=(inround_s_epoch, self.num_inround_s_epoch))
        
        attpool_current_loss = float(train_log_s[train_log_s.find('loss->') + 6: train_log_s.find(', train')])
        if attpool_current_loss <= self.attpool_stop_loss:
            acc_maintain_epoch += 1
        else:
            acc_maintain_epoch = 0
            
        if acc_maintain_epoch >= self.attpool_stop_maintains and inround_s_epoch >= self.num_init_s_epoch:
            ''' this flag means the initial round has finished'''
            att_epoch_stop = True
        if att_epoch_stop == True and attpool_current_loss <= self.overall_stop_loss:
            overall_round_stop = True
            
        print(train_log_s)
        inround_s_epoch += 1
        return inround_s_epoch, acc_maintain_epoch, att_epoch_stop, overall_round_stop
    
    def tile_pos_epoch(self, inround_t_epoch, train_attK_tile_loader):
        
        train_log_t = train_epoch(self.encoder, train_attK_tile_loader,
                                  self.criterion_t, self.optimizer_t,
                                  epoch_info=(inround_t_epoch, self.num_inround_t_epoch))
        print(train_log_t)
        inround_t_epoch += 1
        return inround_t_epoch
        
    def optimize(self):
        '''
        training the models in this algorithm
        '''
        if self.train_slidemat_set is None:
            warnings.warn('test mode, can not running training!')
            return
        
        print('![Training Stage]')
        now_round = self.check_point_t['round'] if self.check_point_t != None else 0
        att_epoch_stop = False  # flag to indicate the stop of first round
        overall_round_stop = False  # flag to indicate the stop of all rounds
        acc_maintain_epoch = 0
        queue_auc = []
        while now_round < self.num_round and overall_round_stop == False:
            if now_round == 0:
                print('### In Initial training... round')
            else:
                print('### In training... round: [%d/%d]' % (now_round, self.num_round - 1))
            
            ''' first stage in one round: training attention pool net on slide matrices '''
            if now_round > 0:
                # refresh after first round
                self.train_slidemat_set.refresh_data(self.train_slidemat_file_sets)
                self.test_slidemat_set.refresh_data(self.test_slidemat_file_sets)
                # reset the optimizer_s and optimizer_t
                self.reset_optim()
                
            inround_s_epoch, inround_t_epoch = self.reload_optim()
            
            train_slidemat_loader = functions.get_data_loader(self.train_slidemat_set, self.batch_size_onslides,
                                                              num_workers=self.slidemat_loader_num_workers, sf=True)
            test_slidemat_loader = functions.get_data_loader(self.test_slidemat_set, self.batch_size_onslides, 
                                                             num_workers=self.slidemat_loader_num_workers, sf=False)
            
            print('[training on slide matrices...]')
            while (inround_s_epoch < self.num_inround_s_epoch or att_epoch_stop == False) and overall_round_stop == False:
                if att_epoch_stop == True and now_round == 0:
                    break # cancel epoch count in the first round
                inround_s_epoch, acc_maintain_epoch, att_epoch_stop, overall_round_stop = self.slide_epoch(inround_s_epoch, train_slidemat_loader,
                                                                                                           acc_maintain_epoch, att_epoch_stop,
                                                                                                           overall_round_stop)
                ''' prediction, testing and record parts '''
                print('>>> In testing...', end=', ')
                ''' prepare a test for this round '''
                y_s_scores, y_labels = self.predict(test_slidemat_loader)
                ''' evaluation '''
                acc_s, _, _, auc_s = functions.regular_evaluation(y_s_scores, y_labels)
                print('>>> on slidemat -> test acc: %.4f, test auc: %.4f' % (acc_s, auc_s))
                
                if now_round > 0:
                    queue_auc.append(auc_s)
                    if len(queue_auc) > self.last_eval_epochs:
                        queue_auc.remove(queue_auc[0])
                
            del train_slidemat_loader
            gc.collect()
            
            '''
            the round-0 is the initializing for attention pool with ImageNet pre-trained resnet embeddings
            training resnet before last round
            '''
            if now_round < self.num_round - 1 and overall_round_stop == False:
                ''' second stage in one round: training classification resnet on attK tile images '''
                print('==> pick the topK attention tiles for look closer and training ==>')
                # get the attention informative tiles, WARNING: need to regenerate a new dataloader and set the shuffle as False
                train_slidemat_loader = functions.get_data_loader(self.train_slidemat_set, self.batch_size_onslides,
                                                                  num_workers=self.slidemat_loader_num_workers, sf=False) 
                train_filter_K_slide_tileidx_dict, reusable_slide_pos_pickpool_dict, reusable_slide_rand_pickpool_dict = filter_slides_posKtiles_fromloader(slidemat_loader=train_slidemat_loader,
                                                                                                                                                            attpool_net=self.aggregator,
                                                                                                                                                            slide_tileidxs_dict=self.train_slide_tileidxs_dict,
                                                                                                                                                            K=self.att_K, R_K=self.rand_K,
                                                                                                                                                            top_range=self.top_range,
                                                                                                                                                            buf_range=self.buf_range)
                del train_slidemat_loader
                gc.collect()
                
                # refresh the tile training set with attK tiles
                self.train_attK_tiles_set.refresh_data(train_filter_K_slide_tileidx_dict)
                
                print('[training on attK attentional tiles...]')
                while inround_t_epoch < self.num_inround_t_epoch:
                    train_attK_tile_loader = functions.get_data_loader(self.train_attK_tiles_set, batch_size=self.batch_size_ontiles,
                                                                       num_workers=self.tile_loader_num_workers, sf=True, p_mem=True)
                    inround_t_epoch = self.tile_pos_epoch(inround_t_epoch, train_attK_tile_loader)
                    
                    if inround_t_epoch < self.num_inround_t_epoch and inround_t_epoch % self.inround_t_refresh_pulse == 0:
                        filter_K_slide_tileidx_dict = repick_filter_slides_Ktiles(reusable_slide_pos_pickpool_dict,
                                                                                     reusable_slide_rand_pickpool_dict,
                                                                                     self.att_K, self.rand_K)
                        self.train_attK_tiles_set.refresh_data(filter_K_slide_tileidx_dict)
                      
                    del train_attK_tile_loader
                    gc.collect()
            
            # record milestone round for visualization
            if now_round in self.history_record_rounds:
                self.record(now_round, inround_s_epoch, inround_t_epoch, overall_round_stop)
            
            # step to next round
            now_round += 1
            
            if now_round < self.num_round and overall_round_stop == False:
                self.refresh_embed()
        
        ''' calculate the final performance '''
        print(queue_auc)
        final_avg_auc = np.average(queue_auc)
        print('### final evaluation on attpool -> average test auc: %.4f' % (final_avg_auc))
    
    def record(self, now_round, inround_s_epoch, inround_t_epoch, overall_round_stop):
        '''
        store the trained models and record the storage log file
        '''
        # record the attpool and resnet model
        alg_milestone_name = self.alg_name + '_[{}]'.format(now_round)
        
        milestone_obj_dict_s = {'epoch': inround_s_epoch, 'round': now_round}
        milestone_filepath_s = store_net(self.apply_tumor_roi, self.model_store_dir, self.aggregator,
                                         alg_milestone_name, self.optimizer_s, milestone_obj_dict_s)
        self.stored_modelpath_dict['milestone']['attpool'].append(milestone_filepath_s)
        
        milestone_filepath_t = 'N/A'
        if now_round < self.num_round - 1 and overall_round_stop == False:
            milestone_obj_dict_t = {'epoch': inround_t_epoch, 'round': now_round}
            milestone_filepath_t = store_net(self.apply_tumor_roi, self.model_store_dir, self.encoder,
                                             alg_milestone_name, self.optimizer_t, milestone_obj_dict_t)
            self.stored_modelpath_dict['milestone']['resnet'].append(milestone_filepath_t)
        
        print('[store the dual-model MILESTONE<{} and {}>]'.format(milestone_filepath_s, milestone_filepath_t))
        
        with open(self.modelpaths_log_path, 'w') as json_f:
            json.dump(self.stored_modelpath_dict, json_f)
            print('*** recorded modelpaths json file is: {} ***'.format(self.modelpaths_log_path))
    
    def predict(self, test_slidemat_loader):
        '''
        perform the prediction based on the trained model
        '''
        self.aggregator.eval()
        
        y_s_scores, y_labels = [], []
        with torch.no_grad():
            for mat_X, bag_dim, y in test_slidemat_loader:
                mat_X = mat_X.cuda()
                bag_dim = bag_dim.cuda()
                y = y.cuda()
                # feed forward
                y_pred, _, _ = self.aggregator(mat_X, bag_dim)
                y_pred = softmax(y_pred, dim=-1)
                # count results
                y_s_scores.extend(y_pred.detach().cpu().numpy()[:, -1].tolist())
                y_labels.extend(y.cpu().numpy().tolist())
            
        return np.array(y_s_scores), np.array(y_labels)
    
    def refresh_embed(self):
        ''' 
        update the slide matrices with trained resnet (in the 2nd stage)
        '''
        print('<---- update slide matrices with trained resnet ---->', end=', ')
        update_time = Time()
        self.train_slidemat_file_sets = functions_attpool.check_load_slide_matrix_files(self.ENV_task, self.batch_size_ontiles,
                                                                                        self.tile_loader_num_workers, encoder_net=self.encoder.backbone,
                                                                                        force_refresh=True, for_train=True, print_info=False)
        self.test_slidemat_file_sets = functions_attpool.check_load_slide_matrix_files(self.ENV_task, self.batch_size_ontiles, self.tile_loader_num_workers,
                                                                                       encoder_net=self.encoder.backbone, force_refresh=True,
                                                                                       for_train=False if not self.ENV_task.DEBUG_MODE else True,
                                                                                       print_info=False)
        print('train: %d, test: %d, time: %s sec' % (len(self.train_slidemat_file_sets), len(self.test_slidemat_file_sets), str(update_time.elapsed())))
   

''' ---------------- function for load trained models' paths from log filename ---------------- '''

        
def load_modelpaths_dict(ENV_task, modelpaths_dict_filename):
    '''
    reload the dictionary of modelpaths
    '''
    modelpaths_dict_filepath = os.path.join(ENV_task.RECORDS_REPO_DIR, modelpaths_dict_filename)
    with open(modelpaths_dict_filepath, 'r') as json_f:
        modelpaths_dict = json.load(json_f)
        
    return modelpaths_dict


''' ---------------- running functions can be directly called ---------------- '''

''' training functions (temporarily not in use) '''
def _run_train_lcsb_gated_attpool_vit_pc(ENV_task, vit_pt_name=None, pt_agt_name=None):
    vit_encoder = ViT_Tiny(image_size=ENV_task.TRANSFORMS_RESIZE,
                           patch_size=int(ENV_task.TILE_H_SIZE / 32), output_dim=2)
    if vit_pt_name is not None:
        vit_encoder, _ = reload_net(vit_encoder, os.path.join(ENV_task.MODEL_STORE_DIR, vit_pt_name))
        
    method = LCSB_MIL(ENV_task, vit_encoder, 'GatedAttPool', pt_agt_name)
    method.optimize()
    
def _run_train_lcsb_gated_attpool_vit_6_8(ENV_task, vit_pt_name=None, pt_agt_name=None):
    vit_encoder = ViT_D6_H8(image_size=ENV_task.TRANSFORMS_RESIZE,
                            patch_size=int(ENV_task.TILE_H_SIZE / 32), output_dim=2)
    if vit_pt_name is not None:
        vit_encoder, _ = reload_net(vit_encoder, os.path.join(ENV_task.MODEL_STORE_DIR, vit_pt_name))
        
    method = LCSB_MIL(ENV_task, vit_encoder, 'GatedAttPool', pt_agt_name)
    method.optimize()

''' training functions '''
def _run_train_lcsb_attpool_resnet18(ENV_task, pt_agt_name=None):
    encoder = BasicResNet18(output_dim=2)
    method = LCSB_MIL(ENV_task, encoder, 'AttPool', pt_agt_name)
    method.optimize()

def _run_train_lcsb_gated_attpool_resnet18(ENV_task, pt_agt_name=None):
    encoder = BasicResNet18(output_dim=2)
    method = LCSB_MIL(ENV_task, encoder, 'GatedAttPool', pt_agt_name)
    method.optimize()
   
''' testing functions ''' 
def _run_test_lcsb_gated_attpool_resnet18(ENV_task, model_s_filename, model_t_filename):
    encoder = BasicResNet18(output_dim=2)
    encoder, _ = reload_net(encoder, os.path.join(ENV_task.MODEL_STORE_DIR, model_t_filename))
    method = LCSB_MIL(ENV_task=ENV_task, encoder=encoder,
                      aggregator_name='GatedAttPool', 
                      model_filename_tuple=(model_s_filename, model_t_filename), 
                      test_mode=True)
    
    test_slidemat_loader = functions.get_data_loader(method.test_slidemat_set, method.batch_size_onslides, 
                                                     num_workers=method.slidemat_loader_num_workers, sf=False)
    y_scores, y_labels = method.predict(test_slidemat_loader)
    acc, fpr, tpr, auc = regular_evaluation(y_scores, y_labels)
    roc_pkl_path = os.path.join(ENV_task.PLOT_STORE_DIR, 'roc_{}-{}.csv'.format(method.alg_name, Time().date))
    store_evaluation_roc(csv_path=roc_pkl_path, roc_set=(acc, fpr, tpr, auc))
    print('Eval %s, Test, BACC: %.4f, AUC: %.4f, store the roc result at: %s' % (method.alg_name, acc, auc, roc_pkl_path) )

    
if __name__ == '__main__':
    pass
