'''
@author: Yang Hu
'''
import gc
import json
import os
import warnings

from models import functions, functions_lcsb, functions_attpool
from models.datasets import Rev_AttK_MIL_Dataset
from models.functions import train_rev_epoch, regular_evaluation, \
    store_evaluation_roc
from models.functions_lcsb import LCSB_MIL, filter_slides_posKtiles_fromloader, \
    repick_filter_slides_Ktiles
from models.networks import ReverseResNet18, store_net, ViT_Tiny, reload_net, \
    ViT_D6_H8, ReverseViT_Tiny, ReverseViT_D6_H8
import numpy as np
from support.tools import Time


def filter_slides_negNtiles(slide_attscores_dict, slide_tileidxs_dict, N, last_range=0.25):
    """
    Args:
        slide_attscores_dict: [key]: slide_id, [value]: numpy slide_attscores
        tileidx_slideid_dict:
        N:
        last_range: will random picked from last (example: 0.25) of all tiles in one slide
    """
    ''' check if the attscores have been cutoff, otherwise: throw the error and return '''
    if len(set(list(len(slide_attscores_dict[slide_id]) for slide_id in slide_attscores_dict.keys()))) == 1:
        warnings.warn('slide_attscores should be cutoff, please check!')
        return
    
    ''' for each slide: tileidx <-> attention_scores: (0, 1, 2, ...) <-> (float, float, float, ...) '''
    filter_negN_slide_tileidx_dict = {}
    reusable_slide_neg_pickpool_dict = {}
    for slide_id in slide_tileidxs_dict.keys():
        slide_tileidx_array = np.array(slide_tileidxs_dict[slide_id])
        slide_attscores = slide_attscores_dict[slide_id]
        order = np.argsort(slide_attscores)
        # from min to max
        slide_tileidx_sort_array = slide_tileidx_array[order]
        idx_size = len(slide_tileidx_array)
        
        slide_neg_tileidx_pickpool = slide_tileidx_sort_array.tolist()[:round(idx_size * last_range)]
        slide_N_tileidxs = functions_lcsb.safe_random_sample(slide_neg_tileidx_pickpool, N)
        
        filter_negN_slide_tileidx_dict[slide_id] = slide_N_tileidxs
        
        reusable_slide_neg_pickpool_dict[slide_id] = slide_neg_tileidx_pickpool
        
    return filter_negN_slide_tileidx_dict, reusable_slide_neg_pickpool_dict


def filter_slides_negNtiles_fromloader(slidemat_loader, attpool_net, slide_tileidxs_dict, N, last_range=0.25):
    """
    filter the side N tiles with lowest attention score for a set of slides
    
    Args:
        slidemat_loader: 
        attpool_net:
        slide_tileidxs_dict:
        N:
    """
    slide_attscores_dict = functions_attpool.query_slides_attscore(slidemat_loader, attpool_net,
                                                                   cutoff_padding=True, norm=True)
    
    filter_negN_slide_tileidx_dict, reusable_slide_neg_pickpool_dict = filter_slides_negNtiles(slide_attscores_dict,
                                                                                               slide_tileidxs_dict,
                                                                                               N, last_range)
            
    return filter_negN_slide_tileidx_dict, reusable_slide_neg_pickpool_dict

class RevgLCSB_MIL(LCSB_MIL):
    '''
    MIL with pos-neg version of Self-Interactive learning algorithm
    with adversarial gradient
    '''
    def __init__(self, ENV_task, encoder=None, aggregator_name='GatedAttPool', pt_agt_name=None,
                 tile_data_augmentation=False, model_filename_tuple=(None, None), test_mode=False):
        super(RevgLCSB_MIL, self).__init__(ENV_task, encoder, aggregator_name, pt_agt_name,
                                           tile_data_augmentation, model_filename_tuple, test_mode)
        '''
        just need to add and change some specific items in addition to the basic LCSB
        '''
        ''' prepare some parames '''
        _env_task_name = self.ENV_task.TASK_NAME
        _env_records_dir = self.ENV_task.RECORDS_REPO_DIR
        pt_prefix = 'pt_' if pt_agt_name is not None else ''
        self.alg_name = '{}{}reLCSB_{}{}_{}'.format(pt_prefix, 'g_' if aggregator_name=='GatedAttPool' else '',
                                                  self.ENV_task.SLIDE_TYPE, self.ENV_task.FOLD_SUFFIX, _env_task_name)
        
        self.num_inround_rev_t_epoch = self.ENV_task.NUM_INROUND_REV_T_EPOCH
        self.inround_rev_t_refresh_pulse = self.ENV_task.NEG_REFRESH_PULSE
        self.revg_N = self.ENV_task.REVERSE_N
        self.last_range = self.ENV_task.NEG_RANGE_RATE
        self.revg_grad_a = self.ENV_task.REVERSE_GRADIENT_ALPHA
        
        ''' 
        if didn't input encoder, use ReverseResNet18 to replace the BasicResNet18 init in LCSB,
        otherwise, no need to renew the encoder
        '''
        if encoder is None:
            # even not be initialized in class <LCSB_MIL>
            self.encoder = ReverseResNet18(output_dim=2)
            self.encoder = self.encoder.cuda()
            
        # add train_negN_tiles_set
        if len(self.train_attK_tiles_set) > 0:
            self.train_negN_tiles_set = Rev_AttK_MIL_Dataset(tiles_list=self.train_tiles_list, 
                                                             label_dict=self.label_dict,
                                                             transform=self.train_transform_augs)
        else:
            self.train_negN_tiles_set = []
        
        # need refresh log_path with new alg_name
        self.modelpaths_log_path = os.path.join(_env_records_dir, 
                                                'recorded_modelpaths_{}{}.json'.format(self.alg_name, '_TROI' if self.apply_tumor_roi == True else ''))
    
    def reload_optim(self):
        ''' 
        check and reload the optimizer_s and optimizer_t
        add inround_r_t_epoch
        '''
        inround_s_epoch, inround_t_epoch = super().reload_optim()
        inround_r_t_epoch = self.check_point_t['epoch_r'] if self.check_point_t != None else 0
        
        return inround_s_epoch, inround_t_epoch, inround_r_t_epoch
    
    def tile_neg_epoch(self, inround_r_t_epoch, train_negN_tile_loader):
        
        train_log_r_t = train_rev_epoch(self.encoder, train_negN_tile_loader,
                                        self.criterion_t, self.optimizer_t,
                                        revg_grad_a=self.revg_grad_a,
                                        epoch_info=(inround_r_t_epoch, self.num_inround_rev_t_epoch))
        print(train_log_r_t)
        inround_r_t_epoch += 1
        return inround_r_t_epoch
    
    def optimize(self):
        '''
        training the models in this algorithm
        '''
        if self.train_slidemat_set is None:
            warnings.warn('test mode, can not running training!')
            return
        
        print('![Training Stage]')
        now_round = self.check_point_t['round'] if self.check_point_t != None else 0
        att_epoch_stop = False # flag to indicate the stop of first round
        overall_round_stop = False # flag to indicate the stop of all rounds
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
            inround_s_epoch, inround_t_epoch, inround_r_t_epoch = self.reload_optim()
            
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
                ''' second stage in one round: training classification resnet on attK and negN tile images '''
                print('==> pick the topK and lowestN attention tiles for look closer and training ==>')
                # get the attention informative tiles, WARNING: need to regenerate a new dataloader and set the shuffle as False
                train_slidemat_loader = functions.get_data_loader(self.train_slidemat_set, self.batch_size_onslides,
                                                                  num_workers=self.slidemat_loader_num_workers, sf=False)
                train_filter_K_slide_tileidx_dict, reusable_slide_pos_pickpool_dict, reusable_slide_rand_pickpool_dict = filter_slides_posKtiles_fromloader(slidemat_loader=train_slidemat_loader,
                                                                                                                                                            attpool_net=self.aggregator,
                                                                                                                                                            slide_tileidxs_dict=self.train_slide_tileidxs_dict,
                                                                                                                                                            K=self.att_K, R_K=self.rand_K,
                                                                                                                                                            top_range=self.top_range,
                                                                                                                                                            buf_range=self.buf_range)
                train_filter_negN_slide_tileidx_dict, reusable_slide_neg_pickpool_dict = filter_slides_negNtiles_fromloader(slidemat_loader=train_slidemat_loader,
                                                                                                                            attpool_net=self.aggregator,
                                                                                                                            slide_tileidxs_dict=self.train_slide_tileidxs_dict,
                                                                                                                            N=self.revg_N, last_range=self.last_range)
                del train_slidemat_loader
                gc.collect()
                
                # refresh the tile training set with attK tiles
                self.train_attK_tiles_set.refresh_data(filter_attK_slide_tileidx_dict=train_filter_K_slide_tileidx_dict)
                self.train_negN_tiles_set.refresh_data(filter_revgN_slide_tileidx_dict=train_filter_negN_slide_tileidx_dict)
                
                print('[training on attK attentional tiles... for %d epochs]' % self.num_inround_t_epoch)         
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
                
                print('[training on negN non-attentional tiles... for %d epochs]' % self.num_inround_rev_t_epoch)
                while inround_r_t_epoch < self.num_inround_rev_t_epoch:
                    train_negN_tile_loader = functions.get_data_loader(self.train_negN_tiles_set, batch_size=self.batch_size_ontiles,
                                                                          num_workers=self.tile_loader_num_workers, sf=True)
                    inround_r_t_epoch = self.tile_neg_epoch(inround_r_t_epoch, train_negN_tile_loader)
                     
                    if inround_r_t_epoch < self.num_inround_rev_t_epoch and inround_r_t_epoch % self.inround_rev_t_refresh_pulse == 0:
                        filter_N_slide_tileidx_dict = repick_filter_slides_Ktiles(reusable_slide_neg_pickpool_dict, {}, self.revg_N, 0)
                        self.train_negN_tiles_set.refresh_data(filter_N_slide_tileidx_dict)
                        
                    del train_negN_tile_loader
                    gc.collect()
            
            # record milestone round for visualization
            if now_round in self.history_record_rounds:
                self.record(now_round, inround_s_epoch, inround_t_epoch, inround_r_t_epoch, overall_round_stop)
                
            # step to next round
            now_round += 1
            
            if now_round < self.num_round and overall_round_stop == False:
                self.refresh_embed()
        
        ''' calculate the final performance '''
        print(queue_auc)
        final_avg_auc = np.average(queue_auc)
        print('### final evaluation on attpool -> average test auc: %.4f' % (final_avg_auc))
            
    def record(self, now_round, inround_s_epoch, inround_t_epoch, inround_r_t_epoch, overall_round_stop):
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
            milestone_obj_dict_t = {'epoch': inround_t_epoch, 'epoch_r': inround_r_t_epoch, 'round': now_round}
            milestone_filepath_t = store_net(self.apply_tumor_roi, self.model_store_dir, self.encoder,
                                             alg_milestone_name, self.optimizer_t, milestone_obj_dict_t)
            self.stored_modelpath_dict['milestone']['resnet'].append(milestone_filepath_t)
        
        print('[store the dual-model MILESTONE<{} and {}>]'.format(milestone_filepath_s, milestone_filepath_t))
        
        with open(self.modelpaths_log_path, 'w') as json_f:
            json.dump(self.stored_modelpath_dict, json_f)
            print('*** recorded modelpaths json file is: {} ***'.format(self.modelpaths_log_path))
                
''' ---------------- running functions can be directly called ---------------- '''

''' training functions (temporarily not in use) '''
def _run_train_relcsb_gated_attpool_vit_pc(ENV_task, vit_pt_name=None, pt_agt_name=None):
    vit_encoder = ReverseViT_Tiny(image_size=ENV_task.TRANSFORMS_RESIZE,
                                  patch_size=int(ENV_task.TILE_H_SIZE / 32), output_dim=2)
    if vit_pt_name is not None:
        vit_encoder, _ = reload_net(vit_encoder, os.path.join(ENV_task.MODEL_STORE_DIR, vit_pt_name))
        
    method = RevgLCSB_MIL(ENV_task, vit_encoder, 'GatedAttPool', pt_agt_name)
    method.optimize()
    
def _run_train_relcsb_gated_attpool_vit_6_8(ENV_task, vit_pt_name=None, pt_agt_name=None):
    vit_encoder = ReverseViT_D6_H8(image_size=ENV_task.TRANSFORMS_RESIZE,
                                   patch_size=int(ENV_task.TILE_H_SIZE / 32), output_dim=2)
    if vit_pt_name is not None:
        vit_encoder, _ = reload_net(vit_encoder, os.path.join(ENV_task.MODEL_STORE_DIR, vit_pt_name))
        
    method = RevgLCSB_MIL(ENV_task, vit_encoder, 'GatedAttPool', pt_agt_name)
    method.optimize()

''' training functions '''
def _run_train_relcsb_attpool_resnet18(ENV_task, pt_agt_name=None):
    encoder = ReverseResNet18(output_dim=2)
    method = RevgLCSB_MIL(ENV_task, encoder, 'AttPool', pt_agt_name)
    method.optimize()

def _run_train_relcsb_gated_attpool_resnet18(ENV_task, pt_agt_name=None):
    encoder = ReverseResNet18(output_dim=2)
    method = RevgLCSB_MIL(ENV_task, encoder, 'GatedAttPool', pt_agt_name)
    method.optimize()
    
''' testing functions '''
def _run_test_relcsb_gated_attpool_resnet18(ENV_task, model_s_filename, model_t_filename):
    encoder = ReverseResNet18(output_dim=2)
    encoder, _ = reload_net(encoder, os.path.join(ENV_task.MODEL_STORE_DIR, model_t_filename))
    method = RevgLCSB_MIL(ENV_task=ENV_task, encoder=encoder,
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