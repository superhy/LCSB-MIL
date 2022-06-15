'''
@author: Yang Hu
'''
import os
import platform


class parames_basic():
    
    def __init__(self, project_name,
                 slide_type='dx',
                 apply_tumor_roi=False,
                 scale_factor=32,
                 tile_size=256, # 512
                 tp_tiles_threshold=70,
                 pil_image_file_format='.png',
                 debug_mode=False):
        """
        Args:
            project_name:, 
            project_dir: use project_name construct the project dir path,
            slide_type: dx or tx, default dx,
            apply_tumor_roi: default False,
            scale_factor: scale ratio when visualization,
            tile_h_size: patch size to separate the whole slide image,
            tile_w_size,
            transforms_resize,
            tp_tiles_threshold,
            pil_image_file_format,
            debug_mode
        """
        
        self.OS_NAME = platform.system()
        self.PROJECT_NAME = project_name
        self.PROJECT_DIR = os.path.join('D:/workspace', self.PROJECT_NAME) if self.OS_NAME == 'Windows' else os.path.join('/well/rittscher/users/lec468/workspace', self.PROJECT_NAME)
        self.SLIDE_TYPE = slide_type
        self.APPLY_TUMOR_ROI = apply_tumor_roi
        self.SCALE_FACTOR = scale_factor
        self.TILE_H_SIZE = tile_size
        self.TILE_W_SIZE = self.TILE_H_SIZE
        self.TRANSFORMS_RESIZE = self.TILE_H_SIZE
        self.TP_TILES_THRESHOLD = tp_tiles_threshold
        self.PIL_IMAGE_FILE_FORMAT = pil_image_file_format
        self.DEBUG_MODE = debug_mode
        
class parames_task(parames_basic):
    
    def __init__(self, project_name, 
                 slide_type, 
                 apply_tumor_roi, 
                 scale_factor, 
                 tile_size, 
                 tp_tiles_threshold, 
                 pil_image_file_format, 
                 debug_mode,
                 task_name,
                 label_type,
                 server_root,
                 pc_root,
                 test_part_prop,
                 fold_suffix,
                 num_cnn_epoch,
                 num_att_epoch,
                 mini_batch_tile=128,
                 tile_dataloader_worker=12,
                 loss_package=('ce', [0.95]),
                 mil_try_k=5,
                 tile_encode_dim=512,
                 mini_batch_slidemat=8,
                 slidemat_dataloader_worker=4,
                 num_last_eval_epochs=5,
                 reset_optim=True,
                 num_round=5,
                 num_init_s_epoch=10,
                 num_inround_s_epoch=5,
                 num_inround_rev_t_epoch=1,
                 num_inround_t_epoch=2,
                 attpool_stop_loss=0.60,
                 attpool_stop_maintains=3,
                 overall_stop_loss=0.45,
                 pos_refersh_pluse=1,
                 neg_refersh_pluse=1,
                 top_range_rate=0.05,
                 sup_range_rate=[0.05, 0.95],
                 neg_range_rate=0.2,
                 att_k=50,
                 sup_k=20,
                 reverse_n=10,
                 reverse_gradient_alpha=1e-4,
                 his_record_rounds=[0, 1, 2, 3, 4],
                 num_pretrain_epoch=20,
                 rand_mat_reload_pluse=5,
                 num_preload_rand_mat=2000,
                 mat_bagdim=5000,
                 num_epoch_rand_pair=400):
        """
        Father Args:
            project_name:, 
            project_dir: use project_name construct the project dir path,
            slide_type: dx or tx, default dx,
            apply_tumor_roi: default False,
            scale_factor: scale ratio when visualization,
            tile_h_size: patch size to separate the whole slide image,
            tile_w_size,
            transforms_resize,
            tp_tiles_threshold,
            pil_image_file_format,
            debug_mode
        
        Args:
            
        """
        super(parames_task, self).__init__(project_name, 
                                           slide_type, 
                                           apply_tumor_roi, 
                                           scale_factor, 
                                           tile_size,
                                           tp_tiles_threshold, 
                                           pil_image_file_format, 
                                           debug_mode)
        
        self.TASK_NAME = task_name
        self.LABEL_TYPE = label_type
        self.SERVER_ROOT = server_root
        self.PC_ROOT = pc_root
        self.DATA_DIR = self.PC_ROOT if self.OS_NAME == 'Windows' else self.SERVER_ROOT
        
        """
        -------------- WSI slides process part ------------------
        """
        self.TEST_PART_PROP = test_part_prop
        ''' REPO: repositories, for example: TCGA '''
        self.ORIGINAL_REPO_DATA_DIR = os.path.join(self.DATA_DIR, 'original/{}'.format(self.SLIDE_TYPE))
        self.SLIDETYPE_DIR = 'example' if self.SLIDE_TYPE == 'tx' else 'example_dx'
        self.PARSE_REPO_DATA_SLIDE_DIR = self.DATA_DIR + '/' + self.SLIDETYPE_DIR + '/slides'
        self.REPO_DATA_TUMOR_MASK_DIR = os.path.join(self.DATA_DIR, self.SLIDETYPE_DIR + '/tumor_mask')
        self.TASK_REPO_DATASET_DIR = os.path.join(self.DATA_DIR, self.SLIDETYPE_DIR + '/{}'.format(self.LABEL_TYPE))
        self.FOLD_SUFFIX = fold_suffix
        self.TILESIZE_DIR = str(self.TILE_H_SIZE) + self.FOLD_SUFFIX
        ''' ------ task name based file system ------ '''
        self.TASK_BASE_REPO_TRAIN_DIR = os.path.join(self.DATA_DIR, '{}/{}/train_tile'.format(self.SLIDETYPE_DIR, self.LABEL_TYPE) if self.TILESIZE_DIR == None else '{}/{}/{}/train_tile'.format(self.SLIDETYPE_DIR, self.LABEL_TYPE, self.TILESIZE_DIR))
        self.TASK_PROCESS_REPO_SLIDE_TILE_PKL_TRAIN_DIR = self.TASK_BASE_REPO_TRAIN_DIR.replace('train_tile', 'train_pkl')
        self.TASK_BASE_REPO_TEST_DIR = os.path.join(self.DATA_DIR, '{}/{}/test_tile'.format(self.SLIDETYPE_DIR, self.LABEL_TYPE) if self.TILESIZE_DIR == None else '{}/{}/{}/test_tile'.format(self.SLIDETYPE_DIR, self.LABEL_TYPE, self.TILESIZE_DIR))
        self.TASK_PROCESS_REPO_SLIDE_TILE_PKL_TEST_DIR =  self.TASK_BASE_REPO_TEST_DIR.replace('test_tile', 'test_pkl')
        ''' ------ file system with tumor ROI mask ------ '''
        self.TASK_PROCESS_REPO_SLIDE_TUMOR_TILE_PKL_TRAIN_DIR = self.TASK_BASE_REPO_TRAIN_DIR.replace('train_tile', 'train_tumor_pkl')
        self.TASK_PROCESS_REPO_SLIDE_TUMOR_TILE_PKL_TEST_DIR = self.TASK_BASE_REPO_TEST_DIR.replace('test_tile', 'test_tumor_pkl')
        ''' ------ file system with non-tumor (background) ROI mask, BACK means background ------ '''
        self.TASK_PROCESS_REPO_SLIDE_BACK_TILE_PKL_TRAIN_DIR = self.TASK_BASE_REPO_TRAIN_DIR.replace('train_tile', 'train_back_pkl')
        self.TASK_PROCESS_REPO_SLIDE_BACK_TILE_PKL_TEST_DIR = self.TASK_BASE_REPO_TEST_DIR.replace('test_tile', 'test_back_pkl')
        
        """
        ---------------- experimental file system ---------------
        """
        ''' meta data file system '''
        # METADATA_REPO_DIR = os.path.join(PROJECT_DIR, 'data/meta_test') if OS_NAME == 'Windows' else os.path.join(PROJECT_DIR, 'data/meta')
        self.METADATA_REPO_DIR = os.path.join(self.PROJECT_DIR, 'data/{}/meta'.format(self.TASK_NAME))
        
        ''' torch network store file system '''
        self.MODEL_STORE_DIR = os.path.join(self.DATA_DIR, 'models')
        # LOG_STORE_DIR = os.path.join(DATA_DIR, 'log')
        
        ''' exp results records file system '''
        self.LOG_REPO_DIR = os.path.join(self.PROJECT_DIR, 'data/{}/logs'.format(self.TASK_NAME))
        self.RECORDS_REPO_DIR = os.path.join(self.PROJECT_DIR, 'data/{}/records'.format(self.TASK_NAME))
        self.HEATMAP_STORE_DIR = os.path.join(self.DATA_DIR, 'visualization/heatmap')
        self.PLOT_STORE_DIR = os.path.join(self.DATA_DIR, 'visualization/plot')
        self.STATISTIC_STORE_DIR = os.path.join(self.DATA_DIR, 'visualization/statistic')
        
        """
        --------------- PyTorch hyper-parameters ---------------
        """
        self.NUM_CNN_EPOCH = num_cnn_epoch
        self.NUM_ATT_EPOCH = num_att_epoch
        self.MINI_BATCH_TILE = int(mini_batch_tile / 8) if self.OS_NAME == 'Windows' else mini_batch_tile
#         self.WEIGHT_CROSSENTROPY_LOSS = weight_crossentropy_loss
        self.TILE_DATALOADER_WORKER = int(tile_dataloader_worker / 3) if self.OS_NAME == 'Windows' else tile_dataloader_worker
        self.EPOCH_START_VAL_TEST = int(self.NUM_ATT_EPOCH * 0.1)
        
        # setup package of loss functions, (name, hyper-parameter list)
        self.LOSS_PACKAGE = loss_package
        # LOSS_PACKAGE = ('focal', [0.75, 2])
        
        ''' parameter in basic MIL algorithm in Thomas J. Fuchs's paper '''
        self.MIL_TRY_K = mil_try_k
        if self.SLIDE_TYPE == 'dx':
            self.MIL_TRY_K = int(self.MIL_TRY_K * 10)
            
        ''' parameter in attention MIL in paper: Attention-based MIL, Maximilian Ilse et al. '''
        self.TILE_ENCODE_DIM = tile_encode_dim # this is depend on the backbone
        self.SLIDE_ENCODES_DIR = 'encodes' if self.SLIDETYPE_DIR == 'example' else 'encodes_dx'
        self.TASK_BASE_PROJECT_TRAIN_DIR = os.path.join(self.PROJECT_DIR, 'data/{}/{}/{}/train_tile'.format(self.TASK_NAME, 
                                                                                                            self.SLIDE_ENCODES_DIR, 
                                                                                                            self.TILESIZE_DIR))
        self.TASK_PRETRAIN_REPO_SLIDE_MATRIX_TRAIN_DIR = self.TASK_BASE_PROJECT_TRAIN_DIR.replace('train_tile', 'train_encode')
        self.TASK_BASE_PROJECT_TEST_DIR = os.path.join(self.PROJECT_DIR, 'data/{}/{}/{}/test_tile'.format(self.TASK_NAME, 
                                                                                                          self.SLIDE_ENCODES_DIR, 
                                                                                                          self.TILESIZE_DIR))
        self.TASK_PRETRAIN_REPO_SLIDE_MATRIX_TEST_DIR = self.TASK_BASE_PROJECT_TEST_DIR.replace('test_tile', 'test_encode')
        self.TASK_PROJECT_TILESIZE_DIR = os.path.join(self.PROJECT_DIR, 'data/{}/{}/{}'.format(self.TASK_NAME, 
                                                                                               self.SLIDE_ENCODES_DIR, 
                                                                                               self.TILESIZE_DIR))
        
        self.MINI_BATCH_SLIDEMAT = int(mini_batch_slidemat / 2) if self.OS_NAME == 'Windows' else mini_batch_slidemat
        if self.SLIDE_TYPE == 'dx':
            self.MINI_BATCH_SLIDEMAT = int(self.MINI_BATCH_SLIDEMAT / 2)
        self.SLIDEMAT_DATALOADER_WORKER = int(slidemat_dataloader_worker / 2) if self.OS_NAME == 'Windows' else slidemat_dataloader_worker
        self.ATTPOOL_RECORD_EPOCHS = [self.NUM_ATT_EPOCH - 1]
        self.NUM_LAST_EVAL_EPOCHS = num_last_eval_epochs
        
        ''' parameter in LCSB MIL (ours) '''
        self.RESET_OPTIMIZER = reset_optim
        self.NUM_ROUND = num_round
        # NUM_ROUND = 10
        self.NUM_INIT_S_EPOCH = num_init_s_epoch
        self.NUM_INROUND_S_EPOCH = num_inround_s_epoch if self.SLIDE_TYPE == 'dx' else int(num_inround_s_epoch * 2)
        self.NUM_INROUND_REV_T_EPOCH = num_inround_rev_t_epoch if self.SLIDE_TYPE == 'dx' else int(num_inround_rev_t_epoch * 4)
        self.NUM_INROUND_T_EPOCH = num_inround_t_epoch if self.SLIDE_TYPE == 'dx' else int(num_inround_t_epoch * 2)
        self.ATTPOOL_STOP_LOSS = attpool_stop_loss
        self.ATTPOOL_STOP_MAINTAINS = attpool_stop_maintains
        self.OVERALL_STOP_LOSS = overall_stop_loss
        # the following 2 parames refer to how often to refresh the tile-level dataset
        self.POS_REFRESH_PULSE = pos_refersh_pluse
        self.NEG_REFRESH_PULSE = neg_refersh_pluse
        
        self.TOP_RANGE_RATE = top_range_rate # the ratio of all tiles in one slides, for [:TOP_RANGE_RATE]
        self.SUP_RANGE_RATE = sup_range_rate # the ratio of all tiles in one slides, for [SUP_RANGE_RATE[0]: SUP_RANGE_RATE[1]]
        self.NEG_RANGE_RATE = neg_range_rate # the ratio of all tiles in one slides, for [NEG_RANGE_RATE:]
        
        self.ATT_K = att_k
        self.SUP_K = sup_k
        self.REVERSE_N = reverse_n
        self.REVERSE_GRADIENT_ALPHA = reverse_gradient_alpha
        
        # if we dont wanna record the milestone, just keep HISTORY_RECORD_ROUNDS as a empty list: []
        # HISTORY_RECORD_ROUNDS = [k - 1 if k > 0 else k for k in range(0, NUM_ROUND + 1, 5)]
        self.HIS_RECORD_ROUNDS = his_record_rounds
        
        ''' parameter in MIL aggregator pre-training alg (ours) '''
        # SSPT: (can used for) self-supervised pre-training
        self.TASK_PRETRAIN_REPO_RAND_MATRIX_SSPT_DIR = os.path.join(self.TASK_PROJECT_TILESIZE_DIR, 'pt_encode')
        self.NUM_PRETRAIN_EPOCH = num_pretrain_epoch
        self.RAND_MAT_RELOAD_PLUSE = rand_mat_reload_pluse # for several epochs, reload the rand_mats once
        self.NUM_PRELOAD_RAND_MAT = num_preload_rand_mat # need to preload (reload regularly) some rand_mats on disk
        self.RAND_MAT_BAGDIM = mat_bagdim
        self.NUM_EPOCH_RAND_PAIR = num_epoch_rand_pair
        
    def refresh_fold_suffix(self, new_fold_suffix):
        '''
        refresh the fold_suffix for validation or training on batch
        and change some necessary parameters
        '''
        self.FOLD_SUFFIX = new_fold_suffix
        self.TILESIZE_DIR = str(self.TILE_H_SIZE) + self.FOLD_SUFFIX
        
        ''' ------ task name based file system ------ '''
        self.TASK_BASE_REPO_TRAIN_DIR = os.path.join(self.DATA_DIR, '{}/{}/train_tile'.format(self.SLIDETYPE_DIR, self.LABEL_TYPE) if self.TILESIZE_DIR == None else '{}/{}/{}/train_tile'.format(self.SLIDETYPE_DIR, self.LABEL_TYPE, self.TILESIZE_DIR))
        self.TASK_PROCESS_REPO_SLIDE_TILE_PKL_TRAIN_DIR = self.TASK_BASE_REPO_TRAIN_DIR.replace('train_tile', 'train_pkl')
        self.TASK_BASE_REPO_TEST_DIR = os.path.join(self.DATA_DIR, '{}/{}/test_tile'.format(self.SLIDETYPE_DIR, self.LABEL_TYPE) if self.TILESIZE_DIR == None else '{}/{}/{}/test_tile'.format(self.SLIDETYPE_DIR, self.LABEL_TYPE, self.TILESIZE_DIR))
        self.TASK_PROCESS_REPO_SLIDE_TILE_PKL_TEST_DIR =  self.TASK_BASE_REPO_TEST_DIR.replace('test_tile', 'test_pkl')
        ''' ------ file system with tumor ROI mask ------ '''
        self.TASK_PROCESS_REPO_SLIDE_TUMOR_TILE_PKL_TRAIN_DIR = self.TASK_BASE_REPO_TRAIN_DIR.replace('train_tile', 'train_tumor_pkl')
        self.TASK_PROCESS_REPO_SLIDE_TUMOR_TILE_PKL_TEST_DIR = self.TASK_BASE_REPO_TEST_DIR.replace('test_tile', 'test_tumor_pkl')
        ''' ------ file system with non-tumor (background) ROI mask, BACK means background ------ '''
        self.TASK_PROCESS_REPO_SLIDE_BACK_TILE_PKL_TRAIN_DIR = self.TASK_BASE_REPO_TRAIN_DIR.replace('train_tile', 'train_back_pkl')
        self.TASK_PROCESS_REPO_SLIDE_BACK_TILE_PKL_TEST_DIR = self.TASK_BASE_REPO_TEST_DIR.replace('test_tile', 'test_back_pkl')
        
        self.TASK_BASE_PROJECT_TRAIN_DIR = os.path.join(self.PROJECT_DIR, 'data/{}/{}/{}/train_tile'.format(self.TASK_NAME, 
                                                                                                            self.SLIDE_ENCODES_DIR, 
                                                                                                            self.TILESIZE_DIR))
        self.TASK_PRETRAIN_REPO_SLIDE_MATRIX_TRAIN_DIR = self.TASK_BASE_PROJECT_TRAIN_DIR.replace('train_tile', 'train_encode')
        self.TASK_BASE_PROJECT_TEST_DIR = os.path.join(self.PROJECT_DIR, 'data/{}/{}/{}/test_tile'.format(self.TASK_NAME, 
                                                                                                          self.SLIDE_ENCODES_DIR, 
                                                                                                          self.TILESIZE_DIR))
        self.TASK_PRETRAIN_REPO_SLIDE_MATRIX_TEST_DIR = self.TASK_BASE_PROJECT_TEST_DIR.replace('test_tile', 'test_encode')
        self.TASK_PROJECT_TILESIZE_DIR = os.path.join(self.PROJECT_DIR, 'data/{}/{}/{}'.format(self.TASK_NAME, 
                                                                                               self.SLIDE_ENCODES_DIR, 
                                                                                               self.TILESIZE_DIR))
        
        self.TASK_PRETRAIN_REPO_RAND_MATRIX_SSPT_DIR = os.path.join(self.TASK_PROJECT_TILESIZE_DIR, 'pt_encode')
        

if __name__ == '__main__':
    pass











