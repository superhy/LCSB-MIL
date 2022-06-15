'''
@author: Yang Hu
'''
import os

from torch.utils.data.dataset import Dataset
from torchvision import transforms

import numpy as np
from support.env import ENV
from support.files import parse_slide_caseid_from_filepath, \
    parse_slideid_from_filepath
from wsi.process import recovery_tiles_list_from_pkl


def load_richtileslist_fromfile(ENV_task, for_train=True):
    """
    laod the full dictionary information of all tiles list for training/test set
    
    need to prepare parmes:
        _env_process_slide_tile_pkl_train_dir,
        _env_process_slide_tile_pkl_test_dir,
        _env_process_slide_tumor_tile_pkl_train_dir,
        _env_process_slide_tumor_tile_pkl_test_dir,
        (label_type)
        
    Args:
        ENV_task: the task environment object
        for_train:
        
    Return:
        tiles_all_list: all tile list with tile-object
        tileidx_slideid_dict: mapping dictionary from tile_idx in tiles_all_list to slide_id
        slide_tileidxs_dict: mapping dictionary from slide_id to tile_idxs (as a list)
    
    """
    
    ''' prepare the parames '''
    _env_process_slide_tile_pkl_train_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TILE_PKL_TRAIN_DIR
    _env_process_slide_tile_pkl_test_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TILE_PKL_TEST_DIR
    _env_process_slide_tumor_tile_pkl_train_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TUMOR_TILE_PKL_TRAIN_DIR
    _env_process_slide_tumor_tile_pkl_test_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TUMOR_TILE_PKL_TEST_DIR
#     label_type = ENV_task.LABEL_TYPE
    
    if ENV.APPLY_TUMOR_ROI == True:
        pkl_dir = _env_process_slide_tumor_tile_pkl_train_dir if for_train == True else _env_process_slide_tumor_tile_pkl_test_dir
    else:
        pkl_dir = _env_process_slide_tile_pkl_train_dir if for_train == True else _env_process_slide_tile_pkl_test_dir
    pkl_files = os.listdir(pkl_dir)
    
    tileidx_slideid_dict = {}
    slide_tileidxs_dict = {}
    tiles_all_list = []
    tileidx = 0
    for pkl_f in pkl_files:
        # each slide each pkl
        tiles_slide_list = recovery_tiles_list_from_pkl(os.path.join(pkl_dir, pkl_f))
        slide_id = tiles_slide_list[0].query_slideid()
        slide_tileidxs_dict[slide_id] = []
        for i in range(len(tiles_slide_list)): 
            tileidx_slideid_dict[tileidx] = slide_id
            slide_tileidxs_dict[slide_id].append(tileidx)
            tileidx += 1
        tiles_all_list.extend(tiles_slide_list)
        
    return tiles_all_list, tileidx_slideid_dict, slide_tileidxs_dict

def load_slides_tileslist(ENV_task, for_train=True):
    """
    a simplify version for above function, only return the dict of slide -> tiles' objects
    
    need to prepare parmes:
        _env_process_slide_tile_pkl_train_dir,
        _env_process_slide_tile_pkl_test_dir,
        _env_process_slide_tumor_tile_pkl_train_dir,
        _env_process_slide_tumor_tile_pkl_test_dir,
        (label_type)
        
    Args:
        ENV_task: the task environment object
        for_train:
        
    Return:
        slide_tiles_dict: dict of slide -> tiles' objects
    
    """
    
    ''' prepare the parames '''
    _env_process_slide_tile_pkl_train_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TILE_PKL_TRAIN_DIR
    _env_process_slide_tile_pkl_test_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TILE_PKL_TEST_DIR
    _env_process_slide_tumor_tile_pkl_train_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TUMOR_TILE_PKL_TRAIN_DIR
    _env_process_slide_tumor_tile_pkl_test_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TUMOR_TILE_PKL_TEST_DIR
#     label_type = ENV_task.LABEL_TYPE
    
    if ENV.APPLY_TUMOR_ROI == True:
        pkl_dir = _env_process_slide_tumor_tile_pkl_train_dir if for_train == True else _env_process_slide_tumor_tile_pkl_test_dir
    else:
        pkl_dir = _env_process_slide_tile_pkl_train_dir if for_train == True else _env_process_slide_tile_pkl_test_dir
    
    pkl_files = os.listdir(pkl_dir)
    
    slide_tiles_dict = {}
    for pkl_f in pkl_files:
        # each slide each pkl
        tiles_list = recovery_tiles_list_from_pkl(os.path.join(pkl_dir, pkl_f))
        slide_id = tiles_list[0].query_slideid()
        slide_tiles_dict[slide_id] = tiles_list
        
    return slide_tiles_dict

    
class Simple_Tile_Dataset(Dataset):
    
    '''
    dataset only provides tiles from a tile list
    
    Args:
        tiles_list: a tile list, can be a slide tile list or a combination tile list
    '''
    
    def __init__(self, tiles_list, transform: transforms):
        self.tiles_list = tiles_list
        
        self.transform = transform
        ''' make slide cache in memory '''
        self.cache_slide = ('none', None)
        
    def __getitem__(self, index):
        tile = self.tiles_list[index]
        
        ''' using slide cache '''
        loading_slide_id = parse_slideid_from_filepath(tile.original_slide_filepath)
        if loading_slide_id == self.cache_slide[0]:
            preload_slide = self.cache_slide[1]
        else:
            _, preload_slide = tile.get_pil_scaled_slide()
            self.cache_slide = (loading_slide_id, preload_slide)
            
        image = tile.get_pil_tile(preload_slide)
        image = self.transform(image)
        return image
        
    def __len__(self):
        return len(self.tiles_list)


'''
------------ datasets for aggregator training -------------
'''

class SlideMatrix_Dataset(Dataset):
    
    def __init__(self, slide_matrix_file_sets, label_dict, preload_slide_matrix_sets=None, batch_loader=False):
        '''
        Args:
            slide_matrix_file_sets: list: [(slide_id, len of tiles in this slide, slide matrix numpy filepath),
                                           (...), ...]
            label_dict: {case_id (can be parse from slide_id): label (0, 1)}
            preload_slide_matrix_sets: list: [(slide_id, len of tiles in this slide, slide matrix numpy ndarray),
                                              (...), ...]
                                       default: None
            batch_loader: if 'True': load the slide matrix in each batch by __getitem__();
                          if 'False': load all slides' matrices once in __init__()
        '''
        self.slide_matrix_file_sets = slide_matrix_file_sets
        self.label_dict = label_dict
        self.slide_matrix_sets = preload_slide_matrix_sets
        self.batch_loader = True if ENV.SLIDE_TYPE == 'dx' else batch_loader
        
        if self.slide_matrix_sets != None:
            self.batch_loader = False
        
        if self.batch_loader == False:
            if self.slide_matrix_sets == None or len(self.slide_matrix_file_sets) != len(self.slide_matrix_sets):
                self.slide_matrix_sets = []
                for file_set in self.slide_matrix_file_sets:
#                     slide_id = file_set[0]
#                     slide_matrix = np.load(file_set[1])
                    self.slide_matrix_sets.append((file_set[0], file_set[1], np.load(file_set[2])))
        else:
            pass
        
    def refresh_data(self, slide_matrix_file_sets):
        '''
        refresh the matrix sets after get new slide matrices
        '''
        self.slide_matrix_file_sets = slide_matrix_file_sets
        if self.slide_matrix_sets == None or len(self.slide_matrix_file_sets) != len(self.slide_matrix_sets):
            self.slide_matrix_sets = []
            for file_set in self.slide_matrix_file_sets:
#                     slide_id = file_set[0]
#                     slide_matrix = np.load(file_set[1])
                self.slide_matrix_sets.append((file_set[0], file_set[1], np.load(file_set[2])))
            
    def __getitem__(self, index):
        if self.batch_loader == False:
            slide_id = self.slide_matrix_sets[index][0]
            case_id = slide_id[:slide_id.find('_') ] if slide_id.find('_') != -1 else slide_id
            matrix = self.slide_matrix_sets[index][2]
            bag_dim = self.slide_matrix_sets[index][1]
            label = self.label_dict[case_id]
            return matrix, bag_dim, label
        else:
            slide_id = self.slide_matrix_file_sets[index][0]
            case_id = slide_id[:slide_id.find('_')] if slide_id.find('_') != -1 else slide_id
            matrix = np.load(self.slide_matrix_file_sets[index][2])
            bag_dim = self.slide_matrix_file_sets[index][1]
            label = self.label_dict[case_id]
            return matrix, bag_dim, label
            
    def __len__(self):
        return len(self.slide_matrix_file_sets)
    
    
class SlideMatrix_Pairs_Dataset(Dataset):
    
    def __init__(self, slide_matrix_pair_file_sets, label_dict):
        '''
        Only using batch_loader, because the pre-training dataset is much bigger
        Args:
            slide_matrix_pair_file_sets: 
                list: [(slide_id_1, bag_dim_1, slide_mat_nd_file_1, slide_id_2, bag_dim_2, slide_mat_nd_file_2),
                        ...]
            label_dict: {case_id (can be parse from slide_id): label (0, 1)}
        '''
        self.slide_matrix_pair_file_sets = slide_matrix_pair_file_sets
        self.label_dict = label_dict
        
    def refresh_data(self, slide_matrix_pair_file_sets):
        '''
        refresh another random cohort of slide_matrix_pair_file_sets
        '''
        self.slide_matrix_pair_file_sets = slide_matrix_pair_file_sets
        
    def __getitem__(self, index):
        slide_id_1 = self.slide_matrix_pair_file_sets[index][0]
        slide_id_2 = self.slide_matrix_pair_file_sets[index][3]
        case_id_1 = slide_id_1[:slide_id_1.find('_') ] if slide_id_1.find('_') != -1 else slide_id_1
        case_id_2 = slide_id_2[:slide_id_2.find('_') ] if slide_id_2.find('_') != -1 else slide_id_2
        mat_1 = np.load(self.slide_matrix_pair_file_sets[index][2])
        mat_2 = np.load(self.slide_matrix_pair_file_sets[index][5])
        bag_dim_1, bag_dim_2 = self.slide_matrix_pair_file_sets[index][1], self.slide_matrix_pair_file_sets[index][4]
        sim_label = 0 if self.label_dict[case_id_1] == self.label_dict[case_id_2] else 1
        
        return mat_1, bag_dim_1, mat_2, bag_dim_2, sim_label
    
    def __len__(self):
        return len(self.slide_matrix_pair_file_sets)
    
'''
    -------------------------------------------------------------------------------
    functions and Dataset mainly for Look Closer to See Better (LCSB) MIL algorithm
        this approach will partly use the same logic previously
    -------------------------------------------------------------------------------
'''
   
class AttK_MIL_Dataset(Dataset):
    
    def __init__(self, tiles_list, label_dict, transform: transforms):
        self.tiles_list = tiles_list
        self.tiles_idxs_train_pool = []
        self.tiles_list_train_pool = tiles_list # will be refresh when training
        self.label_dict = label_dict
        
        self.transform = transform
        ''' make slide cache in memory '''
        self.cache_slide = ('none', None)
        
    def refresh_data(self, filter_attK_slide_tileidx_dict):
        '''
        must be called before training
        '''
        
        self.tiles_idxs_train_pool = []
        for _, tile_idx_list in filter_attK_slide_tileidx_dict.items():
            for idx in tile_idx_list:
                self.tiles_idxs_train_pool.append(idx)
                    
        self.tiles_list_train_pool = []
        for idx in self.tiles_idxs_train_pool:
            self.tiles_list_train_pool.append(self.tiles_list[idx])
        print('Dataset Info: [refresh training tile list, now with: %d tiles...]' % len(self.tiles_list_train_pool))
    
    def __getitem__(self, index):
        tile = self.tiles_list_train_pool[index]
        case_id = parse_slide_caseid_from_filepath(tile.original_slide_filepath)
        
        ''' using slide cache '''
        loading_slide_id = parse_slideid_from_filepath(tile.original_slide_filepath)
        if loading_slide_id == self.cache_slide[0]:
            preload_slide = self.cache_slide[1]
        else:
            _, preload_slide = tile.get_pil_scaled_slide()
            self.cache_slide = (loading_slide_id, preload_slide)
            
        image = tile.get_pil_tile(preload_slide)
        image = self.transform(image)
        label = self.label_dict[case_id]
        
        return image, label
        
    def __len__(self):
        return len(self.tiles_list_train_pool)
    
'''
    -------------------------------------------------------------------------------
    functions and Dataset mainly for dual (reversed) Look Closer to See Better (LCSB) MIL algorithm
        this approach will partly use the same logic previously
    -------------------------------------------------------------------------------
'''


class Rev_AttK_MIL_Dataset(Dataset):
     
    def __init__(self, tiles_list, label_dict, transform: transforms):
        self.tiles_list = tiles_list
        self.tiles_idxs_train_pool_neg = []
        self.tiles_list_train_pool_neg = tiles_list
         
        self.label_dict = label_dict
         
        self.transform = transform
        ''' make slide cache in memory '''
        self.cache_slide_neg = ('none', None)
         
    def refresh_data(self, filter_revgN_slide_tileidx_dict):
        '''
        must be called before training
        '''
 
        self.tiles_idxs_train_pool_neg = []
        for _, tile_idx_list in filter_revgN_slide_tileidx_dict.items():
            for idx in tile_idx_list:
                self.tiles_idxs_train_pool_neg.append(idx) 
                     
        self.tiles_list_train_pool_neg = []
        for idx in self.tiles_idxs_train_pool_neg:
            self.tiles_list_train_pool_neg.append(self.tiles_list[idx])
        print('Dataset Info: [refresh reversed gradient training tile list, now with: %d negative tiles...]' % (len(self.tiles_idxs_train_pool_neg)))
     
    def __getitem__(self, index):
        tile_neg = self.tiles_list_train_pool_neg[index]
         
        case_id_neg = parse_slide_caseid_from_filepath(tile_neg.original_slide_filepath)
         
        ''' using slide cache '''
        loading_slide_neg_id = parse_slideid_from_filepath(tile_neg.original_slide_filepath)
         
        if loading_slide_neg_id == self.cache_slide_neg[0]:
            preload_slide_neg = self.cache_slide_neg[1]
        else:
            _, preload_slide_neg = tile_neg.get_pil_scaled_slide()
            self.cache_slide_neg = (loading_slide_neg_id, preload_slide_neg)
             
        image_neg = tile_neg.get_pil_tile(preload_slide_neg)
        image_neg = self.transform(image_neg)
        label_neg = self.label_dict[case_id_neg]
         
        return image_neg, label_neg
         
    def __len__(self):
        return len(self.tiles_list_train_pool_neg)



if __name__ == '__main__':
    pass




