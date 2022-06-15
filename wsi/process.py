'''
@author: Yang Hu
'''

import csv
import os
import pickle
import random
import sys

import numpy as np
from support.files import parse_slideid_from_filepath, \
    parse_filesystem_slide, clear_dir, parse_slide_caseid_from_filepath, \
    parse_filenames_tilespkl, parse_slideid_from_tilepklname
from wsi import slide_tools, filter_tools, tiles_tools
from support.metadata import query_task_label_dict_fromcsv


sys.path.append("..")       
        
def generate_tiles_list_pkl_filepath(slide_filepath, label_type,
                                     _env_process_slide_tile_pkl_train_dir,
                                     _env_process_slide_tile_pkl_test_dir,
                                     for_train=True):
    """
    generate the filepath of pickle 
    """
    
    tiles_list_pkl_dir = _env_process_slide_tile_pkl_train_dir if for_train == True else _env_process_slide_tile_pkl_test_dir
    
    slide_id = parse_slideid_from_filepath(slide_filepath)
    tiles_list_pkl_filename = slide_id + '-(tiles_list)' + '.pkl'
    if not os.path.exists(tiles_list_pkl_dir):
        os.makedirs(tiles_list_pkl_dir)
    
    pkl_filepath = os.path.join(tiles_list_pkl_dir, tiles_list_pkl_filename)
    
    return pkl_filepath

def generate_t_b_roi_tiles_list_pkl_filepath(slide_filepath, label_type, 
                                             _env_process_slide_t_b_tile_pkl_train_dir, 
                                             _env_process_slide_t_b_tile_pkl_test_dir,
                                             for_train=True):
    """
    generate the filepath of pickle 
    """
    
    if label_type == 'emt':
        tiles_list_pkl_dir = _env_process_slide_t_b_tile_pkl_train_dir if for_train == True else _env_process_slide_t_b_tile_pkl_test_dir    
    
    slide_id = parse_slideid_from_filepath(slide_filepath)
    tiles_list_pkl_filename = slide_id + '-(tiles_tumor_list)' + '.pkl'
    if not os.path.exists(tiles_list_pkl_dir):
        os.makedirs(tiles_list_pkl_dir)
    
    pkl_filepath = os.path.join(tiles_list_pkl_dir, tiles_list_pkl_filename)
    
    return pkl_filepath

 
def recovery_tiles_list_from_pkl(pkl_filepath):
    """
    load tiles list from [.pkl] file on disk
    (this function is for some other module)
    """
    with open(pkl_filepath, 'rb') as f_pkl:
        tiles_list = pickle.load(f_pkl)
    return tiles_list

        
def _run_singleprocess_slide_tiles_split_keep_object(ENV_task, test_num_set: tuple=None,
                                                     delete_old_files=True):
    """
    conduct the whole pipeline of slide's tiles split, by Sequential process
    store the tiles Object [.pkl] on disk
    
    need to prepare some parames:
        _env_parse_data_slide_dir, _env_process_slide_tile_pkl_train_dir,
        _env_process_slide_tile_pkl_test_dir, _env_tp_tiles_threshold,
        _env_tile_w_size, _env_tile_h_size,
    
    Args:
        ENV_task: the task environment object
    """
    
    ''' preparing some file parames '''
    test_prop = ENV_task.TEST_PART_PROP
    label_type = ENV_task.LABEL_TYPE
    _env_parse_data_slide_dir = ENV_task.PARSE_REPO_DATA_SLIDE_DIR
    _env_process_slide_tile_pkl_train_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TILE_PKL_TRAIN_DIR
    _env_process_slide_tile_pkl_test_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TILE_PKL_TEST_DIR
    _env_tp_tiles_threshold = ENV_task.TP_TILES_THRESHOLD
    _env_tile_w_size = ENV_task.TILE_W_SIZE
    _env_tile_h_size = ENV_task.TILE_H_SIZE
    
    slide_dir = _env_parse_data_slide_dir
    slide_path_list = parse_filesystem_slide(slide_dir, original_download=False)
    
    # remove the old train and test pkl dir
    if delete_old_files == True:
        clear_dir([_env_process_slide_tile_pkl_train_dir, _env_process_slide_tile_pkl_test_dir])
    
    label_dict = query_task_label_dict_fromcsv(ENV_task)
        
    pos_neg_path_dict = {'POS': [], 'NEG': []}
    for slide_path in slide_path_list:
        case_id = parse_slide_caseid_from_filepath(slide_path)
        if case_id not in label_dict.keys():
            continue
        if label_dict[case_id] == 1:
            pos_neg_path_dict['POS'].append(slide_path)
        elif label_dict[case_id] == 0:
            pos_neg_path_dict['NEG'].append(slide_path)
            
    # shuffle the POS and NEG slide_tiles_list
    pos_neg_path_dict['POS'] = random.sample(pos_neg_path_dict['POS'], len(pos_neg_path_dict['POS']))
    pos_neg_path_dict['NEG'] = random.sample(pos_neg_path_dict['NEG'], len(pos_neg_path_dict['NEG']))
    
    # count the test(train) number for positive and negative samples
    if test_num_set == None:
        test_num_set = (round(len(pos_neg_path_dict['POS']) * test_prop), round(len(pos_neg_path_dict['NEG']) * test_prop))
    
    # write train set
    print('<---------- store the train tiles list ---------->')
    for train_slide_path in pos_neg_path_dict['POS'][:-test_num_set[0]] + pos_neg_path_dict['NEG'][:-test_num_set[1]]:
        np_small_img, large_w, large_h, small_w, small_h = slide_tools.slide_to_scaled_np_image(train_slide_path)
        np_small_filtered_img = filter_tools.apply_image_filters(np_small_img)
        
        shape_set_img = (large_w, large_h, small_w, small_h)
        tiles_list = tiles_tools.get_slide_tiles(np_small_filtered_img, shape_set_img, train_slide_path,
                                                   _env_tile_w_size, _env_tile_h_size,
                                                   t_p_threshold=_env_tp_tiles_threshold, load_small_tile=False)
        print('generate tiles for slide: %s, keep [%d] tile objects in (.pkl) list.' % (train_slide_path, len(tiles_list)))
        if len(tiles_list) == 0:
            continue
        
        pkl_path = generate_tiles_list_pkl_filepath(train_slide_path, label_type,
                                                    _env_process_slide_tile_pkl_train_dir,
                                                    _env_process_slide_tile_pkl_test_dir)
        with open(pkl_path, 'wb') as f_pkl:
            pickle.dump(tiles_list, f_pkl)
            
    # write test set
    print('<---------- store the test tiles list ---------->')
    for test_slide_path in pos_neg_path_dict['POS'][-test_num_set[0]:] + pos_neg_path_dict['NEG'][-test_num_set[1]: ]:
        np_small_img, large_w, large_h, small_w, small_h = slide_tools.slide_to_scaled_np_image(test_slide_path)
        np_small_filtered_img = filter_tools.apply_image_filters(np_small_img)
        
        shape_set_img = (large_w, large_h, small_w, small_h)
        tiles_list = tiles_tools.get_slide_tiles(np_small_filtered_img, shape_set_img, test_slide_path,
                                                   _env_tile_w_size, _env_tile_h_size,
                                                   t_p_threshold=_env_tp_tiles_threshold, load_small_tile=False)
        print('generate tiles for slide: %s, keep [%d] tile objects in (.pkl) list.' % (test_slide_path, len(tiles_list)))
        if len(tiles_list) == 0:
            continue
        
        pkl_path = generate_tiles_list_pkl_filepath(test_slide_path, label_type,
                                                    _env_process_slide_tile_pkl_train_dir,
                                                    _env_process_slide_tile_pkl_test_dir, 
                                                    for_train=False)
        with open(pkl_path, 'wb') as f_pkl:
            pickle.dump(tiles_list, f_pkl)
            
            
def _run_singleprocess_slide_t_b_roi_tiles_split_keep_object(ENV_task, 
                                                             clone_org_train_test=True, tumor_or_backgorund=True,
                                                             test_num_set: tuple=None, delete_old_files=True):
    """
    (this function is only available to OV_EMT task for the time being)
    conduct the whole pipeline of slide's tiles split, by Sequential process
    with the filter for tumor-area mask/background-tissue mask, 
    remove the non-tumor/tumor regions on slides
    
    store the tiles Object [.pkl] on disk
    
    need to prepare some parames:
        _env_process_slide_tumor_tile_pkl_train_dir, _env_process_slide_tumor_tile_pkl_test_dir,
        _env_process_slide_tile_pkl_train_dir, _env_process_slide_tile_pkl_test_dir,
        _env_parse_data_slide_dir, _env_data_tumor_mask_dir,
        _env_tile_w_size, _env_tile_h_size, _env_tp_tiles_threshold,
    
    Args:
        ENV_task: the task environment object
        clone_org_train_test: 
            if True: keep the separation of train/test set on regular tiles (without tumor ROI) split, only add tumor mask for them
            if False: re-conduct the process of tiles split and train/test set random selection, with tumor ROI filter
        tumor_or_backgorund: 
            True: only left tumor area; False: only left non-tumor (background) tissue area
    """
    
    ''' preparing some file parames '''
    test_prop = ENV_task.TEST_PART_PROP
    label_type = ENV_task.LABEL_TYPE
    _env_process_slide_tumor_tile_pkl_train_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TUMOR_TILE_PKL_TRAIN_DIR
    _env_process_slide_tumor_tile_pkl_test_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TUMOR_TILE_PKL_TEST_DIR
    _env_process_slide_back_tile_pkl_train_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_BACK_TILE_PKL_TRAIN_DIR
    _env_process_slide_back_tile_pkl_test_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_BACK_TILE_PKL_TEST_DIR
    _env_process_slide_tile_pkl_train_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TILE_PKL_TRAIN_DIR
    _env_process_slide_tile_pkl_test_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TILE_PKL_TEST_DIR
    _env_parse_data_slide_dir = ENV_task.PARSE_REPO_DATA_SLIDE_DIR
    _env_data_tumor_mask_dir = ENV_task.REPO_DATA_TUMOR_MASK_DIR
    _env_tile_w_size = ENV_task.TILE_W_SIZE
    _env_tile_h_size = ENV_task.TILE_H_SIZE
    _env_tp_tiles_threshold = ENV_task.TP_TILES_THRESHOLD
    
    slide_dir = _env_parse_data_slide_dir
    slide_path_list = parse_filesystem_slide(slide_dir, original_download=False)
    
    # remove the old train and test pkl dir
    if label_type == 'emt':
        if delete_old_files == True:
            clear_list = [_env_process_slide_tumor_tile_pkl_train_dir, _env_process_slide_tumor_tile_pkl_test_dir] \
                if tumor_or_backgorund == True else [_env_process_slide_back_tile_pkl_train_dir, _env_process_slide_back_tile_pkl_test_dir]
            clear_dir(clear_list)
            
    label_dict = query_task_label_dict_fromcsv(ENV_task)
    
    pos_neg_path_dict = {'POS': [], 'NEG': []}
    slide_id_path_dict = {}
    for slide_path in slide_path_list:
        case_id = parse_slide_caseid_from_filepath(slide_path)
        if case_id not in label_dict.keys(): # TODO: updata on rescomp
            continue
        if label_dict[case_id] == 1:
            pos_neg_path_dict['POS'].append(slide_path)
        elif label_dict[case_id] == 0:
            pos_neg_path_dict['NEG'].append(slide_path)
        
        # make a <slide_id: slide_path> dict
        slide_id = parse_slideid_from_filepath(slide_path)
        slide_id_path_dict[slide_id] = slide_path
        
    if clone_org_train_test == True:
        train_tilepkl_filename_list = parse_filenames_tilespkl(_env_process_slide_tile_pkl_train_dir)
        test_tilepkl_filename_list = parse_filenames_tilespkl(_env_process_slide_tile_pkl_test_dir)
        # make the tile pkl for positive and negative slides together, with the same separation with regular train/test set
        train_slide_path_list, test_slide_path_list = [], []
        for tilepkl_filename in train_tilepkl_filename_list:
            train_slide_path_list.append(slide_id_path_dict[parse_slideid_from_tilepklname(tilepkl_filename)])
        for tilepkl_filename in test_tilepkl_filename_list:
            test_slide_path_list.append(slide_id_path_dict[parse_slideid_from_tilepklname(tilepkl_filename)])
    else:
        pos_neg_path_dict['POS'] = random.sample(pos_neg_path_dict['POS'], len(pos_neg_path_dict['POS']))
        pos_neg_path_dict['NEG'] = random.sample(pos_neg_path_dict['NEG'], len(pos_neg_path_dict['NEG']))
        
        if test_num_set == None:
            test_num_set = (round(len(pos_neg_path_dict['POS']) * test_prop), round(len(pos_neg_path_dict['NEG']) * test_prop))
        # put positive and negative slides together
        train_slide_path_list = pos_neg_path_dict['POS'][:-test_num_set[0]] + pos_neg_path_dict['NEG'][:-test_num_set[1]]
        test_slide_path_list = pos_neg_path_dict['POS'][-test_num_set[0]:] + pos_neg_path_dict['NEG'][-test_num_set[1]:]
        
    # write train set with prepared train_slide_path_list
    print('<---------- store the train tiles list ---------->')
    for train_slide_path in train_slide_path_list:
        np_small_img, large_w, large_h, small_w, small_h = slide_tools.slide_to_scaled_np_image(train_slide_path)
        # get tumor-ROI annotation json filepath
        slide_id = parse_slideid_from_filepath(train_slide_path)
        tumor_roi_jsonpath = os.path.join(_env_data_tumor_mask_dir, 'AIDA_annotation_{}.json'.format(slide_id))
        
        np_small_filtered_img = filter_tools.apply_image_filters(np_small_img, 
                                                                 tumor_region_jsonpath=tumor_roi_jsonpath,
                                                                 tumor_or_background=tumor_or_backgorund)
        
        shape_set_img = (large_w, large_h, small_w, small_h)
        tiles_list = tiles_tools.get_slide_tiles(np_small_filtered_img, shape_set_img, train_slide_path,
                                                 _env_tile_w_size, _env_tile_h_size,
                                                 t_p_threshold=_env_tp_tiles_threshold, load_small_tile=False)
        print('generate tiles for slide: %s, keep [%d] tile objects in (.pkl) list.' % (train_slide_path, len(tiles_list)))
        if len(tiles_list) == 0:
            continue
        
        if tumor_or_backgorund == True:
            pkl_path = generate_t_b_roi_tiles_list_pkl_filepath(train_slide_path, label_type,
                                                                _env_process_slide_tumor_tile_pkl_train_dir, 
                                                                _env_process_slide_tumor_tile_pkl_test_dir)
        else:
            pkl_path = generate_t_b_roi_tiles_list_pkl_filepath(train_slide_path, label_type,
                                                                _env_process_slide_back_tile_pkl_train_dir, 
                                                                _env_process_slide_back_tile_pkl_test_dir)
            
        with open(pkl_path, 'wb') as f_pkl:
            pickle.dump(tiles_list, f_pkl)
            
    # write test set with prepared test_slide_path_list
    print('<---------- store the test tiles list ---------->')
    for test_slide_path in test_slide_path_list:
        np_small_img, large_w, large_h, small_w, small_h = slide_tools.slide_to_scaled_np_image(test_slide_path)
        # get tumor-ROI annotation json filepath
        slide_id = parse_slideid_from_filepath(test_slide_path)
        tumor_roi_jsonpath = os.path.join(_env_data_tumor_mask_dir, 'AIDA_annotation_{}.json'.format(slide_id))
        
        np_small_filtered_img = filter_tools.apply_image_filters(np_small_img,
                                                                 tumor_region_jsonpath=tumor_roi_jsonpath,
                                                                 tumor_or_background=tumor_or_backgorund)
        
        shape_set_img = (large_w, large_h, small_w, small_h)
        tiles_list = tiles_tools.get_slide_tiles(np_small_filtered_img, shape_set_img, test_slide_path,
                                                 _env_tile_w_size, _env_tile_h_size,
                                                 t_p_threshold=_env_tp_tiles_threshold, load_small_tile=False)
        print('generate tiles for slide: %s, keep [%d] tile objects in (.pkl) list.' % (test_slide_path, len(tiles_list)))
        if len(tiles_list) == 0:
            continue
        
        if tumor_or_backgorund == True:
            pkl_path = generate_t_b_roi_tiles_list_pkl_filepath(test_slide_path, label_type, 
                                                                _env_process_slide_tumor_tile_pkl_train_dir, 
                                                                _env_process_slide_tumor_tile_pkl_test_dir,
                                                                for_train=False)
        else:
            pkl_path = generate_t_b_roi_tiles_list_pkl_filepath(test_slide_path, label_type, 
                                                                _env_process_slide_back_tile_pkl_train_dir, 
                                                                _env_process_slide_back_tile_pkl_test_dir,
                                                                for_train=False)
        
        with open(pkl_path, 'wb') as f_pkl:
            pickle.dump(tiles_list, f_pkl) 


def _show_tiles_images(slide_pkl_filepath, tile_show_folderpath):
    """
    show some example for tiles in the inputed slide
    """
    tiles_list = recovery_tiles_list_from_pkl(pkl_filepath=slide_pkl_filepath)
    for tile in tiles_list:
        tile_pil = tile.get_pil_tile()
        tile_name = 'h{}w{}.png'.format(tile.h_id, tile.w_id)
        _ = tile_pil.save(os.path.join(tile_show_folderpath, tile_name))
        print('Store tile: {} in folder: {}'.format(tile_name, tile_show_folderpath))
        
            
def _print_data_info(ENV_task):
    """
    1. slide number, training slide number, testing/validation slide number
    2. total training tile number, total testing/validation tile number,
        average tile number in training set, testing/validation set
    3. number of positive/negative slide in training set, testing/validation set
    4. EMT high/low score threshold (only for OV_EMT task)
    
    need to prepare some parames:
        _env_process_slide_tile_pkl_train_dir,
        _env_process_slide_tile_pkl_test_dir,
        _env_process_slide_tile_pkl_valid_dir,
        _env_parse_data_slide_dir,
        _env_metadata_dir
        
    Args:
        ENV_task: the task environment object
    """
    
    ''' preparing some parmes'''
    _env_process_slide_tile_pkl_train_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TILE_PKL_TRAIN_DIR
    _env_process_slide_tile_pkl_test_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TILE_PKL_TEST_DIR
    _env_process_slide_tile_pkl_valid_dir = ENV_task.TASK_PROCESS_REPO_SLIDE_TILE_PKL_TEST_DIR
    _env_parse_data_slide_dir = ENV_task.PARSE_REPO_DATA_SLIDE_DIR
    _env_metadata_dir = ENV_task.METADATA_REPO_DIR
    
    ''' 1 '''
    nb_training_slide = len(os.listdir(_env_process_slide_tile_pkl_train_dir))
    nb_testing_slide = len(os.listdir(_env_process_slide_tile_pkl_test_dir))
    nb_validation_slide = len(os.listdir(_env_process_slide_tile_pkl_valid_dir)) - nb_testing_slide
    nb_slide = nb_training_slide + nb_testing_slide + nb_validation_slide
    
    print('slide number: {}, training slide number: {}, testing/validation slide number : {}/{}'.format(nb_slide, nb_training_slide, nb_testing_slide, nb_validation_slide))
    
    ''' 2 '''
    nb_training_tile = 0
    for pkl_filepath in os.listdir(_env_process_slide_tile_pkl_train_dir):
        with open(os.path.join(_env_process_slide_tile_pkl_train_dir, pkl_filepath), 'rb') as f_pkl:
            tiles_list = pickle.load(f_pkl)
            nb_training_tile += len(tiles_list)
    
    nb_testing_tile = 0
    nb_validation_tile = 0
    for pkl_filepath in os.listdir(_env_process_slide_tile_pkl_test_dir)[ :nb_testing_slide]:
        with open(os.path.join(_env_process_slide_tile_pkl_test_dir, pkl_filepath), 'rb') as f_pkl:
            tiles_list = pickle.load(f_pkl)
            nb_testing_tile += len(tiles_list)
    for pkl_filepath in os.listdir(_env_process_slide_tile_pkl_valid_dir)[nb_testing_slide: ]:
        with open(os.path.join(_env_process_slide_tile_pkl_valid_dir, pkl_filepath), 'rb') as f_pkl:
            tiles_list = pickle.load(f_pkl)
            nb_validation_tile += len(tiles_list)
    
    print('training tile number: {}, testing/validation tile number: {}/{}'.format(nb_training_tile, nb_testing_tile, nb_validation_tile))
    print('average tile number in training set: %.2f, testing/validation set: %.2f/%.2f' % (nb_training_tile * 1.0 / nb_training_slide,
                                                                                          nb_testing_tile * 1.0 / nb_testing_slide,
                                                                                          nb_validation_tile * 1.0 / nb_validation_slide))
    
    ''' 3 '''
    label_dict = query_task_label_dict_fromcsv(ENV_task)
    nb_pos_slide = 0
    nb_neg_slide = 0
    for slide_filepath in os.listdir(_env_parse_data_slide_dir):
        case_id = slide_filepath[ :slide_filepath.find('_')]
        nb_pos_slide += (1 if label_dict[case_id] == 1 else 0)
        nb_neg_slide += (1 if label_dict[case_id] == 0 else 0)
    
    print('{}-high slide number: {}, {}-low slide number: {}'.format(ENV_task.LABEL_TYPE, nb_pos_slide, ENV_task.LABEL_TYPE, nb_neg_slide))
    
    ''' 4 (only for OV_EMT task) '''
    if ENV_task.LABEL_TYPE == 'emt':
        for f in os.listdir(_env_metadata_dir):
            if f.startswith('2019') and f.endswith('.csv'):
                print('automatically find CSV file: {}'.format(f))
                csv_from_filename = f
                break
        
        csv_from_filepath = os.path.join(_env_metadata_dir, csv_from_filename)
        with open(csv_from_filepath, 'r', newline='') as csv_from_file:
            csv_reader = csv.reader(csv_from_file)
            row_list = list(row for row in csv_reader)[1:]
            # get the median of the EMT score
            EMT_score_list = list(map(float, list(row[3] for row in row_list)))
            EMT_score_median = np.median(np.array(EMT_score_list))
            
        print('EMT high/low score threshold: %.6f' % (EMT_score_median))
    

if __name__ == '__main__': 
#     _run_singleprocess_slide_tiles_split_keep_object(test_prop=0.3, label_type='emt')
    
    '''
    pkl_dir = EMT_PROCESS_TCGA_SLIDE_TILE_PKL_TRAIN_DIR
    files = os.listdir(pkl_dir)
    tiles_list = recovery_tiles_list_from_pkl(os.path.join(pkl_dir, files[0]))
    print(tiles_list)
    print(len(tiles_list))
    '''
    
#     _print_data_info()
    
    slide_pkl_filepath = 'D:\\TCGA_OV_dataset\\example_dx\\emt\\512\\train_pkl\\TCGA-13-A5FT_DX1-(tiles_list).pkl'
    tile_show_folderpath = 'D:\\TCGA_OV_dataset\\example_dx\\emt\\512\\show_tiles'
    _show_tiles_images(slide_pkl_filepath, tile_show_folderpath)
    
    
    
