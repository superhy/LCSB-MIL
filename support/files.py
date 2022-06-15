'''
@author: yang hu


'''

import os
import shutil

import openslide

import pandas as pd
from support.env import ENV
from support.env_liver_nash import ENV_LIVER_NASH
from support.env_lvi import ENV_LVI
from support.env_ov_emt import ENV_OV_EMT
from support.metadata import query_task_label_dict_fromcsv
from support.tools import Time


def parse_filesystem_slide(slide_dir, original_download=False):
    slide_path_list = []
    for root, dirs, files in os.walk(slide_dir):
        for f in files:
            if f.endswith('.svs') or f.endswith('.tiff') or f.endswith('.tif'):
                slide_path = os.path.join(root, f)
                if original_download == True:
                    print('found slide file from TCGA_original: ' + slide_path)
                slide_path_list.append(slide_path)
                
    return slide_path_list

def parse_filenames_tilespkl(tilespkl_dir):
    tilepkl_filename_list = []
    for root, dirs, files in os.walk(tilespkl_dir):
        for f in files:
            if f.endswith('.pkl'):
                print('read slide tiles pkl file: ' + f)
                tilepkl_filename_list.append(f)
                
    return tilepkl_filename_list


def clear_dir(abandon_dirs):
    """
    """
    print('remove the old dirs: {}.'.format(abandon_dirs))
    for dir in abandon_dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)

def move_file(src_path, dst_path, mode='move'):
    """
    move single file from src_path to dst_dir
    
    Args -> mode: 
        'move' -> move the file
        'copy' or other string -> copy the file (for test)
    """
    if mode == 'move':
        shutil.move(src_path, dst_path)
    else:
        shutil.copy(src_path, dst_path)
    
def move_TCGA_download_file_rename_batch(ENV_task, tcga_slide_path_list,
                                         mode='move', filter_annotated_slide=True):
    """
    Move the svs or tif file first,
    and rename according to the case id
    
    Args:
        ENV_task: the task environment object
    
    """
    
    label_dict = query_task_label_dict_fromcsv(ENV_task)
    
    _env_parse_data_slide_dir = ENV_task.PARSE_REPO_DATA_SLIDE_DIR
    if not os.path.exists(_env_parse_data_slide_dir):
        os.makedirs(_env_parse_data_slide_dir)
    else:
        clear_dir([_env_parse_data_slide_dir])
        os.makedirs(_env_parse_data_slide_dir)  
    
    count, filted = 0, 0
    for slide_path in tcga_slide_path_list:
        # parse case id from original slide_path
        case_id = parse_slide_caseid_from_filepath(slide_path)
        if filter_annotated_slide == True and case_id not in label_dict.keys():
            filted += 1
            continue
        
        # parse slide type and id
        slide_type_id = parse_slide_typeid_from_filepath(slide_path)
        suffix = slide_path[len(slide_path) - 4 : ]
        slide_new_name = case_id + slide_type_id + suffix
        print('move slide from: ' + slide_path + ' -> ' + os.path.join(_env_parse_data_slide_dir, slide_new_name))
        move_file(slide_path, os.path.join(_env_parse_data_slide_dir, slide_new_name), mode)
        count += 1
    print('moved {} slide files, filted {} slides.'.format(count, filted))
   
    
''' ---------- id parse functions for different datasets ---------- '''    
    
def parse_TCGA_slide_typeid_from_filepath(slide_filepath):
    """
    get the type id from slide's filepath, for TCGA dataset
    
    PS: with '_' + string of typeid
    """
    slide_type_id = '_' + slide_filepath[slide_filepath.find('.') - 3: slide_filepath.find('.')]
    return slide_type_id

def parse_TCGA_slide_caseid_from_filepath(slide_filepath):
    """
    get the case id from slide's filepath, for TCGA dataset
    
    Args:
        slide_filepath: as name
        cut_range: filepath string cut range to get the TCGA case_id
    """
    cut_range = 15 if ENV.SLIDE_TYPE == 'tx' else 12
    case_id = slide_filepath[slide_filepath.find('TCGA-'): slide_filepath.find('TCGA-') + cut_range]
    return case_id

def parse_LVI_slide_bioid_from_filepath(slide_filepath):
    """
    get the type id from slide's filepath, for LVI dataset
    
    PS: with '_' + string of typeid
    """
    slide_name = slide_filepath.split('slides')[-1][1:]
    id_comps = slide_name.split('_', 2)
    slide_bio_id = id_comps[-1][: id_comps[-1].find('.')]
    return slide_bio_id

def parse_LVI_slide_caseid_from_filepath(slide_filepath):
    """
    get the case id from slide's filepath, for TCGA dataset
    
    Args:
        slide_filepath: as name
    """
    slide_name = slide_filepath.split('slides')[-1][1:]
    id_comps = slide_name.split('_', 2)
    case_id = '_'.join([id_comps[0], id_comps[1]]) + '_'
    return case_id

def parse_GTEX_slide_caseid_from_filepath(slide_filepath):
    """
    get the case id from slide's filepath, for LIVER dataset
    """
    slide_name = slide_filepath.split('slides')[-1][1:]
    case_id = slide_name[:slide_name.find('.svs')]
    return case_id


''' ------------ uniform id parse functions, automatically switch for different datasets ------------ '''
def parse_slide_typeid_from_filepath(slide_filepath):
    """
    get the type id from slide's filepath, for all available dataset
    """
    if slide_filepath.find('TCGA') != -1:
        slide_type_id = parse_TCGA_slide_typeid_from_filepath(slide_filepath)
    elif slide_filepath.find('LVI') != -1:
        slide_type_id = parse_LVI_slide_bioid_from_filepath(slide_filepath)
    elif slide_filepath.find('GTEX') != -1:
        # there is no type_id in GTEX tissues
        slide_type_id = ''
    else:
        raise NameError('cannot detect right dataset indicator!')
    
    return slide_type_id
        
def parse_slide_caseid_from_filepath(slide_filepath):
    """
    get the case id from slide's filepath
    
    Args:
        slide_filepath: as name
    """
    if slide_filepath.find('TCGA') != -1:
        case_id = parse_TCGA_slide_caseid_from_filepath(slide_filepath)
    elif slide_filepath.find('LVI') != -1:
        case_id = parse_LVI_slide_caseid_from_filepath(slide_filepath)
    elif slide_filepath.find('GTEX') != -1:
        case_id = parse_GTEX_slide_caseid_from_filepath(slide_filepath)
    else:
        raise NameError('cannot detect right dataset indicator!')
    
    return case_id

def parse_slideid_from_filepath(slide_filepath):
    """
    get the whole slideid from slide's filepath
    
    PS combine the previous 2 functions
    """
    return parse_slide_caseid_from_filepath(slide_filepath) + parse_slide_typeid_from_filepath(slide_filepath)

def parse_slideid_from_tilepklname(tilepkl_filename):
    """
    get the slide_id from previous made tiles pkl's filename
    """
    slide_id = tilepkl_filename.split('-(')[0]
    return slide_id


def download_gtex_liver_tissues(ENV_task, gtex_meta_csv_name):
    '''
    warning: only running on linux
    '''
    csv_liver_filepath = os.path.join(ENV_task.METADATA_REPO_DIR, gtex_meta_csv_name)
    liver_slides_download_dir = ENV_task.PARSE_REPO_DATA_SLIDE_DIR
    df_liver_meta = pd.read_csv(csv_liver_filepath)
    
    tissue_ids = []
    for i in range(len(df_liver_meta)):
        tissue_ids.append(df_liver_meta.loc[i]['Tissue Sample ID'])
    
    tissue_ids = tissue_ids[:3]
    for i, t_id in enumerate(tissue_ids):
        download_time = Time()
        print('>>> downloading image: {}'.format(t_id))
        
        # like: wget -P /well/rittscher/users/lec468/LIVER_NASH/example_dx/slides/ https://brd.nci.nih.gov/brd/imagedownload/GTEX-111VG-0826
        os.system('wget -P {}/ https://brd.nci.nih.gov/brd/imagedownload/{}'.format(liver_slides_download_dir, t_id))
        # like: mv /well/rittscher/projects/LIVER_NASH_project/example_dx/slides/GTEX-11DYG-1726 /well/rittscher/projects/LIVER_NASH_project/example_dx/slides/GTEX-11DYG-1726.svs
        os.system('mv {}/{} {}/{}.svs'.format(liver_slides_download_dir, t_id, liver_slides_download_dir, t_id))
        print("### download image to: {}/{}.svs, time: {}".format(liver_slides_download_dir, t_id, str(download_time.elapsed() ) ) )
        
''' check damaged slides and re-download it until it's successfully transmitted '''
def check_slides_damaged(slide_filepath):
    ''' check if a tissue slides is damaged '''
    slide_damaged = False
    try:
        _ = openslide.open_slide(slide_filepath)
    except:
        slide_damaged = True
        print('found damaged slide: %s' % slide_filepath)
    return slide_damaged
def list_all_damaged_slides(ENV_task):
    ''' return a list of names of all damaged  '''
    slide_filenames = os.listdir(ENV_task.PARSE_REPO_DATA_SLIDE_DIR)
    damaged_slide_names = []
    for slide_name in slide_filenames:
        if check_slides_damaged(os.path.join(ENV_task.PARSE_REPO_DATA_SLIDE_DIR, slide_name)):
            damaged_slide_names.append(slide_name)
    print('collect damaged slides list:', damaged_slide_names)
    
    return damaged_slide_names

def forced_download_liver_tissues(ENV_task, gtex_meta_csv_name, redownload_try=10):
    '''
    '''
    csv_liver_filepath = os.path.join(ENV_task.METADATA_REPO_DIR, gtex_meta_csv_name)
    liver_slides_download_dir = ENV_task.PARSE_REPO_DATA_SLIDE_DIR
    df_liver_meta = pd.read_csv(csv_liver_filepath)
    
    tissue_ids = []
    for i in range(len(df_liver_meta)):
        tissue_ids.append(df_liver_meta.loc[i]['Tissue Sample ID'])
        
    cannot_download_slide_list = []
    for i, t_id in enumerate(tissue_ids):
        download_time = Time()
        
        tissue_slide_path = '{}/{}.svs'.format(liver_slides_download_dir, t_id)
        
        need_redownload = True
        if os.path.exists(tissue_slide_path):
            if check_slides_damaged(tissue_slide_path) == False:
                print('### slide: {} successfully download already.'.format(t_id + '.svs'))
                need_redownload = False
            else:
                os.system('rm {}'.format(tissue_slide_path))
                print('> remove damaged slide: {}'.format(t_id + '.svs'))
        else:
            print('! not found slide: {}'.format(t_id + '.svs'))
            
        if need_redownload == False:
            continue
        
        redownload_try_times, redownload_success = 0, False
        while redownload_try_times < redownload_try and redownload_success == False:
            os.system('wget -P {}/ https://brd.nci.nih.gov/brd/imagedownload/{}'.format(liver_slides_download_dir, t_id))
            os.system('mv {}/{} {}/{}.svs'.format(liver_slides_download_dir, t_id, liver_slides_download_dir, t_id))
            if check_slides_damaged(tissue_slide_path) is False:
                redownload_success = True
                print("### successfully repair image to: {}/{}.svs".format(liver_slides_download_dir, t_id) )
            else:
                os.system('rm {}'.format(tissue_slide_path) )
                print('! redownload unsuccessful on slide: {}'.format(t_id + '.svs'))
            redownload_try_times += 1
        if redownload_success == False:
            print('!!! tried many times, still cannot download slide: {}'.format(t_id + '.svs'))
            cannot_download_slide_list.append(t_id)
            
        print('used time: {}'.format(str(download_time.elapsed() ) ) )
    
    return cannot_download_slide_list
            
    
def _run_parsedir_move_TCGA_slide(ENV_task):
    
    _env_original_data_dir = ENV_task.ORIGINAL_REPO_DATA_DIR
    _env_parse_data_slide_dir = ENV_task.PARSE_REPO_DATA_SLIDE_DIR
    _env_task_name = ENV_task.TASK_NAME
    
    slide_path_list = parse_filesystem_slide(_env_original_data_dir)
    move_TCGA_download_file_rename_batch(_env_parse_data_slide_dir,
                                         _env_task_name,
                                         slide_path_list, mode='copy')
    
def _run_download_gtex_liver_tissues(ENV_task):
    
    gtex_meta_csv_name = 'GTEx_liver_samples.csv'
    download_gtex_liver_tissues(ENV_task, gtex_meta_csv_name)
    
def _run_forced_repair_gtex_liver_tissues(ENV_task):
    
    gtex_meta_csv_name = 'GTEx_liver_samples.csv'
    cannot_download_slide_list = forced_download_liver_tissues(ENV_task, gtex_meta_csv_name)
    print(cannot_download_slide_list)
    
        

if __name__ == '__main__':
    # for test
#     _run_parsedir_move_TCGA_slide()
#     clear_dir([env.EMT_PROCESS_TCGA_SLIDE_TILE_PKL_TRAIN_DIR, 
#                env.EMT_PROCESS_TCGA_SLIDE_TILE_PKL_TEST_DIR])

    slide_filepath = os.path.join(ENV_LVI.PARSE_REPO_DATA_SLIDE_DIR, '46000_19_100401004_HE_20191028_141306.tiff')
#     slide_filepath = os.path.join(ENV_OV_EMT.PARSE_REPO_DATA_SLIDE_DIR, 'TCGA-13-A5FT_DX1.svs')
#     slide_filepath = 'N/A'
#     slide_id = parse_slideid_from_filepath(slide_filepath=slide_filepath)
#     print(slide_id)
#     tilepkl_filename_list = parse_filenames_tilespkl(env_ov_emt.EMT_PROCESS_TCGA_SLIDE_TILE_PKL_TRAIN_DIR)
#     print(tilepkl_filename_list)
#     for tilepkl_filename in tilepkl_filename_list:
#         print(tilepkl_filename)
#         slide_id = parse_slideid_from_tilepklname(tilepkl_filename)
#         print(slide_id)

    print((check_slides_damaged(slide_filepath = os.path.join(ENV_LIVER_NASH.PARSE_REPO_DATA_SLIDE_DIR, 'GTEX-ZWKS-1726.svs')) is False))
#     list_all_damaged_slides(ENV_task=ENV_LIVER_NASH)
    
    
    
    
    
    