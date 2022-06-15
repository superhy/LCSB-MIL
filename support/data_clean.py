'''
@author: Yang Hu

some function mainly running on WINDOWS (PC)
for EMT and other datasets and other tasks

'''


''' 
----------------- specifically for EMT ----------------- 
'''

import csv
import os
import warnings

import numpy as np
from support.files import parse_slide_caseid_from_filepath
from support.metadata import load_json_file, query_task_label_dict_fromcsv


def parse_task_label_csv2csv(ENV_task, csv_to_filename, csv_from_filename=None):
    """
    parse and extract EMT label from TCGA deconvolution results .csv file
    write the EMT annotations in another .csv file separately
    
    Args:
        ENV_task: the task environment object 
        csv_to_filename:
        csv_from_filename:
    """
    if csv_from_filename == None:
        for f in os.listdir(ENV_task.METADATA_REPO_DIR):
            if f.startswith('2019') and f.endswith('.csv'):
                print('automatically find CSV file: {}'.format(f))
                csv_from_filename = f
                break
    
    csv_from_filepath = os.path.join(ENV_task.METADATA_REPO_DIR, csv_from_filename)
    with open(csv_from_filepath, 'r', newline='') as csv_from_file:
        csv_reader = csv.reader(csv_from_file)
        row_list = list(row for row in csv_reader)[1:]
        # get the median of the EMT score
        EMT_score_list = list(map(float, list(row[3] for row in row_list)))
        EMT_score_median = np.median(np.array(EMT_score_list))
        
        case_EMT_set = []
        for csv_line in row_list:
            case_EMT_set.append((csv_line[0], 1 if float(csv_line[3]) >= EMT_score_median else 0))
        print('read {}, with EMT score median: {}'.format(csv_from_filepath, EMT_score_median))
        
    csv_to_filepath = os.path.join(ENV_task.METADATA_REPO_DIR, csv_to_filename)
    with open(csv_to_filepath, 'w', newline='') as csv_to_file:
        csv_writer = csv.writer(csv_to_file)
        for i, EMT_tuple in enumerate(case_EMT_set):
            csv_writer.writerow([EMT_tuple[0], EMT_tuple[1]])
        print('write label {}'.format(csv_to_filepath))

def filter_emt_dx_label_csv2csv(ENV_task, label_filename, caseid_json_filename):
    """
    filtering the caseid with dx slides in EMT-annotation list
    (some of the dx slides are not in the EMT-annotation list)
    
    Args:
        ENV_task: the task environment object 
    
    """
    json_data = load_json_file(os.path.join(ENV_task.METADATA_REPO_DIR, caseid_json_filename))
    dx_json_caseid_list = []
    for i, element in enumerate(json_data):
        caseid = element['submitter_id']
        dx_json_caseid_list.append(caseid)
        
    caseid_label_dict = {}
    csv_filepath = os.path.join(ENV_task.METADATA_REPO_DIR, label_filename)
    with open(csv_filepath, 'r', newline='') as csv_EMT_file:
        csv_reader = csv.reader(csv_EMT_file)
        for csv_line in csv_reader:
            caseid, label = csv_line[0][:-3], csv_line[1]
            caseid_label_dict[caseid] = label
    
    filted_dx_caseid_list = list(set(dx_json_caseid_list).intersection(set(caseid_label_dict.keys())))

    filted_label_filename =  label_filename.replace('EMT', 'dx_EMT')
    filted_csv_filepath = os.path.join(ENV_task.METADATA_REPO_DIR, filted_label_filename)
    
    nb_EMT_high, nb_EMT_low = 0, 0
    with open(filted_csv_filepath, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        if label_filename.find('cot') != -1:
            for i, caseid in enumerate(filted_dx_caseid_list):
                dx_cot_EMT_label = float(caseid_label_dict[caseid])
                if dx_cot_EMT_label >= 0.5:
                    nb_EMT_high += 1
                else:
                    nb_EMT_low += 1
                print('write annotation info: ', caseid, dx_cot_EMT_label)
                csv_writer.writerow([caseid, dx_cot_EMT_label])
        else:
            for i, caseid in enumerate(filted_dx_caseid_list):
                dx_EMT_label = int(caseid_label_dict[caseid])
                nb_EMT_high += dx_EMT_label
                nb_EMT_low += (1 - dx_EMT_label)
                print('write annotation info: ', caseid, dx_EMT_label)
                csv_writer.writerow([caseid, dx_EMT_label])
    print('number of case in EMT-high: %d, and EMT-low: %d, (in conti, count >= 0.5)' % (nb_EMT_high, nb_EMT_low))
        
def _run_parse_emt_label_csv2csv():
#     csv_to_filename, csv_from_filename = 'TCGA_OV_EMT_annotations.csv', None
    csv_to_filename, csv_from_filename = 'TCGA_OV_EMT_annotations.csv', '20191125TCGA_deconvolutionResults.csv'
    parse_task_label_csv2csv(csv_to_filename, csv_from_filename)
    
def _run_filt_dx_emt_label_json4csv():
    csv_filted_filename, json_used_filename = 'TCGA_OV_EMT_annotations.csv', 'ov_dx_cases.2021-05-25.json'
    filter_emt_dx_label_csv2csv(csv_filted_filename, json_used_filename)
    
    
''' ---------------- other tasks -------------- '''

def _parse_other_task_label_csv2csv(rough_label_filepaths: list,
                                    dx_label_store_dirs: list):
    '''
    suggested to directly call it
    make the annotation meta files for other datasets and tasks on them
    
    only running on WINDOWS
    
    Args:
        rough_label_filepaths: batch of rough label files need to parse
        dx_label_store_dirs: batch of final dx_label annotation file store dir
            if its length is not mapping with rough label file num
            report error
    '''
    if not len(rough_label_filepaths) == len(dx_label_store_dirs):
        warnings.warn('number of source file and target folder is not mapping!')
        return
    
    for i, rough_label_path in enumerate(rough_label_filepaths):
        with open(rough_label_path, 'r', newline='') as csv_from_file:
            csv_reader = csv.reader(csv_from_file)
            # get all rows
            row_list = list(row for row in csv_reader)
            # get the label string of each tile
            case_ids = list(row[0] for row in row_list)
            rough_labels = list(row[1] for row in row_list)
            print('read {}, get case ids and their rough label'.format(rough_label_path))
            
        dx_label_filename = rough_label_path.split('/')[-1].replace('rough', 'dx')
        dx_label_filepath = os.path.join(dx_label_store_dirs[i], dx_label_filename)
        with open(dx_label_filepath, 'w', newline='') as csv_to_file:
            csv_writer = csv.writer(csv_to_file)
            for j, case_id in enumerate(case_ids):
                if rough_labels[j].startswith('Pos') or rough_labels[j].startswith('YES'):
                    csv_writer.writerow([case_id, 1])
                elif rough_labels[j].startswith('Neg') or rough_labels[j].startswith('NO'):
                    csv_writer.writerow([case_id, 0])
                else:
                    continue
            print('write dx label {}, with 0/1 annotations \n'.format(dx_label_filepath))
            
def _filter_dataprotal_links_csv2txt(dx_label_filepaths: list,
                                     manifest_file_sets: list):
    '''
    suggested to directly call it
    filter the tcga data protal links with available annotations and
    write the new manifest txt file for slides download
    
    only running on WINDOWS
    
    Args:
        dx_label_filepaths: batch of final dx_label annotation files
        manifest_file_sets: batch of manifest info set
            with format: (file dir path, original filename, filtered filename)
    '''
    if not len(dx_label_filepaths) == len(manifest_file_sets):
        warnings.warn('number of source file and target set is not mapping!')
        return
    
    for i, dx_label_path in enumerate(dx_label_filepaths):
        with open(dx_label_path, 'r', newline='') as csv_from_file:
            csv_reader = csv.reader(csv_from_file)
            # get all rows
            row_list = list(row for row in csv_reader)
            # get the task available case ids
            case_ids = list(row[0] for row in row_list)
            
        org_manifest_filepath = os.path.join(manifest_file_sets[i][0], manifest_file_sets[i][1])
        # load org link lines
        with open(org_manifest_filepath, 'r') as org_manifest_file:
            lines = list(line for line in org_manifest_file.readlines())
            headline = lines[0][:-1]
            org_link_lines = lines[1:]
        print('load org link %d lines from %s' % (len(org_link_lines), org_manifest_filepath))
        # prepare available link lines, and their string for writing into txt
        new_link_lines = [headline]
        for candi_line in org_link_lines:
#             print(candi_line)
            candi_line = candi_line[:-1]
            svs_case_id = parse_slide_caseid_from_filepath(candi_line)
            if svs_case_id in case_ids:
                new_link_lines.append(candi_line)
        new_links_write_string = '\n'.join(new_link_lines)
        print('filter new link %d lines' % (len(new_link_lines) ))

        new_manifest_filepath = os.path.join(manifest_file_sets[i][0], manifest_file_sets[i][2])        
        # write new available links file
        with open(new_manifest_filepath, 'w') as new_manifest_file:
            new_manifest_file.write(new_links_write_string)
        print('write new link lines to %s \n' % (new_manifest_filepath))
        

def count_each_label_slides(ENV_task, annotation_filename, slide_record_path):
    '''
    count the amount of pos/neg
    
    Args:
        ENV_task: the task environment object
        annotation_filename: the task_csv_filename (or the whole path to task_csv_filepath is OK) 
    '''
    
    label_dict = query_task_label_dict_fromcsv(ENV_task, annotation_filename)
    
    _no_1, _no_0 = 0, 0
    with open(slide_record_path, 'r') as slide_record_file:
        lines = list(line for line in slide_record_file.readlines())
        for slide_rec in lines:
            case_id = slide_rec.split('_')[0]
            if label_dict[case_id] == 1:
                _no_1 += 1
            else:# label_dict[case_id] == 0:
                _no_0 += 1
                
    print('Pos number: %d, Neg number: %d' % (_no_1, _no_0))


if __name__ == '__main__':
    
    from support.env_br_her2 import ENV_BR_HER2
    ENV_task = ENV_BR_HER2
    annotation_path = os.path.join(ENV_task.METADATA_REPO_DIR, 'BR-HER2-dx_label.csv')
    slide_record_path = os.path.join(ENV_task.METADATA_REPO_DIR, 'br-her2.rec')
                                 
    count_each_label_slides(ENV_task, annotation_path, slide_record_path)
    
    
    