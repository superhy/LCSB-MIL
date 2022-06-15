'''

@author: yang hu
'''

import csv
import json
import os

import numpy as np
import pandas as pd
from support.env_liver_nash import ENV_LIVER_NASH


def load_json_file(json_filepath):
    with open(json_filepath, 'r') as json_file:
        json_text = json_file.read()
        json_data = json.loads(json_text)
        
        return json_data

''' ---------------- functions usually used for parse OV-EMT metadatas ----------------- '''

def parsing_slide_json_data(json_data):
    slide_meta_dict = {}
    for i, element in enumerate(json_data):
        """
        case id in this function is uu_id
        without prefix 'TCGA-'
        """
        case_uu_id = element['cases'][0]['case_id']
#         print(case_uu_id)
        if case_uu_id not in slide_meta_dict.keys():
            slide_meta_dict[case_uu_id] = []
            
        slide_single_dict = {}
        slide_single_dict['file_id'] = element['file_id']
        slide_single_dict['file_name'] = element['file_name']
        slide_single_dict['data_format'] = element['data_format']
        slide_single_dict['file_size'] = element['file_size']
        
        slide_meta_dict[case_uu_id].append(slide_single_dict)
    
    return slide_meta_dict

def parse_EMT_label_fromcsv_2csv(ENV_task, csv_to_filename, csv_from_filename=None):
    """
    parse and extract EMT label from TCGA deconvolution results .csv file
    write the EMT annotations in another .csv file separately
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
        
def trans_slide_meta_dict2csv(csv_filepath, slide_meta_dict):
    """
    some slide meta data parsing
    only for review
    """ 
    with open(csv_filepath, 'w', newline='') as csv_file:
        """
        case id in this function is uu_id
        without prefix 'TCGA-'
        """
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['case uu id', 'file_id', 'file_name', 'data_format', 'file_size'])
        
        for i, case_uu_id in enumerate(slide_meta_dict):
            for j, slide in enumerate(slide_meta_dict[case_uu_id]):
                csv_writer.writerow([case_uu_id, slide['file_id'],
                                     slide['file_name'],
                                     slide['data_format'],
                                     slide['file_size']])
                
                
''' -------------- functions usually used for parse Liver metadatas --------------- '''

def filter_withlabel_liver_meta_csv2csv(ENV_task, csv_to_filename, csv_from_filename):
    '''
    filter the tissue slides without any label
    on GTEx Liver dataset
    '''
    
    csv_from_filepath = os.path.join(ENV_task.METADATA_REPO_DIR, csv_from_filename)
    csv_from_data = pd.read_csv(csv_from_filepath, usecols=['Tissue Sample ID', 'Pathology Categories'])
    filter_data = []
    for i in range(len(csv_from_data)):
        data_line = csv_from_data.loc[i]
        if pd.isnull(data_line['Pathology Categories']) == False:
            filter_data.append({'Tissue Sample ID': data_line['Tissue Sample ID'], 
                                'Pathology Categories': data_line['Pathology Categories']
                                })
    print('>>> load csv from: %s, with %d lines, left %d lines, filtered %d non-labeled lines.' % (csv_from_filepath,
                                                                                                   len(csv_from_data), 
                                                                                                   len(filter_data),
                                                                                                   len(csv_from_data) - len(filter_data) ))
            
    csv_to_df = pd.DataFrame(filter_data)
    print('>>> reproduce filtered dataframe as: \n', csv_to_df)
    csv_to_filepath = os.path.join(ENV_task.METADATA_REPO_DIR, csv_to_filename)
    csv_to_df.to_csv(csv_to_filepath, index=False)
    print('### write filtered dataframe to: {}'.format(csv_to_filepath))
    
def filter_nashlabel_liver_meta_csv2csv(ENV_task, csv_to_filename_a, csv_to_filename_i, csv_from_filename,
                                        nash_labels=['fibrosis', 'inflammation', 'steatosis, ']):
    '''
    '''
    csv_from_filepath = os.path.join(ENV_task.METADATA_REPO_DIR, csv_from_filename)
    csv_from_data = pd.read_csv(csv_from_filepath)
    nash_filter_data_a, nash_filter_data_i = [], []
    for i in range(len(csv_from_data)):
        label_line = csv_from_data.loc[i]
        label_dict = {'Tissue Sample ID': label_line['Tissue Sample ID']}
        labels = label_line['Pathology Categories'].split(', ')
        
        withnash = 0
        for nash_lbl in nash_labels:
            label_dict[nash_lbl] = 1 if nash_lbl in labels else 0
            withnash += label_dict[nash_lbl]
        
        nash_filter_data_a.append(label_dict)
        if withnash > 0:
            nash_filter_data_i.append(label_dict)
    print('>>> reformat nash label dataframe as:')
            
    csv_to_df_a = pd.DataFrame(nash_filter_data_a)
    csv_to_df_i = pd.DataFrame(nash_filter_data_i)
    print(csv_to_df_a, '\n------\n', csv_to_df_i)
    
    csv_to_filepath_a = os.path.join(ENV_task.METADATA_REPO_DIR, csv_to_filename_a)
    csv_to_filepath_i = os.path.join(ENV_task.METADATA_REPO_DIR, csv_to_filename_i)
    csv_to_df_a.to_csv(csv_to_filepath_a, index=False)
    csv_to_df_i.to_csv(csv_to_filepath_i, index=False)
    
    print('### write nash label dataframe to: {} and {}'.format(csv_to_filepath_a, csv_to_filename_i))
    
    
"""-----------------------------------------------------------------------------------------"""
        
def query_task_label_dict_fromcsv(ENV_task, task_csv_filename=None):
    f_start_string = ENV_task.TASK_NAME + '-dx'
    if task_csv_filename == None:
        for f in os.listdir(ENV_task.METADATA_REPO_DIR):
            if f.startswith(f_start_string) and f.endswith('.csv'):
                print('automatically find CSV file: {}'.format(f))
                task_csv_filename = f
                break
            
    task_csv_filepath = os.path.join(ENV_task.METADATA_REPO_DIR, task_csv_filename)
    task_label_dict = {}
    with open(task_csv_filepath, 'r', newline='') as task_csv_file:
        csv_reader = csv.reader(task_csv_file)
        for csv_line in csv_reader:
            task_label_dict[csv_line[0]] = int(csv_line[1])
            
    return task_label_dict


if __name__ == '__main__':
    
#     annotation_path = os.path.join(env_ov_emt.METADATA_REPO_DIR, 'TCGA_OV_dx_EMT_annotations.csv')
#     slide_record_path = os.path.join(env_ov_emt.METADATA_REPO_DIR, 'ov-emt.rec')
#     annotation_path = os.path.join(env_lu_egfr.METADATA_REPO_DIR, 'LU-EGFR-dx_label.csv')
#     slide_record_path = os.path.join(env_lu_egfr.METADATA_REPO_DIR, 'lu-egfr.rec')
#     annotation_path = os.path.join(env_co_kras.METADATA_REPO_DIR, 'CO-KRAS-dx_label.csv')
#     slide_record_path = os.path.join(env_co_kras.METADATA_REPO_DIR, 'co-kras.rec')

#     filter_withlabel_liver_meta_csv2csv(ENV_LIVER_NASH, 'Pathology_liver_samples.csv', 'GTEx_liver_samples.csv')
    
    filter_nashlabel_liver_meta_csv2csv(ENV_LIVER_NASH, 
                                        'LIVER-NASH_samples_a.csv', 
                                        'LIVER-NASH_samples_i.csv', 
                                        'Pathology_liver_samples.csv',
                                        nash_labels=['fibrosis', 'inflammation', 'steatosis'])
 
    
