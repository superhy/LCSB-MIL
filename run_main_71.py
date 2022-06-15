'''
@author: Yang Hu
'''


import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import sys

from models.functions_relcsb import _run_train_relcsb_gated_attpool_resnet18
from run_main import Logger
from support import tools
from support.env_ov_emt import ENV_OV_EMT, ENV_OV_EMT_pc
from support.env_lu_egfr import ENV_LU_EGFR
from support.env_co_kras import ENV_CO_KRAS
from support.env_br_her2 import ENV_BR_HER2


task_ids = [71]
task_str = '-' + '-'.join([str(id) for id in task_ids])

if __name__ == '__main__':
    
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    
    ENV_task = ENV_OV_EMT
#     ENV_task = ENV_LU_EGFR
#     ENV_task = ENV_CO_KRAS
#     ENV_task = ENV_BR_HER2

    log_name = 'running_log{}-{}-{}.log'.format(ENV_task.FOLD_SUFFIX,
                                                ENV_task.LABEL_TYPE + task_str,
                                                str(tools.Time().start)[:13].replace(' ', '-'))
    sys.stdout = Logger(os.path.join(ENV_task.LOG_REPO_DIR, log_name))
    
    if 71 in task_ids:
        _run_train_relcsb_gated_attpool_resnet18(ENV_task)
        
        
        