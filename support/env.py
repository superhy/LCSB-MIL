'''
@author: Yang Hu
'''
import torch

from support.parames import parames_basic


devices = torch.device('cuda')
devices_cpu = torch.device('cpu')

ENV = parames_basic(
        project_name='LCSB-path-V15',
        slide_type='dx',
        apply_tumor_roi=False,
        scale_factor=32,
        tile_size=256, # 512
        tp_tiles_threshold=60,
        pil_image_file_format='.png',
        debug_mode=False
    )

if __name__ == '__main__':
    pass