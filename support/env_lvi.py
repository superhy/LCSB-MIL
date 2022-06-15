'''
@author: Yang Hu
'''
from support.env import ENV
from support.parames import parames_task


ENV_LVI = parames_task(
                project_name=ENV.PROJECT_NAME,
                slide_type=ENV.SLIDE_TYPE, 
                apply_tumor_roi=ENV.APPLY_TUMOR_ROI, 
                scale_factor=ENV.SCALE_FACTOR, 
                tile_size=ENV.TILE_H_SIZE, 
                tp_tiles_threshold=ENV.TP_TILES_THRESHOLD, 
                pil_image_file_format=ENV.PIL_IMAGE_FILE_FORMAT, 
                debug_mode=ENV.DEBUG_MODE,
                task_name='LVI',
                label_type='lvi',
                server_root='',
                pc_root='D:/LVI_dataset',
                test_part_prop=0.3,
                fold_suffix='-0',
                num_cnn_epoch=10,
                num_att_epoch=80,
                mini_batch_tile=128,
                tile_dataloader_worker=12,
                loss_package=('ce'),
                mil_try_k=5,
                tile_encode_dim=512,
                mini_batch_slidemat=8,
                slidemat_dataloader_worker=4,
                num_last_eval_epochs=5,
                reset_optim=False,
                num_round=5,
                num_init_s_epoch=20,
                num_inround_s_epoch=5,
                num_inround_rev_t_epoch=1,
                num_inround_t_epoch=1,
                attpool_stop_loss=0.40,
                attpool_stop_maintains=3,
                overall_stop_loss=0.05,
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
                num_pretrain_epoch=100,
                rand_mat_reload_pluse=5,
                num_preload_rand_mat=5000,
                mat_bagdim=5000,
                num_epoch_rand_pair=3000
            )

if __name__ == '__main__':
    pass