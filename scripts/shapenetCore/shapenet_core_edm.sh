accelerate launch --multi_gpu --num_processes=4 --main_process_port=12588 train.py \
--train_batch_size 4 \
--train_num_workers 0 \
--num_sample 2 \
--num_train_epochs 96 \
--lr 3e-2 \
--obj_cat shapenet_core \
--precision_level manifold \
--normalization_type group \
--max_verts 60000 \
--num_groups 200 \
--k_eig 128 \
--group shapenet_core \
--job 8_blocks_all_chairs_03001627 \
--name gn_200_edm_bs4_e96 \
--N_block 8 \
--visible_devices "0,1,2,3" \
--output_dir ./experiment/develop/ \
--gradient_accumulation_steps 1 \
--pipe edm \
--wandb True \
--save_interval 2 \
--synsets '03001627' \
--object_set shapenet_core \
--shapenet_path datasets/objects/shapenetCore \
--preprocessed_data datasets/preprocessed/op_cache/ \
--split_file splits/shapenet_core/split.json \
--splits train \



