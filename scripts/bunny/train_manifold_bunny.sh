
python train.py \
--train_batch_size 4 \
--num_sample 1 \
--visible_devices "0" \
--num_train_epochs 40 \
--lr 1e-4 \
--obj_cat bunny \
--precision_level manifold \
--normalization_type group \
--num_groups 64 \
--group develop \
--job manifold_bunny \
--name test \
--wandb False \
--N_block 8
