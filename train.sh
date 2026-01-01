
# for num_cct in {2..5}; do
#     for repeat in {1..3}; do
#         CUDA_VISIBLE_DEVICES=3 python3 train.py \
#             --dataset_name C45V2 \
#             --wsol_method GLFormer \
#             --backbone conformer-s \
#             --save_dir rslogs/C45V2_conformer-s_cross_cosrefine_cpk01/num_cct${num_cct}/cos_repeat${repeat} \
#             --num_val_sample_per_class 10 \
#             --cam_batch_size 32 \
#             --Unfreeze_batch_size 32 \
#             --num_classes 16 \
#             --lr_decay_type cos \
#             --optimizer_type adam \
#             --Init_lr 3e-4 \
#             --Min_lr 1e-5 \
#             --num_cct ${num_cct} \
#             --eval_period 1 \
#             --conv_attn_refine True \
#             --attention_type mct 
#     done
# done

# for num_cct in {2..5}; do
#     for repeat in {1..3}; do
#         CUDA_VISIBLE_DEVICES=3 python3 train.py \
#             --dataset_name PN2 \
#             --wsol_method GLFormer \
#             --backbone conformer-s \
#             --save_dir rslogs/PN2_conformer-s_cross_cosrefine_cpk01/num_cct${num_cct}/cos_repeat${repeat} \
#             --num_val_sample_per_class 10 \
#             --cam_batch_size 32 \
#             --Unfreeze_batch_size 32 \
#             --num_classes 11 \
#             --lr_decay_type cos \
#             --optimizer_type adam \
#             --Init_lr 3e-4 \
#             --Min_lr 1e-5 \
#             --num_cct ${num_cct} \
#             --eval_period 1 \
#             --conv_attn_refine True \
#             --attention_type mct 
#     done
# done

# for repeat in {1..3}; do
#         CUDA_VISIBLE_DEVICES=3 python3 train.py \
#             --dataset_name PN2 \
#             --wsol_method GLFormer \
#             --backbone conformer-s \
#             --save_dir rslogs/PN2_conformer-s_cosrefine_cpk01/cos_repeat${repeat} \
#             --num_val_sample_per_class 10 \
#             --cam_batch_size 32 \
#             --Unfreeze_batch_size 32 \
#             --num_classes 11 \
#             --lr_decay_type cos \
#             --optimizer_type adam \
#             --Init_lr 3e-4 \
#             --Min_lr 1e-5 \
#             --num_cct 3 \
#             --eval_period 1 \
#             --conv_attn_refine True \
#             --attention_type mct 
# done

# for repeat in {1..3}; do
#         CUDA_VISIBLE_DEVICES=2 python3 train.py \
#             --dataset_name C45V2 \
#             --wsol_method GLFormer \
#             --backbone conformer-s \
#             --save_dir rslogs/C45V2_conformer-s_cosrefine_cpk01/cos_repeat${repeat} \
#             --num_val_sample_per_class 10 \
#             --cam_batch_size 32 \
#             --Unfreeze_batch_size 32 \
#             --num_classes 16 \
#             --lr_decay_type cos \
#             --optimizer_type adam \
#             --Init_lr 3e-4 \
#             --Min_lr 1e-5 \
#             --num_cct 3 \
#             --eval_period 1 \
#             --conv_attn_refine True \
#             --attention_type mct 
# done

# for repeat in {1..3}; do
#         CUDA_VISIBLE_DEVICES=3 python3 train.py \
#             --dataset_name PN2 \
#             --wsol_method GLFormer \
#             --backbone conformer-s \
#             --save_dir rslogs/PN2_conformer-s_cos16x_cpk01/cos_repeat${repeat} \
#             --num_val_sample_per_class 10 \
#             --cam_batch_size 32 \
#             --Unfreeze_batch_size 32 \
#             --num_classes 11 \
#             --lr_decay_type cos \
#             --optimizer_type adam \
#             --Init_lr 3e-4 \
#             --Min_lr 1e-5 \
#             --num_cct 3 \
#             --eval_period 1 \
#             --attention_type mct 
# done

# for repeat in {1..3}; do
#         CUDA_VISIBLE_DEVICES=2 python3 train.py \
#             --dataset_name C45V2 \
#             --wsol_method GLFormer \
#             --backbone conformer-s \
#             --save_dir rslogs/C45V2_conformer-s_cos16x_cpk01/cos_repeat${repeat} \
#             --num_val_sample_per_class 10 \
#             --cam_batch_size 32 \
#             --Unfreeze_batch_size 32 \
#             --num_classes 16 \
#             --lr_decay_type cos \
#             --optimizer_type adam \
#             --Init_lr 3e-4 \
#             --Min_lr 1e-5 \
#             --num_cct 3 \
#             --eval_period 1 \
#             --attention_type mct 
# done

# for repeat in {1..3}; do
#         CUDA_VISIBLE_DEVICES=2 python3 train.py \
#             --dataset_name C45V2 \
#             --wsol_method GLFormer \
#             --backbone conformer-s \
#             --save_dir rslogs/C45V2_conformer-s_cos16x_cpk01/cos_repeat${repeat} \
#             --num_val_sample_per_class 10 \
#             --cam_batch_size 32 \
#             --Unfreeze_batch_size 32 \
#             --num_classes 16 \
#             --lr_decay_type cos \
#             --optimizer_type adam \
#             --Init_lr 3e-4 \
#             --Min_lr 1e-5 \
#             --num_cct 3 \
#             --eval_period 1 \
#             --attention_type mct 
# done

# for num_cct in {3..5}; do
#     for repeat in {1..3}; do
#         CUDA_VISIBLE_DEVICES=0 python3 train.py \
#             --dataset_name C45V2 \
#             --wsol_method GLFormer \
#             --backbone conformer-t \
#             --save_dir rslogs/C45V2_conformer-t_cross_cosrefine_cpk01/num_cct${num_cct}/cos_repeat${repeat} \
#             --num_val_sample_per_class 10 \
#             --cam_batch_size 32 \
#             --Unfreeze_batch_size 32 \
#             --num_classes 16 \
#             --lr_decay_type cos \
#             --optimizer_type adam \
#             --Init_lr 3e-4 \
#             --Min_lr 1e-5 \
#             --num_cct ${num_cct} \
#             --eval_period 1 \
#             --conv_attn_refine True \
#             --attention_type mct 
#     done
# done

for num_cct in {3..5}; do
    for repeat in {1..3}; do
        CUDA_VISIBLE_DEVICES=1 python3 train.py \
            --dataset_name PN2 \
            --wsol_method GLFormer \
            --backbone conformer-s \
            --save_dir rslogs/PN2_conformer-t_cross_cosrefine_cpk01/num_cct${num_cct}/cos_repeat${repeat} \
            --num_val_sample_per_class 10 \
            --cam_batch_size 32 \
            --Unfreeze_batch_size 32 \
            --num_classes 11 \
            --lr_decay_type cos \
            --optimizer_type adam \
            --Init_lr 3e-4 \
            --Min_lr 1e-5 \
            --num_cct ${num_cct} \
            --eval_period 1 \
            --conv_attn_refine True \
            --attention_type mct 
    done
done