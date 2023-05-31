# python -m train_pose_vqvae \
#     --gpus 4 --gpu_ids "0,1,2,3" \
#     --init_lr 2e-4 \
#     --embedding_dim 128 \
#     --batchSize 12 \
#     --n_codes 2048 \
#     --data_path "Data/ProgressiveTransformersSLP" \
#     --vocab_file "Data/ProgressiveTransformersSLP/src_vocab.txt" \
#     --resume_ckpt "" \
#     --default_root_dir "experiments/pose_vqvae/joint" \
#     --max_steps 300000 \
#     --max_frames_num 300 \


# pose_vqvae_sep

python -m train_pose_vqvae \
    --gpus 8 --gpu_ids "0,1,2,3,4,5,6,7" \
    --init_lr 2e-4 \
    --embedding_dim 128 \
    --batchSize 12 \
    --n_codes 1024 \
    --data_path "Data/ProgressiveTransformersSLP" \
    --vocab_file "Data/ProgressiveTransformersSLP/src_vocab.txt" \
    --resume_ckpt "" \
    --default_root_dir "experiments/pose_vqvae/separate" \
    --max_steps 300000 \
    --max_frames_num 300 \