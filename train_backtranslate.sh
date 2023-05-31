# python -m train_text2pose \
#     --gpus 1 \
#     --batchSize 5 \
#     --data_path "Data/how2sign" \
#     --text_path "data/text2gloss/" \
#     --vocab_file "data/text2gloss/how2sign_vocab.txt" \
#     --pose_vqvae "logs/phoneix_spl_seperate_SeqLen_1/lightning_logs/version_3/checkpoints/epoch=123-step=50095.ckpt" \
#     --hparams_file "logs/phoneix_spl_seperate_SeqLen_1/lightning_logs/version_3/hparams.yaml" \
#     --resume_ckpt "" \
#     --default_root_dir "text2pose_logs/test" \
#     --max_steps 300000 \
#     --max_frames_num 200 \
#     --gpu_ids "0" \



python -m train_backtranslate \
    --gpus 4 --gpu_ids "4,5,6,7" \
    --batchSize 2 \
    --data_path "Data/ProgressiveTransformersSLP" \
    --vocab_file "Data/ProgressiveTransformersSLP/src_vocab.txt" \
    --resume_ckpt "" \
    --default_root_dir "experiments/backmodel" \
    --max_steps 300000 \
    --max_frames_num 300 \
    --num_workers 32 \
    