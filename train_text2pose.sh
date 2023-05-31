kill $(ps aux | grep "/opt/tiger/start/hold_gpu.py" | grep -v grep | awk '{print $2}')

python -m train_text2pose --gpus 4 --gpu_ids "0,1,2,3" \
    --stage2_model "configs/stage2_model/vq_diffusion_codeunet.yaml"  \
    --default_root_dir "experiments/text2pose/vq_diffusion_codeunet"

# python -m train_text2pose --gpus 4 --gpu_ids "0,1,2,3" \
#     --stage2_model "configs/stage2_model/vq_diffusion.yaml"  \
#     --default_root_dir "experiments/text2pose/vq_diffusion"

# python -m train_text2pose --gpus 4 --gpu_ids "4,5,6,7" \
#     --stage2_model "configs/stage2_model/vq_cold_diffusion.yaml"  \
#     --default_root_dir "experiments/text2pose/vq_cold_diffusion"


# python -m train_text2pose --gpus 4 --gpu_ids "0,1,2,3" \
#     --stage2_model "configs/stage2_model/mask_predict.yaml"  \
#     --default_root_dir "experiments/text2pose/mask_predict"  

cd /opt/tiger/start/ && bash hold_gpu.sh
