kill $(ps aux | grep "/opt/tiger/start/hold_gpu.py" | grep -v grep | awk '{print $2}')

python -m train_text2pose --gpus 8 --gpu_ids "0,1,2,3,4,5,6,7" \
    --stage2_model "configs/stage2_model/vq_diffusion_codeunet.yaml"  \
    --default_root_dir "experiments/text2pose/test"

# cd /opt/tiger/start/ && bash hold_gpu.sh
