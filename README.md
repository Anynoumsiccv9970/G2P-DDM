# G2P-DDM

This repository contains the implementation of our paper "G2P-DDM: Generating Sign Pose Sequence from Gloss Sequence with Discrete Diffusion Model".

# requirements

```bash
pip install -r requirements.txt
```
# Traning

## Stage 1: Pose-VQVAE for Reconstruction

```bash
bash train_pose_vqvae.sh
```

## Stage 2: Discrete Diffusion Model for Latent Prior Learning

```bash
bash train_text2pose.sh
```

# Inference

```
bash test_test2pose.sh
```

