#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 0-00:10:00
##SBATCH -a 1-350%350
#SBATCH --gres gpu:1
#SBATCH -o logs/inference-%j.out
#SBATCH --signal=B:SIGTERM@120

echo "Available GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Working Directory: $PWD"


CONDAPATH_0=/HPS/prao2/work/anaconda3/envs/eg3d/bin/python
CONDAPATH_1=/CT/LS_FRM/work/miniconda3/envs/lite2relight/bin/python
CONDAPATH_2=/CT/LS_FRM01/work/miniforge3/envs/3dpr/bin/python

## Inference
OUTDIR=results/release/

CKPT_DIR=checkpoints/
MODE=AFA
ID=ID00600

$CONDAPATH_2 infer_relit.py --relight --multi_view \
--data sample/dataset/${ID}/ \
--G_ckpt=/CT/VORF_GAN3/work/code/goae-inversion-photoapp/pretrained_models/ffhqrebalanced512-128.pkl \
--E_ckpt=/CT/VORF_GAN3/work/code/goae-inversion-photoapp/pretrained_models/encoder_FFHQ.pt \
--AFA_ckpt=/CT/VORF_GAN3/work/code/goae-inversion-photoapp/pretrained_models/afa_FFHQ.pt \
--R_ckpt=${CKPT_DIR}/network-snapshot-000027.pkl \
--outdir ${OUTDIR} \
--fix_density --num_emaps=10 \
--emap_mode='emaps_ds' \
# --edit --edit_attr='age' --alpha=0.95 \