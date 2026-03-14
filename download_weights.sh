#!/bin/bash
#
# Download pretrained weigths
mkdir -p /home/renku/work/isles24/data-local/pretrained
wget -O /home/renku/work/isles24/data-local/pretrained/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt \
https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt