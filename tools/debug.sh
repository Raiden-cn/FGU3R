#/bin/bash
CUDA_VISIBLE_DEVICES=$1 python train.py --cfg_file cfgs/kitti_models/pseudorcnn_shareconv.yaml \
--extra_tag debug --batch_size 1