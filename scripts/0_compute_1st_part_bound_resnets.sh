# This script can be used to reproduce Table 2 in https://arxiv.org/abs/2310.01225.
# It computes the first part of the bound for ResNet architectures on ImageNet.

# Below, we use the pre-computed value of B := max L^\infty norm of the input images in ImageNet,
# normalized for inference. If you wish to re-compute B:
# 1. Set data_dir, batch_size, workers.
# 2. Comment the argument --B.



################ To change if not using pre-computed B ################
data_dir=/path/to/ImageNet/
batch_size=1024
workers=16
#######################################################################


python3 src/pathnorm/path_norm/compute_1st_part_bound_resnets.py \
    --data-dir $data_dir \
    --batch-size $batch_size \
    --workers $workers \
    --B 2.640000104904175 # comment to first re-compute B