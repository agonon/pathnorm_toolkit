# This script can be used to reproduce Tables 3, 5 and Figure 3 in https://arxiv.org/abs/2310.01225.
# It computes the path-norms and margins of pre-trained ResNets on ImageNet.

# Below, we use the pre-computed margins on ImageNet.
# If you wish to re-compute the margins:
# 1. Specify another saving_dir, where the margins will be saved.
# 2. Set data_dir, batch_size, workers and uncomment the corresponding arguments.
# 3. Comment the argument --margins-already-computed and put it as the last line of the script to make the bash script correct.


################ To change if not using pre-computed margins ################
saving_dir=results/1_compute_pretrained_pathnorm_margins_resnets/
data_dir=/path/to/ImageNet/
batch_size=1024
workers=16
#############################################################################


python3 src/pathnorm/path_norm/pathnorm_and_margins_pretrained_resnets.py \
    --saving-dir $saving_dir \
    --data-dir $data_dir \
    --batch-size $batch_size \
    --workers $workers \
    --margins-already-computed \
