# This script can be used to plot Figure 5 in https://arxiv.org/abs/2310.01225.
# It plots the accuracy and the path-norms of a ResNet trained on increasing subsets of ImageNet.

# Below, we use pre-computed data corresponding to the training of a resnet18
# on ImageNet for increasing training set sizes.
# If you wish to re-train the model, first run the script 4_train_increasing_dataset.sh,
# second, run this script with results_training_dir set to the saving_dir used in 4_train_increasing_dataset.sh.

num_epochs=90
num_seeds=3

################ To change if not using pre-computed data ################
# The following variables are only used to compute the paths
lr=0.1
wd=0.0001
lr_scheduler=multi-step
imp_iters=0
arch=resnet18
size_dataset=0 # arbitrary integer, will be replaced by the different sizes used in 4_train_increasing_dataset.sh when plotting
seed=0 # arbitrary integer, will be replaced by the different seeds used in 4_train_increasing_dataset.sh when plotting
percentage_pruning=""
# Computing the paths
results_training_dir=results/4_train_increasing_dataset/seed=${seed}/${arch}/size_dataset=${size_dataset}/lr=${lr}_wd=${wd}_epochs=${num_epochs}_scheduler=${lr_scheduler}_percentage_pruning=${percentage_pruning}_imp_iters=${imp_iters}
saving_dir=results/5_plot_increasing_dataset/num_seeds=${num_seeds}/${arch}/lr=${lr}_wd=${wd}_epochs=${num_epochs}_scheduler=${lr_scheduler}_percentage_pruning=${percentage_pruning}_imp_iters=${imp_iters}
##########################################################################

python3 src/pathnorm/plot_increasing_dataset.py \
    --num-epochs $num_epochs \
    --num-seeds $num_seeds \
    --saving-dir $saving_dir \
    --results-training-dir $results_training_dir \
