# This script can be used to plot Figure 4 in https://arxiv.org/abs/2310.01225.
# It plots the accuracy and the path-norms of a ResNet trained on ImageNet for each epoch of 20 iterations of magnitude pruning.

# Below, we use pre-computed data corresponding to the training of a resnet18
# on ImageNet for 20 iterations of magnitude pruning.
# If you wish to re-train the model, first run the script 2_train_imp.sh,
# second, run this script with results_training_dir set to the saving_dir used in 2_train_imp.sh.

num_seeds=1

################ To change iif not using pre-computed data ################
# The following variables are only used to compute the paths
arch=resnet18
epochs=90
lr=0.1
wd=0.0001
lr_scheduler=multi-step
percentage_pruning=0.2
imp_iters=20
seed=0 # arbitrary integer, will be replaced by the different seeds used in 2_train_imp.sh when plotting
# Computing the paths
results_training_dir=results/2_train_imp/seed=${seed}/${arch}/lr=${lr}_wd=${wd}_epochs=${epochs}_scheduler=${lr_scheduler}_percentage_pruning=${percentage_pruning}_imp_iters=${imp_iters}/
saving_dir=results/3_plot_imp/num_seeds=${num_seeds}/${arch}/lr=${lr}_wd=${wd}_epochs=${num_epochs}_scheduler=${lr_scheduler}_percentage_pruning=${percentage_pruning}_imp_iters=${imp_iters}/
##########################################################################

python3 src/pathnorm/plot_imp.py \
    --num-seeds $num_seeds \
    --saving-dir $saving_dir \
    --results-training-dir $results_training_dir
