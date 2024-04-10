# This script can be used to generate the data for Figure 4 in https://arxiv.org/abs/2310.01225.
# It trains a ResNet on ImageNet for 20 iterations of magnitude pruning and rewind, and save the path-norms and the accuracy at each epoch.

# Before use : set data_dir, saving_dir, batch_size and workers accordingly.

epochs=90
lr=0.1
wd=0.0001
lr_scheduler=multi-step
imp_iters=2
percentage_pruning=0.2
start_imp_iter=0
seed=0
arch=resnet18 #resnet34 resnet50 resnet101 resnet152

################ To change  ################
workers=16
batch_size=1024
data_dir=/path/to/ImageNet/
saving_dir=results/2_train_imp/seed=${seed}/${arch}/lr=${lr}_wd=${wd}_epochs=${epochs}_scheduler=${lr_scheduler}_percentage_pruning=${percentage_pruning}_imp_iters=${imp_iters}
############################################


python3 src/pathnorm/train_imagenet.py $data_dir \
    --arch $arch \
    --epochs $epochs \
    --workers $workers \
    --batch-size $batch_size \
    --lr $lr \
    --wd $wd \
    --lr-scheduler $lr_scheduler \
    --tensorboard \
    --saving-dir $saving_dir \
    --IMP-iters $imp_iters \
    --percentage-pruning $percentage_pruning \
    --no-data-parallel \
    --test-after-train \
    --seed $seed \
    --start-IMP-iter $start_imp_iter \
    # --evaluate-before-train \
    # --resume $resume_path/checkpoint.pth.tar \
