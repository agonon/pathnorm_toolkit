# This script can be used to generate the data for Figure 5 in https://arxiv.org/abs/2310.01225.
# It trains a ResNet on increasing subsets of ImageNet and save the path-norms and the accuracy at each epoch.

# Before use : set data_dir, saving_dir, batch_size and workers accordingly.


################ To change (see also below) ################
workers=16
batch_size=1024
############################################################

epochs=90
lr=0.1
wd=0.0001
lr_scheduler=multi-step
imp_iters=0

arch=resnet18 #resnet34 resnet50 resnet101 resnet152
for seed in 0 1 2
do
    for size_dataset in 39636 79272 158544 317089 634178 #99% of ImageNet = 1268355, divided by 2, 4, 8, 16, 32
    do
        ################ To change  ################
        data_dir=/path/to/ImageNet/
        saving_dir=results/4_train_increasing_dataset/seed=${seed}/${arch}/size_dataset=${size_dataset}/lr=${lr}_wd=${wd}_epochs=${epochs}_scheduler=${lr_scheduler}_percentage_pruning=${percentage_pruning}_imp_iters=${imp_iters}
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
            --no-data-parallel \
            --test-after-train \
            --size-dataset $size_dataset \
            --seed $seed
            # --evaluate-before-train \
    done
done
