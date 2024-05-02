import json
import os
import torch
from torchvision import models as tm
from pathnorm.path_norm.compute_path_norm import get_path_norm


def main():
    os.environ['TORCH_HOME'] = 'models'
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Loop over the models that are compatible with path-norm
    with open("ok_models.json", 'r') as in_file:
        data = json.load(in_file)
    # from torchinfo import summary
    for k in data.keys():
        print("Compute path-norm of {0:s}".format(k))
        if k == 'alexnet':
            model = tm.alexnet(weights=tm.AlexNet_Weights.IMAGENET1K_V1).to(device)
        elif k == "deeplabv3_resnet101":
            model = tm.segmentation.deeplabv3_resnet101(weights=tm.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT).to(device)
        elif k == "deeplabv3_resnet50":
            model = tm.segmentation.deeplabv3_resnet50(weights=tm.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT).to(device)
        elif k == "fcn_resnet101":
            model = tm.segmentation.fcn_resnet101(weights=tm.segmentation.FCN_ResNet101_Weights.DEFAULT).to(device)
        elif k == "fcn_resnet50":
            model = tm.segmentation.fcn_resnet50(weights=tm.segmentation.FCN_ResNet50_Weights.DEFAULT).to(device)
        elif k == "inception_v3":
            model = tm.inception_v3(weights=tm.Inception_V3_Weights.IMAGENET1K_V1).to(device)
        elif k == "mnasnet0_5":
            model = tm.mnasnet0_5(weights=tm.MNASNet0_5_Weights.IMAGENET1K_V1).to(device)
        elif k == "mnasnet0_75":
            model = tm.mnasnet0_75(weights=tm.MNASNet0_75_Weights.IMAGENET1K_V1).to(device)
        elif k == "mnasnet1_0":
            model = tm.mnasnet1_0(weights=tm.MNASNet1_0_Weights.IMAGENET1K_V1).to(device)
        elif k == "mnasnet1_3":
            model = tm.mnasnet1_3(weights=tm.MNASNet1_3_Weights.IMAGENET1K_V1).to(device)
        elif k == "regnet_x_16gf":
            model = tm.regnet_x_16gf(weights=tm.RegNet_X_16GF_Weights.IMAGENET1K_V1).to(device)
        elif k == "regnet_x_1_6gf":
            model = tm.regnet_x_1_6gf(weights=tm.RegNet_X_1_6GF_Weights.IMAGENET1K_V1).to(device)
        elif k == "regnet_x_32gf":
            model = tm.regnet_x_32gf(weights=tm.RegNet_X_32GF_Weights.IMAGENET1K_V1).to(device)
        elif k == "regnet_x_3_2gf":
            model = tm.regnet_x_3_2gf(weights=tm.RegNet_X_3_2GF_Weights.IMAGENET1K_V1).to(device)
        elif k == "regnet_x_400mf":
            model = tm.regnet_x_400mf(weights=tm.RegNet_X_400MF_Weights.IMAGENET1K_V1).to(device)
        elif k == "regnet_x_800mf":
            model = tm.regnet_x_800mf(weights=tm.RegNet_X_800MF_Weights.IMAGENET1K_V1).to(device)
        elif k == "regnet_x_8gf":
            model = tm.regnet_x_8gf(weights=tm.RegNet_X_8GF_Weights.IMAGENET1K_V1).to(device)
        elif k == 'resnet18':
            model = tm.resnet18(weights=tm.ResNet18_Weights.IMAGENET1K_V1).to(device)
        elif k == 'resnet34':
            model = tm.resnet34(weights=tm.ResNet34_Weights.IMAGENET1K_V1).to(device)
        elif k == 'resnet50':
            model = tm.resnet50(weights=tm.ResNet50_Weights.IMAGENET1K_V2).to(device)
        elif k == 'resnet101':
            model = tm.resnet101(weights=tm.ResNet101_Weights.IMAGENET1K_V2).to(device)
        elif k == 'resnet152':
            model = tm.resnet152(weights=tm.ResNet152_Weights.IMAGENET1K_V2).to(device)
        elif k == 'resnext101_32x8d':
            model = tm.resnext101_32x8d(weights=tm.ResNeXt101_32X8D_Weights.IMAGENET1K_V1).to(device)
        elif k == 'resnext101_64x4d':
            model = tm.resnext101_64x4d(weights=tm.ResNeXt101_64X4D_Weights.IMAGENET1K_V1).to(device)
        elif k == 'resnext50_32x4d':
            model = tm.resnext50_32x4d(weights=tm.ResNeXt50_32X4D_Weights.IMAGENET1K_V1).to(device)
        elif k == 'squeezenet1_0':
            model = tm.squeezenet1_0(weights=tm.SqueezeNet1_0_Weights.IMAGENET1K_V1).to(device)
        elif k == 'squeezenet1_1':
            model = tm.squeezenet1_1(weights=tm.SqueezeNet1_1_Weights.IMAGENET1K_V1).to(device)
        elif k == 'vgg11':
            model = tm.vgg11(weights=tm.VGG11_Weights.IMAGENET1K_V1).to(device)
        elif k == 'vgg11_bn':
            model = tm.vgg11_bn(weights=tm.VGG11_BN_Weights.IMAGENET1K_V1).to(device)
        elif k == 'vgg13':
            model = tm.vgg13(weights=tm.VGG13_Weights.IMAGENET1K_V1).to(device)
        elif k == 'vgg13_bn':
            model = tm.vgg13_bn(weights=tm.VGG13_BN_Weights.IMAGENET1K_V1).to(device)
        elif k == 'vgg16':
            model = tm.vgg16(weights=tm.VGG16_Weights.IMAGENET1K_V1).to(device)
        elif k == 'vgg16_bn':
            model = tm.vgg16_bn(weights=tm.VGG16_BN_Weights.IMAGENET1K_V1).to(device)
        elif k == 'vgg19':
            model = tm.vgg19(weights=tm.VGG19_Weights.IMAGENET1K_V1).to(device)
        elif k == 'vgg19_bn':
            model = tm.vgg19_bn(weights=tm.VGG19_BN_Weights.IMAGENET1K_V1).to(device)
        elif k == 'wide_resnet101_2':
            model = tm.wide_resnet101_2(weights=tm.Wide_ResNet101_2_Weights.IMAGENET1K_V1).to(device)
        elif k == 'wide_resnet50_2':
            model = tm.wide_resnet50_2(weights=tm.Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device)
        else:
            raise NotImplementedError
        print("L1-path-norm = {0:f}".format(get_path_norm(model, k, device, exponent=1, in_place=True)))


if __name__ == "__main__":
    main()
