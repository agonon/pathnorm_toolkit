#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import os
import sys
import torch
from torchvision import models as tm
from torchvision.models.feature_extraction import create_feature_extractor
from torchinfo import summary


def get_all_layers(name: str, instances: list = []):
    """Return a dict of all the layers of the model.
    If instances is not empty, return all the layers
    that have instance in instances.

    Args: str
        Name of the model.

    Returns: tuple
        dict, int
    """
    match, n_layers = 0, 0
    data = {}
    data["layers"] = {}
    data["idx"] = {}

    # Loop over the modules and check each layer instance.
    # Count +1 match if ...
    for n, m in tm.get_model(name).named_modules():
        count = 0
        for nn, mm in m.named_children():
            count += 1
        if count > 0:
            # Store the parent
            tmp = n
        else:
            # Store each layer in a dict
            data["layers"][n] = {
                "parent": tmp,
                "name": n,
                "str": str(m),
                "instance": m,
                "suffix": n.replace(tmp, ""),
            }
            data["idx"][str(n_layers)] = n
            # Check instance of current layer
            match += int(
                isinstance(m, torch.nn.BatchNorm2d)
                or isinstance(m, torch.nn.Conv2d)
                or isinstance(m, torch.nn.MaxPool2d)
                or isinstance(m, torch.nn.AdaptiveAvgPool2d)
                or isinstance(m, torch.nn.ReLU)
                or isinstance(m, torch.nn.Linear)
                or isinstance(m, torch.nn.Dropout)
            )
            n_layers += 1

    return (data, match, n_layers)


def get_in_out_AdaptiveAvgPool2d(name: str):
    """For each AdaptiveAvgPool2d layer compute input and output shape.
    It also returns the number of channels.

    Args: name
        Name of the model.

    Returns: tuple
        Input (tuple), output (tuple) shape and number of channels.
    """
    # Input shape determined from first layer
    # shape = list(list(tm.get_model(name).parameters())[0].shape)
    # x = torch.rand(*shape)
    x = torch.rand(2, 3, 224, 224)
    model = tm.get_model(name)
    # tmp is one layer late
    tmp = None
    # Get all layers
    data, match, count = get_all_layers(name)
    # Loop over layers
    for k in data["layers"].keys():
        # print(name, k, data["layers"][k])
        if isinstance(
            data["layers"][k]["instance"], torch.nn.AdaptiveAvgPool2d
        ):
            # Compute out_shape
            eval_layer = tm.feature_extraction.create_feature_extractor(
                model,
                [data["layers"][k]["name"]],
                # train_return_nodes=[n],
                # eval_return_nodes=[n],
            )
            for e in eval_layer(x).keys():
                shape = eval_layer(x)[e].shape
                out_shape = (shape[2], shape[3])
            # Because tmp is one layer late we
            # use it to compute in_shape.
            if tmp in data["layers"].keys():
                print("one layer late", tmp, data["layers"][tmp])
                eval_layer = tm.feature_extraction.create_feature_extractor(
                    model,
                    [data["layers"][tmp]["name"]],
                    # train_return_nodes=[n],
                    # eval_return_nodes=[n],
                )
                for e in eval_layer(x).keys():
                    shape = eval_layer(x)[e].shape
                    # print(shape, shape[2], shape[3])
                    in_shape = (shape[2], shape[3])
                    in_channels = shape[1]
        # tmp is one layer late
        tmp = k

    return in_shape, out_shape, in_channels


def get_kernel_shape_AdaptiveAvgPool2d(in_d: tuple, out_d: tuple):
    # https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work/63603993#63603993
    strides = (int(in_d[0] / out_d[0]), int(in_d[1] / out_d[1]))
    kernel_shape = (
        in_d[0] - (out_d[0] - 1) * strides[0],
        in_d[1] - (out_d[1] - 1) * strides[1],
    )
    return strides, kernel_shape


def main():

    os.environ['TORCH_HOME'] = 'models'

    models = tm.list_models()

    ok_models = {}

    # train_nodes, eval_nodes = tm.feature_extraction.get_graph_node_names(
    #     tm.get_model("alexnet")
    # )
    # for i in train_nodes:
    #     print(i)
    # sys.exit()

    # name = 'resnet18'#'googlenet'#'regnet_x_16gf'
    # with open(name + '.out', 'w') as out_file:
    #     for n, m in tm.get_model(name).named_modules():
    #         n_childrens = 0
    #         for nn, mm in m.named_children():
    #             n_childrens += 1
    #         if n_childrens == 0:
    #             out_file.write(str(n) + ' ' + str(m) + '\n')

    # print(summary(tm.get_model('alexnet'), (1, 3, 224, 224)))
    # sys.exit()

    with open("not_ok_models.out", 'w') as out_file:
        pass
    with open("ok_models.out", 'w') as out_file:
        pass

    for name in models:
        if name == "googlenet":
            continue

        print("model={0:s}".format(name))

        data, match, count = get_all_layers(name)

        # Does the number of layers match layers path-norm toolkit handles ?
        if match != count:
            print(
                name,
                "model does not satisfy path-norm toolkit conditions."
            )
            with open("not_ok_models.out", 'a') as out_file:
                out_file.write(name + '\n')
            continue
        else:
            with open("ok_models.out", 'a') as out_file:
                out_file.write(name + '\n')
            ok_models[name] = {}
            data["valid"] = 1

        # Does model use MaxPool2d layer ?
        nAdaptiveAvgPool2d = 0
        nMaxPool2d = 0
        for k in range(count):
            adaptive_output_size = None
            kernel_size = None
            in_channels, out_channels = None, None
            layer_name = data["idx"][str(k)]
            if isinstance(
                data["layers"][layer_name]["instance"], torch.nn.MaxPool2d
            ):
                tmp_name = data['layers'][layer_name]['name']
                if name not in ok_models.keys():
                    ok_models[name] = {}
                if "MaxPool2d" not in ok_models[name].keys():
                    ok_models[name]["MaxPool2d"] = {}
                # It might be possible that the number of MaxPool2d per
                # network is more than one.
                if tmp_name not in ok_models[name]["MaxPool2d"].keys():
                    ok_models[name]["MaxPool2d"][tmp_name] = {}

                # Before MaxPool2d layer
                if k > 0:
                    for i in range(k - 1, -1, -1):
                        tmp = data["idx"][str(i)]
                        if data["layers"][tmp]["str"].startswith(
                            "BatchNorm2d"
                        ):
                            in_channels = data["layers"][tmp][
                                "instance"
                            ].num_features
                            break
                        if data["layers"][tmp]["str"].startswith(
                            "BasicConv2d"
                        ):
                            in_channels = data["layers"][tmp][
                                "instance"
                            ].conv.out_channels
                            # kernel_size = data["layers"][tmp][
                            #     "instance"
                            # ].conv.kernel_size
                            break
                        if data["layers"][tmp]["str"].startswith("Conv2d"):
                            in_channels = data["layers"][tmp][
                                "instance"
                            ].out_channels
                            # kernel_size = data["layers"][tmp][
                            #     "instance"
                            # ].kernel_size
                            break
                        if data["layers"][tmp]["str"].startswith(
                            "FrozenBatchNorm2d"
                        ):
                            in_channels = data["layers"][tmp][
                                "instance"
                            ].weight.shape[0]
                            break
                        if data["layers"][tmp]["str"].startswith(
                            "AdaptiveAvgPool2d"
                        ):
                            in_channels = data["layers"][tmp][
                                "instance"
                            ].output_size
                            print(dir(data["layers"][tmp]["instance"]))
                            break

                # Current MaxPool2d layer
                print(
                    "current",
                    data["idx"][str(k)],
                    data["layers"][layer_name]["instance"],
                )

                # After MaxPool2d layer
                for i in range(k + 1, count, 1):
                    tmp = data["idx"][str(i)]
                    if data["layers"][tmp]["str"].startswith("BatchNorm2d"):
                        out_channels = data["layers"][tmp][
                            "instance"
                        ].num_features
                        break
                    if data["layers"][tmp]["str"].startswith("BasicConv2d"):
                        out_channels = data["layers"][tmp][
                            "instance"
                        ].conv.in_channels
                        # kernel_size = data["layers"][tmp][
                        #     "instance"
                        # ].conv.kernel_size
                        break
                    if data["layers"][tmp]["str"].startswith("Conv2d"):
                        out_channels = data["layers"][tmp][
                            "instance"
                        ].in_channels
                        # kernel_size = data["layers"][tmp][
                        #     "instance"
                        # ].kernel_size
                        break
                    if data["layers"][tmp]["str"].startswith(
                        "FrozenBatchNorm2d"
                    ):
                        out_channels = data["layers"][tmp][
                            "instance"
                        ].weight.shape[0]
                        break
                    if data["layers"][tmp]["str"].startswith("AdaptiveAvgPool2d"):
                        in_d, out_d, out_channels = get_in_out_AdaptiveAvgPool2d(name)
                        strides, kernel_shape = get_kernel_shape_AdaptiveAvgPool2d(in_d, out_d)
                        break

                # Because of model reset (from compute path norm)
                # store parameters of MaxPool2d layer.
                ok_models[name]["MaxPool2d"][tmp_name] = {
                    "name": data["idx"][str(k)],
                    # "kernel_size": kernel_size,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "number": nMaxPool2d,
                    "reset": {
                        "kernel_size": data["layers"][layer_name]["instance"].kernel_size,
                        "stride": data["layers"][layer_name]["instance"].stride,
                        "padding": data["layers"][layer_name]["instance"].padding,
                        "dilation": data["layers"][layer_name]["instance"].dilation
                    }
                }
                with open("ok_models.json", "w") as out_file:
                    json.dump(ok_models, out_file, indent=1)

                nMaxPool2d += 1

        # Does model use AdaptiveAvgPool2d layer ?
        for k in range(count):
            layer_name = data["idx"][str(k)]
            if isinstance(
                data["layers"][layer_name]["instance"],
                torch.nn.AdaptiveAvgPool2d,
            ):
                if name not in ok_models.keys():
                    ok_models[name] = {}
                if "AdaptiveAvgPool2d" not in ok_models[name].keys():
                    ok_models[name]["AdaptiveAvgPool2d"] = {}
                # It might be possible that the number of AdaptiveAvgPool2d per
                # network is more than one.
                if tmp_name not in ok_models[name]["AdaptiveAvgPool2d"].keys():
                    ok_models[name]["AdaptiveAvgPool2d"][tmp_name] = {}

                # Current AdaptiveAvgPool2d layer
                print(
                    "current",
                    data["idx"][str(k)],
                    data["layers"][layer_name]["instance"],
                )

                in_d, out_d, in_channels = get_in_out_AdaptiveAvgPool2d(name)
                strides, kernel_shape = get_kernel_shape_AdaptiveAvgPool2d(in_d, out_d)
                ok_models[name]["AdaptiveAvgPool2d"][
                    tmp_name
                ] = {
                    "name": data["idx"][str(k)],
                    "in_channels": in_channels,
                    "in": in_d,
                    "out": out_d,
                    "strides": strides,
                    "kernel_shape": kernel_shape,
                    "integer_multiple": (in_d[0] % out_d[0]) == 0
                    and (in_d[1] % out_d[1]) == 0,
                    "number": nAdaptiveAvgPool2d,
                }

                with open("ok_models.json", "w") as out_file:
                    json.dump(ok_models, out_file, indent=1)

                nAdaptiveAvgPool2d += 1


if __name__ == "__main__":
    main()  # sys.argv[1:])
