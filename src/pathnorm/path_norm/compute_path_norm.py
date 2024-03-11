import torch
from torch.nn.utils import prune

import copy
import torch


def replace_maxpool2d_with_conv2d(
    list_in_out_channels, model, device, in_place=False
):
    """
    Replace max-pooling layers with convolutional layers with weights constant
    equal to one. Works with the usual ResNet PyTorch class, and it is expected
    to be easily adaptable to work with other architectures.

    Args:
        list_in_out_channels (list): List of tuples representing input and
        output channels of each max-pooling layer being replaced.
        model (torch.nn.Module): Input model.
        device (torch.device): Device on which the model should be placed.
        in_place (bool, optional): If True, modify the input model in place.
        If False, create a deep copy of the model and modify the copy.
        Defaults to False.

    Returns:
        torch.nn.Module: Modified model.
    """
    if in_place is False:
        try:
            new_model = copy.deepcopy(model)
        except Exception as e:
            print(e)
            raise RuntimeError(
                "Deep copy failed, set 'in_place=True' to run the function with in place modification of the model"
            )
    else:
        new_model = model
    i = 0
    for n, m in new_model.named_modules():
        if isinstance(m, torch.nn.MaxPool2d):
            new_model._modules[n] = torch.nn.Conv2d(
                in_channels=list_in_out_channels[i][0],
                out_channels=list_in_out_channels[i][1],
                kernel_size=m.kernel_size,
                stride=m.stride,
                padding=m.padding,
                padding_mode="zeros",
                dilation=m.dilation,
                groups=list_in_out_channels[i][0],
                bias=False,
                device=device,
            )
            new_model._modules[n].weight.data.fill_(1)
            nb_param = 0
            for p in new_model._modules[n].parameters():
                nb_param += p.numel()
            i += 1
    return new_model


def set_weights_for_path_norm(
    model, exponent=1, provide_original_weights=True
):
    """
    Applies $w\\mapsto |w|^{exponent}$ to all weights $w$ of the model
    for path-norm computation. It handles cases where the weights of the
    convolutional and linear layers are pruned with the torch.nn.utils.prune
    library (but not when their biases are pruned).

    Args:
        model (torch.nn.Module): Input model.
        exponent (float, optional): Exponent for weight transformation.
        Defaults to 1.
        provide_original_weights (bool, optional): If True, provide the
        original weights for resetting later. Defaults to True.

    Returns:
        dict: Original weights of the model if
        `provide_original_weights` is True; otherwise, empty dict.
    """
    # If a module is pruned, its original weights are in weight_orig instead of weight
    orig_weights = {}
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            if prune.is_pruned(m):
                if provide_original_weights:
                    orig_weights[n + ".weight"] = (
                        m.weight_orig.detach().clone()
                    )
                m.weight_orig.data = torch.abs(m.weight_orig.detach())
                if exponent != 1:
                    m.weight_orig.data = torch.pow(
                        m.weight_orig.detach(), exponent
                    )
            else:
                if provide_original_weights:
                    orig_weights[n + ".weight"] = m.weight.detach().clone()
                m.weight.data = torch.abs(m.weight.detach())
                if exponent != 1:
                    m.weight.data = torch.pow(m.weight.detach(), exponent)
            if m.bias is not None:
                if provide_original_weights:
                    orig_weights[n + ".bias"] = m.bias.detach().clone()
                m.bias.data = torch.abs(m.bias.detach())
                if exponent != 1:
                    m.bias.data = torch.pow(m.bias.detach(), exponent)
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data = torch.abs(m.weight.detach())
            m.bias.data = torch.abs(m.bias.detach())
            m.running_mean.data = torch.abs(m.running_mean.detach())

            if provide_original_weights:
                orig_weights[n + ".weight"] = m.weight.detach().clone()
                orig_weights[n + ".bias"] = m.bias.detach().clone()
                orig_weights[n + ".running_mean"] = (
                    m.running_mean.detach().clone()
                )
                orig_weights[n + ".running_var"] = (
                    m.running_var.detach().clone()
                )

            if exponent != 1:
                m.weight.data = torch.pow(m.weight.detach(), exponent)
                m.bias.data = torch.pow(m.bias.detach(), exponent)
                m.running_mean.data = torch.pow(
                    m.running_mean.detach(), exponent
                )
                m.running_var.data = torch.pow(
                    m.running_var.detach(), exponent
                )
    return orig_weights


def reset_model(arch, model, orig_weights, in_place=False):
    """
    Reset weights and maxpool layer of a ResNet.

    Args:
        arch (str): Architecture of the model. Should be one of 'resnet18',
        'resnet34', 'resnet50', 'resnet101', 'resnet152'.
        model (torch.nn.Module): Input model.
        orig_weights (dict): Original weights of the model.
        in_place (bool, optional): If True, modify the input model in place.
        Defaults to False.
    """
    if in_place:
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(
                m, torch.nn.Linear
            ):
                if prune.is_pruned(m):
                    m.weight_orig.data = orig_weights[n + ".weight"]
                else:
                    m.weight.data = orig_weights[n + ".weight"]
                if m.bias is not None:
                    m.bias.data = orig_weights[n + ".bias"]
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data = orig_weights[n + ".weight"]
                m.bias.data = orig_weights[n + ".bias"]
                m.running_mean.data = orig_weights[n + ".running_mean"]
                m.running_var.data = orig_weights[n + ".running_var"]

        if (
            arch == "resnet18"
            or arch == "resnet34"
            or arch == "resnet50"
            or arch == "resnet101"
            or arch == "resnet152"
        ):
            model.maxpool = torch.nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                ceil_mode=False,
            )
        else:
            raise NotImplementedError


def get_path_norm(
    model,
    arch,
    device,
    exponent=1,
    constant_input=1,
    in_place=True,
):
    """
    Get the $L^{\textrm{exponent}}$ path-norm of a ResNet model. It does so by
    applying the following recipe given by Theorem A.1 of
    https://arxiv.org/abs/2310.01225.
    1. Modify the model (in_place or not): replace max-pooling-neurons with
    identity ones.
    2. Set all the weights $w$ to $|w|^{\textrm{exponent}}$.
    3. Compute the path norm of the model with a forward-pass with constant
    input equal to one.
    4. Reset the model to its original state.

    Args:
        model (torch.nn.Module): Input model.
        arch (str): Architecture of the model. Should be one of 'resnet18',
        'resnet34', 'resnet50', 'resnet101', 'resnet152'.
        device (torch.device): Device on which the model should be placed.
        exponent (float, optional): Exponent for weight transformation.
        Defaults to 1.
        constant_input (float, optional): Constant input value. Defaults to 1.
        in_place (bool, optional): If True, modify the input model in place.
        Defaults to True.

    Returns:
        float: Path-norm of the model.
    """

    # NOTA BENE: if pruned, we do not remove the pruned framework, and then
    # reapply it afterwards by pruning amount=0 # since this would lose the
    # masks

    if (
        arch == "resnet18"
        or arch == "resnet34"
        or arch == "resnet50"
        or arch == "resnet101"
        or arch == "resnet152"
    ):
        number_antecedents_average_pool = 7 * 7
        weights_average_pool = 1 / number_antecedents_average_pool
        list_in_out_channels = [(64, 64)]
    else:
        raise ValueError(
            "arch should be one of resnet18, resnet34, resnet50, resnet101, resnet152"
        )

    # 1. Modify the model (in_place or not): replace max-pooling-neurons with identity ones (as prescribed in Theorem A.1 of https://arxiv.org/abs/2310.01225).

    new_model = replace_maxpool2d_with_conv2d(
        list_in_out_channels, model, device, in_place=in_place
    )

    # 2. Set all the weights w to |w|^exponent.
    orig_weights = set_weights_for_path_norm(
        new_model, exponent=exponent, provide_original_weights=in_place
    )

    # 3. Compute the path norm of the model with constant input equal to one.

    new_model.eval()
    batch_size = 1
    input_size = (batch_size, 3, 224, 224)
    x = constant_input * torch.ones(input_size)
    x = x.to(device)
    with torch.no_grad():
        path_norm = torch.pow(new_model(x).sum(), 1 / exponent).item()
        path_norm *= weights_average_pool ** (exponent - 1)

    # 4. Reset the model to its original state.
    reset_model(arch, model, orig_weights, in_place=in_place)
    return path_norm


def compute_path_norms(model, arch, exponents, device, data_parallel):
    """
    Compute path-norms for different exponents of a ResNet model.

    Args:
        model (torch.nn.Module): The ResNet model.
        arch (str): Architecture of the model. Should be one of 'resnet18',
        'resnet34', 'resnet50', 'resnet101', 'resnet152'.
        exponents (list): List of exponents for computing path-norms.
        device (torch.device): Device on which the model should be placed.
        data_parallel (bool): If True, the model is parallelized using
        DataParallel. Currently not supported for path-norm computation.

    Returns:
        list: List of path-norms corresponding to different exponents.
    Raises:
        ValueError: If data_parallel is set to True. Data parallelism is not
        supported yet for the computation of path-norm. A special treatment
        would be needed to replace the maxpool layer with a conv layer.
    """

    path_norms = []

    if data_parallel:
        raise ValueError(
            "Data parallelism is not supported yet for the computation of path-norm. A special treatment would be needed to replace the maxpool layer with a conv layer."
        )

    for exponent in exponents:
        path_norm = get_path_norm(
            model,
            arch,
            device,
            exponent=exponent,
        )
        path_norms.append(path_norm)

    return path_norms
