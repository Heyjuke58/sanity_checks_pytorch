import torch
from torch.nn import BatchNorm2d, Conv2d, Linear
from torch.nn.modules.container import Sequential
from torchvision.models.resnet import BasicBlock
from captum.attr import IntegratedGradients, Saliency, InputXGradient, GuidedBackprop, DeepLift
import numpy as np
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import copy


def rand_layers(model, module_paths):
    for module_path in module_paths:
        cur = model
        for name in module_path:
            cur = getattr(cur, name)
        randomize(cur)


def randomize(layer):
    if isinstance(layer, (Conv2d, Linear, BatchNorm2d)):
        # use previous statistical values for the randomization of that specific layer
        std, mean = torch.std_mean(layer.weight)
        layer.weight = torch.nn.Parameter(torch.empty(layer.weight.size()).normal_(mean=mean.item(),std=std.item()))
    elif isinstance(layer, (BasicBlock, Sequential)):
        for child in layer.children():
            randomize(child)


def attribute_image_features(model, algorithm, input, label, **kwargs):
    model.zero_grad()
    tensor_attributions = algorithm.attribute(input, target=label, **kwargs)

    return tensor_attributions


# Given a NN model, ...
def visualize_saliency_method(saliency_kwargs, image, plt_fig_axis):
    image.requires_grad = True
    attrs = attribute_image_features(**saliency_kwargs)
    attrs = np.transpose(attrs.squeeze(0).cpu().detach().numpy(), (1,2,0))
    original_image = np.transpose(image.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    return viz.visualize_image_attr(attrs, original_image, method="heat_map",sign="absolute_value",
                          plt_fig_axis=plt_fig_axis, cmap="Blues", show_colorbar=False,
                          use_pyplot=False)


def get_kwargs(saliency_method, model, image, label):
    sal_kwargs = {
        'algorithm': saliency_method(model),
        'model': model,
        'input': image,
        'label': label,
    }

    if saliency_method == IntegratedGradients:
        pass
    elif saliency_method == Saliency:
        pass
    elif saliency_method == InputXGradient:
        pass
    elif saliency_method == GuidedBackprop:
        pass
    elif saliency_method == DeepLift:
        pass
    else:
        raise Exception("Saliency method not supported :(")

    return sal_kwargs


def visualize_cascading_randomization(model, module_paths, examples, saliency_method):
    model_copy = copy.deepcopy(model)

    # make plt plot
    nrows = len(examples)
    ncols = len(module_paths) + 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows))
    fig.subplots_adjust(hspace=0, wspace=0)

    # show input image at the very left
    for (image, _), row in zip(examples, range(nrows)):
        npimg = np.squeeze(np.squeeze(image.numpy()))
        axs[row, 0].imshow(npimg, cmap='gray')
        axs[row, 0].axis('off')

    # show visualizations before scrambling the model
    for (image, label), row in zip(examples, range(nrows)):
        pred = model_copy(image).argmax(axis=1).item()
        sal_kwargs = get_kwargs(saliency_method, model_copy, image, label)
        fig, _ = visualize_saliency_method(sal_kwargs, image, (fig, axs[row, 1]))

    # cascading randomization and visualization of IG
    # start with 1 because 0th column is unscrambled model
    for path, col in zip(module_paths, range(2, ncols)):
        rand_layers(model_copy, [path])
        for (image, label), row in zip(examples, range(nrows)):
            pred = model_copy(image).argmax(axis=1).item()
            sal_kwargs = get_kwargs(saliency_method, model_copy, image, label)
            fig, _ = visualize_saliency_method(sal_kwargs, image, (fig, axs[row, col]))

    # set titles for each column
    col_titles = ['input', 'normal model'] + [x for x in map((lambda x: '_'.join(x)), module_paths)]
    for ax, col in zip(axs[0], col_titles):
        ax.set_title(col)

    # set title for the whole thing
    fig.suptitle(saliency_method.__name__)

    return fig, axs


def visualize_cascading_randomization_single_example(model, module_paths, example, saliency_method):
    model_copy = copy.deepcopy(model)

    image, label = example

    # make plt plot
    ncols = len(module_paths) + 2
    fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(3*ncols, 3*1+0.5))
    fig.subplots_adjust(hspace=0, wspace=0)

    # show input image at the very left
    npimg = np.squeeze(np.squeeze(image.numpy()))
    axs[0].imshow(npimg, cmap='gray')
    axs[0].axis('off')

    # show visualizations before scrambling the model
    pred = model_copy(image).argmax(axis=1).item()
    sal_kwargs = get_kwargs(saliency_method, model_copy, image, label)
    fig, _ = visualize_saliency_method(sal_kwargs, image, (fig, axs[1]))

    # cascading randomization and visualization of IG
    # start with 1 because 0th column is unscrambled model
    for path, col in zip(module_paths, range(2, ncols)):
        rand_layers(model_copy, [path])
        pred = model_copy(image).argmax(axis=1).item()
        sal_kwargs = get_kwargs(saliency_method, model_copy, image, label)
        fig, _ = visualize_saliency_method(sal_kwargs, image, (fig, axs[col]))

    # set titles for each column
    col_titles = ['input', 'normal model'] + [x for x in map((lambda x: '_'.join(x)), module_paths)]
    for ax, col in zip(axs, col_titles):
        ax.set_title(col)

    # set title for the whole thing
    fig.suptitle(saliency_method.__name__)

    return fig, axs


def visualize_cascading_randomization2(model, module_paths, example, sal_methods, sal_method_names):
    model_copy = copy.deepcopy(model)
    image, label = example

    # make plt plot
    nrows = len(sal_methods)
    ncols = len(module_paths) + 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows))
    fig.subplots_adjust(hspace=0, wspace=0)

    # show input image at the very left
    for row in range(nrows):
        npimg = np.squeeze(np.squeeze(image.numpy()))
        axs[row, 0].imshow(npimg, cmap='gray')
        axs[row, 0].axis('on')
        axs[row, 0].set_xticks([])
        axs[row, 0].set_yticks([])

    # show visualizations before scrambling the model
    for sal_method, row in zip(sal_methods, range(nrows)):
        pred = model_copy(image).argmax(axis=1).item()
        sal_kwargs = get_kwargs(sal_method, model_copy, image, label)
        fig, _ = visualize_saliency_method(sal_kwargs, image, (fig, axs[row, 1]))

    # cascading randomization and visualization
    # start with 1 because 0th column is unscrambled model
    for path, col in zip(module_paths, range(2, ncols)):
        rand_layers(model_copy, [path])
        for sal_method, row in zip(sal_methods, range(nrows)):
            pred = model_copy(image).argmax(axis=1).item()
            sal_kwargs = get_kwargs(sal_method, model_copy, image, label)
            fig, _ = visualize_saliency_method(sal_kwargs, image, (fig, axs[row, col]))

    # set titles for each column
    col_titles = ['input', 'normal model'] + [x for x in map((lambda x: '_'.join(x)), module_paths)]
    for ax, col in zip(axs[0], col_titles):
        ax.set_title(col)

    # set titles for each row
    for ax, name in zip(axs[:,0], sal_method_names):
        ax.set_ylabel(name, rotation=90, size='large')

    # set title for the whole thing
    fig.suptitle("Cascading Randomization")

    return fig, axs