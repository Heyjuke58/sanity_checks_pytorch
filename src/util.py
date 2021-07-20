import torch
from torch.nn import BatchNorm2d, Conv2d, Linear
from torch.nn.modules.container import Sequential
from torchvision.models.resnet import BasicBlock
from captum.attr import IntegratedGradients, Saliency, InputXGradient, GuidedBackprop, DeepLift, NoiseTunnel
import numpy as np
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import copy
from skimage.measure import compare_ssim
from typing import Tuple, Any

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# visualization stuff
TIGHT_LAYOUT_PARAMS = {
        "pad": 0,
        "h_pad": 0,
        "w_pad": 0,
        "rect": (0, 0, 1, 1)
    }

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
def visualize_saliency_method(saliency_kwargs, image, plt_fig_axis, viz_method):
    image.requires_grad = True
    attrs = attribute_image_features(**saliency_kwargs)
    attrs = np.transpose(attrs.squeeze(0).cpu().detach().numpy(), (1,2,0))
    original_image = np.transpose(image.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    return viz.visualize_image_attr(attrs, original_image, method=viz_method, sign="absolute_value",
                          plt_fig_axis=plt_fig_axis, cmap="Reds", show_colorbar=False,
                          use_pyplot=False, alpha_overlay=0.9)


def get_kwargs(saliency_method: Tuple[Any, bool], model, image, label):
    saliency_method, noise_tunnelling = saliency_method
    sal_kwargs = { # these are the default args for attribute_image_features
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

    if noise_tunnelling:
        sal_kwargs.update({
            'algorithm': NoiseTunnel(sal_kwargs['algorithm']),
            'nt_type': 'smoothgrad', # unsure about this one, see p.3 (SmoothGrad) in paper
            'nt_samples': 50, # from their code
            'stdevs': 0.15 # from their code
        })

    return sal_kwargs


# topn: How many top prediction results should be printed out for each image
def visualize_cascading_randomization(model, module_paths, saliency_method, examples, originals=None, cls_index_to_name=None, topn=1, viz_method="heat_map"):
    model_copy = copy.deepcopy(model)

    # make plt plot
    nrows = len(examples)
    ncols = len(module_paths) + 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows), tight_layout=TIGHT_LAYOUT_PARAMS)

    # show input image at the very left
    for (original, label), (image, _), row in zip(examples if not originals else originals, examples, range(nrows)):
        npimg = original.squeeze(0).permute(1, 2, 0).numpy()
        axs[row, 0].imshow(npimg, cmap='gray')
        axs[row, 0].axis('on')
        axs[row, 0].set_xticks([])
        axs[row, 0].set_yticks([])
        # show true label for each row
        # probabilities = torch.nn.functional.softmax(output[0], dim=0)
        # # Read the categories
        # with open("imagenet_classes.txt", "r") as f:
        #     categories = [s.strip() for s in f.readlines()]
        # # Show top categories per image
        # top5_prob, top5_catid = torch.topk(probabilities, 5)
        # for i in range(top5_prob.size(0)):
        #     print(categories[top5_catid[i]], top5_prob[i].item())
        top_probs, top_idxs = torch.topk(torch.nn.functional.softmax(model(image)[0], dim=0), topn)
        if cls_index_to_name:
            # lookup title name
            title = cls_index_to_name[label.item()]
            preds = "\n".join([f"{cls_index_to_name[top_idx.item()]} (p={top_prob:.2f})" for top_idx, top_prob in zip(top_idxs, top_probs)])
        else:
            title = str(label.item())
            preds = "\n".join([f"{str(top_idx.item())} (p={top_prob:.2f})" for top_idx, top_prob in zip(top_idxs, top_probs)])
        # pred = model(image)
        axs[row, 0].set_ylabel("true: " + title + "\npreds: " + str(preds), rotation=90, size='large')


    # show visualizations before scrambling the model
    for (image, label), row in zip(examples, range(nrows)):
        pred = model_copy(image).argmax(axis=1).item()
        sal_kwargs = get_kwargs(saliency_method, model_copy, image, label)
        fig, _ = visualize_saliency_method(sal_kwargs, image, (fig, axs[row, 1]), viz_method)

    # cascading randomization and visualization
    # start with 1 because 0th column is unscrambled model
    for (i, path), col in zip(enumerate(module_paths), range(2, ncols)):
        rand_layers(model_copy, module_paths[:i+1])
        for (image, label), row in zip(examples, range(nrows)):
            pred = model_copy(image).argmax(axis=1).item()
            sal_kwargs = get_kwargs(saliency_method, model_copy, image, label)
            fig, _ = visualize_saliency_method(sal_kwargs, image, (fig, axs[row, col]), viz_method)

    # set titles for each column
    col_titles = ['input', 'normal model'] + [x for x in map((lambda x: '_'.join(x)), module_paths)]
    for ax, col in zip(axs[0], col_titles):
        ax.set_title(col)

    # set title for the whole thing
    fig.suptitle(saliency_method[0].__name__ + " (Smoothing = " + str(saliency_method[1]) + ")", y=1.02, fontsize=36)

    return fig, axs


def visualize_cascading_randomization2(model, module_paths, sal_methods, sal_method_names, example, original=None, viz_method="heat_map"):
    model_copy = copy.deepcopy(model)
    image, label = example
    original = image if original is None else original # for showing unnormalized image

    # make plt plot
    nrows = len(sal_methods)
    ncols = len(module_paths) + 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows), tight_layout=TIGHT_LAYOUT_PARAMS)

    # show input image at the very left
    for row in range(nrows):
        npimg = original.squeeze(0).permute(1, 2, 0).numpy()
        axs[row, 0].imshow(npimg, cmap='gray')
        axs[row, 0].axis('on')
        axs[row, 0].set_xticks([])
        axs[row, 0].set_yticks([])

    # visualization before scrambling the model, and cascading randomization and visualization
    # start with 2 because 1st column is unscrambled model
    for (i, path), col in zip(enumerate([None] + module_paths), range(1, ncols)):
        if col != 1:
            rand_layers(model_copy, module_paths[:i])
        for sal_method, row in zip(sal_methods, range(nrows)):
            # pred = model_copy(image).argmax(axis=1).item()
            print("Working on " + sal_method[0].__name__)
            sal_kwargs = get_kwargs(sal_method, model_copy, image, label)
            fig, _ = visualize_saliency_method(sal_kwargs, image, (fig, axs[row, col]), viz_method)

    # set titles for each column
    col_titles = ['input', 'normal model'] + [x for x in map((lambda x: '_'.join(x)), module_paths)]
    for ax, col in zip(axs[0], col_titles):
        ax.set_title(col)

    # set titles for each row
    for ax, name in zip(axs[:,0], sal_method_names):
        ax.set_ylabel(name, rotation=90, size='large')

    # set title for the whole thing
    fig.suptitle("Cascading Randomization", y=1.02, fontsize=36)

    return fig, axs


# Normalize all values in tensor to [0, 1] range
def normalize_0_1(tensor):
    tensor -= tensor.min()
    tensor /= tensor.max()
    return tensor


#
def ssim_saliency_comparison(model, module_paths, sal_methods, sal_method_names, data_loader):
    model_copy = copy.deepcopy(model)

    # get all original explanations
    original_explanations = {} # key: (image_id, sal_method_id)
    for img_id, (img, label) in enumerate(data_loader):
        for sal_id, sal_method in enumerate(sal_methods):
            attribution = attribute_image_features(**get_kwargs(sal_method, model_copy, img, label))
            attribution = normalize_0_1(attribution.squeeze().permute(1, 2, 0)).detach().numpy()
            original_explanations[(img_id, sal_id)] = attribution

    # iterate over scrambled versions of the model
    ssim_similarities = {} # key: (model_scramble_stage, sal_method_id)
    for i, path in enumerate(module_paths):
        rand_layers(model_copy, module_paths[:i + 1])
        for (sal_id, sal_method), sal_method_name in zip(enumerate(sal_methods), sal_method_names):
            ssim_sum = 0
            errors = 0
            if not sal_method_name in ssim_similarities:
                ssim_similarities[sal_method_name] = {}
            for img_id, (img, label) in enumerate(data_loader):
                try:
                    attribution = attribute_image_features(**get_kwargs(sal_method, model_copy, img, label))
                    attribution = normalize_0_1(attribution.squeeze().permute(1, 2, 0)).detach().numpy()
                    #attrs = _normalize_image_attr(np.transpose(attrs.squeeze(0).detach().numpy(), (1,2,0)), "absolute_value")
                    # TODO: use skimage ssim with 3 channels
                    # TODO: skimage: scikit-image 0.15.0
                    ssim_sum += compare_ssim(original_explanations[(img_id, sal_id)], attribution, multichannel=True, gaussian_weights=True) # calculate ssim score with original attribution and add to sum
                except Exception as e:
                    print(e)
                    print(f'Error with image: {img_id}, sal method: {sal_method_name}, path: {"_".join(path)}')
                    errors += 1
            ssim_similarities[sal_method_name]['_'.join(path)] = (ssim_sum / (len(data_loader) - errors)).item() # save ssim similarity to dict

    return ssim_similarities