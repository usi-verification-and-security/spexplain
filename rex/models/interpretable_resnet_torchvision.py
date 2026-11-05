#!/usr/bin/env python3
import io
import re
from contextlib import redirect_stdout

from PIL import Image

import copy
import inspect
import numpy as np
import torch as tt
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights, get_model
import torchvision.transforms as T

class MaybeToTensor(object):

    def __init__(self):
        self.to_tensor = T.ToTensor()

    def __call__(self, x):
        if isinstance(x, (Image.Image, np.ndarray)):
            return self.to_tensor(x)
        elif isinstance(x, tt.Tensor):
            return x
        else:
            raise TypeError(f"Unsupported input type: {type(x)}. Expected PIL Image, numpy.ndarray, or torch.Tensor.")
        


def plot_images_in_single_axes(tensor, grid_dim=8, figsize=(12, 12), wspace=0.05, hspace=0.05, cmap='viridis', interpolation='nearest', title=None):
    """
    Plots a multi-channel tensor as a grid of images with proper spacing and zoom capability.
    Specifically designed for visualizing conv layer outputs/filters.

    Args:
        tensor: A tensor with shape (C, H, W) or (H, W, C) where C is the number of channels/filters
               For conv filter visualization, typically shape is (64, H, W) for 64 filters
        grid_dim (int): The dimension of the grid (e.g., 8 for an 8x8 grid to display 64 filters)
        figsize (tuple): The figure size passed to plt.figure()
        wspace (float): Width spacing between subplots, relative to subplot width
        hspace (float): Height spacing between subplots, relative to subplot height
        cmap (str): Colormap to use for the images
        interpolation (str): Interpolation method for imshow
        title (str, optional): Title for the plot. If None, a default title is generated
    """

    
    # Determine number of channels/filters to display
    num_channels = min(grid_dim * grid_dim, tensor.shape[0])
    
    # Create figure with subplots grid
    fig, axes = plt.subplots(grid_dim, grid_dim, figsize=figsize)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    

    # Plot each channel/filter
    for i in range(num_channels):
        img = tensor[i]
        print("Value of img:", np.unique(img))

        # Normalize each filter for better visualization
        img_min, img_max = img.min(), img.max()
        if img_min != img_max:  # Avoid division by zero
            img = (img - img_min) / (img_max - img_min)
        
        # Plot the image
        im = axes[i].imshow(img, cmap=cmap)
        
        # Remove ticks for cleaner look
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
        # Optional: Add filter/channel number
        axes[i].set_title(f"{i}", fontsize=8)
    
    # Hide any unused subplots
    for i in range(num_channels, grid_dim * grid_dim):
        axes[i].axis('off')
    
    if title is None:
        title = f'Visualization of {num_channels} filters/channels'
    
    fig.suptitle(title)
    plt.tight_layout()
    
    # Add a colorbar that applies to all subplots
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.show()
    return fig, axes

class InterpretableResNet50(nn.Module):
    def __init__(self, caltech256 = False, pascal_voc = False):
        
        super(InterpretableResNet50, self).__init__()

        # Load the pretrained ResNet50 model
        self.model = get_model('resnet50', weights="DEFAULT")
        
        if caltech256 or pascal_voc:
            if pascal_voc:
                num_classes = 20
            elif caltech256:
                num_classes = 257
            self.transforms = T.Compose([
                T.Resize((232, 232)),
                T.CenterCrop((224, 224)),
                MaybeToTensor(),
                T.ConvertImageDtype(tt.float32),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            if pascal_voc:
                self.model.fc = nn.Sequential(
                    nn.Linear(self.model.fc.in_features, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, num_classes)
                )
            elif caltech256:
                self.model.fc = nn.Sequential(
                    nn.Linear(self.model.fc.in_features, 512),
                    nn.ReLU(),
                    nn.Linear(512, num_classes)
                )


        else:
            self.transforms = ResNet50_Weights.DEFAULT.transforms()
            
        # self.transforms = T.Compose([
        #     T.Resize((224, 224)),
        #     T.ToTensor(),
        #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        self.model.eval()

    def masked_forward(self, input, layer, mask = None):
        if mask is None:
            mask = tt.ones_like(input)
        output = input

        # output = tt.where(mask != 0, output, tt.zeros_like(output))

        for bottleneck in layer:
            shortcut = output
            shortcut_mask = mask

            weights = bottleneck.conv1.weight
            bottleneck.conv1.weight = nn.Parameter(tt.ones_like(weights)/tt.numel(weights[0]))
            mask = bottleneck.conv1(mask)
            bottleneck.conv1.weight = weights
            mask = tt.where(mask != 0, 1.0, 0.0)

            output = bottleneck.conv1(output)
            output = bottleneck.bn1(output)
            output = tt.where(mask != 0.0, output, tt.zeros_like(output))
            output = self.model.relu(output)

            weights = bottleneck.conv2.weight
            bottleneck.conv2.weight = nn.Parameter(tt.ones_like(weights)/tt.numel(weights[0]))
            mask = bottleneck.conv2(mask)
            bottleneck.conv2.weight = weights
            mask = tt.where(mask != 0, 1.0, 0.0)

            output = bottleneck.conv2(output)
            output = bottleneck.bn2(output)
            output = tt.where(mask != 0.0, output, tt.zeros_like(output))
            output = self.model.relu(output)

            weights = bottleneck.conv3.weight
            bottleneck.conv3.weight = nn.Parameter(tt.ones_like(weights)/tt.numel(weights[0]))
            mask = bottleneck.conv3(mask)
            bottleneck.conv3.weight = weights
            mask = tt.where(mask != 0, 1.0, 0.0)

            output = bottleneck.conv3(output)
            output = bottleneck.bn3(output)
            output = tt.where(mask != 0.0, output, tt.zeros_like(output))


            if bottleneck.downsample is not None:
                weights = bottleneck.downsample[0].weight
                bottleneck.downsample[0].weight = nn.Parameter(tt.ones_like(weights)/tt.numel(weights[0]))
                shortcut_mask = bottleneck.downsample[0](shortcut_mask)
                bottleneck.downsample[0].weight = weights
                shortcut_mask = tt.where(shortcut_mask != 0, 1.0, 0.0)

                shortcut = bottleneck.downsample(shortcut)
                shortcut = tt.where(shortcut_mask != 0.0, shortcut, tt.zeros_like(shortcut))

            output += shortcut
            output = bottleneck.relu(output)
        return output, mask


    def forward(self, x,
                 explanation_mode = False,
                 masking_value = None,
                 explanation_mask = None):
        
        if explanation_mode:
            assert explanation_mask is not None or masking_value is not None, "Explanation_mask or masking_value must be provided in explanation mode"

            if masking_value is not None:
                explanation_mask = tt.where(x[:, 1:2, :, :] == masking_value, 0, 1.0)
            
            if len(explanation_mask.shape) == 3 or explanation_mask.shape[1] == 1:
                explanation_mask = tt.stack([explanation_mask] * 3, dim=1)
            
            #Since bias term is actually zero, this is truly faithful to the model
            x = tt.where(explanation_mask == 0, 0, x)

            # Applying the Conv Operator and MaxPooling to the Mutant Mask
            weights = self.model.conv1.weight
            self.model.conv1.weight = nn.Parameter(tt.ones_like(weights)/tt.numel(weights[0]))
            explanation_mask = self.model.conv1(explanation_mask) 
            explanation_mask = tt.where(explanation_mask != 0, 1.0, 0.0)
            self.model.conv1.weight = weights

            output = self.model.conv1(x)
            output = self.model.bn1(output)           
            output = tt.where(explanation_mask == 0, 0, output)
            output = self.model.relu(output)
            output = self.model.maxpool(output)

            explanation_mask = self.model.maxpool(explanation_mask)
            
            output, explanation_mask = self.masked_forward(input = output, layer = self.model.layer1, mask = explanation_mask)
            output, explanation_mask = self.masked_forward(input = output, layer = self.model.layer2, mask = explanation_mask)
            output, explanation_mask = self.masked_forward(input = output, layer = self.model.layer3, mask = explanation_mask)
            output, explanation_mask = self.masked_forward(input = output, layer = self.model.layer4, mask = explanation_mask)
            output = self.model.avgpool(output).squeeze(-1).squeeze(-1)
            output = self.model.fc(output)

            return output
        else:
            return self.model(x)
