import os
os.environ["HF_HOME"] = "../models"

import numpy as np
from PIL import Image


import torch
import timm
import torch.nn as nn
import sys
import torchvision.transforms as T

class MaybeToTensor(object):

    def __init__(self):
        self.to_tensor = T.ToTensor()

    def __call__(self, x):
        if isinstance(x, (Image.Image, np.ndarray)):
            return self.to_tensor(x)
        elif isinstance(x, torch.Tensor):
            return x
        else:
            raise TypeError(f"Unsupported input type: {type(x)}. Expected PIL Image, numpy.ndarray, or torch.Tensor.")
        

class InterpretableRegNetY(nn.Module):
    def __init__(self, caltech256 = False, pascal_voc = False):
        
        super(InterpretableRegNetY, self).__init__()

        # Load the pretrained RegNet-Y model
        self.model = timm.create_model('regnety_120.sw_in12k_ft_in1k', pretrained=True)        
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        self.model.eval()

        if caltech256 or pascal_voc:
            if pascal_voc:
                num_classes = 20
            elif caltech256:
                num_classes = 257
            num_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Sequential(
                nn.Linear(num_features, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_classes)
            )

            self.transforms = T.Compose([
                T.Resize((384, 384), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop(384),
                MaybeToTensor(),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


    def masked_stage_forward(self, input, stage, mask):
        
        output = input
        for bottleneck in list(stage.children()):
            shortcut = output
            shortcut_mask = mask

            weights = bottleneck.conv1.conv.weight
            bottleneck.conv1.conv.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
            mask = bottleneck.conv1.conv(mask)
            bottleneck.conv1.conv.weight = weights
            mask = torch.where(mask != 0, 1.0, 0.0)
            
            output = bottleneck.conv1(output)
            output = torch.where(mask != 0, output, torch.zeros_like(output))

            weights = bottleneck.conv2.conv.weight
            bottleneck.conv2.conv.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
            mask = bottleneck.conv2.conv(mask)
            bottleneck.conv2.conv.weight = weights
            mask = torch.where(mask != 0, 1.0, 0.0)
            
            output = bottleneck.conv2(output)
            output = torch.where(mask != 0, output, torch.zeros_like(output))

            #Now the SE module, has a bias layer
            #First the FC1 Layer
            weights = bottleneck.se.fc1.weight
            bias = bottleneck.se.fc1.bias
            bottleneck.se.fc1.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
            bottleneck.se.fc1.bias = nn.Parameter(torch.zeros_like(bias))
            mask = bottleneck.se.fc1(mask)
            bottleneck.se.fc1.weight = weights
            bottleneck.se.fc1.bias = bias
            mask = torch.where(mask != 0, 1.0, 0.0)

            output_se = output.mean((2,3), keepdim=True)
            if bottleneck.se.add_maxpool:
                output_se = 0.5 * output_se + 0.5 * output_se.amax((2,3), keepdim=True)
            
            output_se = bottleneck.se.fc1(output_se)
            output_se = bottleneck.se.bn(output_se)
            output_se = bottleneck.se.act(output_se)
            output_se = torch.where(mask != 0, output_se, torch.zeros_like(output_se))


            #Now the FC2 Layer
            weights = bottleneck.se.fc2.weight
            bias = bottleneck.se.fc2.bias
            bottleneck.se.fc2.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
            bottleneck.se.fc2.bias = nn.Parameter(torch.zeros_like(bias))
            mask = bottleneck.se.fc2(mask)
            bottleneck.se.fc2.weight = weights
            bottleneck.se.fc2.bias = bias
            mask = torch.where(mask != 0, 1.0, 0.0)
            
            output_se = bottleneck.se.fc2(output_se)
            output_se = torch.where(mask != 0, output_se, torch.zeros_like(output_se))

            output_se = bottleneck.se.gate(output_se)
            output_se = torch.where(output_se != 0, output_se, torch.zeros_like(output_se)) #Have to suppress sigmoid at masked regions

            output = output_se * output

            weights = bottleneck.conv3.conv.weight
            bottleneck.conv3.conv.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
            mask = bottleneck.conv3.conv(mask)
            bottleneck.conv3.conv.weight = weights
            mask = torch.where(mask != 0, 1.0, 0.0)
            
            output = bottleneck.conv3(output)
            output = torch.where(mask != 0, output, torch.zeros_like(output))

            if not isinstance(bottleneck.downsample, nn.Identity):
                weights = bottleneck.downsample.conv.weight
                bottleneck.downsample.conv.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
                shortcut_mask = bottleneck.downsample.conv(shortcut_mask)
                bottleneck.downsample.conv.weight = weights
                shortcut_mask = torch.where(shortcut_mask != 0, 1.0, 0.0)
                
                shortcut = bottleneck.downsample(shortcut)
                shortcut = torch.where(shortcut_mask != 0.0, shortcut, torch.zeros_like(shortcut))

            if bottleneck.drop_path is not None:
                output = bottleneck.drop_path(output)
                
            output += shortcut    
            output = bottleneck.act3(output)
        return output, mask
        


    def forward(self, x,
                 explanation_mode = False,
                 masking_value = None,
                 explanation_mask = None):
        
        if explanation_mode:
            # assert explanation_mask is not None or masking_value is not None, "Explanation_mask or masking_value must be provided in explanation mode"
            with torch.no_grad():
                if masking_value is not None:
                    explanation_mask = torch.where(x[:, 1:2, :, :] == masking_value, 0, 1.0)

                if masking_value is None and explanation_mask is None:
                    explanation_mask = torch.ones_like(x)
                
                if len(explanation_mask.shape) == 3 or explanation_mask.shape[1] == 1:
                    explanation_mask = torch.stack([explanation_mask] * 3, dim=1)
                
                x = torch.where(explanation_mask == 0, 0, x)

                weights = self.model.stem.conv.weight
                self.model.stem.conv.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))

                explanation_mask = self.model.stem.conv(explanation_mask) 
                explanation_mask = torch.where(explanation_mask != 0, 1.0, 0.0)
                
                self.model.stem.conv.weight = weights
                output = self.model.stem(x) #Conv + BN + ReLU
                                    
                output = torch.where(explanation_mask == 0, 0, output)

                output, explanation_mask  = self.masked_stage_forward(input = output, stage = self.model.s1, mask = explanation_mask)
                output, explanation_mask  = self.masked_stage_forward(input = output, stage = self.model.s2, mask = explanation_mask)
                output, explanation_mask  = self.masked_stage_forward(input = output, stage = self.model.s3, mask = explanation_mask)
                output, explanation_mask  = self.masked_stage_forward(input = output, stage = self.model.s4, mask = explanation_mask)

                output = self.model.final_conv(output)
                output = self.model.head(output)
           
                return output
        else:
            with torch.no_grad():
                return self.model(x)
