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
        

class InterpretableEfficientNetV2(nn.Module):
    def __init__(self, caltech_256 = False, pascal_voc = False):
        
        super(InterpretableEfficientNetV2, self).__init__()

        # Load the pretrained EfficientnetV2 model
        self.model = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', pretrained=True)        
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

        if caltech_256 or pascal_voc:
            if pascal_voc:
                num_classes = 20
            elif caltech_256:
                num_classes = 257
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(num_features, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_classes)
            )

            self.transforms = T.Compose([
                T.Resize((300, 300), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop((300, 300)),
                MaybeToTensor(),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=[0.500, 0.500, 0.500], std=[0.500, 0.500, 0.500])
            ])

        self.model.eval()


    def masked_first_stage_forward(self, input, stage, mask):
        output = input
        for block in stage:
            shortcut = output
            shortcut_mask = mask

            weight = block.conv.weight
            block.conv.weight = nn.Parameter(torch.ones_like(weight)/torch.numel(weight[0]))
            mask = block.conv(mask)
            block.conv.weight = weight
            mask = torch.where(mask != 0, 1.0, 0.0)

            output = block.conv(output)
            output = block.bn1(output) #BN + SiLU
            output = torch.where(mask != 0, output, torch.zeros_like(output))

            if block.aa is not None:
                output = block.aa(output)
            
            if block.has_skip:
                shortcut = torch.where(shortcut_mask != 0.0, shortcut, torch.zeros_like(shortcut))
                output = block.drop_path(output) + shortcut
            
        return output, mask


    def masked_edge_residual_block(self, input, stage, mask):
        
        output = input
        for edge_residual_block in stage:
            shortcut = output
            shortcut_mask = mask

            weights = edge_residual_block.conv_exp.weight
            edge_residual_block.conv_exp.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
            mask = edge_residual_block.conv_exp(mask)
            edge_residual_block.conv_exp.weight = weights
            mask = torch.where(mask != 0, 1.0, 0.0)
            
            output = edge_residual_block.conv_exp(output)
            output = edge_residual_block.bn1(output)
            output = torch.where(mask != 0, output, torch.zeros_like(output))

            output = edge_residual_block.aa(output)   

            if not isinstance(edge_residual_block.se, nn.Identity):
                output_se = output.mean((2,3), keepdim=True)
                mask_se = mask.mean((2,3), keepdim=True)

                weights = edge_residual_block.se.conv_reduce.weight
                edge_residual_block.se.conv_reduce.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
                mask_se = edge_residual_block.se.conv_reduce(mask_se)
                edge_residual_block.se.conv_reduce.weight = weights
                mask_se = torch.where(mask_se != 0, 1.0, 0.0)

                output_se = edge_residual_block.se.conv_reduce(output_se)
                output_se = edge_residual_block.se.act1(output_se)
                output_se = torch.where(mask_se == 0, 0, output_se)

                weights = edge_residual_block.se.conv_expand.weight
                edge_residual_block.se.conv_expand.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
                mask_se = edge_residual_block.se.conv_expand(mask_se)
                edge_residual_block.se.conv_expand.weight = weights
                mask_se = torch.where(mask_se != 0, 1.0, 0.0)

                output_se = edge_residual_block.se.conv_expand(output_se)
                output_se = edge_residual_block.se.gate(output_se)
                output_se = torch.where(output_se != 0, output_se, torch.zeros_like(output_se)) #Have to suppress sigmoid at masked regions
                output = output_se * output
            else:
                output = edge_residual_block.se(output)

            weights = edge_residual_block.conv_pwl.weight
            edge_residual_block.conv_pwl.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
            mask = edge_residual_block.conv_pwl(mask)
            edge_residual_block.conv_pwl.weight = weights
            mask = torch.where(mask != 0, 1.0, 0.0)

            output = edge_residual_block.conv_pwl(output)
            output = edge_residual_block.bn2(output)

            if edge_residual_block.has_skip:
                shortcut = torch.where(shortcut_mask != 0.0, shortcut, torch.zeros_like(shortcut))
                output = edge_residual_block.drop_path(output) + shortcut
            
        return output, mask
    
    def masked_inverted_residual_block(self, input, stage, mask):
        
        output = input
        for inverted_residual_block in stage:
            shortcut = output
            shortcut_mask = mask

            if inverted_residual_block.conv_s2d is not None:
                weights = inverted_residual_block.conv_s2d.weight
                inverted_residual_block.conv_s2d.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
                mask = inverted_residual_block.conv_s2d(mask)
                inverted_residual_block.conv_s2d.weight = weights
                mask = torch.where(mask != 0, 1.0, 0.0)
                
                output = inverted_residual_block.conv_s2d(output)
                output = inverted_residual_block.bn_s2d(output)
                output = torch.where(mask != 0, output, torch.zeros_like(output))
                
            weights = inverted_residual_block.conv_pw.weight
            inverted_residual_block.conv_pw.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
            mask = inverted_residual_block.conv_pw(mask)
            inverted_residual_block.conv_pw.weight = weights
            mask = torch.where(mask != 0, 1.0, 0.0)
            
            output = inverted_residual_block.conv_pw(output)
            output = inverted_residual_block.bn1(output)
            output = torch.where(mask != 0, output, torch.zeros_like(output))

            weights = inverted_residual_block.conv_dw.weight
            inverted_residual_block.conv_dw.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
            mask = inverted_residual_block.conv_dw(mask)
            inverted_residual_block.conv_dw.weight = weights
            mask = torch.where(mask != 0, 1.0, 0.0)

            output = inverted_residual_block.conv_dw(output)
            output = inverted_residual_block.bn2(output)
            output = torch.where(mask != 0, output, torch.zeros_like(output))

            output = inverted_residual_block.aa(output)

            if not isinstance(inverted_residual_block.se, nn.Identity):
                output_se = output.mean((2,3), keepdim=True)
                mask_se = mask.mean((2,3), keepdim=True)
                
                weights = inverted_residual_block.se.conv_reduce.weight
                inverted_residual_block.se.conv_reduce.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
                mask_se = inverted_residual_block.se.conv_reduce(mask_se)
                inverted_residual_block.se.conv_reduce.weight = weights
                mask_se = torch.where(mask_se != 0, 1.0, 0.0)
                
                output_se = inverted_residual_block.se.conv_reduce(output_se)
                output_se = inverted_residual_block.se.act1(output_se)
                output_se = torch.where(mask_se == 0, 0, output_se)

                weights = inverted_residual_block.se.conv_expand.weight
                inverted_residual_block.se.conv_expand.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
                mask_se = inverted_residual_block.se.conv_expand(mask_se)
                inverted_residual_block.se.conv_expand.weight = weights
                mask_se = torch.where(mask_se != 0, 1.0, 0.0)

                output_se = inverted_residual_block.se.conv_expand(output_se)
                output_se = inverted_residual_block.se.gate(output_se)
                output_se = torch.where(output_se != 0, output_se, torch.zeros_like(output_se))
                output = output_se * output
            else:
                output = inverted_residual_block.se(output)

            weights = inverted_residual_block.conv_pwl.weight
            inverted_residual_block.conv_pwl.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
            mask = inverted_residual_block.conv_pwl(mask)
            inverted_residual_block.conv_pwl.weight = weights
            mask = torch.where(mask != 0, 1.0, 0.0)

            output = inverted_residual_block.conv_pwl(output)
            output = inverted_residual_block.bn3(output)
            output = torch.where(mask != 0, output, torch.zeros_like(output))

            if inverted_residual_block.has_skip:
                shortcut = torch.where(shortcut_mask != 0.0, shortcut, torch.zeros_like(shortcut))
                output = inverted_residual_block.drop_path(output) + shortcut
            
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

                weights = self.model.conv_stem.weight
                self.model.conv_stem.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))

                explanation_mask = self.model.conv_stem(explanation_mask) 
                explanation_mask = torch.where(explanation_mask != 0, 1.0, 0.0)
                
                self.model.conv_stem.weight = weights
                output = self.model.conv_stem(x)
                output = self.model.bn1(output) #Conv + BN + SiLU
                                    
                output = torch.where(explanation_mask == 0, 0, output)

                output, explanation_mask = self.masked_first_stage_forward(input = output, stage = self.model.blocks[0], mask = explanation_mask)
                output, explanation_mask  = self.masked_edge_residual_block(input = output, stage = self.model.blocks[1], mask = explanation_mask)
                output, explanation_mask  = self.masked_edge_residual_block(input = output, stage = self.model.blocks[2], mask = explanation_mask)
                output, explanation_mask  = self.masked_inverted_residual_block(input = output, stage = self.model.blocks[3], mask = explanation_mask)
                output, explanation_mask  = self.masked_inverted_residual_block(input = output, stage = self.model.blocks[4], mask = explanation_mask)
                output, explanation_mask  = self.masked_inverted_residual_block(input = output, stage = self.model.blocks[5], mask = explanation_mask)
                weights = self.model.conv_head.weight
                self.model.conv_head.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
                explanation_mask = self.model.conv_head(explanation_mask)
                explanation_mask = torch.where(explanation_mask != 0, 1.0, 0.0)

                self.model.conv_head.weight = weights

                output = self.model.conv_head(output)
                output = self.model.bn2(output) #BN + SiLU
                output = torch.where(explanation_mask == 0, 0, output)

                output = self.model.global_pool(output)
                output = self.model.classifier(output)
           
                return output
        else:
            with torch.no_grad():
                return self.model(x)
