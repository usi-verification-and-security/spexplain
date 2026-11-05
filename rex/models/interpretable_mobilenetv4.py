import os
os.environ["HF_HOME"] = "../models"

import torch
import timm
import torch.nn as nn
import sys
import torchvision.transforms as T

class InterpretableMobileNetV4(nn.Module):
    def __init__(self, caltech_256 = False, pascal_voc = False):
        
        super(InterpretableMobileNetV4, self).__init__()

        # Load the pretrained MobileNetV4 model
        self.model = timm.create_model('mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k', pretrained=True)        
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        self.model.eval()

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
                T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop((448, 448)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


    def masked_final_stage_forward(self, input, stage, mask):
        for block in stage:
            weights = block.conv.weight
            block.conv.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
            mask = block.conv(mask)
            block.conv.weight = weights
            mask = torch.where(mask != 0, 1.0, 0.0)

            output = block.conv(input)
            output = block.bn1(output)
            output = torch.where(mask != 0, output, torch.zeros_like(output))

            if block.aa is not None:
                output = block.aa(output)
                mask = block.aa(mask)
                mask = torch.where(mask != 0, 1.0, 0.0)

                output = torch.where(mask != 0, output, torch.zeros_like(output))
                            
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
            mask = edge_residual_block.aa(mask)
            mask = torch.where(mask != 0, 1.0, 0.0)

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
    
    def masked_universal_inverted_residual_block(self, input, stage, mask):
        output = input
        for uib_residual_block in stage:
            shortcut = output
            shortcut_mask = mask

            weights = uib_residual_block.dw_start.conv.weight
            uib_residual_block.dw_start.conv.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
            mask = uib_residual_block.dw_start.conv(mask)
            uib_residual_block.dw_start.conv.weight = weights
            mask = torch.where(mask != 0, 1.0, 0.0)

            output = uib_residual_block.dw_start(output)
            output = torch.where(mask != 0, output, torch.zeros_like(output))

            weights = uib_residual_block.pw_exp.conv.weight
            uib_residual_block.pw_exp.conv.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
            mask = uib_residual_block.pw_exp.conv(mask)
            uib_residual_block.pw_exp.conv.weight = weights
            mask = torch.where(mask != 0, 1.0, 0.0)

            output = uib_residual_block.pw_exp(output)
            output = torch.where(mask != 0, output, torch.zeros_like(output))

            if not isinstance(uib_residual_block.dw_mid, nn.Identity):
                weights = uib_residual_block.dw_mid.conv.weight
                uib_residual_block.dw_mid.conv.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
                mask = uib_residual_block.dw_mid.conv(mask)
                uib_residual_block.dw_mid.conv.weight = weights
                mask = torch.where(mask != 0, 1.0, 0.0)
            
            else:
                output = uib_residual_block.dw_mid(output)

            if not isinstance(uib_residual_block.dw_mid, nn.Identity):
                if uib_residual_block.dw_mid.aa is not None:
                    mask = uib_residual_block.dw_mid.aa(mask)
                    mask = torch.where(mask != 0, 1.0, 0.0)

            output = uib_residual_block.dw_mid(output)
            output = torch.where(mask != 0, output, torch.zeros_like(output))


            if not isinstance(uib_residual_block.se, nn.Identity):
                output_se = output.mean((2,3), keepdim=True)
                mask_se = mask.mean((2,3), keepdim=True)

                weights = uib_residual_block.se.conv_reduce.weight
                uib_residual_block.se.conv_reduce.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
                mask_se = uib_residual_block.se.conv_reduce(mask_se)
                uib_residual_block.se.conv_reduce.weight = weights
                mask_se = torch.where(mask_se != 0, 1.0, 0.0)

                output_se = uib_residual_block.se.conv_reduce(output_se)
                output_se = uib_residual_block.se.act1(output_se)
                output_se = torch.where(mask_se == 0, 0, output_se)

                weights = uib_residual_block.se.conv_expand.weight
                uib_residual_block.se.conv_expand.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
                mask_se = uib_residual_block.se.conv_expand(mask_se)
                uib_residual_block.se.conv_expand.weight = weights
                mask_se = torch.where(mask_se != 0, 1.0, 0.0)

                output_se = uib_residual_block.se.conv_expand(output_se)
                output_se = uib_residual_block.se.gate(output_se)
                output_se = torch.where(output_se != 0, output_se, torch.zeros_like(output_se)) #Have to suppress sigmoid at masked regions
                output = output_se * output
            else:
                output = uib_residual_block.se(output)

            weights = uib_residual_block.pw_proj.conv.weight
            uib_residual_block.pw_proj.conv.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
            mask = uib_residual_block.pw_proj.conv(mask)
            uib_residual_block.pw_proj.conv.weight = weights
            mask = torch.where(mask != 0, 1.0, 0.0)

            output = uib_residual_block.pw_proj(output)
            output = torch.where(mask != 0, output, torch.zeros_like(output))

            output = uib_residual_block.dw_end(output)
            output = uib_residual_block.layer_scale(output)

            if uib_residual_block.has_skip:
                shortcut = torch.where(shortcut_mask != 0.0, shortcut, torch.zeros_like(shortcut))
                output = uib_residual_block.drop_path(output) + shortcut

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
                output = self.model.bn1(output) #Conv + BN + ReLU
                                    
                output = torch.where(explanation_mask == 0, 0, output)

                output, explanation_mask = self.masked_edge_residual_block(input = output, stage = self.model.blocks[0], mask = explanation_mask)
                output, explanation_mask  = self.masked_universal_inverted_residual_block(input = output, stage = self.model.blocks[1], mask = explanation_mask)
                output, explanation_mask  = self.masked_universal_inverted_residual_block(input = output, stage = self.model.blocks[2], mask = explanation_mask)
                output, explanation_mask  = self.masked_universal_inverted_residual_block(input = output, stage = self.model.blocks[3], mask = explanation_mask)
                output, explanation_mask  = self.masked_final_stage_forward(input = output, stage = self.model.blocks[4], mask = explanation_mask)

                output = self.model.global_pool(output)
                explanation_mask = self.model.global_pool(explanation_mask)
                explanation_mask = torch.where(explanation_mask != 0, 1.0, 0.0)

                output = torch.where(explanation_mask != 0, output, torch.zeros_like(output))

                weights = self.model.conv_head.weight
                self.model.conv_head.weight = nn.Parameter(torch.ones_like(weights)/torch.numel(weights[0]))
                explanation_mask = self.model.conv_head(explanation_mask)
                explanation_mask = torch.where(explanation_mask != 0, 1.0, 0.0)

                self.model.conv_head.weight = weights
                output = self.model.conv_head(output)

                output = self.model.norm_head(output) #BN + SiLU
                output = torch.where(explanation_mask != 0, output, torch.zeros_like(output))

                output = self.model.act2(output)
                output = self.model.flatten(output)
                output = self.model.classifier(output)
           
                return output
        else:
            with torch.no_grad():
                return self.model(x)
