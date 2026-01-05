import pandas as pd
import numpy as np

from data.models.models import *
import torch
from pgd_counterfactual import *

import torch

# ---- helper: register hooks to capture post-ReLU activations ----
def get_activation_hook(activations_dict, name):
    def hook(module, input, output):
        # output is already post-activation (after ReLU)
        activations_dict[name] = output.detach()
    return hook

def register_relu_hooks(model):
    activations = {}
    handles = []

    # after l1 -> relu
    handles.append(model.relu.register_forward_hook(
        get_activation_hook(activations, 'relu1')))
    # but we reuse the same ReLU module; better to split the forward or hook at linear layers.
    # So instead we define a wrapper model below.

    return activations, handles

import torch
import torch.nn as nn

def get_layer_activations_generic(model, x):
    """
    Run a forward pass and return a dict of activations after each
    non-linear layer of specified types (default: ReLU).

    Args:
        model            : any feed-forward nn.Module
        x                : input tensor (batch, features)
        activation_types : tuple of activation classes to track

    Returns:
        activations: dict[name] = tensor (batch, features_after_layer)
                     name pattern: "act_{idx}_{module_class}"
    """
    model.eval()
    activations = {}
    curr = x

    # go through all submodules in forward order
    idx = 0
    for module in model.children():
        curr = module(curr)
        if isinstance(module, nn.Linear):
            name = f"act_{idx}_{module.__class__.__name__}"
            activations[name] = model.relu(curr.detach())
            idx += 1

    return activations


def compare_activations_generic(model, x_orig, x_cf, threshold=0.0):
    """
    Compare activation patterns (zero vs >threshold) between original
    and counterfactual inputs, for any feed-forward model.

    Args:
        model            : nn.Module
        x_orig, x_cf     : inputs (batch=1, features)
        threshold        : > threshold means "active"

    Returns:
        changes: dict[layer_name] = {
                    'orig_active'    : bool mask (n_neurons,)
                    'cf_active'      : bool mask
                    'flipped_on_idx' : indices of 0 -> >0
                    'flipped_off_idx': indices of >0 -> 0
                    'any_change_idx' : indices where state differs
                    'not_flipped_on_idx' : indices of >0 -> >0
                    'not_flipped_off_idx': indices of 0 -> 0
                    'fixed_idx'      : indices with no change
                }
    """
    device = next(model.parameters()).device
    x_orig = x_orig.to(device)
    x_cf = x_cf.to(device)

    act_orig = get_layer_activations_generic(model, x_orig)
    act_cf   = get_layer_activations_generic(model, x_cf)

    changes = {}

    for layer_name in act_orig.keys():
        a_o = act_orig[layer_name].squeeze(0)  # (n_neurons,)
        a_c = act_cf[layer_name].squeeze(0)

        orig_active = a_o > threshold
        cf_active   = a_c > threshold

        flipped_on  = (~orig_active) & cf_active
        flipped_off = orig_active & (~cf_active)
        any_change  = orig_active ^ cf_active
        not_flipped_on  = orig_active & cf_active
        not_flipped_off = (~orig_active) & (~cf_active)


        changes[layer_name] = {
            'orig_active': orig_active,
            'cf_active': cf_active,
            'flipped_on_idx': torch.nonzero(flipped_on, as_tuple=False).flatten(),
            'flipped_off_idx': torch.nonzero(flipped_off, as_tuple=False).flatten(),
            'any_change_idx': torch.nonzero(any_change, as_tuple=False).flatten(),
            'not_flipped_on_idx': torch.nonzero(not_flipped_on, as_tuple=False).flatten(),
            'not_flipped_off_idx': torch.nonzero(not_flipped_off, as_tuple=False).flatten(),
            'fixed_idx': torch.nonzero(~any_change, as_tuple=False).flatten(),
        }

    return changes

activations_changes_file = 'data/activation_change/mnist/activation_change-mnist-6x50.txt'
# Load dataset
df = pd.read_csv('data/datasets/mnist/mnist_short.csv')
# lables = df['target']
# df.drop('target', axis=1, inplace=True)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Mnist6xS(input_size=784, hidden_size=50, output_size=10)
pytorchFile = "data/models/mnist/pth_files/mnist_model_6x50.pth"
state_dict = torch.load(pytorchFile, map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()

idx = 0 # sample index
for row in df.itertuples(index=False):
    idx += 1
    pixels = np.array(row[:-1], dtype=np.float32)
    img_flat = torch.from_numpy(pixels).to(device).view(1, -1)

    with torch.no_grad():
        orig_pred = model(img_flat).argmax(dim=1)
    if orig_pred.item() != row[-1]:
        print("****** miss-classified example! Target: %s ******", row[-1] )

    # target = torch.tensor(7, device=device)  # e.g., want class 7
    x_cf, cf_pred = pgd_counterfactual(
            model,
            img_flat,
            true_label=orig_pred,
            # target_label=target,
            epsilon=0.3 * 255.0,
            step_size=0.01 * 255.0,
            num_steps=40,
            clamp_min=0.0,
            clamp_max=255.0,
            targeted=False
    )

    print("Original:", orig_pred.item(), "Counterfactual:", cf_pred.item())
    if cf_pred.item() == orig_pred.item():
        print("********** Failed to generate valid counterfactual. **********")
        continue

    # Print difference in input
    input_diff = x_cf - img_flat
    print("Input difference (L_inf):", torch.max(torch.abs(input_diff)).item())
    print("Input difference (L_2):", torch.norm(input_diff, p=2).item())

    changes = compare_activations_generic(model, img_flat, x_cf)

    # Print activation changes per layer
    for layer, info in changes.items():
        print(f"=== {layer} ===")
        print("flipped ON (0 -> >0):", info['flipped_on_idx'].tolist())
        print("flipped OFF (>0 -> 0):", info['flipped_off_idx'].tolist())
        print("any change:", info['any_change_idx'].tolist())

    # Append sample and per-layer any_change masks to `activation_changes.txt`
    # Place this inside the loop after `changes = compare_activations_generic(...)`
    with open(activations_changes_file, 'a') as out_f:
        out_f.write(f"Sample_index:{idx}\n")
        # sample_pixels = ','.join(str(int(p)) for p in row[:-1])
        # out_f.write(sample_pixels + "\n")

        for layer_idx, (layer_name, info) in enumerate(list(changes.items())[:-1]):
            # any_mask = (info['orig_active'] ^ info['cf_active']).to('cpu').numpy().astype(int)
            line = ' '.join(str(x+1) for x in info['fixed_idx'].tolist())
            out_f.write(line + " 0" + "\n")

        # separate samples
        out_f.write("\n")
