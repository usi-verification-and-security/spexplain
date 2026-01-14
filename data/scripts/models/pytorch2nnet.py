import torch

from python_scripts.nnet_utils.writeNNet import writeNNet
# from Models import *
from CIFAR10_models import *
import torch


import torch

########################
# Hyperparameters
########################
model_task = "heart_attack"  # options: "mnist", "cifar10", "gtsrb", "heart_attack"

pytorchFile = "data/models/heart_attack/heart_attack_50x10.pth"
nnetFile="data/models/heart_attack/heart_attack_50x10.nnet"
checkpoint = torch.load(pytorchFile, map_location=torch.device('cpu'))

# Extract hyperparameters from checkpoint
input_dim = checkpoint["input_dim"]
hidden_size = checkpoint["hidden_size"]
num_layers = checkpoint["num_layers"]
num_classes = checkpoint["num_classes"]

print(f"\nModel saved to: {pytorchFile} with parameters:")
print(f"Input Dimension: {input_dim}")
print(f"Hidden Size: {hidden_size}")
print(f"Number of Layers: {num_layers}")
print(f"Number of Classes: {num_classes}")
print("==============================\n")

model = FCNet(input_dim, hidden_size=hidden_size,
                    num_layers=num_layers, num_classes=num_classes)
if model_task == "heart_attack":
    input_min = [29,0,0,94,126,0,0,71,0,0,0,0,0]
    input_max = [77,1,3,200,594,1,2,202,1,6.2,2,4,3]
elif model_task in ['mnist', 'cifar10', 'gtsrb']:
    input_min= 0
    input_max =255
elif model_task == "obesity":
    input_min = [-1.0,-1.38,-2.62,-2.07,-2.54,-2.63,-2.19,-3.87,-0.14,-1.66,-0.24,-1.16,-1.08,-2.46,-1.98]
    input_max = [1.0,4.16,2.45,0.49,0.4,1.12,1.71,2.39,7.6,1.63,4.31,2.27,2.15,1.46,1.31]
else:
    input_min = -1000000
    input_max = 1000000
def pytorch2nnet(model, nnetFile, input_min=-1000000, input_max=1000000):
    # Extract weights and biases from the model
    weights = []
    biases = []

    # Check if model has a Sequential container
    if hasattr(model, 'net') and isinstance(model.net, torch.nn.Sequential):
        layers_to_iterate = model.net.children()
    else:
        layers_to_iterate = model.children()

    for layer in layers_to_iterate:
        if isinstance(layer, torch.nn.Linear):
            weights.append(layer.weight.data.numpy())
            biases.append(layer.bias.data.numpy())

    # Determine the number of inputs and outputs
    num_inputs = weights[0].shape[1]
    num_outputs = biases[-1].shape[0]

    # Min and max values used to bound the inputs
    if isinstance(input_min, (int, float)):
        input_mins = [input_min for _ in range(num_inputs)]
    else:
        input_mins = input_min
    if isinstance(input_max, (int, float)):
        input_maxes = [input_max for _ in range(num_inputs)]
    else:
        input_maxes = input_max

    # Mean and range values for normalizing the inputs and outputs
    means = [0 for _ in range(num_inputs + num_outputs)]
    ranges = [1 for _ in range(num_inputs + num_outputs)]

    # Convert the file
    writeNNet(weights, biases, input_mins, input_maxes, means, ranges, nnetFile)


# Initialize the model architecture
# model = Mnist3(input_size=784, hidden_size=50, output_size=10)



# Load the state_dict into the model

# state_dict = torch.load(pytorchFile, map_location=torch.device('cpu'))
try:
    model.load_state_dict(checkpoint["model_state_dict"])
except:
    state_dict = torch.load(pytorchFile, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

# print number of parameters
num_params = sum(p.numel() for p in model.parameters())
print('number of parameters: ', num_params)
model.eval()
pytorch2nnet(model=model, nnetFile=nnetFile, input_min=input_min, input_max=input_max)
