import torch
import timm
import csv
import numpy as np
import os
from datetime import datetime

def get_model_family(model_name):
    # Define a list of known model family prefixes if necessary
    known_families = ['resnet', 'mobilenet', 'beit', 'vit', 'efficientnet', 
                      'densenet', 'coatnet', 'cait', 'inception', 'inception_resnet',
                      'coat', 'convit', 'convnet', 'convnext', 'convit', 'convmixer',
                      'crossvit', 'darknet', 'deit', 'densenet', 'dla', 'nfnet',
                      'dpn', 'botnext', 'edgenext', 'edgenet', 'efficientformer', 'vovnet',
                      'fbnet', 'gcvit', 'gernet', 'senet', 'xception', 'gm',
                      'hardcorenas', 'hrnet', 'nest', 'lcnet', 'levit', 'maxvit',
                      'mixer', 'mixnet', 'nasnet', 'mvit', 'regnet', 'pit',
                      'poolformer', 'pvt', 'regnet', 'res2net', 'vgg', 'resmlp',
                      'resnest', 'rexnet', 'halonet', 'secsls', 'sequencer', 
                      'swin', 'tinynet', 'tnt', 'twins', 'visformer', 'vit',
                      'relpos', 'volo', 'xcit', 'resnext', 'botnet', 'halonext',
                      'ghostnet', 'res2next', 'selecsls']
    
    known_families = sorted(known_families, key=len, reverse=True)

    # Infer the family by splitting the model name and checking known families
    for family in known_families:
        if family in model_name:
            return family  # Return the capitalized family name for consistency
    
    # Default to using the first part of the model name as the family, or a custom rule
    return 'UNKNOWN'

def is_attention_or_recurrent_component(module):
    """
    Checks if the module is a part of attention mechanisms or recurrent architectures.
    This includes both standard PyTorch layers and custom implementations that might be named accordingly.
    """
    attention_related = (torch.nn.MultiheadAttention,)
    recurrent_related = (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)
    
    # Check for custom implementations based on naming conventions
    custom_related = 'Transformer' in module.__class__.__name__ or 'Attention' in module.__class__.__name__
    
    return isinstance(module, attention_related + recurrent_related) or custom_related


def get_parameter_statistics(model):
    """Calculate statistics for the model's weights."""
    weights = [param.data.cpu().numpy() for param in model.parameters() if param.requires_grad]
    if not weights:  # If the model has no trainable parameters
        return {}

    # Flatten the weights to a single array for overall statistics
    all_weights = np.concatenate([w.flatten() for w in weights])
    
    return {
        'Weight Mean': np.mean(all_weights),
        'Weight Std': np.std(all_weights),
        'Weight Min': np.min(all_weights),
        'Weight Max': np.max(all_weights),
    }

def is_likely_dimension_matching(module):
    """
    Heuristic to guess if the module potentially matches input-output dimensions.
    This is a simplified check and may not cover all cases or architectures.
    """
    for layer in module.modules():
        if isinstance(layer, torch.nn.Conv2d):
            # Check if the convolution is likely to maintain spatial dimensions
            if layer.stride == (1, 1) and layer.kernel_size in [(1, 1), (3, 3)]:
                # Assuming padding is set to maintain dimensions for kernel_size 3
                return True
            # Additional checks for other layers or configurations can be added here
    return False

def is_likely_residual_module(module):
    """
    Combines checks for residual-like structure and dimension matching.
    """
    return has_residual_connection(module) and is_likely_dimension_matching(module)

def has_residual_connection(module):
    # List known residual block types here
    residual_block_types = ['Bottleneck', 'BasicBlock']
    return module.__class__.__name__ in residual_block_types

def is_residual_module(module):
    # Add logic here to determine if a module is a residual block
    # This is a placeholder; you might use the module's class name or structure as a clue
    return module.__class__.__name__ in ['Bottleneck', 'BasicBlock']

def get_output_features(model):
    """
    Attempts to dynamically determine the number of output features (e.g., classes) from the model's final layer.
    """
    # Attempt to find the final layer by inspecting the model's forward method (if defined)
    final_layer = None
    for layer in model.modules():
        if hasattr(layer, 'out_features'):  # Common attribute for linear layers
            final_layer = layer
        elif hasattr(layer, 'out_channels') and not isinstance(layer, torch.nn.Sequential):  # For convolutional layers
            final_layer = layer

    # Determine the number of output features based on the layer type
    if final_layer is not None:
        if hasattr(final_layer, 'out_features'):
            return final_layer.out_features
        elif hasattr(final_layer, 'out_channels'):
            return final_layer.out_channels
    return None

def summarize_model(model_name):
    model = timm.create_model(model_name, pretrained=False).to(torch.device("cuda"))
    total_params = sum(p.numel() for p in model.parameters())

    model.eval()
    # Get parameter statistics
    #param_stats = get_parameter_statistics(model)

    # Initialize counters for various layer types
    layer_counts = {
        'residual': 0,
        'conv': 0,
        'dense': 0,
        'recurrent': 0,
        'bottlenecks': 0,
        'pooling': 0,
        'normalization': 0,
    }

    rf = 1  # Receptive field starts at 1
    stride_product = 1  # Cumulative product of strides
    layers_rf = []  # Store (layer_name, rf_value) tuples
    max_rf = 0  # Track the maximum receptive field

    # Initialize counters and thresholds for parameter percentages
    cumulative_params = 0
    thresholds = [0.25 * total_params, 0.5 * total_params, 0.75 * total_params]
    percent_layers = {25: 'N/A', 50: 'N/A', 75: 'N/A'}
    current_threshold_index = 0
    
    num_layers = 0
    output_features = get_output_features(model)
    for name, module in model.named_modules():
        num_layers += 1
        module_params = sum(p.numel() for p in module.parameters())
        cumulative_params += module_params

        # Check for layer types
        if isinstance(module, torch.nn.Conv2d):
            layer_counts['conv'] += 1
        
        if isinstance(module, torch.nn.Linear):
            layer_counts['dense'] += 1
            output_features = module.out_features
        
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
            layer_counts['normalization'] += 1
        
        if isinstance(module, (torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
            layer_counts['pooling'] += 1

        if 'Bottleneck' in module.__class__.__name__:
            layer_counts['bottlenecks'] += 1

        if has_residual_connection(module):
            layer_counts['residual'] += 1
        
        if is_attention_or_recurrent_component(module):
            layer_counts['recurrent'] += 1


        while current_threshold_index < len(thresholds) and cumulative_params >= thresholds[current_threshold_index]:
            percent_layers[25 if current_threshold_index == 0 else 50 if current_threshold_index == 1 else 75] = name
            current_threshold_index += 1

        if isinstance(module, (torch.nn.Conv2d, torch.nn.MaxPool2d)):
            kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, (tuple, list)) else module.kernel_size
            stride = module.stride[0] if isinstance(module.stride, (tuple, list)) else module.stride
            rf += (kernel_size - 1) * stride_product
            stride_product *= stride
            layers_rf.append((name, rf))
            max_rf = max(max_rf, rf)
    
    # Find the layer closest to 25% of the max_rf
    target_rf = 0.25 * max_rf
    closest_layer = min(layers_rf, key=lambda x: abs(x[1] - target_rf))

    # Calculate layer's percentage position
    total_layers = len(layers_rf)
    target_layer_index = layers_rf.index(closest_layer) + 1  # +1 for 1-based indexing
    percentage_position = (target_layer_index / total_layers) * 100

    default = model.default_cfg
    family = get_model_family(model_name)

    if output_features == 1000 or output_features == 768 or output_features == 384:
        dataset_used = 'ImageNet-1K'
    elif output_features == 21841 or output_features == 21843:
        dataset_used = 'ImageNet-21K'
    elif output_features == 11221:
        dataset_used = 'ImageNet-21K-P'
    elif output_features == 512 or output_features == 1024:
        dataset_used = 'LAION-2B'
    else:
        dataset_used = 'ERROR UNKNOWN DATASET'

    try:
        crop = default['crop_pct']
    except:
        crop = 'None'

    summary = {
        'Model': model_name,
        'Parameters': total_params,
        'Layers': num_layers,
        'Residual Blocks': layer_counts['residual'],
        'Conv Layers': layer_counts['conv'],
        'Dense Layers': layer_counts['dense'],
        'Bottlenecks': layer_counts['bottlenecks'],
        'Pooling Layers': layer_counts['pooling'],
        'Normalization Layers': layer_counts['normalization'],
        'Recurrent Layers': layer_counts['recurrent'],
        'Output Features': output_features,
        'Training Dataset': dataset_used,
        'RF25 Layer Number': target_layer_index,
        'RF25 Layer Name': closest_layer[0],
        'Architecture': default['architecture'],
        'Crop Percentage': crop,
        'Family': family,
        #**param_stats,
    }

    return summary

def give_summaries():

    model_names = timm.list_models(pretrained=True)
    model_summaries = [summarize_model(model_name) for model_name in model_names]


    csv_columns = list(model_summaries[0].keys())  # Assuming all summaries have the same keys
    csv_file = "../data_storage/results/models_summary.csv"

    try:
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in model_summaries:
                writer.writerow(data)
    except IOError:
        print("I/O error")


print('\n\n\n\n', 'Start:', datetime.now(), '\n')
give_summaries()
print('\n\n\n\n', 'End:', datetime.now())