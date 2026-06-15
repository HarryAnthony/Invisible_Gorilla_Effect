from typing import List

import torch


def get_layer_names(model: torch.nn.Module, types: List):
    """
    Retrieve layer names for modules matching any type in ``types``.
    """
    layer_names = []

    for name, layer in model.named_modules():
        for layer_definition in types:
            if isinstance(layer, layer_definition) or issubclass(layer.__class__, layer_definition):
                if name not in layer_names:
                    layer_names.append(name)

    return layer_names
