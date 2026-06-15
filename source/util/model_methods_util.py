import torch
from torch import nn
from collections import OrderedDict

def add_methods_to_model(net):
    """
    Adds methods to the model for:
    `backbone`: returns the output of the last layer before the head.
    `head`: returns the output of the last hidden layer.
    `get_head_weights`: returns the weights of the head (last fully connected layer).
    `set_head_weights`: sets new weights and biases to the head layer.
    `get_activation`: returns the activation of a specified layer.
    `list_layers`: returns a list of all layers with their names.
    """

    # Define backbone and head based on architecture
    if hasattr(net, 'avgpool') and hasattr(net, 'fc'): #resnet
        def backbone(self,input):
            x = net.conv1(input)
            x = net.bn1(x)
            x = net.relu(x)
            x = net.maxpool(x)

            x = net.layer1(x)
            x = net.layer2(x)
            x = net.layer3(x)
            x = net.layer4(x)
            return x

        def head(self,input):
            x = net.backbone(input)
            x = net.avgpool(x)
            x = torch.flatten(x, 1)
            return x

        def get_head_weights(self):
            return net.fc.weight, net.fc.bias

        def set_head_weights(self,new_weight, new_bias=None):
            net.fc.weight.data = new_weight
            if new_bias is not None:
                net.fc.bias.data = new_bias

        def apply_head(self,features):
            """
            Applies the last layer (fc) on the input features to produce logits.

            Args:
                features (torch.Tensor): A tensor of shape [batch_size, 2048] for ResNet50.
                                         The input shape must match the input size expected by the fc layer.

            Returns:
                torch.Tensor: The logits produced by the head layer.
            """
            return net.fc(features)
        
        def apply_layers_after_backbone(self,x):
            x = net.avgpool(x)
            x = torch.flatten(x, 1)
            return net.fc(x)
        
        

    elif hasattr(net, 'classifier') and isinstance(net.classifier, nn.Sequential): #vgg
        def backbone(self,input):
            x = net.features(input)
            x = net.avgpool(x) if hasattr(net, 'avgpool') else x
            #x = torch.flatten(x, 1)
            return x

        def head(self,input):
            x = net.backbone(input)
            x = torch.flatten(x, 1)
            x = net.classifier[:-1](x)
            return x

        def get_head_weights(self):
            return net.classifier[-1].weight, net.classifier[-1].bias

        def set_head_weights(self,new_weight, new_bias=None):
            net.classifier[-1].weight.data = new_weight
            if new_bias is not None:
                net.classifier[-1].bias.data = new_bias

        def apply_head(self,features):
            """
            Applies the last layer in the classifier on the input features to produce logits.

            Args:
                features (torch.Tensor): A tensor with the correct shape for the final classifier layer.
                                         The input shape must match the expected input size of the last layer.

            Returns:
                torch.Tensor: The logits produced by the head layer.
            """
            return net.classifier[-1](features)
        
        def apply_layers_after_backbone(self,x):
            x = torch.flatten(x, 1)
            x = net.classifier[:-1](x)
            return net.classifier[-1](x)

    elif hasattr(net, 'encoder') and hasattr(net, 'conv_proj'): #vision transformer
        
        def backbone(self, input):
            """
            Processes the input up to the encoder layer and returns the encoder's output.
            """
            # Ensure the input has 3 channels; repeat channels if grayscale
            if input.shape[1] == 1:
                input = input.repeat(1, 3, 1, 1)

            # Step 1: Pass through initial projection layer
            x = self.conv_proj(input)  # [batch_size, hidden_dim, height, width]

            # Step 2: Flatten spatial dimensions to create a sequence of patches
            x = x.flatten(2).transpose(1, 2)  # Shape: [batch_size, num_patches, hidden_dim]

            # Step 3: Add the class token (CLS) to the beginning of the sequence
            cls_token = self.class_token.expand(x.shape[0], -1, -1)  # Shape: [batch_size, 1, hidden_dim]
            x = torch.cat((cls_token, x), dim=1)  # Concatenate along the sequence dimension

            # Step 4: Add positional embeddings and apply dropout
            x = x + self.encoder.pos_embedding
            x = self.encoder.dropout(x)

            # Step 5: Forward pass through all transformer encoder layers
            x = self.encoder(x)  # Process through the encoder layers

            return x

        #    return self.get_activation(input,layer_name='encoder')
           

        def head(self, input):
            x = self.backbone(input)
            return x.unsqueeze(1)

        def get_head_weights(self):
            return net.heads.head.weight, net.heads.head.bias

        def set_head_weights(self, new_weight, new_bias=None):
            net.heads.head.weight.data = new_weight
            if new_bias is not None:
                net.heads.head.bias.data = new_bias

        def apply_head(self, features):
            #input(features.shape)
            if features.dim() == 4:
                features = features.squeeze(1)  # Remove the extra dimension added by the head method
            cls_token = features[:, 0]  # Shape: [batch_size, hidden_dim]

            logits = self.heads.head(cls_token)  # Shape: [batch_size, num_classes]
            #input(logits.shape)
            return logits
        
        def apply_layers_after_backbone(self,x):
            x.unsqueeze(1)
            if x.dim() == 4:
                x = x.squeeze(1)  # Remove the extra dimension added by the head method
            cls_token = x[:, 0]  # Shape: [batch_size, hidden_dim]

            logits = self.heads.head(cls_token)  # Shape: [batch_size, num_classes]
            #input(logits.shape)
            return logits
        

    elif hasattr(net, 'stages') and hasattr(net, 'head'):  # swin transformer
        def backbone(self, input):
            """
            Extract features from the input using the Swin Transformer backbone up to the stage outputs.

            Args:
                input (torch.Tensor): Input tensor with shape [batch_size, 3, height, width].

            Returns:
                torch.Tensor: The output features from the final stage of the Swin Transformer.
            """
            x = net.patch_embed(input)  # Initial patch embedding
            if net.absolute_pos_embed is not None:
                x = x + net.absolute_pos_embed
            x = net.pos_drop(x)  # Apply positional dropout

            # Pass through each stage of the Swin Transformer
            for stage in net.stages:
                x = stage(x)

            return x

        def head(self, input):
            """
            Pass the extracted features through the Swin Transformer head to produce logits.

            Args:
                input (torch.Tensor): Feature tensor from the backbone with shape [batch_size, num_patches, hidden_dim].

            Returns:
                torch.Tensor: Logits tensor with shape [batch_size, num_classes].
            """
            x = net.norm(input)  # Normalize features
            x = net.avgpool(x.transpose(1, 2))  # Global average pooling, transpose to [batch_size, hidden_dim, num_patches]
            x = torch.flatten(x, 1)  # Flatten to [batch_size, hidden_dim]
            x = net.head(x)  # Pass through the final head (linear layer)
            return x

        def get_head_weights(self):
            """
            Retrieve the weights and bias of the final classification layer.

            Returns:
                tuple: A tuple containing the weight and bias tensors of the final linear layer.
            """
            return net.head.weight, net.head.bias

        def set_head_weights(self, new_weight, new_bias=None):
            """
            Set new weights and bias for the final classification layer.

            Args:
                new_weight (torch.Tensor): New weights for the head.
                new_bias (torch.Tensor, optional): New bias for the head. Default is None.
            """
            net.head.weight.data = new_weight
            if new_bias is not None:
                net.head.bias.data = new_bias

        def apply_head(self, features):
            """
            Applies the classification head on extracted features to produce logits.

            Args:
                features (torch.Tensor): Input tensor with shape [batch_size, hidden_dim].

            Returns:
                torch.Tensor: The logits produced by the head layer.
            """
            return net.head(features)


    else:
        raise ValueError("The model architecture is not recognized. Cannot determine the last hidden layer.")

    # Method to get activation of any layer
    def get_activation(self,input,layer_name):
        activations = {}

        def hook_fn(module, input, output):
            activations[layer_name] = output

        submodule = dict(net.named_modules()).get(layer_name)
        if submodule is None:
            raise ValueError(f"Layer '{layer_name}' not found in the model.")
        
        handle = submodule.register_forward_hook(hook_fn)
        net(input)
        handle.remove()
        return activations[layer_name]
    
    def apply_remaining_layers(self, activation, start_layer_name):
            """
            Applies the remaining layers after the specified layer on an activation input,
            correctly handling skip connections and adaptive pooling for models like ResNet.

            Args:
                activation (torch.Tensor): The activation output from the start_layer_name.
                start_layer_name (str): The name of the layer from which to start applying remaining layers.

            Returns:
                torch.Tensor: The output after applying the remaining layers.
            """
            # Get all layers of the model as an ordered dictionary
            all_layers = list(self.named_children())

            # Verify the start layer is valid and find its position
            start_layer_idx = None
            for idx, (name, _) in enumerate(all_layers):
                if name == start_layer_name:
                    start_layer_idx = idx
                    break
            if start_layer_idx is None:
                raise ValueError(f"Layer '{start_layer_name}' not found in the model.")

            # Build a submodel for the remaining layers
            class RemainingLayersModel(nn.Module):
                def __init__(self, layers, has_avgpool, has_fc):
                    super(RemainingLayersModel, self).__init__()
                    self.layers = nn.ModuleList(layers)
                    self.has_avgpool = has_avgpool
                    self.has_fc = has_fc

                def forward(self, x):
                    for layer in self.layers:
                        x = layer(x)
                    
                    # Apply avgpool and flatten if needed before the fully connected layer
                    if self.has_avgpool:
                        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
                    if self.has_fc:
                        x = torch.flatten(x, 1)
                    return x

            # Determine if avgpool and fc are part of the remaining layers
            has_avgpool = 'avgpool' in dict(all_layers[start_layer_idx + 1:])
            has_fc = 'fc' in dict(all_layers[start_layer_idx + 1:])

            # Collect remaining layers starting from start_layer_name
            remaining_layers = [layer for _, layer in all_layers[start_layer_idx + 1:]]
            remaining_model = RemainingLayersModel(remaining_layers, has_avgpool, has_fc)

            # Forward pass through the remaining layers model
            output = remaining_model(activation)
            
            # Apply final fully connected layer if it exists
            if has_fc:
                output = self.fc(output)  # Assumes the final layer is called `fc`

            return output

    #Bind the new method to the model instance
    net.apply_remaining_layers = apply_remaining_layers.__get__(net)


    # Method to list all layers in the model with names
    def list_layers(self):
        return [name for name, _ in self.named_modules() if "out_proj" not in name][1:]
        #return [name for name, _ in net.named_modules()][1:]
        #return [(name, module) for name, module in net.named_modules()]

    def list_layers_with_modules(self):
        return [(name, module) for name, module in net.named_modules() if "out_proj" not in name][1:]


    def get_all_activations(self, input):
        """
        Runs the model on the input and returns a dictionary with activations
        for all layers.

        Args:
            input (torch.Tensor): The input to the model.

        Returns:
            dict: A dictionary with layer names as keys and corresponding activations as values.
        """
        activations = {}

        # Hook function to store each layer's output

        def hook_fn(name):
            #print(name)
            def hook(module, input, output):
                # If the layer is a self-attention layer, only take the first element of the tuple
                if "self_attention" in name and isinstance(output, tuple):
                    #print(name)
                    output = output[0]
                if isinstance(output, torch.Tensor):
                    #print(name)
                    #activations.append(output)
                    activations[name] = output
                else:
                    raise TypeError(f"Expected output to be a torch.Tensor, but got {type(output)} in layer {name}")
            return hook

        # Register hooks for all layers in the model
        handles = []
        for name_idx, (name, layer) in enumerate(self.named_modules()):
            if name_idx != 0:  # Skip the first layer (input)
                handles.append(layer.register_forward_hook(hook_fn(name)))

        # Run a forward pass to capture activations
        self(input)

        # Remove all hooks after forward pass
        for handle in handles:
            handle.remove()

        return activations
    
    def feature_extractor(self, input, module_names = []):
        """
        Extracts features from the model for the specified modules.

        Args:
            input (torch.Tensor): The input to the model.
            module_names (list): A list of module names for which features are to be extracted.

        Returns:
            dict: A dictionary with module names as keys and corresponding activations as values.
        """
        activations = {}

        # Hook function to store each layer's output
        def hook_fn(name):
            def hook(module, input, output):
                # If the layer is a self-attention layer, only take the first element of the tuple
                if "self_attention" in name and isinstance(output, tuple):
                    output = output[0]

                if isinstance(output, torch.Tensor):
                    #print(name)
                    #activations.append(output)
                    activations[name] = output
                else:
                    raise TypeError(f"Expected output to be a torch.Tensor, but got {type(output)} in layer {name}")
            return hook

        # Register hooks for specified layers in the model
        handles = []
        for name_idx, (name, layer) in enumerate(self.named_modules()):
            if name in module_names:
                handles.append(layer.register_forward_hook(hook_fn(name)))

        # Run a forward pass to capture activations
        self(input)

        # Remove all hooks after forward pass
        for handle in handles:
            handle.remove()

        # Sort activations to match the order of module_names
        activations = {name: activations[name] for name in module_names if name in activations}


        #activations_tensor = torch.cat([act.unsqueeze(0) for act in activations], dim=0)

        return activations

    # Bind the new methods to the model instance
    net.backbone = backbone.__get__(net)
    net.head = head.__get__(net)
    net.get_head_weights = get_head_weights.__get__(net)
    net.set_head_weights = set_head_weights.__get__(net)
    net.get_activation = get_activation.__get__(net)
    net.list_layers = list_layers.__get__(net)
    net.get_all_activations = get_all_activations.__get__(net)
    net.apply_head = apply_head.__get__(net)
    net.apply_remaining_layers = apply_remaining_layers.__get__(net)
    net.feature_extractor = feature_extractor.__get__(net)
    net.list_layers_with_modules = list_layers_with_modules.__get__(net)
    net.apply_layers_after_backbone = apply_layers_after_backbone.__get__(net)



