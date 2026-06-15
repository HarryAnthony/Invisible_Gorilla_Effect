import torch
import torch.nn as nn
import torch.nn.functional as F

# Global hyperparameters (matching the Keras settings)
USE_BIAS = False
WEIGHT_DECAY = 0.0005
# In PyTorch, we use kaiming (He) initialization later.

###############################################################################
# Wide Residual Basic Block (supports both down and up directions)
###############################################################################
class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dropout_probability=0.0, direction='down'):
        """
        For down direction:
          - First conv: kernel_size=3, stride=stride, padding=1.
          - Second conv: kernel_size=3, stride=1, padding=1.
          - Shortcut: if in_planes != out_planes then use 1x1 conv with stride=stride;
                      else if stride != 1 use AvgPool2d.
        For up direction:
          - First conv: kernel_size=3, stride=1, padding=1, then upsample (scale factor=stride).
          - Second conv: kernel_size=3, stride=1, padding=1.
          - Shortcut: if in_planes != out_planes then use 1x1 conv then upsample; otherwise upsample.
        """
        super(WideBasicBlock, self).__init__()
        self.direction = direction
        self.dropout_probability = dropout_probability
        self.relu = nn.ReLU(inplace=True)

        if direction == 'up':
            # Both conv layers use stride 1; we later upsample after the first conv.
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=USE_BIAS)
            self.upsample = nn.Upsample(scale_factor=stride, mode='nearest')
            self.bn2 = nn.BatchNorm2d(out_planes)
            self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=USE_BIAS)
        else:
            # Downward block: first conv uses the given stride.
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=USE_BIAS)
            self.bn2 = nn.BatchNorm2d(out_planes)
            self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=USE_BIAS)

        if dropout_probability > 0:
            self.dropout = nn.Dropout(dropout_probability)
        else:
            self.dropout = None

        # Shortcut connection
        if in_planes != out_planes:
            if direction == 'up':
                self.shortcut_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=USE_BIAS)
                self.shortcut_upsample = nn.Upsample(scale_factor=stride, mode='nearest')
            else:
                self.shortcut_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=USE_BIAS)
        else:
            self.shortcut_conv = None
            # When dimensions match but stride > 1, use pooling (for down) or upsampling (for up)
            if stride != 1:
                if direction == 'up':
                    self.shortcut_upsample = nn.Upsample(scale_factor=stride, mode='nearest')
                else:
                    self.avg_pool = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        if self.direction == 'up':
            # Branch: BN -> ReLU -> Conv -> Upsample -> BN -> ReLU -> (Dropout) -> Conv
            out = self.bn1(x)
            out = self.relu(out)
            out = self.conv1(out)
            out = self.upsample(out)
            out = self.bn2(out)
            out = self.relu(out)
            if self.dropout is not None:
                out = self.dropout(out)
            out = self.conv2(out)
        else:
            # Downward branch: BN -> ReLU -> Conv -> BN -> ReLU -> (Dropout) -> Conv
            out = self.bn1(x)
            out = self.relu(out)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.relu(out)
            if self.dropout is not None:
                out = self.dropout(out)
            out = self.conv2(out)

        # Shortcut branch
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(x)
            if self.direction == 'up':
                shortcut = self.shortcut_upsample(shortcut)
        else:
            if self.direction == 'up' and hasattr(self, 'shortcut_upsample'):
                shortcut = self.shortcut_upsample(x)
            elif self.direction == 'down' and hasattr(self, 'avg_pool'):
                shortcut = self.avg_pool(x)
            else:
                shortcut = x

        return out + shortcut

###############################################################################
# Helper function to stack residual blocks (like the Keras _layer)
###############################################################################
def make_layer(block, in_planes, out_planes, count, stride, dropout_probability=0.0, direction='down'):
    layers = []
    layers.append(block(in_planes, out_planes, stride, dropout_probability, direction))
    for _ in range(1, count):
        layers.append(block(out_planes, out_planes, stride=1, dropout_probability=dropout_probability, direction=direction))
    return nn.Sequential(*layers)

###############################################################################
# Wide Residual Encoder–Decoder Network (decoder variant)
#
# This implementation corresponds to the create_wide_residual_network_dec
# function in the TensorFlow code.
#
# Parameters:
#   - input_channels: number of input channels (e.g. 1 for grayscale)
#   - num_classes: number of output channels; for self-supervised regression,
#                  set this to 1.
#   - depth: overall depth (must satisfy (depth - 6) % 10 == 0)
#   - k: widening factor (default 4)
#   - dropout_probability: dropout rate
#   - final_activation: 'sigmoid', 'softmax', or 'linear'
###############################################################################

class WideResNetDec(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, depth=16, k=4, dropout_probability=0.0, final_activation='sigmoid'):
        super(WideResNetDec, self).__init__()
        print(num_classes)
        assert (depth - 6) % 10 == 0, "depth should be 10n+6"
        n = (depth - 6) // 10
        # Define stage channel sizes
        n_stages = [16, 16*k, 32*k, 64*k, 64*k, 64*k]

        # Encoder
        self.conv1 = nn.Conv2d(input_channels, n_stages[0], kernel_size=3, stride=1, padding=1, bias=USE_BIAS)
        self.layer1 = make_layer(WideBasicBlock, n_stages[0], n_stages[1], count=n, stride=1, dropout_probability=dropout_probability, direction='down')
        self.layer2 = make_layer(WideBasicBlock, n_stages[1], n_stages[2], count=n, stride=2, dropout_probability=dropout_probability, direction='down')
        self.layer3 = make_layer(WideBasicBlock, n_stages[2], n_stages[3], count=n, stride=2, dropout_probability=dropout_probability, direction='down')
        self.layer4 = make_layer(WideBasicBlock, n_stages[3], n_stages[4], count=n, stride=2, dropout_probability=dropout_probability, direction='down')
        self.layer5 = make_layer(WideBasicBlock, n_stages[4], n_stages[5], count=n, stride=2, dropout_probability=dropout_probability, direction='down')

        # Decoder (using blocks with direction 'up')
        # For the decoder, we use one block per upsampling stage.
        self.upconv1 = make_layer(WideBasicBlock, n_stages[5], n_stages[2], count=1, stride=2, dropout_probability=dropout_probability, direction='up')
        self.upconv2 = make_layer(WideBasicBlock, n_stages[2], n_stages[1], count=1, stride=2, dropout_probability=dropout_probability, direction='up')
        self.upconv3 = make_layer(WideBasicBlock, n_stages[1], n_stages[0], count=1, stride=2, dropout_probability=dropout_probability, direction='up')
        self.upconv4 = make_layer(WideBasicBlock, n_stages[0], num_classes, count=1, stride=2, dropout_probability=dropout_probability, direction='up')

        # Final activation
        if final_activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif final_activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif final_activation == 'linear':
            self.activation = nn.Identity()
        else:
            raise ValueError("Unsupported final activation: " + final_activation)

        # Weight initialization (Kaiming Normal)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Encoder
        out = self.conv1(x)           # (B, n_stages[0], H, W)
        out = self.layer1(out)        # (B, n_stages[1], H, W)
        out = self.layer2(out)        # (B, n_stages[2], H/2, W/2)
        out = self.layer3(out)        # (B, n_stages[3], H/4, W/4)
        out = self.layer4(out)        # (B, n_stages[4], H/8, W/8)
        out = self.layer5(out)        # (B, n_stages[5], H/16, W/16)
        # Decoder
        out = self.upconv1(out)       # (B, n_stages[2], H/8, W/8)
        out = self.upconv2(out)       # (B, n_stages[1], H/4, W/4)
        out = self.upconv3(out)       # (B, n_stages[0], H/2, W/2)
        out = self.upconv4(out)       # (B, num_classes, H, W)
        out = self.activation(out)
        return out

import torch
import torch.nn as nn


class WideResNetEncoder(nn.Module):
    def __init__(self, input_channels=3, feature_dim=32, depth=16, k=4, dropout_probability=0.0, input_size=224, num_classes=10):
        super(WideResNetEncoder, self).__init__()

        assert (depth - 6) % 10 == 0, "Depth should be 10n+6"
        n = (depth - 6) // 10

        # Define stage channel sizes
        n_stages = [16, 16 * k, 32 * k, 64 * k, 64 * k, 64 * k]

        # Encoder layers
        self.conv1 = nn.Conv2d(input_channels, n_stages[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.layer1 = make_layer(WideBasicBlock, n_stages[0], n_stages[1], count=n, stride=1, dropout_probability=dropout_probability, direction='down')
        self.layer2 = make_layer(WideBasicBlock, n_stages[1], n_stages[2], count=n, stride=2, dropout_probability=dropout_probability, direction='down')
        self.layer3 = make_layer(WideBasicBlock, n_stages[2], n_stages[3], count=n, stride=2, dropout_probability=dropout_probability, direction='down')
        self.layer4 = make_layer(WideBasicBlock, n_stages[3], n_stages[4], count=n, stride=2, dropout_probability=dropout_probability, direction='down')
        self.layer5 = make_layer(WideBasicBlock, n_stages[4], n_stages[5], count=n, stride=2, dropout_probability=dropout_probability, direction='down')

        # Compute final feature map size dynamically
        self.final_channels = n_stages[5]
        self.feature_map_size = self._get_feature_map_size(input_size)

        # Fully connected layer for classification (used only in classifier mode)
        self.fc = nn.Linear(self.final_channels * self.feature_map_size, feature_dim)

        # Output layer for classification
        self.classifier = nn.Linear(feature_dim, num_classes)  # Only used for classification mode

        # Weight Initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_feature_map_size(self, input_size):
        """Dynamically compute the spatial size after downsampling."""
        size = input_size
        for _ in range(4):  # 5 downsampling layers (stride=2)
            size = size // 2
        return size * size  # Final spatial area

    def forward(self, x, mode="feature"):
        """
        mode:
        - "feature": Return feature maps (for autoencoder)
        - "embedding": Return flattened embedding (for SVDD)
        - "classification": Return class logits (for classification tasks)
        """
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)



        if mode == "feature":
            return out  # Return feature maps for autoencoder decoding
        

        #print(f"Shape before flattening: {out.shape}")

        out = out.view(out.size(0), -1)  # Flatten feature maps

        #print(f"Shape after flattening: {out.shape}")

        if mode == "embedding":
            return self.fc(out)  # Return embedding for Deep SVDD

        elif mode == "classification":
            out = self.fc(out)  # Convert feature maps to feature vector
            return self.classifier(out)  # Return class logits

        else:
            raise ValueError("Invalid mode. Choose from ['feature', 'embedding', 'classification']")


    
class WideResNetDecoder(nn.Module):
    def __init__(self, feature_dim=32, output_channels=3, depth=16, k=4, dropout_probability=0.0, final_activation='linear'):
        super(WideResNetDecoder, self).__init__()

        assert (depth - 6) % 10 == 0, "Depth should be 10n+6"
        n = (depth - 6) // 10

        # Define stage channel sizes (reversed)
        n_stages = [16, 16 * k, 32 * k, 64 * k, 64 * k, 64 * k]

        # Upsampling layers
        #self.fc = nn.Linear(feature_dim, n_stages[5] * 14 * 14)  # Adjust based on spatial resolution
        self.upconv1 = make_layer(WideBasicBlock, n_stages[5], n_stages[2], count=1, stride=2, dropout_probability=dropout_probability, direction='up')
        self.upconv2 = make_layer(WideBasicBlock, n_stages[2], n_stages[1], count=1, stride=2, dropout_probability=dropout_probability, direction='up')
        self.upconv3 = make_layer(WideBasicBlock, n_stages[1], n_stages[0], count=1, stride=2, dropout_probability=dropout_probability, direction='up')
        self.upconv4 = make_layer(WideBasicBlock, n_stages[0], output_channels, count=1, stride=2, dropout_probability=dropout_probability, direction='up')

        # Final convolution layer (ensure output matches input channels)
        #self.final_conv = nn.Conv2d(n_stages[0], output_channels, kernel_size=3, stride=1, padding=1)

        # Final activation function
        if final_activation == 'sigmoid':  
            self.activation = nn.Sigmoid()
        elif final_activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif final_activation == 'linear':  
            self.activation = nn.Identity()
        else:
            raise ValueError("Unsupported final activation: " + final_activation)

    def forward(self, x):
        # Decode the feature representation
        out = self.upconv1(x)
        out = self.upconv2(out)
        out = self.upconv3(out)
        out = self.upconv4(out)
        #out = self.final_conv(out)
        #out = self.activation(out)  # Apply final activation

        return out  # Reconstructed image


class WideResNetAutoencoder(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, feature_dim=32, depth=16, k=4, dropout_probability=0.0, final_activation='linear', num_classes=2):
        super(WideResNetAutoencoder, self).__init__()

        self.encoder = WideResNetEncoder(
            input_channels=input_channels,
            feature_dim=feature_dim,
            depth=depth,
            k=k,
            dropout_probability=dropout_probability,
            num_classes=num_classes  # Needed for classification mode
        )

        self.decoder = WideResNetDecoder(
            feature_dim=feature_dim,
            output_channels=output_channels,
            depth=depth,
            k=k,
            dropout_probability=dropout_probability,
            final_activation=final_activation
        )

    def forward(self, x, mode="autoencoder"):
        """
        Modes:
        - "autoencoder": Train as an autoencoder (default).
        - "embedding": Use encoder to extract features for Deep SVDD.
        - "classification": Use encoder for classification (output class logits).
        """
        if mode == "autoencoder":
            features = self.encoder(x, mode="feature")  # Get feature maps
            reconstructed = self.decoder(features)  # Decode to reconstruct
            return reconstructed

        elif mode in ["embedding", "classification"]:
            return self.encoder(x, mode=mode)  # Extract features for SVDD or classification

        else:
            raise ValueError("Invalid mode. Choose from ['autoencoder', 'embedding', 'classification']")

    def get_encoder(self):
        """Extract the encoder for Deep SVDD or other feature extraction tasks."""
        return self.encoder
