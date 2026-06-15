import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms.functional as T


def batch_random_rotation(images):
    """Apply random 0°, 90°, 180°, or 270° rotations to a batch of images."""
    batch_size = images.size(0)
    angles = [0, 90, 180, 270]
    
    # Generate random rotation labels for the batch
    rotation_labels = torch.randint(0, 4, (batch_size,))  # 0, 1, 2, or 3 corresponding to 0°, 90°, 180°, 270°
    
    # Apply rotations in a vectorized way
    rotated_images = torch.stack([T.rotate(img, angles[label]) for img, label in zip(images, rotation_labels)])
    
    return rotated_images, rotation_labels


class RotPredModel(nn.Module):
    def __init__(self, base_model, num_classes, use_rotation_head=True):
        super(RotPredModel, self).__init__()

        self.use_rotation_head = use_rotation_head
        
        # Remove the last layer (fully connected) to extract features
        if isinstance(base_model, models.ResNet):
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # Remove final FC layer
            in_features = base_model.fc.in_features
        elif isinstance(base_model, models.VGG):
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # Remove final classifier layer
            in_features = base_model.classifier[-1].in_features
        else:
            raise ValueError("Unsupported model. Use ResNet or VGG.")

        # Fully connected layer for classification
        self.classifier_head = nn.Linear(in_features, num_classes)

        # Fully connected layer for rotation prediction (4 classes: 0°, 90°, 180°, 270°)
        self.rotation_head = nn.Linear(in_features, 4)

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        features = torch.flatten(features, start_dim=1)  # Flatten before FC layers

        class_logits = self.classifier_head(features)

        if self.use_rotation_head:
            rot_logits = self.rotation_head(features)
            return class_logits, rot_logits
        
        return class_logits


