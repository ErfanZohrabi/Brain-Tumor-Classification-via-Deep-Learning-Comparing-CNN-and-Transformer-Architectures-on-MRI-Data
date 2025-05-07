"""
Model definitions for Brain Tumor Classification.
"""

import torch
import torch.nn as nn
from torchvision.models import (
    vit_b_16, ViT_B_16_Weights,
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)
import timm  # Added timm support


class ViTClassifierTorchvision(nn.Module):
    """
    Vision Transformer model from torchvision for brain tumor classification.
    (Uni-modal 2D input)
    """
    def __init__(self, num_classes: int, img_size: int = 384):
        super().__init__()
        # Load ViT-B/16 with recommended weights (ImageNet1K + SWAG)
        w = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        self.net = vit_b_16(weights=w, image_size=img_size)
        # Replace the classifier head to match our number of classes
        self.net.heads.head = nn.Linear(self.net.heads.head.in_features, num_classes)

    def forward(self, x):
        return self.net(x)


class ResNetClassifierTorchvision(nn.Module):
    """
    ResNet-based model from torchvision for brain tumor classification.
    (Uni-modal 2D input)
    """
    def __init__(self, num_classes: int):
        super().__init__()
        # Load ResNet50 with pretrained weights
        w = ResNet50_Weights.IMAGENET1K_V2
        self.net = resnet50(weights=w)
        # Replace the final fully connected layer
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.net(x)


class EfficientNetClassifierTorchvision(nn.Module):
    """
    EfficientNet-based model from torchvision for brain tumor classification.
    (Uni-modal 2D input)
    """
    def __init__(self, num_classes: int):
        super().__init__()
        # Load EfficientNet with pretrained weights
        w = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.net = efficientnet_b0(weights=w)
        # Replace the classifier
        self.net.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes)
        )
        
    def forward(self, x):
        return self.net(x)


# --- New Hybrid Model Definition ---
class SwinTransformerHybridClassifier(nn.Module):
    """
    Hybrid classifier using a Swin Transformer backbone and a simple classification head.
    (Uni-modal 2D input)

    Args:
        swin_model_name: Name of the Swin Transformer model from timm (e.g., 'swin_base_patch4_window7_224')
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights for the Swin backbone
        img_size: Input image size (should match Swin pretraining if using pretrained weights)
    """
    def __init__(self, swin_model_name: str, num_classes: int, pretrained: bool = True, img_size: int = 224):
        super().__init__()

        # Load the Swin Transformer backbone without the classifier head
        self.backbone = timm.create_model(swin_model_name, pretrained=pretrained, num_classes=0, global_pool='') # num_classes=0 removes the head, global_pool removes pooling before head

        # Get the number of features from the backbone's output
        # This depends on the Swin model and global_pool setting
        # With global_pool='', we get feature maps, need to figure out channels
        # A common approach is to global average pool manually or flatten
        # Let's use global average pooling on the output features
        # The feature info should tell us the output channels of the last stage
        # Alternatively, inspect the model structure or use timm's feature info
        # For Swin, the last stage output before pooling is usually flattened
        # Let's get the number of features that would go into a head
        feature_info = timm.get_feature_info(swin_model_name)
        num_features = feature_info[-1]['num_chs'] # Get channels of the last stage


        # Define the classification head
        # This can be simple (GAP + Linear) or more complex
        self.classifier = nn.Sequential(
            # Optional: Add a Conv layer or other CNN-like processing here
            # nn.Conv2d(num_features, some_intermediate_features, kernel_size=1),
            # nn.AdaptiveAvgPool2d(1), # If backbone output is spatial feature maps
            nn.Flatten(), # Swin's output before head is often flattened tokens
            nn.Linear(num_features, num_classes)
        )
        print(f"Initialized Swin Transformer backbone '{swin_model_name}' with custom classification head.")


    def forward(self, x):
        # Pass input through the backbone
        # Note: Swin might output feature maps or pooled features depending on global_pool
        # With global_pool='', it outputs feature maps/tokens from the last stage
        features = self.backbone(x)

        # Pass features through the classification head
        logits = self.classifier(features)
        return logits


# Factory function to create models by name
def create_model(model_name: str, num_classes: int, pretrained: bool = True, **kwargs):
    """
    Create a model by name.

    Args:
        model_name: Name of the model ('se_resnet50', 'swin_hybrid',
                                      'vit_torchvision', 'resnet_torchvision',
                                      'efficientnet_torchvision', 'convnext_base',
                                      'efficientnet_b0', 'efficientnet_b3', 'densenet121',
                                      'regnety_032', 'maxvit_tiny_rw_224', 'inception_v3',
                                      'coat_lite_medium', or any other timm model)
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
        **kwargs: Additional arguments to pass to the model constructor (e.g., img_size, swin_model_name for 'swin_hybrid')

    Returns:
        Instantiated model
    """
    # torchvision model map (optional)
    torchvision_model_map = {
        'vit_torchvision': ViTClassifierTorchvision,
        'resnet_torchvision': ResNetClassifierTorchvision,
        'efficientnet_torchvision': EfficientNetClassifierTorchvision,
    }

    if model_name in torchvision_model_map:
        # Handle specific args like img_size for torchvision ViT
        if model_name == 'vit_torchvision' and 'img_size' in kwargs:
             return torchvision_model_map[model_name](num_classes, img_size=kwargs['img_size'])
        return torchvision_model_map[model_name](num_classes)

    elif model_name == 'se_resnet50':
         # SE-ResNet50 from timm (Uni-modal 2D)
         # Using 'resnet50.gluon_se' or similar from timm that includes SE blocks
         # Check timm.list_models('*se*resnet*') for exact names
         try:
             model = timm.create_model('resnet50.gluon_se', pretrained=pretrained, num_classes=num_classes, **kwargs)
             print(f"Successfully loaded SE-ResNet50 from timm (pretrained={pretrained}).")
             return model
         except Exception as e:
             raise ValueError(f"Could not load SE-ResNet50 from timm. Check model name or timm version. Error: {e}")

    elif model_name == 'swin_hybrid':
         # Swin Transformer Hybrid model (Uni-modal 2D)
         # Requires 'swin_model_name' in kwargs, e.g., 'swin_base_patch4_window7_224'
         swin_model_name = kwargs.get('swin_model_name', 'swin_base_patch4_window7_224') # Default Swin base
         img_size = kwargs.get('img_size', 224) # Default img_size for Swin base

         if 'swin' not in swin_model_name:
             raise ValueError(f"Model name '{swin_model_name}' is not a Swin model for 'swin_hybrid'.")

         print(f"Creating Swin Hybrid model with backbone '{swin_model_name}'.")
         return SwinTransformerHybridClassifier(
             swin_model_name=swin_model_name,
             num_classes=num_classes,
             pretrained=pretrained,
             img_size=img_size # Pass img_size to hybrid class if needed internally (though timm handles it mostly)
         )
    
    # New benchmark models - direct access through timm
    elif model_name in ['convnext_base', 'efficientnet_b0', 'efficientnet_b3', 
                         'densenet121', 'regnety_032', 'maxvit_tiny_rw_224', 
                         'inception_v3', 'coat_lite_medium']:
        # Direct access through timm
        try:
            # Handle image size for some models that require it
            if 'vit' in model_name or 'swin' in model_name or 'maxvit' in model_name:
                img_size = kwargs.get('img_size', 224)
                model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, img_size=img_size)
            else:
                model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
            
            print(f"Successfully loaded {model_name} from timm (pretrained={pretrained}).")
            return model
        except Exception as e:
            raise ValueError(f"Could not load {model_name} from timm. Error: {e}")

    else:
        # Fallback to loading any other model name directly from timm
        # Need to handle img_size for ViT/Swin style models if not default
        try:
            if any(m in model_name for m in ['vit', 'swin', 'beit']) and 'img_size' in kwargs:
                 model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, img_size=kwargs['img_size'])
            else:
                 model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, **kwargs)

            print(f"Successfully loaded model '{model_name}' from timm (pretrained={pretrained}).")
            return model
        except Exception as e:
            benchmark_models = ['convnext_base', 'efficientnet_b0', 'efficientnet_b3', 
                               'densenet121', 'regnety_032', 'maxvit_tiny_rw_224', 
                               'inception_v3', 'coat_lite_medium', 'vit_base_patch16_224', 
                               'swin_base_patch4_window7_224']
            raise ValueError(
                f"Unknown model: {model_name}. Not found in torchvision map, specific names, or timm. Error: {e}. "
                f"Available torchvision models: {list(torchvision_model_map.keys())}. "
                f"Supported specific timm names: {['se_resnet50', 'swin_hybrid']}. "
                f"Supported benchmark models: {benchmark_models}. "
                f"For other timm models, ensure the name is correct (check timm.list_models()). "
                f"For 'swin_hybrid', ensure 'swin_model_name' is provided in kwargs if not default."
            ) 