import torch
import torch.nn as nn
from torchvision import models


class GatedAttentionMIL(nn.Module):
    def __init__(self, backbone='b0', D=512, K=1):
        super().__init__()

        # 1. Dynamic Backbone Selection
        model_map = {
            'b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 1280),
            'b1': (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT, 1280),
            'b2': (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT, 1408),
            'b3': (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT, 1536),
            'b4': (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT, 1792),
            'b5': (models.efficientnet_b5, models.EfficientNet_B5_Weights.DEFAULT, 2048),
            'b6': (models.efficientnet_b6, models.EfficientNet_B6_Weights.DEFAULT, 2304),
            'b7': (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT, 2560),
        }

        if backbone not in model_map:
            raise ValueError(f"Backbone {backbone} not supported.")

        model_fn, weight_fn, self.L = model_map[backbone]
        base = model_fn(weights=weight_fn)

        # Feature Extractor (removes the original classifier)
        self.feature_extractor = nn.Sequential(
            base.features,
            base.avgpool
        )

        # Freeze backbone to start (as per your logic)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # 2. Gated Attention (Using D=512 for higher capacity)
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, D),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, D),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(D, K)

        # 3. Classifier (Deepened for large backbones)
        self.classifier = nn.Sequential(
            nn.Linear(self.L, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Added dropout for medical imaging robustness
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.attention_V, self.attention_U, self.attention_w, self.classifier]:
            if isinstance(m, nn.Sequential):
                for layer in m:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        print(f"--- MIL [{self.L} features] Attention and Classifier weights initialized ---")

    def forward(self, x):
        # x shape: [num_frames, 3, H, W]
        h = self.feature_extractor(x)
        h = torch.flatten(h, 1)  # [num_frames, L]

        # Gated Attention Mechanism
        a_v = self.attention_V(h)
        a_u = self.attention_U(h)
        a = self.attention_w(a_v * a_u)  # [num_frames, K]
        weights = torch.softmax(a, dim=0)  # Attention over frames

        # Bag representation (Weighted sum of frames)
        bag_representation = torch.sum(weights * h, dim=0)  # [L]

        # Classification
        logits = self.classifier(bag_representation.unsqueeze(0))  # [1, 1]

        return logits, weights