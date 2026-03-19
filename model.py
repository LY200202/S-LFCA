import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeatureGrouping(nn.Module):
    def __init__(self, in_channels, num_classes, projection_dim=128):
        super(FeatureGrouping, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes))

        self.projection = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim)
        )

    def forward(self, patch_tokens, cls_token):
        # Input shape: (B, C, H, W)
        batch_size, channels, h, w = patch_tokens.shape

        # (B, C, H, W) -> (B, H*W, C)
        features = patch_tokens.reshape(batch_size, channels, -1).transpose(1, 2)

        # Compute classification logits: (B, H*W, N)
        logits = self.classifier(features)

        # Compute assignment weights with Softmax
        # weights shape: (B, H*W, N)
        weights = F.softmax(logits, dim=-1)

        # Use the first N-1 groups for feature aggregation and reserve
        # the last group as an background category.
        effective_weights = weights[:, :, :-1]
        # effective_weights = weights

        # Feature aggregation by weighted summation
        # (B, N-1, H*W) @ (B, H*W, C) -> (B, N-1, C)
        features = self.projection(features)
        aggregated_features = torch.bmm(effective_weights.transpose(1, 2), features)

        # Compute the weighted average
        weight_sum = effective_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1, N-1)
        aggregated_features = aggregated_features / weight_sum.transpose(1, 2)

        # Flatten the N-1 grouped features: (B, (N-1) * C)
        # Normalize local features
        aggregated_features = F.normalize(aggregated_features, dim=-1)
        local_feature = aggregated_features
        output = aggregated_features.reshape(batch_size, -1)

        # Concatenate global and local features
        cls_token = F.normalize(cls_token, dim=-1)
        output = torch.cat([cls_token, output], dim=1)

        return output, local_feature


DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}


class Model(nn.Module):

    def __init__(
            self,
            model_name='dinov2_vitb14',
            num_trainable_blocks=4,
            norm_layer=True,
            use_FeatureGrouping=True,
            num_classes=64,
            projection_dim=128,
    ):
        super().__init__()

        assert model_name in DINOV2_ARCHS
        self.use_FeatureGrouping = use_FeatureGrouping
        self.num_channels = DINOV2_ARCHS[model_name]
        self.norm_layer = norm_layer

        # -------- DINOv2 backbone --------
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)

        total_blocks = len(self.model.blocks)
        assert 0 <= num_trainable_blocks <= total_blocks, \
            f"num_trainable_blocks must be in [0, {total_blocks}]"
        self.num_trainable_blocks = num_trainable_blocks

        if self.num_trainable_blocks < total_blocks:
            for blk in self.model.blocks[:-self.num_trainable_blocks] if self.num_trainable_blocks > 0 else self.model.blocks:
                for p in blk.parameters():
                    p.requires_grad = False

        # -------- FeatureGrouping --------
        if use_FeatureGrouping:
            self.featuregrouping = FeatureGrouping(
                in_channels=self.num_channels,
                num_classes=num_classes,
                projection_dim=projection_dim
            )

        self.logit_scale1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.05))
        self.logit_scale2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.20))
        self.logit_scale3 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_config(self):
        patch_size = getattr(self.model.patch_embed, "patch_size", 14)
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]

        return {
            "input_size": (3, None, None),
            "patch_size": patch_size,
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "interpolation": "bicubic",
            "crop_pct": 1.0,
        }

    def _forward_single(self, img):

        B, _, H, W = img.shape

        assert H % 14 == 0 and W % 14 == 0, "Input size must be divisible by 14"

        x = self.model.prepare_tokens_with_masks(img)  # [B, 1+N, C]

        # ---- Forward through frozen blocks ----
        if self.num_trainable_blocks < len(self.model.blocks):
            frozen_blocks = self.model.blocks[:-self.num_trainable_blocks] if self.num_trainable_blocks > 0 else self.model.blocks
            with torch.no_grad():
                for blk in frozen_blocks:
                    x = blk(x)
            x = x.detach()

        # ---- Forward through trainable blocks ----
        if self.num_trainable_blocks > 0:
            for blk in self.model.blocks[-self.num_trainable_blocks:]:
                x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)

        cls_token = x[:, 0]  # [B, C]
        patch_tokens = x[:, 1:]  # [B, N, C]

        h = H // 14
        w = W // 14

        patch_tokens = (
            patch_tokens
            .reshape(B, h, w, self.num_channels)
            .permute(0, 3, 1, 2)
        )  # [B, C, h, w]

        if self.use_FeatureGrouping:
            feature, local_feature = self.featuregrouping(patch_tokens, cls_token)
            if self.training:
                return feature, local_feature
            else:
                return feature
        else:
            return F.normalize(cls_token, dim=-1)

    def forward(self, img1, img2=None):
        if img2 is not None:
            f1 = self._forward_single(img1)
            f2 = self._forward_single(img2)
            return f1, f2
        else:
            return self._forward_single(img1)