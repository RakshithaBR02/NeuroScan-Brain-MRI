import torch
import torch.nn as nn
import timm


# ------------------------------------------
# CNN FEATURE EXTRACTION BRANCH
# ------------------------------------------

class CNNBranch(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        x = self.cnn(x)
        return self.pool(x)


# ------------------------------------------
# CROSS ATTENTION
# ------------------------------------------

class CrossAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn_to_swin = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True
        )

        self.swin_to_cnn = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True
        )

        self.norm_cnn = nn.LayerNorm(256)
        self.norm_swin = nn.LayerNorm(256)

    def forward(self, cnn_features, swin_features):

        cnn_updated, _ = self.cnn_to_swin(
            cnn_features,
            swin_features,
            swin_features
        )

        swin_updated, _ = self.swin_to_cnn(
            swin_features,
            cnn_features,
            cnn_features
        )

        cnn_updated = self.norm_cnn(
            cnn_features + cnn_updated
        )

        swin_updated = self.norm_swin(
            swin_features + swin_updated
        )

        return cnn_updated, swin_updated


# ------------------------------------------
# SWINCT FEATURE EXTRACTOR
# ------------------------------------------

class SwinCTFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=0
        )

        self.cnn_branch = CNNBranch()

        self.swin_projection = nn.Linear(768, 256)

        self.cross_attention = CrossAttention()

    def forward(self, x):

        cnn_features = self.cnn_branch(x)

        cnn_features = cnn_features.flatten(2).transpose(1, 2)

        swin_features = self.swin.forward_features(x)

        swin_features = self.swin_projection(swin_features)

        swin_features = swin_features.reshape(
            swin_features.shape[0], -1, 256
        )

        cnn_features, swin_features = self.cross_attention(
            cnn_features,
            swin_features
        )

        cnn_vector = cnn_features.mean(dim=1)
        swin_vector = swin_features.mean(dim=1)

        combined = torch.cat(
            [cnn_vector, swin_vector],
            dim=1
        )

        return combined


# ------------------------------------------
# SWINCT CLASSIFIER
# ------------------------------------------

class SwinCTClassifier(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()

        self.feature_extractor = SwinCTFeatureExtractor()

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        features = self.feature_extractor(x)

        predictions = self.classifier(features)

        return predictions