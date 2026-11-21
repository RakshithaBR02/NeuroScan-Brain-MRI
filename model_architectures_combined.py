import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualAttentionBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = F.relu(out + identity)

        return out * self.attention(out)

class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = ResidualAttentionBlock(
            in_channels, out_channels
        )

    def forward(self, x):
        return self.block(x)

class GatedAttention(nn.Module):

    def __init__(self, skip_channels, gate_channels, inter_channels):
        super().__init__()

        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )

        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )

        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, skip, gate):

        skip_projected = self.skip_conv(skip)
        gate_projected = self.gate_conv(gate)

        if gate_projected.shape[2:] != skip_projected.shape[2:]:
            gate_projected = F.interpolate(
                gate_projected,
                size=skip_projected.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        attention = self.attention(
            skip_projected + gate_projected
        )

        return skip * attention

class DecoderAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):
        super().__init__()

        self.gate = GatedAttention(
            skip_channels,
            in_channels,
            out_channels
        )

        self.conv = ResidualAttentionBlock(
            in_channels + skip_channels,
            out_channels
        )

        self.decoder_attention = DecoderAttention(
            out_channels
        )

    def forward(self, x, skip):

        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        skip = self.gate(skip, x)

        x = torch.cat([x, skip], dim=1)

        x = self.conv(x)

        x = self.decoder_attention(x)

        return x

class MDUNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.enc1 = EncoderBlock(in_channels, 32)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)

        self.bottleneck = ResidualAttentionBlock(
            128, 256
        )

        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 32, 32)

        self.output = nn.Conv2d(
            32, out_channels, kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.output(d1)

class UndercompleteAutoencoder(nn.Module):
    def __init__(self, input_dim=229, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.ReLU(),
            nn.Linear(192, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 192),
            nn.ReLU(),
            nn.Linear(192, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

class ChannelAttention(nn.Module):
    def __init__(self, channels=128):
        super().__init__()

        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // 8),
            nn.ReLU(),
            nn.Linear(channels // 8, channels)
        )

    def forward(self, x):
        attention = torch.sigmoid(
            self.shared_mlp(x) + self.shared_mlp(x)
        )
        return x * attention

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        sequence = x.unsqueeze(-1)

        avg = sequence.mean(dim=1, keepdim=True)
        maximum = sequence.max(dim=1, keepdim=True)[0]

        attention = torch.sigmoid(
            self.conv(torch.cat([avg, maximum], dim=1))
        )

        return (sequence * attention).squeeze(-1)

class CBAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel_attention = ChannelAttention(128)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        return self.spatial_attention(x)

class CapsuleNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.primary_capsules = nn.Linear(128, 512)

        self.W = nn.Parameter(
            torch.randn(32, 4, 16, 16) * 0.01
        )

        self.routing_iterations = 3

    def squash(self, x, dim=-1):
        squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)

        return scale * x / torch.sqrt(squared_norm + 1e-8)

    def forward(self, x):
        batch_size = x.size(0)

        primary = self.primary_capsules(x)
        primary = primary.view(batch_size, 32, 16)
        primary = self.squash(primary)

        predictions = torch.einsum(
            "bid,icod->bico",
            primary,
            self.W
        )

        routing = torch.zeros(
            batch_size, 32, 4, device=x.device
        )

        for iteration in range(self.routing_iterations):
            coupling = torch.softmax(routing, dim=-1)

            outputs = (
                coupling.unsqueeze(-1) * predictions
            ).sum(dim=1)

            outputs = self.squash(outputs)

            if iteration < self.routing_iterations - 1:
                agreement = (
                    predictions * outputs.unsqueeze(1)
                ).sum(dim=-1)

                routing = routing + agreement

        return torch.sqrt(
            (outputs ** 2).sum(dim=-1) + 1e-8
        )

class HCACClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.autoencoder = UndercompleteAutoencoder()
        self.cbam = CBAM()
        self.capsule_network = CapsuleNetwork()

    def forward(self, x):
        latent, reconstructed = self.autoencoder(x)
        attended = self.cbam(latent)
        predictions = self.capsule_network(attended)

        return predictions, reconstructed, latent

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