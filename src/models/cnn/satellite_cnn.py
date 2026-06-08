import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ChangeDetectionModule(nn.Module):
    """
    Computes element-wise feature-space absolute differences between before and current images.
    """
    def __init__(self):
        super(ChangeDetectionModule, self).__init__()
        
    def forward(self, feat_before: torch.Tensor, feat_current: torch.Tensor) -> torch.Tensor:
        return torch.abs(feat_current - feat_before)


class SatelliteSiameseCNN(nn.Module):
    """
    Siamese CNN with a ResNet-50 backbone for physical change detection and progress monitoring.
    Inputs:
        - img_before: Image at previous timestamp (B, C, H, W)
        - img_current: Image at current timestamp (B, C, H, W)
    Outputs:
        - progress: Regression output for construction progress (0.0 to 1.0)
        - phase_logits: Classification output for construction phase (logits for classes)
        - anomaly_logits: Binary classification output for construction anomalies (logits for normal vs anomaly)
    """
    def __init__(self, num_classes=5, in_channels=13):
        super(SatelliteSiameseCNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.change_detection = ChangeDetectionModule()
        
        # Sentinel-2 has 13 spectral bands. We project to 3 channels to use pretrained ResNet-50
        if in_channels != 3:
            self.channel_projector = nn.Conv2d(in_channels, 3, kernel_size=1)
        else:
            self.channel_projector = None
            
        try:
            # Try to load pretrained weights
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            # Fallback to un-pretrained model if offline
            resnet = models.resnet50(weights=None)
            
        # Extract features (exclude average pool and FC layer)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Regression head for construction progress (MAPE target < 15%)
        self.regression_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Progress output scaled [0.0, 1.0]
        )
        
        # Classification head for construction phase
        self.classification_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # Logits for [normal, anomaly]
        )
        
    def forward_once(self, x):
        if self.channel_projector is not None:
            x = self.channel_projector(x)
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
        
    def forward(self, img_before, img_current):
        feat_before = self.forward_once(img_before)
        feat_current = self.forward_once(img_current)
        
        # Element-wise absolute difference represents change between the two timestamps
        diff = self.change_detection(feat_before, feat_current)
        
        progress = self.regression_head(diff)
        phase_logits = self.classification_head(diff)
        anomaly_logits = self.anomaly_head(diff)
        
        return progress, phase_logits, anomaly_logits
        
    def save_model(self, path):
        """Saves model state dict."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        
    def load_model(self, path):
        """Loads model state dict with CPU safety fallback."""
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        else:
            raise FileNotFoundError(f"Model weights file not found at {path}")

    def train_on_data(self, train_loader, epochs=5, lr=1e-3):
        """
        Simulates model training with standard Adam optimizer, L1 loss, and cross-entropy loss.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        l1_loss_fn = nn.L1Loss()
        cls_loss_fn = nn.CrossEntropyLoss()
        
        self.train()
        for epoch in range(epochs):
            for batch in train_loader:
                img_before, img_current, target_progress, target_phase, target_anomaly = batch
                img_before = img_before.to(self.regression_head[0].weight.device)
                img_current = img_current.to(self.regression_head[0].weight.device)
                
                optimizer.zero_grad()
                progress, phase_logits, anomaly_logits = self(img_before, img_current)
                
                loss_reg = l1_loss_fn(progress, target_progress.to(progress.device))
                loss_cls = cls_loss_fn(phase_logits, target_phase.to(phase_logits.device))
                loss_anom = cls_loss_fn(anomaly_logits, target_anomaly.to(anomaly_logits.device))
                
                loss = loss_reg + loss_cls + loss_anom
                loss.backward()
                optimizer.step()

    def evaluate_model(self, test_loader) -> float:
        """
        Evaluates the model on progress estimation and computes the Mean Absolute Percentage Error (MAPE).
        """
        import numpy as np
        self.eval()
        mape_list = []
        with torch.no_grad():
            for batch in test_loader:
                img_before, img_current, target_progress, _, _ = batch
                img_before = img_before.to(self.regression_head[0].weight.device)
                img_current = img_current.to(self.regression_head[0].weight.device)
                
                progress, _, _ = self(img_before, img_current)
                
                y_pred = progress.cpu().numpy().ravel()
                y_true = target_progress.numpy().ravel()
                
                for pred, true in zip(y_pred, y_true):
                    if true > 0:
                        mape_list.append(np.abs(pred - true) / true)
                        
        if len(mape_list) == 0:
            return 0.0
        return float(np.mean(mape_list) * 100)
