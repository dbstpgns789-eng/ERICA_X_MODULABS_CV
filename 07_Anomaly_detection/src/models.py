"""이상탐지 모델 세 가지: PaDiM, AutoEncoder, One-Class SVM. 노트북 셀 [20]~[25]를 옮긴 것."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from tqdm import tqdm

PATCH = 28  # PaDiM이 특징맵을 맞추는 크기. 28x28 = 784칸


class FeatureExtractor(nn.Module):
    """사전학습 CNN의 중간 층 출력을 hook으로 꺼낸다. 가중치는 고정."""

    def __init__(self, model_name='resnet18', layers_to_extract=['layer1', 'layer2', 'layer3'], pretrained=True):
        super().__init__()
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        elif model_name == 'wide_resnet50':
            self.model = models.wide_resnet50_2(pretrained=pretrained)
        else:
            raise ValueError(f"지원하지 않는 모델: {model_name}")

        self.layers_to_extract = layers_to_extract
        self.features = {}
        for name, module in self.model.named_modules():
            if name in layers_to_extract:
                module.register_forward_hook(self.save_features(name))

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def save_features(self, name):
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def forward(self, x):
        self.features.clear()
        self.model(x)
        return self.features


def patch_features(features, i):
    """사진 i의 층별 특징을 28x28로 맞춰 [784, 채널합]으로 붙인다."""
    parts = []
    for feat in features.values():
        f = F.adaptive_avg_pool2d(feat[i:i+1], (PATCH, PATCH))
        parts.append(f.permute(0, 2, 3, 1).reshape(PATCH * PATCH, -1))
    return torch.cat(parts, dim=1)


class PaDiM:
    """패치 위치마다 정상 특징의 평균·공분산을 저장하고, 거리로 이상 점수를 낸다.

    score: 사진 점수를 784칸의 'max'로 낼지 'mean'으로 낼지
    distance: 'mahalanobis' 또는 'euclidean'
    실험 6에서 max·mahalanobis가 가장 높았다 (0.972, mean 0.931, euclidean max 0.909).
    """

    def __init__(self, feature_extractor, device, d_reduced=100, score='max', distance='mahalanobis'):
        self.feature_extractor = feature_extractor
        self.device = device
        self.d_reduced = d_reduced
        self.score = score
        self.distance = distance
        self.means = []
        self.inv_covariances = []
        self.projection_matrix = None

    def fit(self, train_loader):
        print("PaDiM 훈련 시작...")
        all_features = []
        for batch in tqdm(train_loader, desc="특징 추출 중"):
            images = batch['image'].to(self.device)
            with torch.no_grad():
                features = self.feature_extractor(images)
                for i in range(images.size(0)):
                    all_features.append(patch_features(features, i))

        # [N*784, 채널합]. 사진마다 784줄씩 순서대로 쌓인다
        all_features = torch.cat(all_features, dim=0)
        print(f"전체 특징 크기: {all_features.shape}")

        if all_features.shape[1] > self.d_reduced:
            self.projection_matrix = torch.randn(all_features.shape[1], self.d_reduced).to(self.device)
            self.projection_matrix = F.normalize(self.projection_matrix, dim=0)
            all_features = torch.mm(all_features, self.projection_matrix)
            print(f"차원 축소 후 크기: {all_features.shape}")

        for patch_idx in range(PATCH * PATCH):
            # 784줄 간격으로 뽑으면 모든 사진의 같은 위치 칸만 모인다
            patch_feats = all_features[patch_idx::PATCH * PATCH]
            mean = torch.mean(patch_feats, dim=0)
            centered = patch_feats - mean
            cov = torch.mm(centered.T, centered) / (patch_feats.shape[0] - 1)
            cov += torch.eye(cov.shape[0]).to(self.device) * 1e-4
            try:
                inv_cov = torch.linalg.inv(cov)
            except RuntimeError:
                inv_cov = torch.linalg.pinv(cov)
            self.means.append(mean)
            self.inv_covariances.append(inv_cov)

        print(f"PaDiM 훈련 완료! (패치 수: {len(self.means)})")

    def predict(self, test_loader):
        anomaly_scores = []
        for batch in tqdm(test_loader, desc="이상 탐지 중"):
            images = batch['image'].to(self.device)
            with torch.no_grad():
                features = self.feature_extractor(images)
                for i in range(images.size(0)):
                    combined = patch_features(features, i)
                    if self.projection_matrix is not None:
                        combined = torch.mm(combined, self.projection_matrix)

                    patch_scores = []
                    for patch_idx in range(PATCH * PATCH):
                        diff = combined[patch_idx] - self.means[patch_idx]
                        if self.distance == 'mahalanobis':
                            d = torch.sqrt(torch.mm(
                                torch.mm(diff.unsqueeze(0), self.inv_covariances[patch_idx]),
                                diff.unsqueeze(1)
                            )).item()
                        else:
                            d = diff.norm().item()
                        patch_scores.append(d)

                    # 불량은 몇 칸뿐이라 평균을 내면 희석된다
                    anomaly_scores.append(max(patch_scores) if self.score == 'max' else float(np.mean(patch_scores)))

        return np.array(anomaly_scores), None

    def state(self):
        """pickle로 저장할 값. FeatureExtractor는 다시 만들면 되므로 뺀다."""
        return {
            'means': self.means,
            'inv_covariances': self.inv_covariances,
            'projection_matrix': self.projection_matrix,
            'd_reduced': self.d_reduced,
            'padim_patch_size': PATCH,
        }


class ConvAutoEncoder(nn.Module):
    """256x256 입력을 8x8 latent로 줄였다가 복원.

    sigmoid=True면 노트북 원본처럼 디코더 끝에 Sigmoid를 붙인다. 입력은 Normalize를 거쳐
    약 -2.1~2.6인데 Sigmoid는 0~1만 낼 수 있어서, 원본은 결함이 아니라 극단 색 픽셀 양을
    재고 있었다 (실험 7: AUROC 0.27, 제거 후 0.93). 그래서 기본은 뺀다.
    """

    def __init__(self, input_channels=3, latent_dim=128, sigmoid=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),  # 128x128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, latent_dim, 4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(),
        )

        decoder = [
            nn.ConvTranspose2d(latent_dim, 256, 4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 128x128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 4, stride=2, padding=1),  # 256x256
        ]
        if sigmoid:
            decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoEncoderAnomalyDetector:
    """정상 사진만으로 복원을 배우고, 복원 오차(MSE 평균)를 이상 점수로 쓴다."""

    def __init__(self, device, latent_dim=128, sigmoid=False):
        self.device = device
        self.model = ConvAutoEncoder(input_channels=3, latent_dim=latent_dim, sigmoid=sigmoid).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def fit(self, train_loader, epochs=50):
        print("AutoEncoder 훈련 시작...")
        self.model.train()
        train_losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = batch['image'].to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(images), images)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

        self.model.eval()
        print("AutoEncoder 훈련 완료!")
        return train_losses

    def predict(self, test_loader):
        self.model.eval()
        anomaly_scores = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="이상 탐지 중"):
                images = batch['image'].to(self.device)
                errors = F.mse_loss(self.model(images), images, reduction='none')
                anomaly_scores.extend(errors.view(images.size(0), -1).mean(dim=1).cpu().numpy())
        return np.array(anomaly_scores)


class OneClassSVMAnomalyDetector:
    """층별 특징을 평균 풀링해 사진당 벡터 하나로 만들고, 정상 무리의 경계를 학습한다."""

    def __init__(self, feature_extractor, device, nu=0.1):
        self.feature_extractor = feature_extractor
        self.device = device
        self.svm = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
        self.scaler = StandardScaler()

    def extract_features(self, data_loader):
        features = []
        for batch in tqdm(data_loader, desc="특징 추출 중"):
            images = batch['image'].to(self.device)
            with torch.no_grad():
                feat_dict = self.feature_extractor(images)
                pooled = [F.adaptive_avg_pool2d(f, (1, 1)).view(f.size(0), -1) for f in feat_dict.values()]
                features.append(torch.cat(pooled, dim=1).cpu().numpy())
        return np.concatenate(features, axis=0)

    def fit(self, train_loader):
        print("One-Class SVM 훈련 시작...")
        train_features = self.scaler.fit_transform(self.extract_features(train_loader))
        self.svm.fit(train_features)
        print("One-Class SVM 훈련 완료!")

    def predict(self, test_loader):
        test_features = self.scaler.transform(self.extract_features(test_loader))
        # decision_function은 정상일수록 크므로 부호를 뒤집어 이상 점수로 쓴다
        return -self.svm.decision_function(test_features)
