"""
Feature Extractor с предобученной моделью.

Использует MobileNetV3-Small или EfficientNet-B0 с заморозкой первых слоёв.
Вход: только пиксели (64x64 grayscale, стек кадров)
"""
import torch
import torch.nn as nn
import torchvision.models as models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np


class MobileNetExtractor(BaseFeaturesExtractor):
    """
    Feature Extractor на основе MobileNetV3-Small.
    
    Предобученная модель с частичной заморозкой слоёв.
    Преобразует grayscale изображения в 3-канальные для MobileNet.
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, 
                 freeze_layers: int = 8):
        """
        Args:
            observation_space: Пространство наблюдений
            features_dim: Размер выходного вектора признаков
            freeze_layers: Сколько слоёв заморозить (0 = не замораживать)
        """
        super().__init__(observation_space, features_dim)
        
        # observation_space.shape = (frame_stack, H, W, n_cameras)
        self.frame_stack = observation_space.shape[0]
        self.height = observation_space.shape[1]
        self.width = observation_space.shape[2]
        self.n_cameras = observation_space.shape[3]
        
        # Загружаем предобученную MobileNetV3-Small
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        # Модифицируем первый слой для нашего входа
        # Оригинальный вход: 3 канала
        # Наш вход: frame_stack * n_cameras каналов (grayscale кадры)
        in_channels = self.frame_stack * self.n_cameras
        
        # Заменяем первый conv слой
        original_conv = mobilenet.features[0][0]
        mobilenet.features[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Инициализируем новые веса усреднением оригинальных RGB весов
        with torch.no_grad():
            # Среднее по RGB каналам
            avg_weights = original_conv.weight.data.mean(dim=1, keepdim=True)
            # Повторяем для всех наших входных каналов
            mobilenet.features[0][0].weight.data = avg_weights.repeat(1, in_channels, 1, 1)
        
        # Берём только features (без classifier)
        self.features = mobilenet.features
        
        # Замораживаем первые слои
        if freeze_layers > 0:
            for i, layer in enumerate(self.features):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(f"  Заморожено {freeze_layers} слоёв из {len(self.features)}")
        
        # Adaptive pooling для фиксированного размера выхода
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Вычисляем размер выхода features
        with torch.no_grad():
            sample = torch.zeros(1, in_channels, self.height, self.width)
            out = self.features(sample)
            feature_size = out.shape[1]
        
        # FC голова
        self.fc = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
        
        # Подсчёт параметров
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"MobileNetExtractor создан:")
        print(f"  - Input: {in_channels} каналов, {self.height}x{self.width}")
        print(f"  - Feature size: {feature_size}")
        print(f"  - Output: {features_dim}")
        print(f"  - Total params: {total_params:,}")
        print(f"  - Trainable params: {trainable_params:,}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, frame_stack, H, W, n_cameras)
        batch_size = observations.shape[0]
        
        # Reshape: (batch, frame_stack * n_cameras, H, W)
        x = observations.permute(0, 1, 4, 2, 3)  # (batch, frame_stack, n_cameras, H, W)
        x = x.reshape(batch_size, -1, self.height, self.width)
        
        # Нормализация [0, 255] -> [0, 1]
        x = x.float() / 255.0
        
        # MobileNet features
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        
        # FC
        x = self.fc(x)
        
        return x


class EfficientNetExtractor(BaseFeaturesExtractor):
    """
    Feature Extractor на основе EfficientNet-B0.
    
    Более мощная модель, но требует больше памяти.
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256,
                 freeze_layers: int = 4):
        super().__init__(observation_space, features_dim)
        
        self.frame_stack = observation_space.shape[0]
        self.height = observation_space.shape[1]
        self.width = observation_space.shape[2]
        self.n_cameras = observation_space.shape[3]
        
        in_channels = self.frame_stack * self.n_cameras
        
        # Загружаем EfficientNet-B0
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Модифицируем первый слой
        original_conv = efficientnet.features[0][0]
        efficientnet.features[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        with torch.no_grad():
            avg_weights = original_conv.weight.data.mean(dim=1, keepdim=True)
            efficientnet.features[0][0].weight.data = avg_weights.repeat(1, in_channels, 1, 1)
        
        self.features = efficientnet.features
        
        # Заморозка
        if freeze_layers > 0:
            for i, layer in enumerate(self.features):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        with torch.no_grad():
            sample = torch.zeros(1, in_channels, self.height, self.width)
            out = self.features(sample)
            feature_size = out.shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"EfficientNetExtractor создан:")
        print(f"  - Input: {in_channels} каналов, {self.height}x{self.width}")
        print(f"  - Feature size: {feature_size}")
        print(f"  - Output: {features_dim}")
        print(f"  - Total params: {total_params:,}")
        print(f"  - Trainable params: {trainable_params:,}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        x = observations.permute(0, 1, 4, 2, 3)
        x = x.reshape(batch_size, -1, self.height, self.width)
        x = x.float() / 255.0
        
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        return x


class SimpleCNNExtractor(BaseFeaturesExtractor):
    """
    Простая CNN без предобучения (для сравнения).
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        self.frame_stack = observation_space.shape[0]
        self.height = observation_space.shape[1]
        self.width = observation_space.shape[2]
        self.n_cameras = observation_space.shape[3]
        
        in_channels = self.frame_stack * self.n_cameras
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            sample = torch.zeros(1, in_channels, self.height, self.width)
            cnn_out = self.cnn(sample).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(cnn_out, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
        print(f"SimpleCNNExtractor: {in_channels} channels -> {cnn_out} -> {features_dim}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        x = observations.permute(0, 1, 4, 2, 3)
        x = x.reshape(batch_size, -1, self.height, self.width)
        x = x.float() / 255.0
        
        x = self.cnn(x)
        x = self.fc(x)
        
        return x
