"""
Custom Feature Extractor с Transfer Learning для Stable-Baselines3.
Использует предобученную MobileNetV3-Small для извлечения признаков из изображений.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np


class MobileNetFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature Extractor с использованием предобученной MobileNetV3-Small.
    Веса сверточной части заморожены, обучается только голова.
    """
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        """
        Args:
            observation_space: Пространство наблюдений (Dict с 'image' и 'joints')
            features_dim: Размерность выходного вектора признаков
        """
        # Вычисляем общую размерность признаков
        super().__init__(observation_space, features_dim)
        
        # Получаем frame_stack из observation_space
        image_shape = observation_space['image'].shape
        self.frame_stack = image_shape[0]
        
        # Загрузка предобученной MobileNetV3-Small
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        
        # Удаляем классификатор, оставляем только feature extractor
        self.cnn = nn.Sequential(*list(mobilenet.children())[:-1])  # Убираем последний слой
        
        # Полная заморозка для скорости (GPU будет работать только с FC слоями)
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # Переводим в eval режим для ускорения
        self.cnn.eval()
        
        # Добавляем AdaptiveAvgPool для фиксированного размера вывода
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Размерность выхода MobileNetV3-Small
        mobilenet_out_dim = 576
        
        # Размерность входа джоинтов
        joints_dim = observation_space['joints'].shape[0]
        
        # Размерность признаков: последний кадр (576) + motion features (576 * (frame_stack-1))
        # Итого: 576 * frame_stack
        total_cnn_features = mobilenet_out_dim * self.frame_stack
        
        # Обучаемая голова (Policy Head) - увеличенная для большей загрузки GPU
        self.fc = nn.Sequential(
            nn.Linear(total_cnn_features + joints_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),  # Регуляризация
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
        
        self.mobilenet_out_dim = mobilenet_out_dim
        self.joints_dim = joints_dim
        
        # Кэшируем тензоры нормализации (создаются при первом forward pass)
        self._norm_mean = None
        self._norm_std = None
        
    def forward(self, observations: dict) -> torch.Tensor:
        """
        Прямой проход через сеть.
        
        Args:
            observations: Словарь с 'image' и 'joints'
            
        Returns:
            Вектор признаков размерности features_dim
        """
        # Извлечение изображений и джоинтов
        images = observations['image']
        joints = observations['joints']
        
        # images: (batch, frame_stack, height, width, channels)
        batch_size = images.shape[0]
        frame_stack = images.shape[1]
        height = images.shape[2]
        width = images.shape[3]
        channels = images.shape[4]
        
        # Переставляем оси: (batch, frame_stack, height, width, channels) -> 
        # (batch, frame_stack, channels, height, width)
        images = images.permute(0, 1, 4, 2, 3).float() / 255.0
        
        # ЭФФЕКТИВНЫЙ BATCH PROCESSING: 
        # Обрабатываем все кадры за один проход через MobileNet
        # Reshape: (batch, frame_stack, channels, H, W) -> (batch * frame_stack, channels, H, W)
        images_flat = images.reshape(batch_size * frame_stack, channels, height, width)
        
        # Преобразование grayscale -> RGB для MobileNet
        if channels == 1:
            images_flat = images_flat.repeat(1, 3, 1, 1)  # (batch*frame_stack, 3, H, W)
        
        # Нормализация ImageNet (кэшируем тензоры для скорости)
        if self._norm_mean is None or self._norm_mean.device != images_flat.device:
            self._norm_mean = torch.tensor([0.485, 0.456, 0.406], device=images_flat.device).view(1, 3, 1, 1)
            self._norm_std = torch.tensor([0.229, 0.224, 0.225], device=images_flat.device).view(1, 3, 1, 1)
        
        images_flat = (images_flat - self._norm_mean) / self._norm_std
        
        # Один проход через замороженную CNN для всех кадров
        with torch.no_grad():
            all_features = self.cnn(images_flat)  # (batch*frame_stack, 576, h', w')
            all_features = self.pool(all_features)  # (batch*frame_stack, 576, 1, 1)
            all_features = all_features.flatten(1)  # (batch*frame_stack, 576)
        
        # Reshape обратно: (batch*frame_stack, 576) -> (batch, frame_stack, 576)
        all_features = all_features.reshape(batch_size, frame_stack, self.mobilenet_out_dim)
        
        # Объединяем признаки всех кадров для понимания движения:
        # Последний кадр (текущее состояние) + дельты между кадрами (motion)
        current_features = all_features[:, -1, :]  # (batch, 576) - последний кадр
        
        # Motion features: разница между последовательными кадрами
        motion_features = []
        for i in range(1, frame_stack):
            delta = all_features[:, i, :] - all_features[:, i-1, :]
            motion_features.append(delta)
        
        # Конкатенируем: текущий кадр + motion features
        if motion_features:
            motion_combined = torch.cat(motion_features, dim=1)  # (batch, 576 * (frame_stack-1))
            cnn_features = torch.cat([current_features, motion_combined], dim=1)  # (batch, 576 * frame_stack)
        else:
            cnn_features = current_features
        
        # Нормализация джоинтов
        joints = joints.float()
        
        # Объединяем признаки изображения и джоинтов
        combined = torch.cat([cnn_features, joints], dim=1)
        
        # Пропускаем через обучаемую голову
        features = self.fc(combined)
        
        return features


class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Простая custom CNN для сравнения (легкая архитектура).
    Обучается с нуля.
    """
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Получаем размеры
        image_shape = observation_space['image'].shape
        joints_dim = observation_space['joints'].shape[0]
        
        # image_shape: (frame_stack, height, width, channels)
        frame_stack = image_shape[0]
        height = image_shape[1]
        width = image_shape[2]
        channels = image_shape[3]
        
        # Входные каналы: frame_stack * channels
        in_channels = frame_stack * channels
        
        # Простая CNN (Nature CNN style)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Вычисляем размер выхода CNN
        with torch.no_grad():
            sample = torch.zeros(1, in_channels, height, width)
            cnn_out_size = self.cnn(sample).shape[1]
        
        # Полносвязные слои
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size + joints_dim, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
        
        self.cnn_out_size = cnn_out_size
        self.joints_dim = joints_dim
        
    def forward(self, observations: dict) -> torch.Tensor:
        images = observations['image']
        joints = observations['joints']
        
        # Преобразование изображений
        batch_size = images.shape[0]
        frame_stack = images.shape[1]
        height = images.shape[2]
        width = images.shape[3]
        channels = images.shape[4]
        
        images = images.permute(0, 1, 4, 2, 3)
        images = images.reshape(batch_size, frame_stack * channels, height, width)
        images = images.float() / 255.0
        
        # CNN
        cnn_features = self.cnn(images)
        
        # Объединение с джоинтами
        joints = joints.float()
        combined = torch.cat([cnn_features, joints], dim=1)
        
        # FC
        features = self.fc(combined)
        
        return features
