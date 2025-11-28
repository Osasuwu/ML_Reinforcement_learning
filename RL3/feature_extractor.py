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
        # MobileNetV3-Small выдает 576 признаков + 7 джоинтов
        super().__init__(observation_space, features_dim)
        
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
        
        # Обучаемая голова (Policy Head) - увеличенная для большей загрузки GPU
        self.fc = nn.Sequential(
            nn.Linear(mobilenet_out_dim + joints_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),  # Регуляризация
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
        
        self.mobilenet_out_dim = mobilenet_out_dim
        self.joints_dim = joints_dim
        
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
        # Преобразуем в (batch, channels * frame_stack, height, width)
        batch_size = images.shape[0]
        frame_stack = images.shape[1]
        height = images.shape[2]
        width = images.shape[3]
        channels = images.shape[4]
        
        # Переставляем оси: (batch, frame_stack, height, width, channels) -> 
        # (batch, frame_stack, channels, height, width) ->
        # (batch, frame_stack * channels, height, width)
        images = images.permute(0, 1, 4, 2, 3)  # (batch, frame_stack, channels, height, width)
        images = images.reshape(batch_size, frame_stack * channels, height, width)
        
        # Нормализация изображений (ImageNet stats)
        # MobileNet ожидает RGB изображения, нормализованные по ImageNet
        images = images.float() / 255.0
        
        # Если grayscale, преобразуем в 3-канальное изображение путем повторения
        if channels == 1:
            # У нас frame_stack каналов grayscale, нужно привести к 3-м каналам RGB
            # Берем среднее по frame_stack каналам и повторяем 3 раза
            images = images.mean(dim=1, keepdim=True)  # (batch, 1, height, width)
            images = images.repeat(1, 3, 1, 1)  # (batch, 3, height, width)
        else:
            # Для RGB берем только последний кадр (или можно усреднить)
            # Берем последние 3 канала (последний RGB кадр)
            images = images[:, -3:, :, :]
        
        # Нормализация ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std
        
        # Пропускаем через CNN (частично заморожен, последние слои обучаются)
        cnn_features = self.cnn(images)  # (batch, 576, h', w')
        cnn_features = self.pool(cnn_features)  # (batch, 576, 1, 1)
        cnn_features = cnn_features.flatten(1)  # (batch, 576)
        
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
