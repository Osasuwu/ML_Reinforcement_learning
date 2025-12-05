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
    
    ВАЖНО: Используем ImageNet нормализацию для совместимости с предобученными весами.
    """
    
    # ImageNet нормализация (усреднённые для grayscale)
    IMAGENET_MEAN = 0.449  # среднее (0.485 + 0.456 + 0.406) / 3
    IMAGENET_STD = 0.226   # среднее (0.229 + 0.224 + 0.225) / 3
    
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
        
        # Простая нормализация [0, 255] -> [-1, 1]
        # ImageNet нормализация НЕ подходит для нашей сцены PyBullet!
        x = x.float() / 127.5 - 1.0
        
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


class MobileNetWithDepthExtractor(BaseFeaturesExtractor):
    """
    Feature Extractor для side+depth и side+wrist режимов.
    
    Обрабатывает Dict observation space с:
    - 'image': (frame_stack, 64, 64, 1) - RGB grayscale
    - 'depth' или 'wrist': (frame_stack, 8, 8, 1) - мини вторичная камера
    
    MobileNet для изображения + маленькая FC сеть для вторичной камеры.
    Экономит память: 64x64 + 8x8 вместо 64x64 + 64x64
    
    ВАЖНО: Используем ImageNet нормализацию для совместимости с предобученными весами.
    """
    
    # ImageNet нормализация (усреднённые для grayscale)
    IMAGENET_MEAN = 0.449
    IMAGENET_STD = 0.226
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256,
                 freeze_layers: int = 8):
        # Для Dict space нужно передать весь features_dim
        super().__init__(observation_space, features_dim)
        
        # Image space
        image_space = observation_space['image']
        self.frame_stack = image_space.shape[0]
        self.image_height = image_space.shape[1]
        self.image_width = image_space.shape[2]
        
        # Secondary space (depth или wrist)
        # Определяем какой ключ используется
        self.secondary_key = 'depth' if 'depth' in observation_space.spaces else 'wrist'
        secondary_space = observation_space[self.secondary_key]
        self.secondary_size = secondary_space.shape[1]  # 8
        
        in_channels_image = self.frame_stack  # 4 кадра grayscale
        in_channels_secondary = self.frame_stack  # 4 кадра secondary
        
        # === MobileNet для изображения ===
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        original_conv = mobilenet.features[0][0]
        mobilenet.features[0][0] = nn.Conv2d(
            in_channels=in_channels_image,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        with torch.no_grad():
            avg_weights = original_conv.weight.data.mean(dim=1, keepdim=True)
            mobilenet.features[0][0].weight.data = avg_weights.repeat(1, in_channels_image, 1, 1)
        
        self.image_features = mobilenet.features
        
        if freeze_layers > 0:
            for i, layer in enumerate(self.image_features):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        with torch.no_grad():
            sample = torch.zeros(1, in_channels_image, self.image_height, self.image_width)
            out = self.image_features(sample)
            image_feature_size = out.shape[1]
        
        # === CNN сеть для secondary камеры ===
        # Для 32x32: используем лёгкую CNN вместо простого FC
        # Для 8x8 или меньше: простой FC достаточно
        if self.secondary_size >= 16:
            # CNN для больших secondary камер (16x16, 32x32)
            self.secondary_net = nn.Sequential(
                nn.Conv2d(in_channels_secondary, 32, kernel_size=3, stride=2, padding=1),  # 32 -> 16
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 16 -> 8
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 8 -> 4
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            secondary_feature_size = 64
        else:
            # Простой FC для маленьких (8x8 и меньше)
            self.secondary_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_channels_secondary * self.secondary_size * self.secondary_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            secondary_feature_size = 64
        
        # === Комбинированная голова ===
        combined_size = image_feature_size + secondary_feature_size
        
        self.fc = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"MobileNetWithDepthExtractor создан:")
        print(f"  - Image: {in_channels_image} кадров, {self.image_height}x{self.image_width}")
        print(f"  - Secondary ({self.secondary_key}): {in_channels_secondary} кадров, {self.secondary_size}x{self.secondary_size}")
        print(f"  - Image features: {image_feature_size}")
        print(f"  - Secondary features: {secondary_feature_size}")
        print(f"  - Combined: {combined_size} -> {features_dim}")
        print(f"  - Total params: {total_params:,}")
        print(f"  - Trainable params: {trainable_params:,}")
        print(f"  - Память secondary: {in_channels_secondary * self.secondary_size * self.secondary_size} значений (vs {in_channels_secondary * self.image_height * self.image_width} если бы 64x64)")
    
    def forward(self, observations: dict) -> torch.Tensor:
        # observations - dict с 'image' и 'depth'/'wrist'
        images = observations['image']  # (batch, frame_stack, H, W, 1)
        secondary = observations[self.secondary_key]  # (batch, frame_stack, 8, 8, 1)
        
        batch_size = images.shape[0]
        
        # === Image branch ===
        # (batch, frame_stack, H, W, 1) -> (batch, frame_stack, H, W)
        x_img = images.permute(0, 1, 4, 2, 3).reshape(batch_size, -1, self.image_height, self.image_width)
        # Простая нормализация [-1, 1] - лучше для нашей сцены
        x_img = x_img.float() / 127.5 - 1.0
        x_img = self.image_features(x_img)
        x_img = self.image_pool(x_img)
        x_img = x_img.flatten(1)
        
        # === Secondary branch ===
        # (batch, frame_stack, 8, 8, 1) -> (batch, frame_stack * 8 * 8)
        x_sec = secondary.permute(0, 1, 4, 2, 3).reshape(batch_size, -1, self.secondary_size, self.secondary_size)
        # Простая нормализация для secondary
        x_sec = x_sec.float() / 127.5 - 1.0
        x_sec = self.secondary_net(x_sec)
        
        # === Combine ===
        x = torch.cat([x_img, x_sec], dim=1)
        x = self.fc(x)
        
        return x


class SimpleCNNWithDepthExtractor(BaseFeaturesExtractor):
    """
    Простая CNN для side+depth режима БЕЗ предобучения.
    
    Преимущества:
    - Все слои обучаются с нуля под нашу задачу
    - Нет "шума" от ImageNet признаков
    - Меньше параметров = быстрее обучение
    - Лучше сохраняет пространственную информацию
    
    Архитектура специально для pick-and-place:
    - Сохраняет пространственное разрешение дольше
    - Использует координатные каналы для позиционной информации
    """
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Image space
        image_space = observation_space['image']
        self.frame_stack = image_space.shape[0]
        self.image_height = image_space.shape[1]
        self.image_width = image_space.shape[2]
        
        # Secondary space
        self.secondary_key = 'depth' if 'depth' in observation_space.spaces else 'wrist'
        secondary_space = observation_space[self.secondary_key]
        self.secondary_size = secondary_space.shape[1]
        
        in_channels = self.frame_stack
        
        # === CNN для изображения с СОХРАНЕНИЕМ пространственной информации ===
        # Используем меньший stride чтобы не потерять позицию объекта
        self.image_cnn = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 8x8 -> 4x4
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # === CNN для secondary (depth) камеры ===
        if self.secondary_size >= 16:
            self.secondary_cnn = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            secondary_out = 64
        else:
            self.secondary_cnn = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_channels * self.secondary_size * self.secondary_size, 64),
                nn.ReLU()
            )
            secondary_out = 64
        
        # === Комбинированная голова ===
        image_out = 128
        combined_size = image_out + secondary_out
        
        self.fc = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"SimpleCNNWithDepthExtractor создан:")
        print(f"  - Image: {in_channels} кадров, {self.image_height}x{self.image_width}")
        print(f"  - Secondary ({self.secondary_key}): {in_channels} кадров, {self.secondary_size}x{self.secondary_size}")
        print(f"  - Image features: {image_out}")
        print(f"  - Secondary features: {secondary_out}")
        print(f"  - Combined: {combined_size} -> {features_dim}")
        print(f"  - Total params: {total_params:,}")
        print(f"  - ВСЕ параметры обучаемые (нет заморозки)")
    
    def forward(self, observations: dict) -> torch.Tensor:
        images = observations['image']
        secondary = observations[self.secondary_key]
        
        batch_size = images.shape[0]
        
        # === Image branch ===
        x_img = images.permute(0, 1, 4, 2, 3).reshape(batch_size, -1, self.image_height, self.image_width)
        x_img = x_img.float() / 127.5 - 1.0
        x_img = self.image_cnn(x_img)
        
        # === Secondary branch ===
        x_sec = secondary.permute(0, 1, 4, 2, 3).reshape(batch_size, -1, self.secondary_size, self.secondary_size)
        x_sec = x_sec.float() / 127.5 - 1.0
        x_sec = self.secondary_cnn(x_sec)
        
        # === Combine ===
        x = torch.cat([x_img, x_sec], dim=1)
        x = self.fc(x)
        
        return x
