"""
Диагностика визуального восприятия модели.

Показывает:
1. Что видит камера (боковая + depth)
2. Как меняется изображение при разных позициях объекта
3. Проверяет различимость объекта на изображении
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from robot_env import RobotEnv


def test_object_visibility():
    """Проверяет видимость объекта в разных позициях"""
    print("=" * 60)
    print("VISUAL DEBUGGING: Object Visibility Test")
    print("=" * 60)
    
    env = RobotEnv(
        use_gui=False,
        image_size=64,
        frame_stack=4,
        camera_mode='side+depth',
        max_steps=200
    )
    
    # Тестируем 4 разных позиции объекта
    positions = [
        (0.35, 0.15, "Close-left"),
        (0.55, 0.15, "Far-left"),
        (0.35, 0.30, "Close-right"),
        (0.55, 0.30, "Far-right"),
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, (x, y, name) in enumerate(positions):
        # Сбрасываем среду с фиксированной позицией
        env.fixed_object_pos = (x, y)
        obs, _ = env.reset()
        
        # Получаем изображения
        if isinstance(obs, dict):
            side_img = obs['image'][0, :, :, 0]  # Первый кадр, grayscale
            depth_img = obs['depth'][0, :, :, 0] if 'depth' in obs else None
        else:
            side_img = obs[0, :, :, 0]
            depth_img = None
        
        # Показываем боковую камеру
        axes[0, i].imshow(side_img, cmap='gray', vmin=0, vmax=255)
        axes[0, i].set_title(f"Side: {name}\nObj at ({x:.2f}, {y:.2f})")
        axes[0, i].axis('off')
        
        # Анализируем различимость объекта
        mean_val = np.mean(side_img)
        std_val = np.std(side_img)
        max_val = np.max(side_img)
        min_val = np.min(side_img)
        
        print(f"\n{name} (x={x:.2f}, y={y:.2f}):")
        print(f"  Side img - mean: {mean_val:.1f}, std: {std_val:.1f}, range: [{min_val}-{max_val}]")
        
        # Показываем depth
        if depth_img is not None:
            axes[1, i].imshow(depth_img, cmap='gray', vmin=0, vmax=255)
            axes[1, i].set_title(f"Depth: {name}")
            axes[1, i].axis('off')
            
            depth_mean = np.mean(depth_img)
            depth_std = np.std(depth_img)
            print(f"  Depth img - mean: {depth_mean:.1f}, std: {depth_std:.1f}")
        else:
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'debug_vision_positions.png'), dpi=150)
    print(f"\nSaved: debug_vision_positions.png")
    plt.show()
    
    env.close()


def test_image_differences():
    """Проверяет насколько РАЗНЫЕ изображения при разных позициях объекта"""
    print("\n" + "=" * 60)
    print("VISUAL DEBUGGING: Image Difference Analysis")
    print("=" * 60)
    
    env = RobotEnv(
        use_gui=False,
        image_size=64,
        frame_stack=4,
        camera_mode='side+depth',
        max_steps=200
    )
    
    # Собираем изображения для разных позиций
    images = []
    positions = []
    
    for _ in range(10):
        env.fixed_object_pos = None  # Случайная позиция
        obs, _ = env.reset()
        
        if isinstance(obs, dict):
            img = obs['image'][0, :, :, 0]
        else:
            img = obs[0, :, :, 0]
        
        obj_pos = env._get_object_pos()
        images.append(img.flatten())
        positions.append((obj_pos[0], obj_pos[1]))
    
    images = np.array(images)
    
    # Вычисляем попарные различия
    print("\nPairwise image differences (L2 norm):")
    print("Position 1 vs Position 2 -> Difference")
    
    diffs = []
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            diff = np.linalg.norm(images[i] - images[j])
            pos_dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                              (positions[i][1] - positions[j][1])**2)
            diffs.append((diff, pos_dist))
            if len(diffs) <= 10:
                print(f"  ({positions[i][0]:.2f},{positions[i][1]:.2f}) vs "
                      f"({positions[j][0]:.2f},{positions[j][1]:.2f}): "
                      f"img_diff={diff:.0f}, pos_dist={pos_dist:.3f}")
    
    # Корреляция между различием изображений и расстоянием позиций
    img_diffs = [d[0] for d in diffs]
    pos_dists = [d[1] for d in diffs]
    correlation = np.corrcoef(img_diffs, pos_dists)[0, 1]
    
    print(f"\n*** CORRELATION (image diff vs position dist): {correlation:.3f} ***")
    print("  > 0.5: Good - images change with position")
    print("  < 0.3: Bad - images too similar, model can't see position!")
    
    env.close()
    return correlation


def test_frame_stack_variance():
    """Проверяет, есть ли различие между кадрами в стеке"""
    print("\n" + "=" * 60)
    print("VISUAL DEBUGGING: Frame Stack Variance")
    print("=" * 60)
    
    env = RobotEnv(
        use_gui=False,
        image_size=64,
        frame_stack=4,
        camera_mode='side+depth',
        max_steps=200
    )
    
    obs, _ = env.reset()
    
    # Делаем несколько шагов
    for _ in range(10):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
    
    if isinstance(obs, dict):
        frames = obs['image']  # (4, 64, 64, 1)
    else:
        frames = obs  # (4, 64, 64, 1)
    
    print("\nFrame-to-frame differences in stack:")
    for i in range(frames.shape[0] - 1):
        diff = np.mean(np.abs(frames[i].astype(float) - frames[i+1].astype(float)))
        print(f"  Frame {i} vs Frame {i+1}: mean diff = {diff:.2f}")
    
    # Общая дисперсия
    variance = np.var(frames)
    print(f"\nTotal variance across frames: {variance:.2f}")
    print("  > 100: Good - frames are different")
    print("  < 50: Bad - frames too similar, no motion info!")
    
    env.close()


def analyze_feature_extractor_output():
    """Проверяет выход feature extractor для разных позиций"""
    print("\n" + "=" * 60)
    print("VISUAL DEBUGGING: Feature Extractor Analysis")
    print("=" * 60)
    
    import torch
    from feature_extractor import MobileNetWithDepthExtractor, SimpleCNNWithDepthExtractor
    from gymnasium import spaces
    
    # Создаём observation space как в среде
    obs_space = spaces.Dict({
        'image': spaces.Box(low=0, high=255, shape=(4, 64, 64, 1), dtype=np.uint8),
        'depth': spaces.Box(low=0, high=255, shape=(4, 32, 32, 1), dtype=np.uint8)
    })
    
    # Тестируем ОБА экстрактора
    extractors = [
        ("SimpleCNN", SimpleCNNWithDepthExtractor(obs_space, features_dim=256)),
        ("MobileNet", MobileNetWithDepthExtractor(obs_space, features_dim=256, freeze_layers=4)),
    ]
    
    env = RobotEnv(
        use_gui=False,
        image_size=64,
        frame_stack=4,
        camera_mode='side+depth',
        max_steps=200
    )
    
    results = {}
    
    for name, extractor in extractors:
        print(f"\n--- {name} ---")
        extractor.eval()
        
        # Собираем features для разных позиций
        features_list = []
        positions = []
        
        with torch.no_grad():
            for _ in range(20):
                env.fixed_object_pos = None
                obs, _ = env.reset()
                
                # Преобразуем в torch tensor
                obs_torch = {
                    'image': torch.from_numpy(obs['image']).unsqueeze(0),
                    'depth': torch.from_numpy(obs['depth']).unsqueeze(0)
                }
                
                features = extractor(obs_torch).numpy().flatten()
                features_list.append(features)
                
                obj_pos = env._get_object_pos()
                positions.append((obj_pos[0], obj_pos[1]))
        
        features_arr = np.array(features_list)
        
        # Анализируем
        print(f"Feature statistics:")
        print(f"  Shape: {features_arr.shape}")
        print(f"  Mean: {np.mean(features_arr):.3f}")
        print(f"  Std: {np.std(features_arr):.3f}")
        print(f"  Range: [{np.min(features_arr):.3f}, {np.max(features_arr):.3f}]")
        
        # Попарные различия features
        feature_diffs = []
        pos_dists = []
        
        for i in range(len(features_list)):
            for j in range(i+1, len(features_list)):
                f_diff = np.linalg.norm(features_list[i] - features_list[j])
                p_dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                                (positions[i][1] - positions[j][1])**2)
                feature_diffs.append(f_diff)
                pos_dists.append(p_dist)
        
        correlation = np.corrcoef(feature_diffs, pos_dists)[0, 1]
        
        print(f"*** FEATURE-POSITION CORRELATION: {correlation:.3f} ***")
        
        # Проверяем активные нейроны
        active_neurons = np.sum(np.std(features_arr, axis=0) > 0.01)
        dead_neurons = np.sum(np.std(features_arr, axis=0) < 0.001)
        print(f"Active neurons (std > 0.01): {active_neurons}/{features_arr.shape[1]}")
        print(f"Dead neurons (std < 0.001): {dead_neurons}/{features_arr.shape[1]}")
        
        results[name] = correlation
    
    env.close()
    return results


if __name__ == "__main__":
    print("Running visual diagnostics...\n")
    
    # 1. Проверяем видимость объекта
    test_object_visibility()
    
    # 2. Проверяем различия изображений
    img_corr = test_image_differences()
    
    # 3. Проверяем frame stack
    test_frame_stack_variance()
    
    # 4. Анализируем feature extractor
    feat_corrs = analyze_feature_extractor_output()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Image-Position Correlation: {img_corr:.3f}")
    print(f"\nFeature-Position Correlations:")
    for name, corr in feat_corrs.items():
        status = "✓ Good" if corr > 0.3 else "⚠️ Weak" if corr > 0.1 else "❌ Bad"
        print(f"  {name}: {corr:.3f} {status}")
    
    best_extractor = max(feat_corrs, key=feat_corrs.get)
    print(f"\n>>> Recommended extractor: {best_extractor}")
    
    if feat_corrs[best_extractor] < 0.1:
        print("\n⚠️ PROBLEM: Features don't encode object position!")
        print("   Check camera placement and object visibility")
