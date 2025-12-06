# RL3 - Робот Pick-and-Place

Обучение робота Franka Panda переносить объекты с помощью камеры.

## Установка

```bash
pip install -r RL3/requirements.txt
```

## Обучение

```bash
python RL3/train.py --steps 1000000 --algo ppo --camera-mode side+depth
```

**Параметры:**
- `--steps` — количество шагов (по умолчанию 10M)
- `--algo` — алгоритм: `ppo` или `sac`
- `--curriculum` — начать с простой позиции объекта
- `--name` — имя эксперимента

## Тестирование

```bash
python RL3/test.py RL3/models/ваша_модель.zip --episodes 5 --gui
```

## Структура

- `robot_env.py` — среда симуляции
- `train.py` — скрипт обучения
- `test.py` — скрипт тестирования
- `models/` — сохранённые модели
- `logs/` — логи обучения
