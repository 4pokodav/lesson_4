## Задание 1: Сравнение CNN и полносвязных сетей (40 баллов)

### 1.1 Сравнение на MNIST (20 баллов)
Обучил три модели: Полносвязная сеть (3-4 слоя); Простая CNN (2-3 conv слоя); CNN с Residual Block;
Сравнил их между собой.
![Сравнение 1](https://github.com/4pokodav/lesson_4/raw/main/plots/fc_residual_mnist_comparison.png)
![Сравнение 2](https://github.com/4pokodav/lesson_4/raw/main/plots/fc_snn_mnist_comparison.png)

Наилучшей моделью на MNIST оказалась CNN с Residual Block (с точностью 0.9943)

### 1.2 Сравнение на CIFAR-10 (20 баллов)
Обучил три модели: Полносвязная сеть; CNN с Residual блоками; CNN с регуляризацией и Residual блоками;
Сравнил их между собой.
![Сравнение 1](https://github.com/4pokodav/lesson_4/raw/main/plots/fc_residual_cifar_comparison.png)
![Сравнение 2](https://github.com/4pokodav/lesson_4/raw/main/plots/fc_snn_cifar_comparison.png)

Наилучшей моделью на CIFAR оказалась x (с точностью x)


![Confusion matrix](https://github.com/4pokodav/lesson_4/raw/main/plots/confusion_mat_cifar_cnn.png)

![Gradient flow](https://github.com/4pokodav/lesson_4/raw/main/plots/gradient_flow.png)

## Задание 2: Анализ архитектур CNN (30 баллов)

### 2.1 Влияние размера ядра свертки (15 баллов)
Исследовал влияние размера ядра свертки:
- 3x3 ядра | Train Acc: 0.9922; Test Acc: 0.9915; Training time: 92.3
- 5x5 ядра | Train Acc: 0.9930; Test Acc: 0.9879; Training time: 100.7
- 7x7 ядра | Train Acc: 0.9928; Test Acc: 0.9915; Training time: 106.78
- Комбинация разных размеров (1x1 + 3x3) | Train Acc: 0.9912; Test Acc: 0.9909; Training time: 101.83

### 2.2 Влияние глубины CNN (15 баллов)
Исследуйте влияние глубины CNN:
- Неглубокая CNN (2 conv слоя) | Train Acc: 0.9883; Test Acc: 0.9893; Training time: 106.51
- Средняя CNN (4 conv слоя) | Train Acc: 0.9898; Test Acc: 0.9924; Training time: 103.17
- Глубокая CNN (6+ conv слоев) | Train Acc: 0.9901; Test Acc: 0.9917; Training time: 121.55
- CNN с Residual связями | Train Acc: 0.9926; Test Acc: 0.9934; Training time: 125.65

Графики прилагаются.

## Задание 3: Кастомные слои и эксперименты (30 баллов)

### 3.1 Реализация кастомных слоев (15 баллов)
Реализовал кастомные слои:
- Кастомный сверточный слой с дополнительной логикой
- Attention механизм для CNN
- Кастомная функция активации
- Кастомный pooling слой

Для каждого слоя:
- Реализовал forward и backward проходы
- Добавил параметры если необходимо
- Протестировал на простых примерах

### 3.2 Эксперименты с Residual блоками (15 баллов)
Проведены эксперименты с тремя типами Residual-блоков:

![Метрики](https://github.com/4pokodav/lesson_4/raw/main/plots/3.2.png)

Графики прилагаются.
