# Домашнее задание к уроку 4: Сверточные сети

## Цель задания
Сравнить эффективность сверточных и полносвязных сетей на задачах компьютерного зрения, изучить преимущества CNN архитектур.

## Задание 1: Сравнение CNN и полносвязных сетей (40 баллов)

### 1.1 Сравнение на MNIST (20 баллов)
Сравнил производительность на MNIST:
- Полносвязная сеть (3-4 слоя) | Train Acc: 0.9852; Test Acc: 0.9766
- Простая CNN (2-3 conv слоя) | Train Acc: 0.9916; Test Acc: 0.9924
- CNN с Residual Block | Train Acc: 0.9932; Test Acc: 0.9943

**Полносвязная сеть и CNN с Residual Block**
![Сравнение 1](https://github.com/4pokodav/lesson_4/raw/main/plots/fc_residual_mnist_comparison.png)

**Полносвязная сеть и Простая CNN**
![Сравнение 2](https://github.com/4pokodav/lesson_4/raw/main/plots/fc_snn_mnist_comparison.png)

Наилучшей моделью на MNIST оказалась CNN с Residual Block (с точностью 0.9943)

### 1.2 Сравнение на CIFAR-10 (20 баллов)
Сравнил производительность на CIFAR-10:
- Полносвязная сеть (глубокая) | Train Acc: 0.5414; Test Acc: 0.5159
- CNN с Residual блоками | Train Acc: 0.8253; Test Acc: 0.8000
- CNN с регуляризацией и Residual блоками | Train Acc: 0.8099; Test Acc: 0.7626

**Полносвязная сеть и CNN с Residual блоками**
![Сравнение 1](https://github.com/4pokodav/lesson_4/raw/main/plots/fc_residual_cifar_comparison.png)

**Полносвязная сеть и CNN с регуляризацией и Residual блоками**
![Сравнение 2](https://github.com/4pokodav/lesson_4/raw/main/plots/fc_snn_cifar_comparison.png)

Наилучшей моделью на CIFAR оказалась CNN с Residual блоками (с точностью 0.8)

![Confusion matrix](https://github.com/4pokodav/lesson_4/raw/main/plots/confusion_mat_cifar_cnn.png)

![Gradient flow](https://github.com/4pokodav/lesson_4/raw/main/plots/gradient_flow.png)

## Задание 2: Анализ архитектур CNN (30 баллов)

### 2.1 Влияние размера ядра свертки (15 баллов)
Исследовал влияние размера ядра свертки:
- 3x3 ядра | Train Acc: 0.9922; Test Acc: 0.9915; Training time: 92.3
- 5x5 ядра | Train Acc: 0.9930; Test Acc: 0.9879; Training time: 100.7
- 7x7 ядра | Train Acc: 0.9928; Test Acc: 0.9915; Training time: 106.78
- Комбинация разных размеров (1x1 + 3x3) | Train Acc: 0.9912; Test Acc: 0.9909; Training time: 101.83

![Kernel size](https://github.com/4pokodav/lesson_4/raw/main/plots/kernel_size_accuracy.png)

Наилучший результат показала модели с ядрами 3x3 и 7x7 (acc = 0.9915)

### 2.2 Влияние глубины CNN (15 баллов)
Исследуйте влияние глубины CNN:
- Неглубокая CNN (2 conv слоя) | Train Acc: 0.9883; Test Acc: 0.9893; Training time: 106.51
- Средняя CNN (4 conv слоя) | Train Acc: 0.9898; Test Acc: 0.9924; Training time: 103.17
- Глубокая CNN (6+ conv слоев) | Train Acc: 0.9901; Test Acc: 0.9917; Training time: 121.55
- CNN с Residual связями | Train Acc: 0.9926; Test Acc: 0.9934; Training time: 125.65

![Depth](https://github.com/4pokodav/lesson_4/raw/main/plots/depth_accuracy.png)

Наилучший результат показала модель CNN с Residual связями (acc = 0.9934)

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
