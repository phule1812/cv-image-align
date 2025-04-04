# Лабораторная работа 2: Выравнивание изображений с использованием ключевых точек и гомографии

## 🎯 Цель работы

Изучить и реализовать полный pipeline по выравниванию изображений с использованием классических методов компьютерного зрения, включая:

- Детектирование ключевых точек (Harris Corner).
- Сопоставление ключевых точек между изображениями.
- Расчёт гомографии с использованием метода RANSAC.
- Перспективное преобразование изображений.

## 📌 Краткое описание

В этой лабораторной работе реализуется поэтапная система, позволяющая привести несколько изображений в соответствие с эталонным (опорным) изображением путём анализа геометрических признаков. Процесс включает:

1. 📥 Загрузка изображений и их предварительное отображение.
2. 🎯 Обнаружение ключевых точек с помощью алгоритма Харриса.
3. 🧬 Извлечение дескрипторов из окрестностей ключевых точек.
4. 🔗 Сопоставление точек между изображениями.
5. 🧠 Построение матрицы гомографии с помощью алгоритма RANSAC.
6. 🔄 Перспективное преобразование изображений на основе найденной гомографии.
7. 🖼️ Визуализация результатов до и после выравнивания.


# 🔍 Обнаружение ключевых точек: Harris Corner и SIFT

В задачах компьютерного зрения (выравнивание изображений, распознавание объектов, панорама) важно находить **ключевые точки** — уникальные, устойчивые признаки на изображении.  
Здесь рассмотрим два популярных метода: **Harris Corner** и **SIFT**.

---

## 🟦 1. Harris Corner Detection

**Harris Corner Detector** — классический метод обнаружения углов, предложенный в 1988 году.  
Метод основан на анализе изменения интенсивности пикселей в разных направлениях.

### 📌 Принцип работы:
- Вычисляются градиенты по x и y.
- Строится матрица авто-корреляции \( M \).
- Вычисляется отклик Harris:

  $$
  R = \det(M) - k \cdot (\mathrm{trace}(M))^2
  $$

- Если значение \( R \) выше порога → это угол (ключевая точка).

### 📷 Иллюстрация:

![Harris Corner](https://tse3.mm.bing.net/th?id=OIP.YYhAiO8SC9os7AbsZn-apwHaEn&pid=Api)

---

## 🟩 2. SIFT — Scale-Invariant Feature Transform

**SIFT** — более продвинутый алгоритм, устойчивый к масштабированию, повороту, и частично — к изменениям освещения.

### 📌 Этапы:
1. Построение scale-space через гауссовы фильтры.
2. Поиск экстремумов на изображениях разности гауссиан (DoG).
3. Определение главного направления градиента.
4. Формирование дескриптора (128-мерного вектора).

### 📷 Иллюстрация:

![SIFT Keypoints](https://cdn.analyticsvidhya.com/wp-content/uploads/2024/09/download-1.webp)

---

## ⚖️ Сравнение: Harris vs SIFT

| Критерий                   | Harris Corner         | SIFT                             |
|----------------------------|------------------------|----------------------------------|
| Инвариантность к масштабу  | ❌ Нет                 | ✅ Да                            |
| Инвариантность к повороту  | ⚠️ Частично            | ✅ Да                            |
| Наличие дескриптора        | ❌ Нет                 | ✅ Да (128-мерный)               |
| Устойчивость к шуму        | Средняя               | Высокая                          |
| Скорость                   | Высокая               | Ниже                            |
| Применение                 | Простые задачи, трекинг | Панорама, распознавание, SLAM   |

---

## ✅ Вывод

- **Harris** — простой и быстрый метод, подходит для базовых задач.
- **SIFT** — более надёжный и мощный, используется в сложных системах с изменениями масштаба, поворота и освещения.

👉 Оба метода можно использовать для дальнейших шагов: сопоставление точек, построение гомографии и трансформация изображений.

---

*📌 Примечание: Изображения используются только в учебных целях из открытых источников.*
