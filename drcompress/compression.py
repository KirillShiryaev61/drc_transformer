# Импорты
import pandas as pd
import numpy as np

# Функция для компрессии данных
def compress(data, threshold, coef, method='power', dry=0):
    '''
    Применяет компрессию к одному признаку (1D numpy array).
    Возвращает сжатый массив той же длины.
    
    Аргументы:
    data : array-like
        Оригинальный признак (pd.Series, np.array и т.п.).
    threshold : float
        Порог срабатывания компрессии (абсолютное значение).
    coef : float
        Для method='power' — степень сжатия (обычно 0 < coef < 1).
        Для method='log' — масштабирующий коэффициент логарифмического сжатия.
    method : {'power', 'log'}, default 'power'
        Метод компрессии.
    dry : float, default 0
        Доля исходных значений в результате (0 = полная компрессия, 
        1 = без изменений). Должен быть в [0, 1].

    Вывод: Скомпрессированный массив (np.ndarray).
    '''

    # Проверка параметров
    if not np.isfinite(threshold):
        raise ValueError("Параметр threshold должен быть конечным числом.")
    if not (0 <= dry <= 1):
        raise ValueError(f"Параметр dry = {dry}, должен быть в диапазоне [0, 1].")
    if coef <= 0:
        raise ValueError(f"Параметр coef = {coef}, должен быть coef > 0.")
    if method not in ['power', 'log']:
        raise ValueError(f"Неподдерживаемый метод: '{method}'. Допустимые значения: 'power', 'log'")

    data = np.asarray(data, dtype=np.float64)
    mask = data > threshold
    compressed = data.copy()
    
    if method == 'power':
        compressed[mask] = threshold + np.power(data[mask] - threshold, coef)
    
    else:
        compressed[mask] = threshold + np.log1p(data[mask] - threshold) * coef
    
    return dry * data + (1 - dry) * compressed