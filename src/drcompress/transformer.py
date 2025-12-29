# Импорты
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

# Код трансформера
class DRCTransformer(TransformerMixin, BaseEstimator):

    # Инициализация трансформера
    def __init__(self, threshold=None, coef=0.5, side='max', method='power', dry=0.0):
        '''
        Аргументы:
        
        threshold : (int, float, list, dict) должен быть threshold > 0, default None.
            Порог срабатывания компрессии (абсолютное значение).     
        
        coef : (int, float, list, dict) должен быть coef > 0, default 0.5.
            Для method='linear' — коэффициент линейного сжатия.
            Для method='power' — степень сжатия.
            Для method='log' — масштабирующий коэффициент логарифмического сжатия. 
            Для method='clip' — неактивен.
        
        side : (str, list, dict), default 'max'.
            Сторона сжатия.
            Доступные методы — {'max', 'min', 'both'}
        
        method : (str, list, dict), default 'power'.
            Метод компрессии.
            Доступные методы — {'linear', 'power', 'log', 'clip'}
        
        dry : (int, float, list, dict) должен быть в диапозоне [0, 1], default 0.
            Доля исходных значений в результате (0 = полная компрессия, 
            1 = без изменений). 
        '''
        self.threshold = threshold
        self.coef = coef
        self.side = side
        self.dry = dry
        self.method = method

    # Функция для компрессии данных
    def _compressing(self, data, threshold, coef, method, dry):
        '''
        Применяет компрессию к одному признаку (1D numpy array).
        Возвращает сжатый массив той же длины.
        '''
        
        # Процесс сжатия
        mask = data > threshold
        compressed = data.copy()
        diff = data[mask] - threshold

        if method == 'linear':
            compressed[mask] = threshold + diff * coef
        
        elif method == 'power':
            compressed[mask] = threshold + np.power(diff + 1, coef) - 1
    
        elif method == 'log':
            compressed[mask] = threshold + np.log1p(diff) * coef

        else: # method == 'clip'
            compressed[mask] = threshold
    
        return dry * data + (1 - dry) * compressed

    # Компрессия одной стороны
    def _side_compress(self, data, threshold, coef, method, dry):
    
        # Расчет минимального
        if data.min() < 0:

            # Избавляемся от отрицательных значений
            minimal = abs(data.min())
            data += minimal
            threshold += minimal

            # Сжатие сигнала
            data = self._compressing(data=data, 
                                     threshold=threshold, 
                                     coef=coef, 
                                     method=method, 
                                     dry=dry)
            
            # Возврат к искомым значениям
            data -= minimal
            return data
            
        else:
            data = self._compressing(data=data, 
                                     threshold=threshold, 
                                     coef=coef, 
                                     method=method, 
                                     dry=dry)
            return data

    # Функция компрессии
    def _compress_array(self, data, threshold, coef, side, method, dry):

        data = np.array(data, dtype=np.float64, copy=True)

        # Сжатие максимальных значений
        if side == 'max':
            # Сжатие сигнала
            data = self._side_compress(data=data, 
                                       threshold=threshold, 
                                       coef=coef, 
                                       method=method, 
                                       dry=dry) 
            return data

        # Сжатие минимальных значений
        elif side == 'min':
            # Инвертирование сигнала
            data = -data     
            threshold = -threshold

            # Сжатие сигнала
            data = self._side_compress(data=data, 
                                       threshold=threshold, 
                                       coef=coef, 
                                       method=method, 
                                       dry=dry)

            # Обратное инвертирование сигнала
            data = -data
            return data

        # Сжатие максимальных и минимальных значений
        else:
            if np.isscalar(threshold):
                threshold = [threshold] * 2
            if np.isscalar(coef):
                coef = [coef] * 2
            if np.isscalar(method):
                method = [method] * 2
            if np.isscalar(dry):
                dry = [dry] * 2

            # Распаковка параметров
            threshold_min, threshold_max = threshold
            coef_min, coef_max = coef
            method_min, method_max = method
            dry_min, dry_max = dry     

            # Сжатие максимальных значений
            data = self._side_compress(data=data, 
                                       threshold=threshold_max, 
                                       coef=coef_max, 
                                       method=method_max, 
                                       dry=dry_max) 

            # Инвертирование сигнала
            data = -data     
            threshold_min = -threshold_min

            # Сжатие сигнала
            data = self._side_compress(data=data, 
                                       threshold=threshold_min, 
                                       coef=coef_min, 
                                       method=method_min, 
                                       dry=dry_min)

            # Обратное инвертирование сигнала
            data = -data
            return data

    # Вспомогательная функция _list_if_pair
    def _list_if_pair(self, param):
        '''
        Преобразование заданных параметров
        '''
        if isinstance(param, (list, tuple)) and len(param) == 2:
            return [param]
        else:
            return param

    # Вспомогательная функция _expand_param
    def _expand_param(self, param, n_features, default):
        '''
        Преобразование заданных параметров.
        '''
        if param is None:
            '''
            Если параметр не задан он определяется, 
            как значение по умолчанию
            '''
            return [default] * n_features
        elif np.isscalar(param):
            '''
            Если параметр задан одним числом, распределяем 
            это значение на все признаки
            '''
            return [param] * n_features
        else:
            '''
            Если значения параметра заданы массивом или списком,
            распределяем значения по порядку
            '''
            param = list(param)
            if len(param) != n_features:
                raise ValueError(f'Длина параметра не совпадает с числом признаков ({n_features}).')
            return param

    # Метод get_feature_names_out()
    def get_feature_names_out(self, input_features=None):
        '''
        Возвращение имен входных признаков
        '''
        if input_features is not None:
            return np.array(input_features, dtype=object)
        
        return np.array([f"col_{i}" for i in range(self.n_features_in_)], dtype=object)
            
    # Метод fit()
    def fit(self, X, y=None):
        '''
        Обучение трансформера.
        Определяем количество колонок и значения параметров для них
        '''
        # Валидация данных
        X_values = validate_data(
            self, X,
            reset=True,
            ensure_all_finite=True,
            ensure_2d=True,
            dtype=np.float64,
        )

        # Преобразование параметров если признак один
        if self.n_features_in_ == 1 and self.side == 'both':
            self.threshold = self._list_if_pair(self.threshold)
            self.coef = self._list_if_pair(self.coef)
            self.method = self._list_if_pair(self.method)
            self.dry = self._list_if_pair(self.dry)

        # Обработка параметра coef
        self.coef_ = self._expand_param(self.coef, self.n_features_in_, default=0.5)
        
        # Валидация параметра coef, должно быть coef > 0
        for i, c in enumerate(self.coef_):
            if isinstance(c, (list, tuple)):
                for j in c:
                    if not np.isfinite(j):
                        raise ValueError(f"Параметр coef для колонки ({i}) должен быть конечным числом.")
                    elif j <= 0:
                        raise ValueError(f"Параметр coef для колонки ({i}) должен быть coef > 0.")
            else: 
                if not np.isfinite(c):
                    raise ValueError(f"Параметр coef для колонки ({i}) должен быть конечным числом.")
                elif c <= 0:
                    raise ValueError(f"Параметр coef для колонки ({i}) должен быть coef > 0.")

        # Обработка параметра method
        self.method_ = self._expand_param(self.method, self.n_features_in_, default='power')

        # Валидация параметра method, должен быть linear, power или log
        for i, m in enumerate(self.method_):
            if isinstance(m, (list, tuple)):
                for j in m:
                    if j not in ['linear', 'power', 'log', 'clip']:
                        raise ValueError(f"Неподдерживаемый метод в колонке ({i}), допустимые значения: 'linear', 'power', 'log', 'clip'")
            else:
                if m not in ['linear', 'power', 'log', 'clip']:
                    raise ValueError(f"Неподдерживаемый метод в колонке ({i}), допустимые значения: 'linear', 'power', 'log', 'clip'")
        
        # Обработка параметра side
        self.side_ = self._expand_param(self.side, self.n_features_in_, default='max')
        
        # Валидация параметра side, должен быть max, min или both
        for i, s in enumerate(self.side_):
            if s not in ['max', 'min', 'both']:
                raise ValueError(f"Неподдерживаемый метод в колонке ({i}), допустимые значения: 'max', 'min', 'both'")
                
        # Обработка параметра dry
        self.dry_ = self._expand_param(self.dry, self.n_features_in_, default=0.0)
        
        # Валидация параметра dry, должно быть [0, 1]
        for i, d in enumerate(self.dry_):
            if isinstance(d, (list, tuple)):
                for j in d:
                    if not np.isfinite(j):
                        raise ValueError(f"Параметр dry для колонки ({i}), должен быть конечным числом.")
                    elif not (0 <= j <= 1):
                        raise ValueError(f"Параметр dry для колонки ({i}), должен быть в диапазоне [0, 1].")
            else:
                if not np.isfinite(d):
                    raise ValueError(f"Параметр dry для колонки ({i}), должен быть конечным числом.")
                elif not (0 <= d <= 1):
                    raise ValueError(f"Параметр dry для колонки ({i}), должен быть в диапазоне [0, 1].")

        # Обработка threshold
        if self.threshold is None:
            '''
            Если порог не задан он определяется, 
            как максимальное, минимальное или оба 
            значения признака в зависимости от 
            указанного параметра side
            '''
            self.threshold_ = []
            for i, side in enumerate(self.side_):
                if side == 'max':
                    self.threshold_.append(np.max(X_values[:, i]))
                elif side == 'min':
                    self.threshold_.append(np.min(X_values[:, i]))
                else: # side == 'both'
                    self.threshold_.append([np.min(X_values[:, i]), np.max(X_values[:, i])])    
        
        elif np.isscalar(self.threshold):
            '''
            Если порог задан одним числом, распределяем 
            это значение на все признаки
            '''
            self.threshold_ = [self.threshold] * self.n_features_in_
        
        else:
            '''
            Если значения заданы массивом или списком,
            распределяем значения по порядку
            '''
            self.threshold_ = list(self.threshold)
            if len(self.threshold_) != self.n_features_in_:
                raise ValueError(f'Длина параметра не совпадает с числом признаков ({self.n_features_in_}).')

        # Валидация параметра threshold, должен быть конечным числом
        for i, h in enumerate(self.threshold_):
            if isinstance(h, (list, tuple)):
                for j in h:
                    if not np.isfinite(j):
                        raise ValueError(f"Параметр threshold для колонки ({i}), threshold должен быть конечным числом.")
            else:
                if not np.isfinite(h):
                    raise ValueError(f"Параметр threshold для колонки ({i}), threshold должен быть конечным числом.")

        return self     

    # Метод transform()
    def transform(self, X):
        '''
        Применение компрессии к данным
        '''
        # Проверка, обучен ли трансформер
        check_is_fitted(self)

        # Валидация данных
        X_values = validate_data(
            self, X,
            reset=False,
            ensure_all_finite=True,
            ensure_2d=True,
            dtype=np.float64,
        )

        X_compressed = X_values.copy()

        for i in range(self.n_features_in_):
            
            X_compressed[:, i] = self._compress_array(
                data=X_values[:, i],
                threshold=self.threshold_[i],
                coef=self.coef_[i],
                side=self.side_[i],
                dry=self.dry_[i],
                method=self.method_[i]
            )
        
        return X_compressed