# Импорты
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Код трансформера
class DRCTransformer(BaseEstimator, TransformerMixin):

    # Инициализация трансформера
    def __init__(self, coef=0.5, threshold=None, dry=0.0, method='power'):
        '''
        Параметры:
        - coef: степень сжатия (float) или словарь {col: coef}
        - threshold: порог (float или dict). Если None — будет вычислен как максимум колонки при fit.
        - dry: доля оригинального сигнала (0.0 = полное сжатие, 1.0 = без изменений)
        - method: 'power' или 'log'
        '''
        self.coef = coef
        self.threshold = threshold
        self.dry = dry
        self.method = method

    # Функция компрессии
    def _compress_array(self, data, threshold, coef, dry, method):
        '''
        Применяет компрессию к одному признаку (1D numpy array).
        Возвращает сжатый массив той же длины.
        '''
        data = np.asarray(data, dtype=np.float64)
        mask = data > threshold
        compressed = data.copy()
    
        if method == 'power':
            compressed[mask] = threshold + np.power(data[mask] - threshold, coef)
    
        else:
            compressed[mask] = threshold + np.log1p(data[mask] - threshold) * coef
    
        return dry * data + (1 - dry) * compressed

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
        elif isinstance(param, dict):
            '''
            Берем значения из словаря, если значение отсутствует,
            берем значение по умолчанию
            '''
            return [param.get(name, default) for name in self.feature_names_in_]
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
        check_is_fitted(self, 'feature_names_in_')
        return np.array(self.feature_names_in_, dtype=object)
            
    # Метод fit()
    def fit(self, X, y=None):
        '''
        Обучение трансформера.
        Определяем количество колонок и значения параметров для них
        '''
        # Сохраняем информацию о колонках, если X — DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_list()
            X_values = X.values
        else:
            X_values = np.asarray(X)
            # Если вход — массив, даём имена "заглушки"
            self.feature_names_in_ = [f'col_{i}' for i in range(X_values.shape[1])]

        n_features = X_values.shape[1]

        # Обработка threshold
        if self.threshold is None:
            '''
            Если порог не задан он определяется, 
            как максимальное значение признака
            '''
            self.threshold_ = np.max(X_values, axis=0).tolist()
        elif isinstance(self.threshold, dict):
            '''
            Берем значения из словаря, если значения отсутствует,
            берем максимальное значение признака
            '''
            self.threshold_ = [
                self.threshold.get(name, np.max(X_values[:, i]))
                for i, name in enumerate(self.feature_names_in_)
            ]
        elif np.isscalar(self.threshold):
            '''
            Если порог задан одним числом, распределяем 
            это значение на все признаки
            '''
            self.threshold_ = [self.threshold] * n_features
        else:
            '''
            Если значения заданы массивом или списком,
            распределяем значения по порядку
            '''
            self.threshold_ = list(self.threshold)
            if len(self.threshold_) != n_features:
                raise ValueError(f'Длина параметра не совпадает с числом признаков ({n_features}).')

        # Обработка остальных параметров (coef, dry, method)
        self.coef_ = self._expand_param(self.coef, n_features, default=0.5)

        # Валидация параметре coef, должно быть coef > 0
        for i, c in enumerate(self.coef_):
            if c <= 0:
                col_name = self.feature_names_in_[i]
                raise ValueError(f"Параметр coef для колонки '{col_name}' = {c}, должен быть coef > 0.")
        
        self.dry_ = self._expand_param(self.dry, n_features, default=0.0)
        
        # Валидация параметра dry, должно быть [0, 1]
        for i, d in enumerate(self.dry_):
            if not (0 <= d <= 1):
                col_name = self.feature_names_in_[i]
                raise ValueError(f"Параметр dry для колонки '{col_name}' = {d}, должен быть в диапазоне [0, 1].")
                
        self.method_ = self._expand_param(self.method, n_features, default='power')

        # Валидация параметра method, должен быть power или log
        for i, m in enumerate(self.method_):
            if m not in ['power', 'log']:
                col_name = self.feature_names_in_[i]
                raise ValueError(f"Неподдерживаемый метод: '{m}'. Допустимые значения: 'power', 'log'")

        return self     

    # Метод transform()
    def transform(self, X):
        '''
        Применение компрессии к данным
        '''
        # Проверка, обучен ли трансформер
        check_is_fitted(self)
        
        X_values = np.asarray(X)
        X_compressed = X_values.copy()

        for i in range(X_values.shape[1]):
            X_compressed[:, i] = self._compress_array(
                data=X_values[:, i],
                threshold=self.threshold_[i],
                coef=self.coef_[i],
                dry=self.dry_[i],
                method=self.method_[i]
            )
        
        return X_compressed