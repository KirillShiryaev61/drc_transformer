# Import
from drcompress.transformer import DRCTransformer
import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

class TestDRCTransformer:
    
    def test_check_estimator_sklearn_compatible(self):
        """Проверка совместимости с API scikit-learn"""
        check_estimator(DRCTransformer())

    def test_basic_compression_max_side(self):
        """Базовая компрессия: method='power', side='max'"""
        X = np.array([[1.0], [2.0], [10.0]])
        
        drc = DRCTransformer(threshold=7.0, 
                             coef=0.5, 
                             method='power', 
                             side='max', 
                             dry=0.0)
        X_comp = drc.fit_transform(X)

        assert X_comp[2, 0] < X[2, 0]
        assert X_comp[2, 0] == 8.0
        assert X_comp[0, 0] == X[0, 0]

    def test_basic_compression_min_side(self):
        """Базовая компрессия: method='power', side='min'"""
        X = np.array([[-10.0], [-2.0], [1.0]])
        
        drc = DRCTransformer(threshold=-7.0, 
                             coef=0.5, 
                             method='power', 
                             side='min', 
                             dry=0.0)
        X_comp = drc.fit_transform(X)

        assert X_comp[0, 0] > X[0, 0]
        assert X_comp[0, 0] == -8.0
        assert X_comp[2, 0] == X[2, 0]

    def test_basic_compression_both_side(self):
        """Базовая компрессия: method='power', side='both'"""
        X = np.array([[-10.0], [0.0], [10.0]])
        
        drc = DRCTransformer(threshold=[-7.0, 7.0], 
                             coef=[0.5, 0.5], 
                             method='power', 
                             side='both', 
                             dry=0.0)
        X_comp = drc.fit_transform(X)

        assert X_comp[0, 0] > X[0, 0]
        assert X_comp[0, 0] == -8.0
        assert X_comp[2, 0] < X[2, 0]
        assert X_comp[2, 0] == 8.0
        assert X_comp[1, 0] == X[1, 0]

    def test_mix_compression_both_side(self):
        """
        Смешанная компрессия:
            Минимальные значения — method='linear', coef=0.5;
            Максимальные значения — method='power', coef=0.5. 
        """
        X = np.array([[-10.0], [0.0], [10.0]])

        drc = DRCTransformer(threshold=[-5.0, 7.0], 
                             coef=[0.5, 0.5], 
                             method=['linear', 'power'], 
                             side='both',
                             dry=0.0)
        X_comp = drc.fit_transform(X)

        assert X_comp[0, 0] > X[0, 0]
        assert X_comp[0, 0] == -7.5
        assert X_comp[2, 0] < X[2, 0]
        assert X_comp[2, 0] == 8.0
        assert X_comp[1, 0] == X[1, 0]

    def test_compression_ndarray(self):
        """Компрессия разных признаков массива одновременно"""
        X = np.array([[0, -45, -10], 
                      [1,   0,   0], 
                      [10, 15,  10]])

        drc = DRCTransformer(threshold=[5, 0, [-7, 7]], 
                             coef=0.5, 
                             method=['linear', ['clip', 'power'], 'power'], 
                             side=['max', 'both', 'both'], 
                             dry=0.0)
        X_comp = drc.fit_transform(X)

        # Первый признак
        assert X_comp[0, 0] == 0.0
        assert X_comp[1, 0] == 1.0
        assert X_comp[2, 0] == 7.5

        # Второй признак
        assert X_comp[0, 1] == 0.0
        assert X_comp[1, 1] == 0.0
        assert X_comp[2, 1] == 3.0

        # Третий признак
        assert X_comp[0, 2] == -8.0
        assert X_comp[1, 2] == 0.0
        assert X_comp[2, 2] == 8.0

    def test_dry_blend(self):
        """Проверка параметра dry"""
        X = np.array([[10.0]])
        
        drc_full = DRCTransformer(threshold=7.0, dry=0.0)
        drc_half = DRCTransformer(threshold=7.0, dry=0.5)

        X_full = drc_full.fit_transform(X)
        X_half = drc_half.fit_transform(X)

        # При dry=0.5 результат должен быть средним между исходным и сжатым
        expected = (X[0, 0] + X_full[0, 0]) / 2
        assert np.isclose(X_half, expected)

    def test_auto_threshold_side_max(self):
        """
        Проверка автоматического порога (threshold=None)
        при side='max'
        """
        X = np.array([[1.0], [2.0], [10.0]])

        drc = DRCTransformer(threshold=None, side='max')
        drc.fit(X)

        # Порог должен быть равен максимуму
        assert drc.threshold_[0] == 10.0

        X_comp = drc.transform(X)
        # Ничего не должно сжиматься
        assert np.allclose(X_comp, X)

    def test_auto_threshold_side_min(self):
        """
        Проверка автоматического порога (threshold=None)
        при side='min'
        """
        X = np.array([[1.0], [2.0], [10.0]])

        drc = DRCTransformer(threshold=None, side='min')
        drc.fit(X)

        # Порог должен быть равен минимуму
        assert drc.threshold_[0] == 1.0

        X_comp = drc.transform(X)
        # Ничего не должно сжиматься
        assert np.allclose(X_comp, X)

    def test_auto_threshold_side_both(self):
        """
        Проверка автоматического порога (threshold=None)
        при side='both'
        """
        X = np.array([[1.0], [2.0], [10.0]])

        drc = DRCTransformer(threshold=None, side='both')
        drc.fit(X)

        # Пороги должны быть равны максимуму и минимуму
        assert drc.threshold_[0][0] == 1.0
        assert drc.threshold_[0][1] == 10.0

        X_comp = drc.transform(X)
        # Ничего не должно сжиматься
        assert np.allclose(X_comp, X)

    def test_get_feature_names_out_with_dataframe(self):
        """Проверка get_feature_names_out с pandas DataFrame"""
        X = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        
        drc = DRCTransformer()
        drc.fit(X)

        names = drc.get_feature_names_out()
        assert list(names) == ['a', 'b']

        # Явная передача input_features
        names_2 = drc.get_feature_names_out(input_features=['x', 'y'])
        assert list(names_2) == ['x', 'y']

    def test_get_feature_names_out_with_numpy_array(self):
        """Проверка get_feature_names_out с numpy array"""
        X = np.array([[1, 2], [3, 4]])

        drc = DRCTransformer()
        drc.fit(X)

        names = drc.get_feature_names_out()
        assert list(names) == ['col_0', 'col_1']

    def test_pipeline_intergration(self):
        """Интеграция в Pipeline с обучением модели"""
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)

        pipe = Pipeline([
            ('drc', DRCTransformer(threshold=1.0, side='both')),
            ('model', LogisticRegression())
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (100,)

    def test_fit_transform_vs_fit_transform_separate(self):
        """fit_transform должен давать тот же результат, что и fit + transform"""
        X = np.random.randn(50, 3)

        drc_1 = DRCTransformer(threshold=1.0, side='max')
        result_1 = drc_1.fit_transform(X)

        drc_2 = DRCTransformer(threshold=1.0, side='max')
        drc_2.fit(X)
        result_2 = drc_2.transform(X)

        assert np.allclose(result_1, result_2)

    def test_single_feature_both_side(self):
        """Особый случай: 1 признак + side='both'"""
        X = np.array([[ -5.0 ], [ 0.0 ], [ 5.0 ]])
        
        drc = DRCTransformer(side='both')
        X_comp = drc.fit_transform(X)

        assert X_comp.shape == X.shape

    def test_invalid_parameters(self):
        """Проверка валидации некорректных параметров"""
        X = np.array([[1.0]])

        # coef <= 0
        with pytest.raises(ValueError, match='coef > 0'):
            DRCTransformer(coef=-1).fit(X)

        # dry вне [0,1]
        with pytest.raises(ValueError, match='должен быть в диапазоне'):
            DRCTransformer(dry=1.5).fit(X)

        # неподдерживаемый method
        with pytest.raises(ValueError, match="linear', 'power', 'log', 'clip'"):
            DRCTransformer(method='unknown').fit(X)

        # несовпадение длины threshold, X имеет 1 признак
        with pytest.raises(ValueError, match='Длина параметра'):
            DRCTransformer(threshold=[1, 2]).fit(X)

    def test_check_is_fitted(self):
        """Проверка, что transform без fit вызывает ошибку"""
        X = np.array([[1.0]])
        drc = DRCTransformer()

        with pytest.raises(ValueError, match='not fitted'):
            drc.transform(X)
    
        

    