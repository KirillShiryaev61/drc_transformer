# DRCTransformer

Eliminate the influence of outliers in data, or `OOD` *(out of distribution)*, in both training and test data.

This method was inspired by the history of the first audio signal compressors on radio stations, which were designed to protect expensive equipment from unexpected signal surges on audio tracks.

*\* The transformer is fully compatible with the **Scikit-Learn** library and is distributed under the **MIT-license**.*

## Description

**DRCTransformer** is a scikit-learn-compatible transformer for robust handling of outliers and `OOD` values without hard clipping and while preserving data informativeness. It is suitable for use in production ML pipelines where stability and interpretability are important.

### Features

- Protection against outliers and `OOD` values without losing gradient information
- Flexible configuration: linear, power, logarithmic compression or clipping
- Support for `side='min'`, `'max'`, `'both'` — processing of distribution tails as desired  
- `dry` parameter — smooth control of compression level (dry/wet blend)
- Fully compatible with `sklearn.Pipeline`, `ColumnTransformer`, `GridSearchCV`
- Works with `numpy.ndarray` and `pandas.DataFrame`

## Installing
### Dependencies

drcompress requires:

- Python (>= 3.8)
- NumPy (>= 1.23)
- Pandas (>= 2.0)
- Scikit-Learn (>=1.4)

### User installation
```bash
pip install git+https://github.com/KirillShiryaev61/drc_transformer.git
```


## Examples

>**Mixed compression:**

```python
>>>import numpy as np
>>>from drcompress import DRCTransformer

>>>X = np.array([[-10.0], [0.0], [10.0]])
>>>drc = DRCTransformer(threshold=[-5.0, 7.0], 
                        coef=[0.5, 0.5], 
                        method=['linear', 'power'], 
                        side='both',
                        dry=0.0)
>>>X_comp = drc.fit_transform(X)
>>>print(X_comp)
[[-7.5]
 [ 0. ]
 [ 8. ]]
```

>**Pipeline compression:**

```python
>>>from sklearn.pipeline import Pipeline
>>>from sklearn.ensemble import RandomForestRegressor
>>>from drcompress import DRCTransformer

>>>pipeline = Pipeline([
    ('drc', DRCTransformer(threshold=95)),
    ('model', RandomForestRegressor())])

>>>pipeline.fit(X_train, y_train)
```

## Documentation

🔗 [Documentation_RUS](https://kirillshiryaev61.github.io/drc_transformer/documentation_rus.md)

## Testing

Unit tests for `DRCTransformer` are located in the [`tests/`](./tests) directory.  
The test suite is configured via `pyproject.toml` to run `test_drctransformer.py` automatically.

To run the tests:

```bash
# Install the package in development mode with test dependencies
pip install -e .[test]

# Run all tests
pytest
```

## Contact

**Kirill Shiryaev**  
GitHub: [@KirillShiryaev61](https://github.com/KirillShiryaev61)  
Telegram: [@govorite_gromche](https://t.me/govorite_gromche)

Feel free to reach out with questions or ideas!