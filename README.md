# TezzAutoML

Just focus on the data, TezzAutoML will do the rest.

You'll need to do all the preprocessing with the data the way you want.

## Installation

## Usage

```python
from tezzautoml.automl import AutoML

automl = AutoML(data=df, target='target', task='classification', n_trials=100, fast_mode=False)
```

When Fast Mode is False, it will use KFold for Regression tasks and StratifiedKFold for Classification
tasks.

When Fast Mode is True, it will use train_test_split for both the tasks.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
