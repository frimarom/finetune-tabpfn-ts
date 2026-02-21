# Explanation of feature transformer in TabPFN-TS
The feature transformer is a crucial component for TabPFN-TS since it helps TabPFN to handle time series regression tasks

## How the code works:
### feature_transformer.py
- contains the basic feature transformer tasks which will later be used to transform the time series data
- gets different feature generator classes from feature_generators.py and applies them one by one to the input data via `transform()` method

### feature_generator_base.py
- contains the base class for all feature generators
- defines the interface for all feature generators, including the `generate()` method that later will be implemented to transform the base data into the new features
- the generate method will later be called by the feature transformer via `apply()`

### TabPFN needs
`(n_samples, batch_size, n_features)`