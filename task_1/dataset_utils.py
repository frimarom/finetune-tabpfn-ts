# split data into batches
# params: sample_size, batch_size, sample offset
# fun split into samples
# datasets into samples by first going vertically trough each time series of dataset and then moving by offset horizontally and repeating

# fun split into batches
# construct batches by grouping samples together which either are in the same time series but far apart or are from different time series