import functools
import tensorflow as tf
import reading_utils  # import the reading_utils from learning to stimulate
import os

# # import metadata from datasets in learning to simulate
metadata = {
 "bounds": [[0.1, 0.9], [0.1, 0.9]],
 "sequence_length": 1000,
 "default_connectivity_radius": 0.015,
 "dim": 2,
 "dt": 0.0025,
 "vel_mean": [1.1927917091800243e-05, -0.0002563314637168018],
 "vel_std": [0.0013973410613251076, 0.00131291713199288],
 "acc_mean": [-1.10709094667326e-08, 8.749365512454699e-08],
 "acc_std": [6.545267379756913e-05, 7.965494666766224e-05]
}

#create a TFRecord object, including .tfrecord data
ds = tf.data.TFRecordDataset([os.path.join("C:/Users/84017/PycharmProjects/pythonProject1", f'train.tfrecord')])
# call the particle function, transfer the metadata as parameters
ds = ds.map(functools.partial(
    reading_utils.parse_serialized_simulation_example, metadata=metadata))
#transfer the ds object to the numpy iterator and print the results
for element in ds.as_numpy_iterator():
    print(element)