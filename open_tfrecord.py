"""
This code is used to open tfrecord files from learning to simulate,
the dataset are saved in a tfrecord with Serialization and the global parameters in json format
before running the code,the data_path and filename should be changed properly

"""


import functools
import tensorflow.compat.v1 as tf
import reading_utils  # import the reading_utils from learning to stimulate
import os
import json
import numpy as np

# #Paste the metadata from json file
# metadata = {
#  "bounds": [[0.01, 0.99], [0.01, 0.99]],
#  "sequence_length": 9,
#  "default_connectivity_radius": 0.0015,
#  "dim": 2,
#  "dt": 0.0002,
#  # "vel_mean": [-3.964619574176163e-05, -0.00026272129664401046],
#  # "vel_std": [0.0013722809722366911, 0.0013119977252142715],
#  # "acc_mean": [2.602686518497945e-08, 1.0721623948191945e-07],
#  # "acc_std": [6.742962470925277e-05, 8.700719180424815e-05]
# }


data_path = r"C:\Users\84017\Desktop\mpm2d"
filename = 'test.tfrecord'

# load the metadata and tfrecord dataset from dictionary
def _read_metadata (data_path):
 with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
  return json.loads(fp.read())
metadata = _read_metadata(data_path)


# create a TFRecord object, including .tfrecord data
ds = tf.data.TFRecordDataset([os.path.join(data_path, filename)])
# call the partial function, transfer the metadata as parameters
ds = ds.map(functools.partial(
    reading_utils.parse_serialized_simulation_example, metadata=metadata))


# transfer the ds object to the numpy iterator and print the results
k = 0  # init K to check how many elements in tfrecord, element means trajectories
for element in ds.as_numpy_iterator():
    k += 1
    print(k)
    print(element)  # this step shows dataset in array

for _ in ds:
    k += 1
print(f" totally {k} element")

# # print all position as in position_shape use for checking if all position are saved in tfrecord
# pos = np.empty([1, 678, 2])  # Position is encoded as [sequence_length, num_particles, dim]
# particle_type = None
# for element in ds.as_numpy_iterator():
#    temp = element[1]['position']
#    temp = np.reshape(temp, [1, 678, 2])
#    pos = np.concatenate([pos, temp], axis=0)
#    particle_type = element[0]['particle_type']
#
# pos = pos[1:, :, :]
# positions = ({'particle_type': particle_type}, {'positions': pos})
# pass