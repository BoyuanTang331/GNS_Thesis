"""

this file is to convert npy files to a tfrecord and generate the metadata in json
before running npy_folder_path should be changed to the numpy folder path,
line 54 changed to path save the tfrecord

"""


import mpmSceneSim
import tensorflow.compat.v1 as tf
import numpy as np
import os
import json

# run the mpmSceneSim/sphSceneSim simulation code
for t in range(20):   # t, number of sim time to get different fluid behavior in npy   (element/trajectory)
    mpmSceneSim.run(t)



particle_types = []
keys = []
positions = []

npy_folder_path = f'./result/'  # npy data folder
npy_files = sorted(os.listdir(npy_folder_path))

# load the numpy files and write keys as numbers of simulations times
for idx, npy_file in enumerate(npy_files):
    position = np.load(os.path.join(npy_folder_path, npy_file))
    positions.append(position)
    keys.append(idx)
    particle_types.append(np.full(position.shape[1], 5, dtype=np.int64))

# The following functions can be used to convert a value to a type compatible with tf.train.Example.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))



# Serialization the data in tfrecord file with in the format of learning to simulate
with tf.python_io.TFRecordWriter(r'C:\Users\84017\Desktop\mpm2d\mpm2d_valid20.tfrecord') as writer:
    for step, (particle_type, key, position) in enumerate(zip(particle_types, keys, positions)):
        seq = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                "particle_type": _bytes_feature(particle_type.tobytes()),
                "key": _int64_feature(key)
            }),
            feature_lists=tf.train.FeatureLists(feature_list={
                'position': tf.train.FeatureList(
                    feature=[_bytes_feature(position.flatten().tobytes())],
                ),
                'step_context': tf.train.FeatureList(
                    feature=[_bytes_feature(np.float32(step).tobytes())]
                ),
            })
        )
        serialized_data = seq.SerializeToString()
        writer.write(seq.SerializeToString())


################################################################
# load the npy files
data_folder = r'C:\Users\84017\Desktop\mpm2d\result_train'
numpy_files = [f for f in os.listdir(data_folder) if f.endswith('.npy')]

# timestep= step*dt in mpm
dt = 0.0025

# initialize the list to save the data
accumulated_velocities = np.array([], dtype=np.float64).reshape(0,2)
accumulated_accelerations = np.array([], dtype=np.float64).reshape(0,2)

for file_name in numpy_files:
    # load numpy array
    file_path = os.path.join(data_folder, file_name)
    positions = np.load(file_path)

    # calculate the velocity
    velocities = np.diff(positions, axis=0) / dt

    # calculate the acceleration
    # Note: This calculation will reduce two time steps, so the resulting time step will be the original timestep-2
    accelerations = np.diff(positions, n=2, axis=0) / dt**2

    # Accumulated velocity and acceleration data
    accumulated_velocities = np.vstack((accumulated_velocities, velocities.reshape(-1, 2)))
    accumulated_accelerations = np.vstack((accumulated_accelerations, accelerations.reshape(-1, 2)))

    stats = {
        'file': file_name,
        'vel_mean': np.mean(velocities, axis=(0, 1)).tolist(),
        'vel_std': np.std(velocities, axis=(0, 1)).tolist(),
        'acc_mean': np.mean(accelerations, axis=(0, 1)).tolist(),
        'acc_std': np.std(accelerations, axis=(0, 1)).tolist(),
    }
print(stats)



# define the metadata dict
metadata = {
    "bounds": [[0.05, 0.95], [0.05, 0.95]],  # bound/n_grid
    "sequence_length": 799,           # timestep in each simulation -1
    "default_connectivity_radius": 0.0015,
    "dim": 2,                        # dimension
    "dt": 0.0025,                    # step*dt
    "vel_mean": stats['vel_mean'],
    "vel_std": stats['vel_std'],
    "acc_mean": stats['acc_mean'],
    "acc_std": stats['acc_std']
}

# save dict as a json file
with open(r'C:\Users\84017\Desktop\mpm2d\metadata.json', 'w') as json_file:
    json.dump(metadata, json_file, indent=4)
