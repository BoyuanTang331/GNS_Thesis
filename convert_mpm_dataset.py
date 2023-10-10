import functools
import os
import json
import pickle

from plyfile import PlyData, PlyElement
import tensorflow.compat.v1 as tf
import numpy as np
import reading_utils

# # Set datapath and validation set
# data_path = 'C:/Users/84017/Downloads'
# filename = 'train_waterdrop.tfrecord'
# # Read metadata
# def _read_metadata(data_path):
#     with open(os.path.join(data_path, 'metadata_waterdrop.json'), 'rt') as fp:
#         return json.loads(fp.read())
# # Fetch metadata
# metadata = _read_metadata(data_path)
# print(metadata)
# # Read TFRecord
# ds_org = tf.data.TFRecordDataset([os.path.join(data_path, filename)])
# ds = ds_org.map(functools.partial(reading_utils.parse_serialized_simulation_example, metadata=metadata))
# # Convert to list
# # @tf.function
# def list_tf(ds):
#     return (list(ds))
# lds = list_tf(ds)


# Define function to read positions from PLY files
def read_positions_from_ply(file_path):
    positions = []
    with open(file_path, 'rb') as f:
        for line in f.readlines():
            if line.startswith(b'element vertex'):
                num_vertices = int(line.split()[-1])
            if line.startswith(b'end_header'):
                break
        for _ in range(num_vertices):
            line = f.readline().strip()
            if not line:  # Check if the line is empty
                continue
            values = line.split()
            if len(values) < 2:  # Check if there are less than 2 values
                continue
            x, y = map(float, f.readline().split()[:2])
            positions.append([x, y])
    return np.array(positions)

# Fetch particle types, keys, and positions from PLY files
ply_folder_path = 'C:/Users/84017/Desktop/mpm'
ply_files = sorted(os.listdir(ply_folder_path))



particle_types = []
keys = []
positions = []
for idx, ply_file in enumerate(ply_files):
    particle_types.append(np.array([5]))  # As per your statement, all particle types are 5
    keys.append(idx)  # Time series starts from 0
    positions.append(read_positions_from_ply(os.path.join(ply_folder_path, ply_file)))

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



# Write TF Record
with tf.python_io.TFRecordWriter('test.tfrecord') as writer:
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

        writer.write(seq.SerializeToString())




# dt = tf.data.TFRecordDataset(['test.tfrecord'])
# dt = dt.map(functools.partial(reading_utils.parse_serialized_simulation_example, metadata=metadata))


#
# # Check if the original TFRecord and the newly generated TFRecord are the same
# for ((_ds_context, _ds_feature), (_dt_context, _dt_feature)) in zip(ds, dt):
#     if not np.allclose(_ds_context["key"].numpy(), _dt_context["key"].numpy()):
#         break
#
#     if not np.allclose(_ds_context["particle_type"].numpy(), _dt_context["particle_type"].numpy()):
#         break
#
#     if not np.allclose(_ds_feature["position"].numpy(), _dt_feature["position"].numpy()):
#         break
#
# else:
#     print("TFRecords are similar!")