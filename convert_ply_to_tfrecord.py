# -*- coding: utf-8 -*-
# @Time : 2023/7/25 0:24
# @Author : Boyuan
# @File : convert.py
from plyfile import PlyData, PlyElement
import tensorflow as tf
import numpy as np
import os

def read_ply_file(file_path):
    ply_data = PlyData.read(file_path)
    vertex_data = ply_data['vertex']
    points = np.column_stack([vertex_data['x'], vertex_data['y']]) #vertex_data['z']])
    points = points[:, :, np.newaxis]
    return points


def numpy_array_to_example(particle_type, key, points):
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'particle_type':tf.train.Feature(int64_list=tf.train.Int64List(value=particle_type)),
            'key':tf.train.Feature(int64_list=tf.train.Int64List(value=[key])),
            'points': tf.train.Feature(float_list=tf.train.FloatList(value=points.flatten()))
        }))
    return example


def save_to_tfrecord(particle_type,keys,data, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(len(data)):
            example = numpy_array_to_example(particle_type,keys[i],data[i])
            writer.write(example.SerializeToString())


data_dir = r'C:\Users\84017\Desktop\mpm'
data_files = os.listdir(data_dir)
data = np.empty((2000, 4, 1),dtype=np.float32)
particle_type = np.full((data.shape[0],), 5, dtype=np.int64)
keys = []

# for f in data_files:
#     #path = os.path.join(data_dir, f)
#     # print(path)
#     points = read_ply_file(path)
#     # print(points.shape)
#     # print(points.dtype)
#     # print('============')
#     # points = np.float32(points)
#     data = np.concatenate([data, points], axis=2)
#     # print(data.dtype)

for i, f in enumerate(data_files):
    points = read_ply_file(os.path.join(data_dir, f))
    data = np.concatenate([data, points], axis=1) #if 2D axis=1 if 3D axis=2
    keys.append(i)

data = data[:, :, 2:]
save_path = r'C:\Users\84017\Desktop\mpm\Water2D_small.tfrecord'
save_to_tfrecord(particle_type, keys, data, save_path)


print("data_files:", data_files)
print("keys:", keys)