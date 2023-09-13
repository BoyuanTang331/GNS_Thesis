# GNS_Thesis

GNS-Thesis

Dataset generation in two method: MPM and SPH

For the MPM method, when converting the .ply file to .tfrecord file, there is a data structure problem about float32 to float64

the .ply file saved in mpm folder, the output33.tfrecord is the convert result using convert(1)
The original dataset from the paper can be downloaded:
https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/water/train.tfrecord

For SPH method, the output file.VTK or .bgeo does not include the position information, the data generated code is the file custom_scene.py
