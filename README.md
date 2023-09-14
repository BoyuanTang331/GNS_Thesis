# GNS_Thesis

GNS-Thesis

Dataset generation in two methods: MPM and SPH

For the MPM method problem, when converting the .ply file to .tfrecord file, there is a data structure problem about float32 to float64, when I try to open the original paper dataset use opentfrecord.py, in my output .tfrecord file has a data structure problem.

the example of .ply file saved in mpm folder, the output33.tfrecord is the converted result using convert(1).py
The original dataset from the GNS paper can be downloaded at: https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/water/train.tfrecord

For the SPH method problem, the output file.VTK or .bgeo does not include the position information, the data generated code is from the file custom_scene.py, needs to get a time sequence file for each time step the position information of all the fluid particles and then write it in .tfrecord form file.

----------------------------------------------------------------------------------------------------------------------------------------
Original paper to re-implement: GNS https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate

SPH data generate library:  https://github.com/InteractiveComputerGraphics/SPlisHSPlasH   and the example for generating the dataset is using custom_scene.py  https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/8454e9f454fef20771dfe98d318904da62764b4c/pySPlisHSPlasH/examples/custom_scene.py#L4

MPM data generate library:  https://github.com/yuanming-hu/taichi_mpm   the example for generating MPM dataset is mpm3d which has a output.ply function https://github.com/taichi-dev/test_actions/blob/e5ed25678acfbe3eff49f4ac05345b183876890f/python/taichi/examples/simulation/mpm3d.py
