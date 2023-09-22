# GNS_Thesis

GNS-Thesis


----Update from 09.23.2023----
For the MPM method problem, how to add an index to my convert_ply_to_tfrecord.py the index[] shows timesteps, particle number, and position, in my file, the features of data were only written in correct form like in GNS (one tuple for dataset, include 2 dicts, first dict is particle_type in array and key for timestep, second dict is position in array. showed in .png) 

For the SPH method problem, the output file *.VTK* or *.bgeo* does not include the position information, the data generated code is from the file custom_scene.py in custom_scene.py I can use * fluid = sim.getFluidModel(0)* to get only the initial position. The author said it can be written in time_step_callback() in callbacks.py. but it didn't show the sequence position. the callbacks func can show a set of *steps*(only words for example) for each timestep, how can use this function to replace steps to my positions

----------------------------------------------------------------------------------------------------------------------------------------
Original paper to re-implement: GNS https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate

Dataset generation in two methods: SPH and MPM

SPH data generate library:  https://github.com/InteractiveComputerGraphics/SPlisHSPlasH   and the example for generating the dataset is using custom_scene.py  https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/8454e9f454fef20771dfe98d318904da62764b4c/pySPlisHSPlasH/examples/custom_scene.py#L4

MPM data generate library:  https://github.com/yuanming-hu/taichi_mpm   The example for generating MPM dataset is mpm3d.py which has a ti.tools.PLYWriter function https://github.com/taichi-dev/test_actions/blob/e5ed25678acfbe3eff49f4ac05345b183876890f/python/taichi/examples/simulation/mpm3d.py

The original dataset from the GNS paper can be downloaded at: https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/water/train.tfrecord or use bash referring to GNS page
