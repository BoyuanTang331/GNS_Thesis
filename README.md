# GNS_Thesis

GNS-Thesis


----Update from 10.16.2023----

bug: when converting the .ply point cloud file to .tfrecord file using tf.SequenceExample function, it shows the buffer size error. 
![2d8fbf7c0399910fc7a777dac7a5d05](https://github.com/BoyuanTang331/GNS_Thesis/assets/117408630/e2b95d4d-214a-4f35-bfe4-b3220a3cd827)


-----------------------------

In order to directly use the model to train, I want to write the dataset in the same form as the author, their dataset is written in *.tfrecord* form. In the dataset reading function, https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/learning_to_simulate/reading_utils.py#L59 shows that their dataset was written in the certain format as tf.SequenceExample https://www.tensorflow.org/api_docs/python/tf/train/SequenceExample What I need do is get the fluid particle position from the traditional physic engine MPM&SPH method, and add the time step for key as well as particle type a full array with 5. 
![image](https://github.com/BoyuanTang331/GNS_Thesis/assets/117408630/751f0f00-beb7-489e-a8c8-bbfe1972445d)

Check if the .tfrecord is correct can use the **opentfrecord.py** code to open, Generate mpm data use **mpm_data_generation.py** code, Some ply example are in **mpm** folder. Convert .ply to .tfrecord code is **convert_mpm_dataset.py**  .tfrecord include buffer size error is **test.tfrecord**  

I found a reply from the author in an issue that if .tfrecord is difficult to achieve， it can be used another method tf.data.Dataset to complete dataset https://github.com/google-deepmind/deepmind-research/issues/199#issuecomment-901040649 

-----------------------------

For the MPM method problem, use *mpm_data_generation.py* can generate a set of ply files, which include 2D/3D point position, each ply file shows the particle position in *step×dt* time interval. the ply file is specifically used for a render software Blender, inside the file it may also include the vertex， edge and surface information. how to extract only the point position information to convert the tfrecord? In https://github.com/BoyuanTang331/GNS_Thesis/blob/b923c6c6ee487335db273528457e73d8f2505fb5/mpm_data_generation.py#L114 this ti.tool.PLYwriter function it convert the numpy array to ply, can I directly use numpy array to convert a tfrecord without considering ply file?

For the SPH method problem, the output file *.VTK* or *.bgeo* may not include the position information, the data generated code is from the file **custom_scene.py**. In this file I can use **fluid = sim.getFluidModel(0)** to get only the initial position. The author said it can be written in time_step_callback() in callbacks.py. but it didn't show the sequence position. the callbacks function can show a set of *steps*(only words for example) for each timestep, how can I use this function to replace steps to my positions? reply from the library author (https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/discussions/275#discussioncomment-7030194 )


----------------------------------------------------------------------------------------------------------------------------------------
Original paper to re-implement: GNS https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate

Dataset generation in two methods: SPH and MPM

SPH data generate library:  https://github.com/InteractiveComputerGraphics/SPlisHSPlasH   and the example for generating the dataset is using custom_scene.py  https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/8454e9f454fef20771dfe98d318904da62764b4c/pySPlisHSPlasH/examples/custom_scene.py#L4

MPM data generate library:  https://github.com/yuanming-hu/taichi_mpm   The example for generating MPM dataset is mpm3d.py which has a ti.tools.PLYWriter function https://github.com/taichi-dev/test_actions/blob/e5ed25678acfbe3eff49f4ac05345b183876890f/python/taichi/examples/simulation/mpm3d.py

The original dataset from the GNS paper can be downloaded at: https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/water/train.tfrecord or use bash referring to GNS page
