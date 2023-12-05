# GNS_Thesis

GNS-Thesis


----Update from 12.05.2023----

data generation work is finished, the dataset can use convert_npy_to_tfrecord.py to run mpmSceneSim.py for mpm and sphSceneSim.py to automatically get the numpy array in .npy files and convert it to tfrecord format and write the global parameter in metadata.json for the model learning phase. the dataset of fluid particle position should be Serialization as [code](https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/learning_to_simulate/reading_utils.py#L22C1-L22C1)  

TO evaluate if the dataset is correct, it needs to run on [learning to simulate](https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/learning_to_simulate/train.py) and train the model, and then get the render result.

the render result for my model is saved in animation.mp4 the result is unexpected, it has some minus position which is beyond the boundary! [image](https://github.com/BoyuanTang331/GNS_Thesis/assets/117408630/67175b2e-850d-435c-86f8-1948a8cf1e1e) and the result shows unstable for rollout predict (rollout gives the model the init position and predict the whole phase/ one step is each step are predict base on the last step on dataset)

I set the batch size=4 and a [exp decay learning rate](https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/learning_to_simulate/train.py#L351) from 1e-6 to 1e-8. I am not sure the model is overfitting or underfitting, the loss has already oscillation from 1e-6 to 1e-7 at step5000 but I still train it 11000 step, because I thought the learning rate will decay and later will train more fine model and the author has a 1e-9 loss.

my loss at end of training is ![image](https://github.com/BoyuanTang331/GNS_Thesis/assets/117408630/d2ac9854-564f-40c1-8aec-872096233b1f)

my [evaluation](https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/learning_to_simulate/train.py) set mode="eval_rollout" and the result ![image](https://github.com/BoyuanTang331/GNS_Thesis/assets/117408630/730eb791-3b5c-4443-bdfa-c023ffceb6a9)

bad predict is in animation.mp4 




----------------------------------------------------------------------------------------------------------------------------------------
Original [paper](https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate) to re-implement

Dataset generation in two methods: DFSPH and MLS-MPM

SPH data generate [library](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH)    and the example for generating the dataset is using [sph_example.py](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/8454e9f454fef20771dfe98d318904da62764b4c/pySPlisHSPlasH/examples/custom_scene.py#L4)  

MPM data generate [library](https://github.com/yuanming-hu/taichi_mpm)    
The example for generating the MPM dataset is [mpm_example.py](https://github.com/taichi-dev/test_actions/blob/e5ed25678acfbe3eff49f4ac05345b183876890f/python/taichi/examples/simulation/mpm3d.py) 

my mpm2d dataset is stored in [Google Drive](https://drive.google.com/drive/folders/1-KhKdztRIIGeD8T_Dw_qaK16eFFbKI-B?usp=drive_link) because of the large size


