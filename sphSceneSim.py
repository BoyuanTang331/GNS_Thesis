"""
this file is use to automatically generate sph fluid simulation and write it as a class for self convert
the waterdrop size and position will change randomly in each trajectory
TODO: find a better way to describe init conditions, set the velocity from (-2.2) and position according abcdef pos
the boundaries should be loaded on meshFile, define a cube scene
code should be running in the dictionary of SPlisHSPlasH because of
some support file such as boundary UnitBox.obj
https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/tree/master/pySPlisHSPlasH/examples

"""

import pysplishsplash as sph
import pysplishsplash.Utilities.SceneLoaderStructs as Scenes
import numpy as np
import random

class SphSceneSim:
    def __init__(self):
        self.particle_positions_over_time = []

    # call at each time step to get the particle positions
    def time_step_callback(self):
        sim = sph.Simulation.getCurrent()
        fluid = sim.getFluidModel(0)
        num_particles = fluid.numActiveParticles()
        current_step_positions = np.zeros((num_particles, 3))  # 初始化当前时间步的位置数组

        for i in range(num_particles):
            current_step_positions[i] = np.array(fluid.getPosition(i))  # 获取第i个粒子的位置

        # add current particle positions in array
        self.particle_positions_over_time.append(current_step_positions)



    def main(self,t):
        base = sph.Exec.SimulatorBase()
        base.init(useGui=True, sceneFile=sph.Extras.Scenes.Empty)
        sim = sph.Simulation.getCurrent()

        # Create an imgui simulator
        gui = sph.GUI.Simulator_GUI_imgui(base)
        base.setGui(gui)
        #base.setValueBool(base.PAUSE, False)
        # base.setVec3ValueReal(base.CAMERA_POSITION, [5, 5, 5])
        # base.setVec3ValueReal(base.CAMERA_LOOKAT, [0, 0, 0])

        base.setValueFloat(base.STOP_AT, 1.0000)
        base.setValueUInt(base.NUM_STEPS_PER_RENDER, 4)
        tm = sph.TimeManager.getCurrent()
        tm.setValueFloat(tm.TIME_STEP_SIZE, 0.0025)

        # Get the scene and add objects
        scene = sph.Exec.SceneConfiguration.getCurrent().getScene()
        scene.boundaryModels.append(
            Scenes.BoundaryData(meshFile="../models/UnitBox.obj", translation=[0., 3.0, 0.], scale=[4., 6., 4.],
                                color=[0.1, 0.4, 0.5, 1.0], isWall=True, mapInvert=True, mapResolution=[25, 25, 25]))

        a, b, c = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(0.0, 1.00)
        d, e, f = a + min(2 - a, 0.4), b + min(2 - b, 0.4), c + min(4 - c, 0.4),
        scene.fluidBlocks.append(Scenes.FluidBlock(id='Fluid', boxMin=[a, b, c], boxMax=[d, e, f], mode=0,
                                                   initialVelocity = [random.uniform(-2, 2) for _ in range(3)]))

        base.initSimulation()
        sim.setValueFloat(sim.CFL_MIN_TIMESTEPSIZE, 0.005)
        sim.setValueFloat(sim.CFL_MAX_TIMESTEPSIZE, 0.005) # default timestep


        base.setTimeStepCB(self.time_step_callback)
        base.runSimulation()
        base.cleanup()

        np.save(f'sph_{t}.npy', np.array(self.particle_positions_over_time))
        self.particle_positions_over_time.clear()

def run(t):
    simulator = SphSceneSim()
    simulator.main(t)