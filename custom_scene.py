# import pysplishsplash as sph
# import pysplishsplash.Utilities.SceneLoaderStructs as Scenes
#
#
# def main():
#     # Set up the simulator
#     base = sph.Exec.SimulatorBase()
#     base.init(useGui=True,  sceneFile=sph.Extras.Scenes.Empty)
#
#     # Create an imgui simulator
#     gui = sph.GUI.Simulator_GUI_imgui(base)
#     base.setGui(gui)
#
#     # Get the scene and add objects
#     scene = sph.Exec.SceneConfiguration.getCurrent().getScene()
#     scene.boundaryModels.append(Scenes.BoundaryData(meshFile="../models/UnitBox.obj", translation=[0., 3.0, 0.], scale=[4., 6., 4.], color=[0.1, 0.4, 0.5, 1.0], isWall=True, mapInvert=True, mapResolution=[25, 25, 25]))
#     scene.fluidBlocks.append(Scenes.FluidBlock(id='Fluid', boxMin = [-1.5, 0.0, -1.5], boxMax = [-0.5, 2.0, -0.5], mode=0, initialVelocity=[0.0, 0.0, 0.0]))
#     scene.fluidBlocks.append(Scenes.FluidBlock(id='Fluid', boxMin = [0.5, 0.0, 0.5], boxMax = [1.5, 2.0, 1.5], mode=0, initialVelocity=[0.0, 0.0, 0.0]))
#
#     # Run the GUI
#     base.run()
#
#
# if __name__ == "__main__":
#     main()

import pysplishsplash as sph
import pysplishsplash.Utilities.SceneLoaderStructs as Scenes

def main():
    # Set up the simulator
    base = sph.Exec.SimulatorBase()
    base.init(useGui=True,  sceneFile=sph.Extras.Scenes.Empty)
    sim = sph.Simulation.getCurrent()
    # Create an imgui simulator
    gui = sph.GUI.Simulator_GUI_imgui(base)
    base.setGui(gui)

    base.setVec3ValueReal(base.CAMERA_POSITION, [10, 5, 5])
    base.setVec3ValueReal(base.CAMERA_LOOKAT, [0, 0, 0])

    # Get the scene and add objects
    scene = sph.Exec.SceneConfiguration.getCurrent().getScene()
    scene.particleRadius = 0.025
    # scene.sim2D = True
    scene.boundaryModels.append(Scenes.BoundaryData(meshFile="../models/UnitBox.obj", translation=[0, 2.0, 0], scale=[4, 4, 4.],
                                                    color=[0.1, 0.4, 0.5, 1.0], isWall=True, mapInvert=True, mapResolution=[5, 5, 5]))
    scene.fluidBlocks.append(Scenes.FluidBlock(id='Fluid', boxMin = [-0.5, 0, -0.5], boxMax = [0.5, 1.0, 0.5], mode=0, initialVelocity=[0.0, 0.0, 0.0]))
    # scene.fluidBlocks.append(Scenes.FluidBlock(id='Fluid', boxMin = [0.5, 0.0, 0.5], boxMax = [1.5, 2.0, 1.5], mode=0, initialVelocity=[0.0, 0.0, 0.0]))

    base.initSimulation()

    fluid = sim.getFluidModel(0)
    pos = []
    for i in range(6859):
        pos.append(fluid.getPosition(i))
    print(pos)

    fluid.setValueFloat(fluid.DENSITY0, 1000.0)

    visco = fluid.getViscosityBase()
    visco.setValueFloat(visco.VISCOSITY_COEFFICIENT, 0.001)


    base.runSimulation()
    base.cleanup()

    # Run the GUI
    # base.run()




if __name__ == "__main__":
    main()


