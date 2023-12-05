"""
write the mpm code as a class so that in convert_npy_to_tfrecord.py will automatically generate the simulation results,
this file is use to generate mpm2d/3d in MLS-MPM approach, the results will be saved in numpy arrays

"""

import numpy as np
import taichi as ti
import random

ti.init(arch=ti.cpu)


@ti.data_oriented
class Fluid():
    def __init__(self):
        self.dim, self.n_grid, self.dt = 2, 128, 2.5e-4

        self.dx = 1 / self.n_grid
        self.p_rho = 1
        self.p_vol = (self.dx * 0.5) ** 2
        self.p_mass = self.p_vol * self.p_rho
        self.gravity = 9.8
        self.bound = 6.4 # boundary = 6.4//128=0.05
        self.E = 400

        self.F_grid_v = ti.Vector.field(self.dim, float, (self.n_grid,) * self.dim)
        self.F_grid_m = ti.field(float, (self.n_grid,) * self.dim)

        self.neighbour = (3,) * self.dim


        self.init_width = [random.uniform(0.1, 0.5), random.uniform(0.1, 0.5)]

    def init_num_particle(self):
        self.n_particles = random.randint(1050, 1100)
        self.F_x = ti.Vector.field(self.dim, float, self.n_particles)
        self.F_v = ti.Vector.field(self.dim, float, self.n_particles)
        self.F_C = ti.Matrix.field(self.dim, self.dim, float, self.n_particles)
        self.F_J = ti.field(float, self.n_particles)


    @ti.kernel
    def substep(self):
        for I in ti.grouped(self.F_grid_m):
            self.F_grid_v[I] = ti.zero(self.F_grid_v[I])
            self.F_grid_m[I] = 0
        ti.loop_config(block_dim=self.n_grid)
        for p in self.F_x:
            Xp = self.F_x[p] / self.dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            stress = -self.dt * 4 * self.E * self.p_vol * (self.F_J[p] - 1) / self.dx**2
            affine = ti.Matrix.identity(float, self.dim) * stress + self.p_mass * self.F_C[p]
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                self.F_grid_v[base + offset] += weight * (self.p_mass * self.F_v[p] + affine @ dpos)
                self.F_grid_m[base + offset] += weight * self.p_mass
        for I in ti.grouped(self.F_grid_m):
            if self.F_grid_m[I] > 0:
                self.F_grid_v[I] /= self.F_grid_m[I]
            self.F_grid_v[I][1] -= self.dt * self.gravity
            cond = (I < self.bound) & (self.F_grid_v[I] < 0) | (I > self.n_grid - self.bound) & (self.F_grid_v[I] > 0)
            self.F_grid_v[I] = ti.select(cond, 0, self.F_grid_v[I])
        ti.loop_config(block_dim=self.n_grid)
        for p in self.F_x:
            Xp = self.F_x[p] / self.dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.zero(self.F_v[p])
            new_C = ti.zero(self.F_C[p])
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                g_v = self.F_grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / self.dx**2
            self.F_v[p] = new_v
            self.F_x[p] += self.dt * self.F_v[p]
            self.F_J[p] *= 1 + self.dt * new_C.trace()
            self.F_C[p] = new_C


    @ti.kernel
    def init(self):
        velocity = [1 * (ti.random() - 0.5), 1 * (ti.random() - 1)]
        for i in range(self.n_particles):
            # TODO: Change a better description of random initial condition and velocity, otherwise the program will out of memory caused by Explicit integration
            self.F_x[i] =  [ti.random() * self.init_width[0] + 0.2, ti.random() * self.init_width[1] + 0.2]
            # self.F_x[i] = [ti.random() * 0.1 + 0.4, ti.random() * 0.2 + 0.1]
            self.F_v[i] = velocity
            self.F_J[i] = 1

    # # 3D visualization of particles in different sight angles
    def T(self, a):
        if self.dim == 2:
            return a

        phi, theta = np.radians(28), np.radians(32)

        a = a - 0.5
        x, y, z = a[:, 0], a[:, 1], a[:, 2]
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        x, z = x * cp + z * sp, z * cp - x * sp
        u, v = x, y * ct + z * st
        return np.array([u, v]).swapaxes(0, 1) + 0.5



def run(t):
    myfluid = Fluid()
    steps = 10      # number of steps in each frame, for instance dt = 2.5ms here
    timestep = 800  # simulate step each time /trajectory length, pos array dimension (800,-1,2)
    myfluid.init_num_particle()
    print(myfluid.n_particles) # check the number of particles
    myfluid.init()

    gui = ti.GUI("water", background_color=0x112F41)

    positions_array = np.zeros((timestep, myfluid.n_particles, myfluid.dim), dtype=np.float32)
    #while gui.running and not gui.get_event(gui.ESCAPE):
    for frame in range(timestep):
        for s in range(steps):
            myfluid.substep()
        pos = myfluid.F_x.to_numpy()
        positions_array[frame] = myfluid.F_x.to_numpy()

        gui.circles(myfluid.T(pos), radius=2.5, color=0x66CCFF)
        gui.show()
    np.save(f'./result/position{t+18}.npy', positions_array)

