import numpy as np


class SimulationConfig:
    def __init__(self, **simulation_config):
        self.folder = None
        self.frames_folder = None

        self.dt = simulation_config['dt']

        self.c = simulation_config['c']

        self.dz = simulation_config['dz']
        self.dx = simulation_config['dx']

        self.grid_size_z = simulation_config['grid_size_z']
        self.grid_size_x = simulation_config['grid_size_x']
        self.grid_size_shape = (self.grid_size_z, self.grid_size_x)

        self.total_time = simulation_config['total_time']

        print(f'CFL = {np.amax(self.c) * self.dt * ((1 / self.dz) + (1 / self.dx))}')
        print(f'Grid Size (px): ({self.grid_size_z}, {self.grid_size_x})')

        # Pressure fields
        self.p_future = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.p_present = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.p_past = np.zeros(self.grid_size_shape, dtype=np.float32)

        # Partial derivatives
        self.dp_1_z = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.dp_1_x = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.dp_2_z = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.dp_2_x = np.zeros(self.grid_size_shape, dtype=np.float32)
        

        """ CPML """
        self.absorption_layer_size = np.int32(25)
        self.damping_coefficient = np.float32(6e8)

        x, z = np.meshgrid(np.arange(self.grid_size_x, dtype=np.float32), np.arange(self.grid_size_z, dtype=np.float32))

        self.is_z_absorption = (z < self.absorption_layer_size) | (z >= self.grid_size_z - self.absorption_layer_size)
        self.is_x_absorption = (x < self.absorption_layer_size) | (x >= self.grid_size_x - self.absorption_layer_size)

        self.absorption_coefficient = np.exp(
            -(self.damping_coefficient * (np.arange(self.absorption_layer_size) / self.absorption_layer_size) ** 2) * self.dt
        ).astype(np.float32)

        self.psi_z = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.psi_x = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.phi_z = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.phi_x = np.zeros(self.grid_size_shape, dtype=np.float32)

        self.absorption_z = np.ones(self.grid_size_shape, dtype=np.float32)
        self.absorption_x = np.ones(self.grid_size_shape, dtype=np.float32)

        self.absorption_z[:self.absorption_layer_size, :] = self.absorption_coefficient[:, np.newaxis][::-1]  # z < layer_size
        self.absorption_z[-self.absorption_layer_size:, :] = self.absorption_coefficient[:, np.newaxis]  # z > (size_z - layer_size)
        self.absorption_x[:, :self.absorption_layer_size] = self.absorption_coefficient[::-1]  # x < layer_size
        self.absorption_x[:, -self.absorption_layer_size:] = self.absorption_coefficient  # x > (size_x - layer_size)

        # Converts boolean array to int array to pass to GPU
        self.is_z_absorption_int = self.is_z_absorption.astype(np.int32)
        self.is_x_absorption_int = self.is_x_absorption.astype(np.int32)
