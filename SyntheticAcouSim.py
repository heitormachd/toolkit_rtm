import numpy as np
import os
from SimulationConfig import SimulationConfig
from WebGpuHandler import WebGpuHandler
from functions import save_image, create_video, load_sources
import matplotlib.pyplot as plt


class SyntheticAcouSim(SimulationConfig):
    def __init__(self, **simulation_config):
        super().__init__(**simulation_config)

        # Create folders
        self.folder = './SyntheticAcouSim'
        self.frames_folder = f'{self.folder}/frames'
        os.makedirs(self.frames_folder, exist_ok=True)

        self.microphone_z = simulation_config['microphone_z']
        self.microphone_x = simulation_config['microphone_x']
        self.microphones_amount = simulation_config['microphones_amount']

        self.source_z = np.atleast_1d(simulation_config['source_z']).astype(np.int32)
        self.source_x = np.atleast_1d(simulation_config['source_x']).astype(np.int32)
        self.source_ids = np.atleast_1d(
            simulation_config.get('source_ids', simulation_config.get('source_id', 0))
        ).astype(np.int32)
        self.sources_amount = np.int32(len(self.source_z))
        if len(self.source_x) != self.sources_amount or len(self.source_ids) != self.sources_amount:
            raise ValueError('source_z, source_x, and source_ids must have the same length.')

        self.reflector_z, self.reflector_x = np.int32(np.where(self.c == 0))
        self.reflectors_amount = np.int32(len(self.reflector_z))

        self.microphones_recording = np.array([[0 for _ in range(self.total_time)] for _ in range(self.microphones_amount)], dtype=np.float32)

        # Source
        self.source = load_sources(self.source_ids, self.total_time)
        self.source_time = np.int32(self.source.shape[1])
        self.source_zx = np.ascontiguousarray(np.concatenate((self.source_z, self.source_x)).astype(np.int32))

        # WebGPU buffer
        self.info_i32 = np.array(
            [
                self.grid_size_z,
                self.grid_size_x,
                self.sources_amount,
                self.source_time,
                0,
            ],
            dtype=np.int32
        )

        # WebGPU buffer
        self.info_f32 = np.array(
            [
                self.dz,
                self.dx,
                self.dt,
            ],
            dtype=np.float32
        )

        self.wgpu_handler = None
        self.setup_gpu()

    def setup_gpu(self):
        self.wgpu_handler = WebGpuHandler(shader_file='./synthetic_acou_sim.wgsl', wsz=self.grid_size_z, wsx=self.grid_size_x)

        self.wgpu_handler.create_shader_module()

        # Data passed to gpu buffers
        wgsl_data = {
            'infoI32': self.info_i32,
            'infoF32': self.info_f32,
            'c': self.c,
            'source': self.source,
            'p_future': self.p_future,
            'p_present': self.p_present,
            'p_past': self.p_past,
            'dp_1_z': self.dp_1_z,
            'dp_1_x': self.dp_1_x,
            'dp_2_z': self.dp_2_z,
            'dp_2_x': self.dp_2_x,
            'psi_z': self.psi_z,
            'psi_x': self.psi_x,
            'phi_z': self.phi_z,
            'phi_x': self.phi_x,
            'absorption_z': self.absorption_z,
            'absorption_x': self.absorption_x,
            'is_z_absorption': self.is_z_absorption_int,
            'is_x_absorption': self.is_x_absorption_int,
            'source_zx': self.source_zx,
        }

        self.wgpu_handler.create_buffers(wgsl_data)

    def run(self, generate_video: bool, animation_step: int):
        if generate_video:
            for frame_name in os.listdir(self.frames_folder):
                if frame_name.startswith('frame_') and frame_name.endswith('.png'):
                    os.remove(os.path.join(self.frames_folder, frame_name))

        compute_forward_diff = self.wgpu_handler.create_compute_pipeline("forward_diff")
        compute_after_forward = self.wgpu_handler.create_compute_pipeline("after_forward")
        compute_backward_diff = self.wgpu_handler.create_compute_pipeline("backward_diff")
        compute_after_backward = self.wgpu_handler.create_compute_pipeline("after_backward")
        compute_sim = self.wgpu_handler.create_compute_pipeline("sim")
        compute_incr_time = self.wgpu_handler.create_compute_pipeline("incr_time")

        for i in range(self.total_time):
            command_encoder = self.wgpu_handler.device.create_command_encoder()
            compute_pass = command_encoder.begin_compute_pass()

            for index, bind_group in enumerate(self.wgpu_handler.bind_groups):
                compute_pass.set_bind_group(index, bind_group, [], 0, 999999)

            compute_pass.set_pipeline(compute_forward_diff)
            compute_pass.dispatch_workgroups(self.grid_size_z // self.wgpu_handler.ws[0],
                                             self.grid_size_x // self.wgpu_handler.ws[1])

            compute_pass.set_pipeline(compute_after_forward)
            compute_pass.dispatch_workgroups(self.grid_size_z // self.wgpu_handler.ws[0],
                                             self.grid_size_x // self.wgpu_handler.ws[1])

            compute_pass.set_pipeline(compute_backward_diff)
            compute_pass.dispatch_workgroups(self.grid_size_z // self.wgpu_handler.ws[0],
                                             self.grid_size_x // self.wgpu_handler.ws[1])

            compute_pass.set_pipeline(compute_after_backward)
            compute_pass.dispatch_workgroups(self.grid_size_z // self.wgpu_handler.ws[0],
                                             self.grid_size_x // self.wgpu_handler.ws[1])

            compute_pass.set_pipeline(compute_sim)
            compute_pass.dispatch_workgroups(self.grid_size_z // self.wgpu_handler.ws[0],
                                             self.grid_size_x // self.wgpu_handler.ws[1])

            compute_pass.set_pipeline(compute_incr_time)
            compute_pass.dispatch_workgroups(1)

            compute_pass.end()
            self.wgpu_handler.device.queue.submit([command_encoder.finish()])

            """ READ BUFFERS """
            self.p_future = (np.asarray(self.wgpu_handler.device.queue.read_buffer(self.wgpu_handler.buffers['b4']).cast("f"))
                             .reshape(self.grid_size_shape))

            self.microphones_recording[:, i] = self.p_future[self.microphone_z[:], self.microphone_x[:]]

            if generate_video and i % animation_step == 0:
                plt.figure()
                plt.imshow(self.p_future, cmap='bwr')
                plt.colorbar()
                plt.scatter(self.microphone_x, self.microphone_z, s=1, color='purple')
                plt.scatter(self.source_x, self.source_z, s=20, color='yellow', edgecolors='black')
                plt.scatter(self.reflector_x, self.reflector_z, s=0.05, color='green')
                plt.grid(True)
                plt.title(f'Synthetic Acoustic Sim - {self.sources_amount} sources - {i}')
                plt.savefig(f'{self.frames_folder}/frame_{i // animation_step}.png')
                plt.close()

            if i % 300 == 0:
                print(f'Synthetic Acoustic Simulation - i={i}')

        print('Synthetic Acoustic Simulation finished.')

        np.save(f'{self.folder}/microphones_recording.npy', self.microphones_recording)

        if generate_video:
            create_video(path=self.frames_folder, output_path=f'{self.folder}/acou_sim.mp4')
