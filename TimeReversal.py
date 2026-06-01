import numpy as np
import os
from InputTest import InputTest
from SimulationConfig import SimulationConfig
from WebGpuHandler import WebGpuHandler
from functions import save_image, create_video
import matplotlib.pyplot as plt


class TimeReversal(SimulationConfig):
    def __init__(self, **simulation_config):
        super().__init__(**simulation_config)

        # Create folders
        self.folder = './TimeReversal'
        self.frames_folder = f'{self.folder}/frames'
        os.makedirs(self.frames_folder, exist_ok=True)

        input_test: InputTest = simulation_config['input_test']
        self.bscan = input_test.bscan
        self.microphones_distance = input_test.microphones_distance
        self.microphones_amount = input_test.microphones_amount
        self.total_time = input_test.total_time
        print(f'Total time: {self.total_time}')

        # Microphones' position
        self.microphone_x = []
        for rp in range(self.microphones_amount):
            self.microphone_x.append((self.microphones_distance * rp) / self.dx)
        self.microphone_x = (np.int32(np.asarray(self.microphone_x))
                           + np.int32((self.grid_size_x - self.microphone_x[-1]) / 2))
        self.microphone_z = np.full(self.microphones_amount, 1, dtype=np.int32)  # Não colocar microfones no índice 0.

        # Save emitter's position to use as source position in Reverse Time Migration
        np.save(f'{self.folder}/emitter_z.npy', self.microphone_z[input_test.fmc_emitter])
        np.save(f'{self.folder}/emitter_x.npy', self.microphone_x[input_test.fmc_emitter])

        # Flip bscan
        self.flipped_bscan = self.bscan[:, ::-1].astype(np.float32)

        # WebGPU buffer
        self.info_i32 = np.array(
            [
                self.grid_size_z,
                self.grid_size_x,
                self.microphones_amount,
                0,
                self.flipped_bscan.shape[1],
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
        self.wgpu_handler = WebGpuHandler(shader_file='./time_reversal.wgsl', wsz=self.grid_size_z, wsx=self.grid_size_x)

        self.wgpu_handler.create_shader_module()

        # Data passed to gpu buffers
        wgsl_data = {
            'infoI32': self.info_i32,
            'infoF32': self.info_f32,
            'c': self.c,
            'microphone_zx': np.ascontiguousarray(np.concatenate((self.microphone_z, self.microphone_x)).astype(np.int32)),
            'p_future': self.p_future,
            'p_present': self.p_present,
            'p_past': self.p_past,
            'dp_1': np.ascontiguousarray(np.concatenate((self.dp_1_z.reshape(-1), self.dp_1_x.reshape(-1)))),
            'dp_2': np.ascontiguousarray(np.concatenate((self.dp_2_z.reshape(-1), self.dp_2_x.reshape(-1)))),
            'psi': np.ascontiguousarray(np.concatenate((self.psi_z.reshape(-1), self.psi_x.reshape(-1)))),
            'phi': np.ascontiguousarray(np.concatenate((self.phi_z.reshape(-1), self.phi_x.reshape(-1)))),
            'absorption': np.ascontiguousarray(np.concatenate((self.absorption_z.reshape(-1), self.absorption_x.reshape(-1)))),
            'is_absorption': np.ascontiguousarray(np.concatenate((self.is_z_absorption_int.reshape(-1), self.is_x_absorption_int.reshape(-1)))),
            'flipped_bscan': np.ascontiguousarray(self.flipped_bscan.reshape(-1)),
        }

        self.wgpu_handler.create_buffers(wgsl_data)

    def run(self, generate_video: bool, animation_step: int):
        compute_forward_diff = self.wgpu_handler.create_compute_pipeline("forward_diff")
        compute_after_forward = self.wgpu_handler.create_compute_pipeline("after_forward")
        compute_backward_diff = self.wgpu_handler.create_compute_pipeline("backward_diff")
        compute_after_backward = self.wgpu_handler.create_compute_pipeline("after_backward")
        compute_sim = self.wgpu_handler.create_compute_pipeline("sim")
        compute_incr_time = self.wgpu_handler.create_compute_pipeline("incr_time")

        l2_norm = np.zeros(self.grid_size_shape, dtype=np.float32)

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
            self.p_future = (np.asarray(self.wgpu_handler.device.queue.read_buffer(self.wgpu_handler.buffers['b5']).cast("f"))
                             .reshape(self.grid_size_shape))

            if generate_video and i % animation_step == 0:
                plt.figure()
                plt.imshow(self.p_future, cmap='viridis', vmax=0.55, vmin=-0.55)
                plt.colorbar()
                plt.scatter(self.microphone_x, self.microphone_z, s=0.05, color='purple')
                plt.grid(True)
                plt.title(f'Time Reversal - {i}')
                plt.savefig(f'{self.frames_folder}/frame_{i // animation_step}.png')
                plt.close()

            l2_norm += np.square(self.p_future)

            # Save last 2 frames (for RTM)
            if i == self.total_time - 2:
                np.save(f'{self.folder}/second_to_last_frame.npy', self.p_future)
            if i == self.total_time - 1:
                np.save(f'{self.folder}/last_frame.npy', self.p_future)

            if i % 300 == 0:
                print(f'Time Reversal - i={i}')

        print('Time Reversal finished.')

        # Save L2-Norm
        np.save(f'{self.folder}/l2_norm.npy', np.sqrt(l2_norm))

        if generate_video:
            create_video(path=self.frames_folder, output_path=f'{self.folder}/tr.mp4')
