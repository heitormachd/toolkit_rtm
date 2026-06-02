import numpy as np
import os
from SimulationConfig import SimulationConfig
from WebGpuHandler import WebGpuHandler
from functions import save_image, create_video, load_sources
import matplotlib.pyplot as plt


class SyntheticTimeReversal(SimulationConfig):
    def __init__(self, **simulation_config):
        super().__init__(**simulation_config)

        self.reflector_z, self.reflector_x = np.int32(np.where(self.c == 0))
        self.reflectors_amount = np.int32(len(self.reflector_z))

        self.c = self.c.copy()

        self.c[self.c == np.float32(0)] = simulation_config['medium_c']

        # Create folders
        self.folder = './SyntheticTR'
        self.frames_folder = f'{self.folder}/frames'
        os.makedirs(self.frames_folder, exist_ok=True)
        self.acou_sim_folder = './SyntheticAcouSim'

        self.bscan = np.load(f'{self.acou_sim_folder}/microphones_recording.npy')
        self.recorded_time = np.int32(self.bscan.shape[1])
        recorded_time = int(self.recorded_time)
        tr_total_time = int(self.total_time)

        # print(f'bscan shape: {self.bscan.shape}')

        # plt.figure()
        # plt.plot(self.bscan[0,:])
        # plt.plot(self.bscan[-1, :])
        # plt.show()

        self.bscan[:, :200] = np.float32(0)
        
        # plt.figure()
        # plt.plot(self.bscan[0,:])
        # plt.plot(self.bscan[-1, :])
        # plt.show()
        
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

        sources = load_sources(self.source_ids, recorded_time)
        source_index = np.any(~np.isclose(sources, 0), axis=0)
        # Cut the recorded source
        self.bscan[:, source_index] = np.float32(0)

        # Flip bscan
        self.flipped_bscan = self.bscan[:, ::-1].astype(np.float32)
        if tr_total_time > recorded_time:
            extra_samples = tr_total_time - recorded_time
            self.flipped_bscan = np.pad(self.flipped_bscan, ((0, 0), (0, extra_samples)), 'constant').astype(np.float32)
        elif tr_total_time < recorded_time:
            self.flipped_bscan = self.flipped_bscan[:, :tr_total_time].astype(np.float32)

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

        max_abs_pressure = np.zeros(self.grid_size_shape, dtype=np.float32)

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
                plt.imshow(self.p_future, cmap='bwr')
                plt.colorbar()
                plt.scatter(self.microphone_x, self.microphone_z, s=0.05, color='purple')
                plt.scatter(self.source_x, self.source_z, s=20, color='yellow', edgecolors='black')
                plt.scatter(self.reflector_x, self.reflector_z, s=0.05, color='green')
                plt.grid(True)
                plt.title(f'Time Reversal - {self.sources_amount} sources - {i}')
                plt.savefig(f'{self.frames_folder}/frame_{i // animation_step}.png')
                plt.close()

                # save_image(self.p_future, f'{self.frames_folder}/frame_{i // animation_step}.png')

            max_abs_pressure = np.maximum(max_abs_pressure, np.abs(self.p_future))

            # Save last 2 frames (for RTM)
            if i == self.total_time - 2:
                np.save(f'{self.folder}/second_to_last_frame.npy', self.p_future)
            if i == self.total_time - 1:
                np.save(f'{self.folder}/last_frame.npy', self.p_future)

            if i % 300 == 0:
                print(f'Time Reversal - i={i}')

        print('Time Reversal finished.')

        # Save peak absolute pressure over all time steps.
        np.save(f'{self.folder}/max_abs_pressure.npy', max_abs_pressure)

        if generate_video:
            create_video(path=self.frames_folder, output_path=f'{self.folder}/tr.mp4')
