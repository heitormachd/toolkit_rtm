import os
import numpy as np
from .InputTest import InputTest
from .SimulationConfig import SimulationConfig
from .WebGpuHandler import WebGpuHandler
from .functions import save_rtm_image, create_video
from .paths import REAL_RTM_OUTPUT_DIR, REAL_TIME_REVERSAL_OUTPUT_DIR, SHADERS_DIR, SOURCES_DIR


class ReverseTimeMigration(SimulationConfig):
    def __init__(self, **simulation_config):
        super().__init__(**simulation_config)

        # Create folders
        self.folder = REAL_RTM_OUTPUT_DIR
        self.frames_folder = self.folder / 'frames'
        self.frames_folder.mkdir(parents=True, exist_ok=True)
        self.tr_folder = REAL_TIME_REVERSAL_OUTPUT_DIR

        input_test: InputTest = simulation_config['input_test']
        self.emitter_index = input_test.fmc_emitter
        self.total_time = input_test.total_time
        print(f'Total time: {self.total_time}')

        # Source
        self.source = np.load(SOURCES_DIR / 'source.npy').astype(np.float32)
        if len(self.source) < self.total_time:
            self.source = np.pad(self.source, (0, self.total_time - len(self.source)), 'constant').astype(np.float32)
        elif len(self.source) > self.total_time:
            self.source = self.source[:self.total_time]

        # Source's position
        self.source_z = np.int32(np.load(self.tr_folder / 'emitter_z.npy'))
        self.source_x = np.int32(np.load(self.tr_folder / 'emitter_x.npy'))

        # Up-going pressure fields (Flipped Time Reversal)
        self.p_future_flipped_tr = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.p_present_flipped_tr = np.load(self.tr_folder / 'second_to_last_frame.npy')
        self.p_past_flipped_tr = np.load(self.tr_folder / 'last_frame.npy')

        # Partial derivatives (Flipped Time Reversal)
        self.dp_1_z_flipped_tr = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.dp_1_x_flipped_tr = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.dp_2_z_flipped_tr = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.dp_2_x_flipped_tr = np.zeros(self.grid_size_shape, dtype=np.float32)

        # CPML (Flipped Time Reversal)
        self.psi_z_flipped_tr = self.psi_z.copy()
        self.psi_x_flipped_tr = self.psi_x.copy()
        self.phi_z_flipped_tr = self.phi_z.copy()
        self.phi_x_flipped_tr = self.phi_x.copy()
        self.absorption_z_flipped_tr = self.absorption_z.copy()
        self.absorption_x_flipped_tr = self.absorption_x.copy()
        self.is_z_absorption_int_flipped_tr = self.is_z_absorption_int.copy()
        self.is_x_absorption_int_flipped_tr = self.is_x_absorption_int.copy()

        # WebGPU buffer
        self.info_i32 = np.array(
            [
                self.grid_size_z,
                self.grid_size_x,
                self.source_z,
                self.source_x,
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
        self.wgpu_handler = WebGpuHandler(shader_file=SHADERS_DIR / 'reverse_time_migration.wgsl', wsz=self.grid_size_z, wsx=self.grid_size_x)

        self.wgpu_handler.create_shader_module()

        # Data passed to gpu buffers
        wgsl_data = {
            'infoI32': self.info_i32,
            'infoF32': self.info_f32,
            'source': self.source,
            'c': self.c,
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
            'p_future_flipped_tr': self.p_future_flipped_tr,
            'p_present_flipped_tr': self.p_present_flipped_tr,
            'p_past_flipped_tr': self.p_past_flipped_tr,
            'dp_1_z_flipped_tr': self.dp_1_z_flipped_tr,
            'dp_1_x_flipped_tr': self.dp_1_x_flipped_tr,
            'dp_2_z_flipped_tr': self.dp_2_z_flipped_tr,
            'dp_2_x_flipped_tr': self.dp_2_x_flipped_tr,
            'psi_z_flipped_tr': self.psi_z_flipped_tr,
            'psi_x_flipped_tr': self.psi_x_flipped_tr,
            'phi_z_flipped_tr': self.phi_z_flipped_tr,
            'phi_x_flipped_tr': self.phi_x_flipped_tr,
            'absorption_z_flipped_tr': self.absorption_z_flipped_tr,
            'absorption_x_flipped_tr': self.absorption_x_flipped_tr,
            'is_z_absorption_flipped_tr': self.is_z_absorption_int_flipped_tr,
            'is_x_absorption_flipped_tr': self.is_x_absorption_int_flipped_tr,
        }

        self.wgpu_handler.create_buffers(wgsl_data)

    def run(self, generate_video: bool, animation_step: int):
        compute_forward_diff = self.wgpu_handler.create_compute_pipeline("forward_diff")
        compute_after_forward = self.wgpu_handler.create_compute_pipeline("after_forward")
        compute_backward_diff = self.wgpu_handler.create_compute_pipeline("backward_diff")
        compute_after_backward = self.wgpu_handler.create_compute_pipeline("after_backward")
        compute_sim_flipped_tr = self.wgpu_handler.create_compute_pipeline("sim_flipped_tr")
        compute_sim = self.wgpu_handler.create_compute_pipeline("sim")
        compute_incr_time = self.wgpu_handler.create_compute_pipeline("incr_time")

        accumulated_product = np.zeros(self.grid_size_shape, dtype=np.float32)

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

            compute_pass.set_pipeline(compute_sim_flipped_tr)
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
            self.p_future_flipped_tr = (np.asarray(self.wgpu_handler.device.queue.read_buffer(self.wgpu_handler.buffers['b19']).cast("f"))
                             .reshape(self.grid_size_shape))

            current_product = self.p_future * self.p_future_flipped_tr
            accumulated_product += current_product

            if generate_video and i % animation_step == 0:
                save_rtm_image(
                    upper_left=self.p_future_flipped_tr,
                    upper_right=current_product,
                    bottom_left=self.p_future,
                    bottom_right=accumulated_product,
                    path=self.frames_folder / f'frame_{i // animation_step}.png'
                )

            if i % 300 == 0:
                print(f'Reverse Time Migration - i={i}')

        print('Reverse Time Migration finished.')

        # Save last frame of accumulated_product
        np.save(self.folder / f'accumulated_product_{self.emitter_index}.npy', accumulated_product)

        if generate_video:
            create_video(path=self.frames_folder, output_path=self.folder / 'rtm.mp4')
