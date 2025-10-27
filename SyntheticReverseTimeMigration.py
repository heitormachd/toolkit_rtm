import os
import numpy as np
from SimulationConfig import SimulationConfig
from WebGpuHandler import WebGpuHandler
from functions import save_rtm_image, create_video
import matplotlib.pyplot as plt


class SyntheticReverseTimeMigration(SimulationConfig):
    def __init__(self, **simulation_config):
        super().__init__(**simulation_config)

        self.reflector_z, self.reflector_x = np.int32(np.where(self.c == 0))
        self.reflectors_amount = np.int32(len(self.reflector_z))

        self.c = self.c.copy()

        self.c[self.c == np.float32(0)] = simulation_config['medium_c']

        self.emitter_index = simulation_config['emitter_index']

        # Create folders
        self.folder = './SyntheticRTM'
        self.frames_folder = f'{self.folder}/frames'
        os.makedirs(self.frames_folder, exist_ok=True)
        self.tr_folder = './SyntheticTR'

        # Source
        self.source = np.load('./source.npy').astype(np.float32)
        if len(self.source) < self.total_time:
            self.source = np.pad(self.source, (0, self.total_time - len(self.source)), 'constant').astype(np.float32)
        elif len(self.source) > self.total_time:
            self.source = self.source[:self.total_time]

        # Source's position
        self.source_z = simulation_config['source_z']
        self.source_x = simulation_config['source_x']

        # Up-going pressure fields (Flipped Time Reversal)
        self.p_future_flipped_tr = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.p_present_flipped_tr = np.load(f'{self.tr_folder}/second_to_last_frame.npy')
        self.p_past_flipped_tr = np.load(f'{self.tr_folder}/last_frame.npy')

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

        self.v_z_present = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.v_x_present = np.zeros(self.grid_size_shape, dtype=np.float32)

        self.v_z_present_flipped_tr = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.v_x_present_flipped_tr = np.zeros(self.grid_size_shape, dtype=np.float32)

        self.rtm_poynting_image = np.zeros(self.grid_size_shape, dtype=np.float32)

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
        self.wgpu_handler = WebGpuHandler(shader_file='./reverse_time_migration.wgsl', wsz=self.grid_size_z, wsx=self.grid_size_x)

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
            'v_z_present': self.v_z_present,
            'v_x_present': self.v_x_present,
            'v_z_present_flipped_tr': self.v_z_present_flipped_tr,
            'v_x_present_flipped_tr': self.v_x_present_flipped_tr,
            'rtm_poynting_image': self.rtm_poynting_image,
        }

        self.wgpu_handler.create_buffers(wgsl_data)

    def run(self, generate_video: bool, animation_step: int):
        compute_forward_diff = self.wgpu_handler.create_compute_pipeline("forward_diff")
        compute_after_forward = self.wgpu_handler.create_compute_pipeline("after_forward")
        compute_backward_diff = self.wgpu_handler.create_compute_pipeline("backward_diff")
        compute_after_backward = self.wgpu_handler.create_compute_pipeline("after_backward")
        compute_sim_flipped_tr = self.wgpu_handler.create_compute_pipeline("sim_flipped_tr")
        compute_sim = self.wgpu_handler.create_compute_pipeline("sim")
        compute_update_velocity = self.wgpu_handler.create_compute_pipeline("update_velocity")
        compute_update_rtm_image = self.wgpu_handler.create_compute_pipeline("update_rtm_image")
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
            
            compute_pass.set_pipeline(compute_update_velocity)
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
            
            compute_pass.set_pipeline(compute_update_rtm_image)
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
            accumulated_product_poynting = (np.asarray(self.wgpu_handler.device.queue.read_buffer(self.wgpu_handler.buffers['b38']).cast("f"))
                                 .reshape(self.grid_size_shape))
            
            current_product = self.p_future * self.p_future_flipped_tr
            accumulated_product += current_product

            L = self.absorption_layer_size
            roi_slice = (slice(None, -L), slice(L, -L))

            in_roi = (self.reflector_x >= L) & (self.reflector_x < (self.grid_size_x - L)) & \
                         (self.reflector_z < (self.grid_size_z - L))
            
            roi_reflector_x = self.reflector_x[in_roi] - L 
            roi_reflector_z = self.reflector_z[in_roi]

            
            if generate_video and i % animation_step == 0:
                fig, axs = plt.subplots(2, 2, figsize=(10, 10))

                axs[0, 0].imshow(self.p_future_flipped_tr, cmap='viridis', interpolation='none')
                axs[0, 0].set_title('Up-Going')
                axs[1, 0].imshow(self.p_future, cmap='viridis', interpolation='none')
                axs[1, 0].set_title('Down-Going')

                axs[0, 1].imshow(accumulated_product[roi_slice], cmap='viridis', interpolation='none')
                axs[0, 1].set_title('Accumulated Standard Product')

                axs[1, 1].imshow(accumulated_product_poynting[roi_slice], cmap='viridis', interpolation='none')
                axs[1, 1].set_title('Accumulated Poynting Product')

                axs[1, 1].scatter(roi_reflector_x, roi_reflector_z, s=0.05, color='red')

                # axs[1, 1].scatter(self.reflector_x, self.reflector_z, s=0.05, color='red')
                plt.savefig(f'{self.frames_folder}/frame_{i // animation_step}.png', bbox_inches='tight', pad_inches=0)
                plt.close()

                # save_rtm_image(
                #     upper_left=self.p_future_flipped_tr,
                #     upper_right=current_product,
                #     bottom_left=self.p_future,
                #     bottom_right=accumulated_product,
                #     path=f'{self.frames_folder}/frame_{i // animation_step}.png'
                # )

            if i % 300 == 0:
                print(f'Reverse Time Migration - i={i}')

        print('Reverse Time Migration finished.')

        # Save last frame of accumulated_product
        np.save(f'{self.folder}/accumulated_product_{self.emitter_index}.npy', accumulated_product[roi_slice])
        np.save(f'{self.folder}/accumulated_product_poynting_{self.emitter_index}.npy', accumulated_product_poynting[roi_slice])

        if generate_video:
            create_video(path=self.frames_folder, output_path=f'{self.folder}/rtm.mp4')
