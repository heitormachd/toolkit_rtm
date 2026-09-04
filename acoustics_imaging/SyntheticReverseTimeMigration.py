import os
import numpy as np
from .SimulationConfig import SimulationConfig
from .WebGpuHandler import WebGpuHandler
from .functions import save_rtm_image, create_video, load_sources
from .paths import SHADERS_DIR, SOURCES_DIR, SYNTHETIC_RTM_OUTPUT_DIR, SYNTHETIC_TR_OUTPUT_DIR
import matplotlib.pyplot as plt


SOURCE_ILLUMINATION_FLOOR_FRACTION = np.float32(1e-3)


def _normalize_by_source_energy(
    image,
    source_energy,
    illumination_floor_fraction=SOURCE_ILLUMINATION_FLOOR_FRACTION,
):
    normalized = np.zeros_like(image)
    max_source_energy = np.max(source_energy)
    if max_source_energy <= 0.0:
        return normalized

    illumination_floor = np.float32(illumination_floor_fraction * max_source_energy)
    stabilized_source_energy = np.maximum(source_energy, illumination_floor)
    epsilon = np.float32(np.finfo(np.float32).eps * max_source_energy)
    np.divide(
        image,
        stabilized_source_energy + epsilon,
        out=normalized,
    )
    return normalized


class SyntheticReverseTimeMigration(SimulationConfig):
    def __init__(self, **simulation_config):
        super().__init__(**simulation_config)

        self.reflector_z, self.reflector_x = np.int32(np.where(self.c == 0))
        self.reflectors_amount = np.int32(len(self.reflector_z))

        self.c = self.c.copy()

        self.c[self.c == np.float32(0)] = simulation_config['medium_c']

        self.emitter_index = int(simulation_config.get('emitter_index', 0))

        # Create folders
        self.folder = SYNTHETIC_RTM_OUTPUT_DIR
        self.frames_folder = self.folder / 'frames'
        self.frames_folder.mkdir(parents=True, exist_ok=True)
        self.tr_folder = SYNTHETIC_TR_OUTPUT_DIR

        # Source
        self.source_ids = np.atleast_1d(
            simulation_config.get('source_ids', simulation_config.get('source_id', 0))
        ).astype(np.int32)
        self.source = load_sources(self.source_ids, self.total_time, source_dir=SOURCES_DIR)
        self.source_time = np.int32(self.source.shape[1])

        # Source's position
        self.source_z = np.atleast_1d(simulation_config['source_z']).astype(np.int32)
        self.source_x = np.atleast_1d(simulation_config['source_x']).astype(np.int32)
        self.sources_amount = np.int32(len(self.source_z))
        if len(self.source_x) != self.sources_amount or len(self.source_ids) != self.sources_amount:
            raise ValueError('source_z, source_x, and source_ids must have the same length.')
        self.source_zx = np.ascontiguousarray(np.concatenate((self.source_z, self.source_x)).astype(np.int32))

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
            'v_z_present': self.v_z_present,
            'v_x_present': self.v_x_present,
            'v_z_present_flipped_tr': self.v_z_present_flipped_tr,
            'v_x_present_flipped_tr': self.v_x_present_flipped_tr,
            'rtm_poynting_image': self.rtm_poynting_image,
            'source_zx': self.source_zx,
        }

        self.wgpu_handler.create_buffers(wgsl_data)

    def run(
        self,
        generate_video: bool,
        animation_step: int,
        use_poynting_vectors: bool = True,
        poynting_validation_steps=(),
    ):
        validation_steps = {int(step) for step in poynting_validation_steps}
        if validation_steps and not use_poynting_vectors:
            raise ValueError('Poynting validation steps require use_poynting_vectors=True.')
        self.poynting_validation_snapshots = {}

        if generate_video:
            for frame_name in os.listdir(self.frames_folder):
                if frame_name.startswith('frame_') and frame_name.endswith('.png'):
                    os.remove(os.path.join(self.frames_folder, frame_name))

        compute_forward_diff = self.wgpu_handler.create_compute_pipeline("forward_diff")
        compute_after_forward = self.wgpu_handler.create_compute_pipeline("after_forward")
        compute_backward_diff = self.wgpu_handler.create_compute_pipeline("backward_diff")
        compute_after_backward = self.wgpu_handler.create_compute_pipeline("after_backward")
        compute_sim_flipped_tr = self.wgpu_handler.create_compute_pipeline("sim_flipped_tr")
        compute_sim = self.wgpu_handler.create_compute_pipeline("sim")
        if use_poynting_vectors:
            compute_update_velocity = self.wgpu_handler.create_compute_pipeline("update_velocity")
            compute_update_rtm_image = self.wgpu_handler.create_compute_pipeline("update_rtm_image")
        compute_incr_time = self.wgpu_handler.create_compute_pipeline("incr_time")


        accumulated_product = np.zeros(self.grid_size_shape, dtype=np.float32)
        accumulated_product_poynting = (
            np.zeros(self.grid_size_shape, dtype=np.float32)
            if use_poynting_vectors
            else None
        )
        accumulated_source_energy = np.zeros(self.grid_size_shape, dtype=np.float32)
        previous_source_pressure = np.zeros(self.grid_size_shape, dtype=np.float32)
        L = self.absorption_layer_size
        roi_slice = (slice(L, -L), slice(L, -L))

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
            
            if use_poynting_vectors:
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
            
            if use_poynting_vectors:
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
            if use_poynting_vectors:
                accumulated_product_poynting = (np.asarray(
                    self.wgpu_handler.device.queue.read_buffer(self.wgpu_handler.buffers['b38']).cast("f")
                ).reshape(self.grid_size_shape))

            if i in validation_steps:
                source_pressure_past = (np.asarray(
                    self.wgpu_handler.device.queue.read_buffer(self.wgpu_handler.buffers['b6']).cast("f")
                ).reshape(self.grid_size_shape))
                receiver_pressure_past = (np.asarray(
                    self.wgpu_handler.device.queue.read_buffer(self.wgpu_handler.buffers['b21']).cast("f")
                ).reshape(self.grid_size_shape))
                source_vz = (np.asarray(
                    self.wgpu_handler.device.queue.read_buffer(self.wgpu_handler.buffers['b34']).cast("f")
                ).reshape(self.grid_size_shape))
                source_vx = (np.asarray(
                    self.wgpu_handler.device.queue.read_buffer(self.wgpu_handler.buffers['b35']).cast("f")
                ).reshape(self.grid_size_shape))
                receiver_vz = (np.asarray(
                    self.wgpu_handler.device.queue.read_buffer(self.wgpu_handler.buffers['b36']).cast("f")
                ).reshape(self.grid_size_shape))
                receiver_vx = (np.asarray(
                    self.wgpu_handler.device.queue.read_buffer(self.wgpu_handler.buffers['b37']).cast("f")
                ).reshape(self.grid_size_shape))

                source_pressure_half = np.float32(0.5) * (source_pressure_past + self.p_future)
                receiver_pressure_half = np.float32(0.5) * (
                    receiver_pressure_past + self.p_future_flipped_tr
                )
                source_jx = np.zeros(self.grid_size_shape, dtype=np.float32)
                source_jz = np.zeros(self.grid_size_shape, dtype=np.float32)
                receiver_jx = np.zeros(self.grid_size_shape, dtype=np.float32)
                receiver_jz = np.zeros(self.grid_size_shape, dtype=np.float32)
                source_jx[:, 1:] = source_pressure_half[:, 1:] * np.float32(0.5) * (
                    source_vx[:, 1:] + source_vx[:, :-1]
                )
                source_jz[1:, :] = source_pressure_half[1:, :] * np.float32(0.5) * (
                    source_vz[1:, :] + source_vz[:-1, :]
                )
                receiver_jx[:, 1:] = -receiver_pressure_half[:, 1:] * np.float32(0.5) * (
                    receiver_vx[:, 1:] + receiver_vx[:, :-1]
                )
                receiver_jz[1:, :] = -receiver_pressure_half[1:, :] * np.float32(0.5) * (
                    receiver_vz[1:, :] + receiver_vz[:-1, :]
                )
                self.poynting_validation_snapshots[i] = {
                    'source_pressure': source_pressure_half,
                    'source_jx': source_jx,
                    'source_jz': source_jz,
                    'receiver_pressure': receiver_pressure_half,
                    'receiver_jx': receiver_jx,
                    'receiver_jz': receiver_jz,
                }
            
            current_product = self.p_future * self.p_future_flipped_tr
            accumulated_product += current_product

            # The shader pairs velocity at n + 1/2 with the average of pressure
            # at n and n + 1. Reuse consecutive source-pressure readbacks to
            # accumulate illumination at that same half-step without another
            # GPU buffer read.
            source_pressure_half = np.float32(0.5) * (previous_source_pressure + self.p_future)
            accumulated_source_energy += source_pressure_half * source_pressure_half
            previous_source_pressure = self.p_future

            if generate_video and i % animation_step == 0:
                fig, axs = plt.subplots(2, 2, figsize=(10, 10))

                standard_normalized_frame = _normalize_by_source_energy(
                    accumulated_product,
                    accumulated_source_energy,
                )[roi_slice]

                axs[0, 0].imshow(self.p_future_flipped_tr, cmap='viridis', interpolation='none')
                axs[0, 0].set_title('Up-Going')
                axs[1, 0].imshow(self.p_future, cmap='viridis', interpolation='none')
                axs[1, 0].set_title('Down-Going')

                if use_poynting_vectors:
                    poynting_normalized_frame = _normalize_by_source_energy(
                        accumulated_product_poynting,
                        accumulated_source_energy,
                    )[roi_slice]
                    display_limit = np.percentile(
                        np.abs(np.concatenate((
                            standard_normalized_frame.ravel(),
                            poynting_normalized_frame.ravel(),
                        ))),
                        99.5,
                    )
                else:
                    display_limit = np.percentile(np.abs(standard_normalized_frame), 99.5)

                if display_limit == 0.0:
                    display_limit = 1.0

                axs[0, 1].imshow(
                    standard_normalized_frame,
                    cmap='seismic',
                    vmin=-display_limit,
                    vmax=display_limit,
                    interpolation='none',
                )
                axs[0, 1].set_title('Normalized Standard RTM')

                if use_poynting_vectors:
                    axs[1, 1].imshow(
                        poynting_normalized_frame,
                        cmap='seismic',
                        vmin=-display_limit,
                        vmax=display_limit,
                        interpolation='none',
                    )
                    axs[1, 1].set_title('Normalized Poynting RTM')
                else:
                    current_limit = np.percentile(np.abs(current_product[roi_slice]), 99.5)
                    axs[1, 1].imshow(
                        current_product[roi_slice],
                        cmap='seismic',
                        vmin=-current_limit,
                        vmax=current_limit,
                        interpolation='none',
                    )
                    axs[1, 1].set_title('Current Product')

                plt.savefig(self.frames_folder / f'frame_{i // animation_step}.png', bbox_inches='tight', pad_inches=0)
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
        np.save(self.folder / f'accumulated_product_{self.emitter_index}.npy', accumulated_product[roi_slice])
        if use_poynting_vectors:
            np.save(
                self.folder / f'accumulated_product_poynting_{self.emitter_index}.npy',
                accumulated_product_poynting[roi_slice],
            )

        accumulated_product_normalized = _normalize_by_source_energy(
            accumulated_product,
            accumulated_source_energy,
        )

        np.save(
            self.folder / f'accumulated_product_normalized_{self.emitter_index}.npy',
            accumulated_product_normalized[roi_slice],
        )
        np.save(
            self.folder / f'accumulated_source_energy_{self.emitter_index}.npy',
            accumulated_source_energy[roi_slice],
        )
        if use_poynting_vectors:
            accumulated_product_poynting_normalized = _normalize_by_source_energy(
                accumulated_product_poynting,
                accumulated_source_energy,
            )
            np.save(
                self.folder / f'accumulated_product_poynting_normalized_{self.emitter_index}.npy',
                accumulated_product_poynting_normalized[roi_slice],
            )
        else:
            print('Poynting vectors disabled; Poynting output files were not updated.')

        if generate_video:
            create_video(path=self.frames_folder, output_path=self.folder / 'rtm.mp4')

        result = {
            'standard_raw': accumulated_product[roi_slice].copy(),
            'source_energy': accumulated_source_energy[roi_slice].copy(),
            'standard_normalized': accumulated_product_normalized[roi_slice].copy(),
        }
        if use_poynting_vectors:
            result['poynting_raw'] = accumulated_product_poynting[roi_slice].copy()
            result['poynting_normalized'] = (
                accumulated_product_poynting_normalized[roi_slice].copy()
            )
        return result
