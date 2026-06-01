import numpy as np
import os
import re
from SimulationConfig import SimulationConfig
from WebGpuHandler import WebGpuHandler
from functions import save_image, create_video
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

        source = np.load('./source.npy').astype(np.float32)
        if len(source) < recorded_time:
            source = np.pad(source, (0, recorded_time - len(source)), 'constant').astype(np.float32)
        elif len(source) > recorded_time:
            source = source[:recorded_time]
        source_index = ~np.isclose(source, 0)
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

        # Inject flipped microphones code into shader string
        matches = re.findall(r'@binding\((\d+)\)', self.wgpu_handler.shader_string)
        last_binding = int(matches[-1])
        aux_string = ''
        for i in range(self.microphones_amount):
            aux_string += f'''@group(0) @binding({i + (last_binding + 1)})
            var<storage,read> flipped_microphone_{i}: array<f32>;\n\n'''
        self.wgpu_handler.shader_string = self.wgpu_handler.shader_string.replace('//FLIPPED_MICROPHONES_BINDINGS', aux_string)
        aux_string = ''
        for i in range(self.microphones_amount):
            aux_string += f'''if (microphone_index == {i})
                    {{
                        p_future[zx(z, x)] += flipped_microphone_{i}[infoI32.i];
                    }}\n'''
        self.wgpu_handler.shader_string = self.wgpu_handler.shader_string.replace('//FLIPPED_MICROPHONES_SIM', aux_string)

        self.wgpu_handler.create_shader_module()

        # Data passed to gpu buffers
        wgsl_data = {
            'infoI32': self.info_i32,
            'infoF32': self.info_f32,
            'c': self.c,
            'microphone_z': np.ascontiguousarray(self.microphone_z),
            'microphone_x': np.ascontiguousarray(self.microphone_x),
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
            **{f'flipped_microphone_{i}': np.ascontiguousarray(self.flipped_bscan[i]) for i in range(self.microphones_amount)}
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
                plt.imshow(self.p_future, cmap='bwr')
                plt.colorbar()
                plt.scatter(self.microphone_x, self.microphone_z, s=0.05, color='purple')
                plt.scatter(self.reflector_x, self.reflector_z, s=0.05, color='green')
                plt.grid(True)
                plt.title(f'Time Reversal - {i}')
                plt.savefig(f'{self.frames_folder}/frame_{i // animation_step}.png')
                plt.close()

                # save_image(self.p_future, f'{self.frames_folder}/frame_{i // animation_step}.png')

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
