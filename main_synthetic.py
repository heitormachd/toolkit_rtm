import numpy as np
from SyntheticAcouSim import SyntheticAcouSim
from SyntheticTimeReversal import SyntheticTimeReversal
from SyntheticReverseTimeMigration import SyntheticReverseTimeMigration
from functions import convert_image_to_matrix

c, microphone_z, microphone_x = convert_image_to_matrix('./map.png')

microphone_z[:] = np.int32(1)

microphones_amount = np.int32(len(microphone_z))

grid_size_z = np.int32(len(c[:, 0]))
grid_size_x = np.int32(len(c[0, :]))
grid_size_shape = (grid_size_z, grid_size_x)

dt = np.float32(3e-7)

# Spatial steps (m/px)
dz = np.float32(1.5e-3)
dx = np.float32(1.5e-3)

for mic in range(microphones_amount):
    simulation_config = {
        'dt': dt,
        'c': c,
        'dz': dz,
        'dx': dx,
        'grid_size_z': grid_size_z,
        'grid_size_x': grid_size_x,
        'total_time': np.int32(4700),
        'medium_c': np.float32(1500),
    }

    synthetic_config = {
        'source_z': microphone_z[mic],
        'source_x': microphone_x[mic],
        'microphones_amount': microphones_amount,
        'microphone_z': microphone_z,
        'microphone_x': microphone_x,
        'emitter_index': int(mic),
    }

    simulation_config.update(synthetic_config)

    print(f'Microfone {mic}/{microphones_amount - 1}')

    if mic == 0:
        # acou_sim = SyntheticAcouSim(**simulation_config)
        # acou_sim.run(generate_video=True, animation_step=15)
        # tr_sim = SyntheticTimeReversal(**simulation_config)
        # tr_sim.run(generate_video=True, animation_step=15)
        rtm_sim = SyntheticReverseTimeMigration(**simulation_config)
        rtm_sim.run(generate_video=True, animation_step=15)
    else:
        acou_sim = SyntheticAcouSim(**simulation_config)
        acou_sim.run(generate_video=False, animation_step=15)
        tr_sim = SyntheticTimeReversal(**simulation_config)
        tr_sim.run(generate_video=False, animation_step=15)
        rtm_sim = SyntheticReverseTimeMigration(**simulation_config)
        rtm_sim.run(generate_video=False, animation_step=15)        

