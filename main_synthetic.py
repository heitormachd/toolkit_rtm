import numpy as np
from SyntheticAcouSim import SyntheticAcouSim
from SyntheticTimeReversal import SyntheticTimeReversal
from SyntheticReverseTimeMigration import SyntheticReverseTimeMigration
from functions import convert_image_to_matrix

IMAGE_PATH = './weird-shape.png'
FORCE_RECEIVERS_TO_SURFACE = False
SURFACE_RECEIVER_Z = np.int32(1)
FORWARD_TOTAL_TIME = np.int32(4000)
TIME_REVERSAL_TOTAL_TIME = np.int32(5000)

c, source_z, source_x, receptor_z, receptor_x = convert_image_to_matrix(IMAGE_PATH)

if FORCE_RECEIVERS_TO_SURFACE:
    receptor_z = np.full_like(receptor_z, SURFACE_RECEIVER_Z)

sources_amount = np.int32(len(source_z))
microphones_amount = np.int32(len(receptor_z))

grid_size_z = np.int32(len(c[:, 0]))
grid_size_x = np.int32(len(c[0, :]))
grid_size_shape = (grid_size_z, grid_size_x)

dt = np.float32(3e-7)

# Spatial steps (m/px)
dz = np.float32(1.5e-3)
dx = np.float32(1.5e-3)

for src in range(sources_amount):
    simulation_config = {
        'dt': dt,
        'c': c,
        'dz': dz,
        'dx': dx,
        'grid_size_z': grid_size_z,
        'grid_size_x': grid_size_x,
        'total_time': FORWARD_TOTAL_TIME,
        'medium_c': np.float32(1500),
    }

    synthetic_config = {
        'source_z': source_z[src],
        'source_x': source_x[src],
        'microphones_amount': microphones_amount,
        'microphone_z': receptor_z,
        'microphone_x': receptor_x,
        'emitter_index': int(src),
    }

    simulation_config.update(synthetic_config)
    time_reversal_config = simulation_config.copy()
    time_reversal_config['total_time'] = TIME_REVERSAL_TOTAL_TIME

    print(f'Fonte {src}/{sources_amount - 1}')

    if src == 0:
        acou_sim = SyntheticAcouSim(**simulation_config)
        acou_sim.run(generate_video=True, animation_step=15)
        tr_sim = SyntheticTimeReversal(**time_reversal_config)
        tr_sim.run(generate_video=True, animation_step=15)
        # rtm_sim = SyntheticReverseTimeMigration(**time_reversal_config)
        # rtm_sim.run(generate_video=True, animation_step=15)
    else:
        acou_sim = SyntheticAcouSim(**simulation_config)
        acou_sim.run(generate_video=False, animation_step=15)
        tr_sim = SyntheticTimeReversal(**time_reversal_config)
        tr_sim.run(generate_video=False, animation_step=15)
        # rtm_sim = SyntheticReverseTimeMigration(**time_reversal_config)
        # rtm_sim.run(generate_video=False, animation_step=15)        
