import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import subprocess
from pathlib import Path


def temporal_spatial_plot(recording_path='./SyntheticAcouSim/microphones_recording.npy', *, cmap='seismic', percentile=99.5):
    recording = np.load(recording_path)
    if recording.ndim != 2:
        raise ValueError(f'Expected a 2D recording array at {recording_path}, got shape {recording.shape}.')

    vmax = np.percentile(np.abs(recording), percentile)
    if vmax == 0:
        vmax = 1

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150, layout='tight')
    im = ax.imshow(recording, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Channel')
    ax.set_title(f'Temporal-Spatial Recording ({recording.shape[0]} channels x {recording.shape[1]} samples)')
    fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.9, label='Amplitude')
    plt.show()

    return fig, ax


def plot_accumulated_product():

    c, _, _, _, _ = convert_image_to_matrix('./map.png')
    reflector_z, reflector_x = np.int32(np.where(c == 0))

    accumulated_product_paths = sorted(Path('./SyntheticRTM').glob('accumulated_product_*.npy'))
    if not accumulated_product_paths:
        raise FileNotFoundError('No accumulated RTM images were found in ./SyntheticRTM.')

    accumulated_product = np.load(accumulated_product_paths[0])
    for accumulated_product_path in accumulated_product_paths[1:]:
        accumulated_product += np.load(accumulated_product_path)

    # accumulated_product_poynting = np.load('./SyntheticRTM/accumulated_product_poynting_0.npy')
    # for i in range(1, 8):
    #     accumulated_product_poynting += np.load(f'./SyntheticRTM/accumulated_product_poynting_{i}.npy')

    L = 45 
  
    grid_size_z, grid_size_x = c.shape 

    in_roi = (reflector_x >= L) & (reflector_x < (grid_size_x - L)) & \
             (reflector_z < (grid_size_z - L))
    
    roi_reflector_x = reflector_x[in_roi] - L 
    roi_reflector_z = reflector_z[in_roi]

    # fig, axs = plt.subplots(1, 2, figsize=(16, 9), dpi=300, layout='tight')

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300, layout='tight')

    abs_standard = np.abs(accumulated_product)
    vmax_standard = np.percentile(abs_standard, 100)
    vmin_standard = np.percentile(abs_standard, 99.)

    im0 = ax.imshow(abs_standard, aspect='auto', vmax=vmax_standard, vmin=vmin_standard)
    ax.set_title('Accumulated Product (Standard RTM)')
    ax.grid()
    fig.colorbar(im0, ax=ax, orientation='vertical', shrink=0.8)
    ax.scatter(roi_reflector_x, roi_reflector_z, s=0.05, color='white')
    # abs_poynting = np.abs(accumulated_product_poynting)
    # vmax_poynting = np.percentile(abs_poynting, 100)
    # vmin_poynting = np.percentile(abs_poynting, 80)
    
    # im1 = axs[1].imshow(abs_poynting, aspect='auto', vmax=vmax_poynting, vmin=vmin_poynting)
    # axs[1].set_title('Accumulated Product (Poynting RTM)')
    # axs[1].grid()
    # fig.colorbar(im1, ax=axs[1], orientation='vertical', shrink=0.8)
    # axs[1].scatter(roi_reflector_x, roi_reflector_z, s=0.05, color='white')

    plt.savefig('rtm.png')
    plt.show()
    
def create_source(
    n_sources=1,
    mu=30,
    sigma=5,
    samples=1000,
    output_dir='.',
    legacy_alias=True,
    delay_between_sources=None,
):
    """Create Gaussian source waveforms with independent first center and spacing."""
    n_sources = int(n_sources)
    samples = int(samples)
    output_dir = Path(output_dir)
    if delay_between_sources is None:
        delay_between_sources = mu

    if n_sources < 1 or n_sources > 100:
        raise ValueError('n_sources must be between 1 and 100.')
    if samples < 1:
        raise ValueError('samples must be at least 1.')
    if sigma <= 0:
        raise ValueError('sigma must be greater than 0.')
    if delay_between_sources < 0:
        raise ValueError('delay_between_sources must be greater than or equal to 0.')

    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(samples, dtype=np.float32)
    source_paths = []

    for source_id in range(n_sources):
        center = np.float32(mu + delay_between_sources * source_id)
        source = np.exp(-0.5 * ((x - center) / np.float32(sigma)) ** 2).astype(np.float32)
        source_path = output_dir / f'source{source_id}.npy'
        np.save(source_path, source)
        source_paths.append(source_path)

        if source_id == 0 and legacy_alias:
            np.save(output_dir / 'source.npy', source)

    return source_paths


def load_source(source_id=0, total_time=None, source_dir='.'):
    source_id = int(source_id)
    source_dir = Path(source_dir)

    if source_id < 0:
        raise ValueError('source_id must be non-negative.')

    expected_source_path = source_dir / f'source{source_id}.npy'
    source_path = expected_source_path
    if not source_path.exists() and source_id == 0:
        source_path = source_dir / 'source.npy'

    if not source_path.exists():
        raise FileNotFoundError(
            f'Could not find waveform for source {source_id}. Expected {expected_source_path}.'
        )

    source = np.load(source_path).astype(np.float32)
    if total_time is None:
        return source

    total_time = int(total_time)
    if len(source) < total_time:
        return np.pad(source, (0, total_time - len(source)), 'constant').astype(np.float32)
    if len(source) > total_time:
        return source[:total_time].astype(np.float32)
    return source


def load_sources(source_ids, total_time, source_dir='.'):
    source_ids = np.atleast_1d(source_ids).astype(np.int32)
    if source_ids.size == 0:
        raise ValueError('At least one source ID is required.')

    sources = [load_source(int(source_id), total_time, source_dir) for source_id in source_ids]
    return np.ascontiguousarray(np.vstack(sources).astype(np.float32))

def plot_source(source_dir='.'):
    source_dir = Path(source_dir)
    source_paths = []

    for source_path in source_dir.glob('source*.npy'):
        source_suffix = source_path.stem.removeprefix('source')
        if source_suffix.isdigit():
            source_paths.append((int(source_suffix), source_path))

    source_paths.sort(key=lambda item: item[0])

    if not source_paths:
        legacy_source_path = source_dir / 'source.npy'
        if not legacy_source_path.exists():
            raise FileNotFoundError(f'No source waveforms found in {source_dir}.')
        source_paths = [(0, legacy_source_path)]

    fig, ax = plt.subplots()
    for source_id, source_path in source_paths:
        source = np.load(source_path).astype(np.float32)
        ax.plot(source, label=f'source{source_id}')

    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.set_title('Source Waveforms')
    ax.legend()
    plt.show()

    return fig, ax

def plot_l2_norm():
    l2_norm = np.load('./SyntheticTR/l2_norm.npy')

    l2_norm[0:10, :] = np.float32(0)

    # l2_norm_log = np.log10(l2_norm + 1e-10)  # Add a small constant to avoid log(0)

    vmax = np.percentile(l2_norm, 99)

    plt.figure()
    plt.imshow(l2_norm, aspect='auto', vmax=vmax, vmin=0)

    plt.colorbar()
    plt.grid()
    plt.title('L2-Norm - Time Reversal')
    plt.show()


def save_image(image, path):
    # Ensure width is even for ffmpeg/libx264
    h, w = image.shape[:2]
    if w % 2 != 0:
        # Crop last column
        image = image[:, :-1]
    plt.imsave(path, image)


def create_video(path, output_path):
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', '25',
        '-i', f'{path}/frame_%d.png',
        # Add the video filter here to ensure even dimensions
        '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2', 
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    subprocess.run(cmd, check=True)

def save_rtm_image(upper_left, upper_right, bottom_left, bottom_right, path):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Upper left subplot
    axs[0, 0].imshow(upper_left, cmap='viridis', interpolation='none')
    axs[0, 0].set_title('Up-Going')

    # Upper right subplot
    axs[0, 1].imshow(upper_right, cmap='viridis', interpolation='none')
    axs[0, 1].set_title('Product')

    # Bottom left subplot
    axs[1, 0].imshow(bottom_left, cmap='viridis', interpolation='none')
    axs[1, 0].set_title('Down-Going')

    # Bottom right subplot
    axs[1, 1].imshow(bottom_right, cmap='viridis', interpolation='none')
    axs[1, 1].set_title('Accumulated Product')

    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def convert_image_to_matrix(image_path, return_source_ids=False):
    rgb_raw_image = np.asarray(imread(image_path))

    if rgb_raw_image.ndim != 3 or rgb_raw_image.shape[2] < 3:
        raise ValueError(f'Expected an RGB image at {image_path}, got shape {rgb_raw_image.shape}.')

    # Marker colors expected in the PNG editor:
    # black   = #000000 -> reflector / obstacle
    # blue    = #0000FF -> background medium (1500 m/s)
    # green   = #00FF00 -> material (3200 m/s)
    # red     = #FF0000 -> material (6400 m/s)
    # source  = #FFFF00..#FFFF99 -> source only, decimal suffix selects sourceN.npy
    # cyan    = #00FFFF -> receptor only
    # white   = #FFFFFF -> colocated source + receptor
    rgb_image = rgb_raw_image[:, :, :3]
    if np.issubdtype(rgb_image.dtype, np.floating):
        rgb_image = np.rint(rgb_image * 255).astype(np.uint8)
    else:
        rgb_image = rgb_image.astype(np.uint8)

    red = rgb_image[:, :, 0]
    green = rgb_image[:, :, 1]
    blue = rgb_image[:, :, 2]

    is_black = (red == 0) & (green == 0) & (blue == 0)
    is_red = (red == 255) & (green == 0) & (blue == 0)
    is_green = (red == 0) & (green == 255) & (blue == 0)
    is_blue = (red == 0) & (green == 0) & (blue == 255)
    is_cyan = (red == 0) & (green == 255) & (blue == 255)
    is_white = (red == 255) & (green == 255) & (blue == 255)

    source_marker_tolerance = 3
    source_candidate = (
        (red >= 255 - source_marker_tolerance) &
        (green >= 255 - source_marker_tolerance) &
        ~is_white
    )
    source_id_tens = blue // 16
    source_id_ones = blue % 16
    has_decimal_source_suffix = (source_id_tens <= 9) & (source_id_ones <= 9)
    is_source = source_candidate & has_decimal_source_suffix
    invalid_source_suffix = source_candidate & ~has_decimal_source_suffix

    if np.any(invalid_source_suffix):
        invalid_positions = np.argwhere(invalid_source_suffix)
        invalid_z, invalid_x = invalid_positions[0]
        invalid_rgb = rgb_image[invalid_z, invalid_x].tolist()
        invalid_hex = '#{:02X}{:02X}{:02X}'.format(*invalid_rgb)
        raise ValueError(
            f'Unsupported source color {invalid_hex} at pixel ({invalid_z}, {invalid_x}) in {image_path}. '
            'Use decimal source colors #FFFF00 through #FFFF99.'
        )

    recognized_mask = is_black | is_red | is_green | is_blue | is_source | is_cyan | is_white
    if not np.all(recognized_mask):
        unknown_positions = np.argwhere(~recognized_mask)
        first_unknown_z, first_unknown_x = unknown_positions[0]
        first_unknown_rgb = rgb_image[first_unknown_z, first_unknown_x].tolist()
        raise ValueError(
            f'Unsupported color {first_unknown_rgb} at pixel ({first_unknown_z}, {first_unknown_x}) in {image_path}.'
        )

    rgb_float = np.full(rgb_image.shape[:2], np.float32(1500), dtype=np.float32)
    rgb_float[is_black] = np.float32(0)
    rgb_float[is_green] = np.float32(3200)
    rgb_float[is_red] = np.float32(6400)

    source_pos = np.where(is_source | is_white)
    receptor_pos = np.where(is_cyan | is_white)

    source_z, source_x = np.int32(source_pos)
    receptor_z, receptor_x = np.int32(receptor_pos)
    source_id_matrix = (source_id_tens.astype(np.int32) * 10) + source_id_ones.astype(np.int32)
    source_ids = source_id_matrix[source_pos].astype(np.int32)
    source_ids[is_white[source_pos]] = np.int32(0)

    if source_z.size == 0:
        raise ValueError(
            f'No sources found in {image_path}. Use #FFFF00 through #FFFF99 for source-only markers or white for colocated source/receptor markers.'
        )
    if receptor_z.size == 0:
        raise ValueError(
            f'No receptors found in {image_path}. Use cyan pixels for receptor-only markers or white for colocated source/receptor markers.'
        )

    if return_source_ids:
        return rgb_float, source_z, source_x, receptor_z, receptor_x, source_ids

    return rgb_float, source_z, source_x, receptor_z, receptor_x

if __name__ == "__main__":
    # plot_accumulated_product()
    create_source(n_sources=3, samples=3500 , delay_between_sources=1000)
    plot_source()
    plot_l2_norm()
    temporal_spatial_plot()
