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
    
# def create_source():

    # source = np.random.rand(1000)

    # np.save('source.npy', source)

def plot_source():

    source = np.load('source.npy')

    plt.figure()
    plt.plot(source)
    plt.show()

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


def convert_image_to_matrix(image_path):
    rgb_raw_image = np.asarray(imread(image_path))

    if rgb_raw_image.ndim != 3 or rgb_raw_image.shape[2] < 3:
        raise ValueError(f'Expected an RGB image at {image_path}, got shape {rgb_raw_image.shape}.')

    # Exact marker colors expected in the PNG editor:
    # black   = #000000 -> reflector / obstacle
    # blue    = #0000FF -> background medium (1500 m/s)
    # green   = #00FF00 -> material (3200 m/s)
    # red     = #FF0000 -> material (6400 m/s)
    # yellow  = #FFFF00 -> source only
    # cyan    = #00FFFF -> receptor only
    # white   = #FFFFFF -> colocated source + receptor
    rgb_image = rgb_raw_image[:, :, :3]
    if np.issubdtype(rgb_image.dtype, np.floating):
        rgb_image = np.rint(rgb_image * 255).astype(np.uint8)
    else:
        rgb_image = rgb_image.astype(np.uint8)

    red = rgb_image[:, :, 0] == 255
    green = rgb_image[:, :, 1] == 255
    blue = rgb_image[:, :, 2] == 255

    is_black = ~red & ~green & ~blue
    is_red = red & ~green & ~blue
    is_green = ~red & green & ~blue
    is_blue = ~red & ~green & blue
    is_yellow = red & green & ~blue
    is_cyan = ~red & green & blue
    is_white = red & green & blue

    recognized_mask = is_black | is_red | is_green | is_blue | is_yellow | is_cyan | is_white
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

    source_pos = np.where(is_yellow | is_white)
    receptor_pos = np.where(is_cyan | is_white)

    source_z, source_x = np.int32(source_pos)
    receptor_z, receptor_x = np.int32(receptor_pos)

    if source_z.size == 0:
        raise ValueError(
            f'No sources found in {image_path}. Use yellow pixels for source-only markers or white for colocated source/receptor markers.'
        )
    if receptor_z.size == 0:
        raise ValueError(
            f'No receptors found in {image_path}. Use cyan pixels for receptor-only markers or white for colocated source/receptor markers.'
        )

    return rgb_float, source_z, source_x, receptor_z, receptor_x

if __name__ == "__main__":
    # plot_accumulated_product()
    # plot_source()
    plot_l2_norm()
    temporal_spatial_plot()
