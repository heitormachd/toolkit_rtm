import numpy as np
import matplotlib.pyplot as plt
import ffmpeg
from matplotlib.image import imread
import subprocess


def plot_accumulated_product():

    c, _, _ = convert_image_to_matrix('./map.png') 
    reflector_z, reflector_x = np.int32(np.where(c == 0))

    accumulated_product = np.load('./SyntheticRTM/accumulated_product_0.npy')
    for i in range(1, 106):
        accumulated_product += np.load(f'./SyntheticRTM/accumulated_product_{i}.npy')

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
    
def plot_source():

    source = np.load('source.npy')

    plt.figure()
    plt.plot(source)
    plt.show()

def plot_l2_norm():
    l2_norm = np.load('./SyntheticTR/l2_norm.npy')

    l2_norm[0:25, :] = np.float32(0)

    plt.figure()
    plt.imshow(l2_norm, aspect='auto')
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
    rgb_raw_image = np.int32(imread(image_path))

    velocity_map = {
        'white': 'receptors',
        'black': '0',
        'blue': '1500',
        'green': '3200',
        'red': '6400',
    }
    binary_color = {
        7: 'white',
        0: 'black',
        1: 'red',
        2: 'green',
        4: 'blue',
    }

    rgb_2d_grid = np.zeros_like(rgb_raw_image[:, :, 0])

    b = 1
    for i in range(3):
        b += i
        rgb_2d_grid += rgb_raw_image[:, :, i] * b

    rgb_string = np.array(rgb_2d_grid, dtype='str')
    for k in binary_color.keys():
        rgb_string[rgb_string == str(k)] = velocity_map[binary_color[k]]

    receptor_pos = np.where(rgb_string == 'receptors')
    rgb_string[receptor_pos] = '1500'
    receptor_z, receptor_x = np.int32(receptor_pos)

    rgb_float = np.array(rgb_string, dtype=np.float32)

    return rgb_float, receptor_z, receptor_x

if __name__ == "__main__":
    plot_accumulated_product()
    # plot_source()
    # plot_l2_norm()