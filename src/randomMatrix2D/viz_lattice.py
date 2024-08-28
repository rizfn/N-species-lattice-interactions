import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import ast
from tqdm import tqdm
import numpy as np

def parse_lattice(s):
    return ast.literal_eval(s)

def update(frame, lines, img, ax):
    step, lattice_str = lines[frame].split('\t')
    lattice = parse_lattice(lattice_str)
    img.set_array(lattice)
    ax.set_title(f'Step: {step}')
    return img,

def viz_with_nutrient():
    N = 50  # Number of species
    L = 512
    theta = 0.1
    filepath = f"src/randomMatrix2D/outputs/latticeTimeseries/N_{N}_L_{L}_theta_{theta}.tsv"

    # Define the color scheme
    light_colors = ['lightgreen', 'lightblue', 'violet', 'tomato', 'wheat']
    dark_colors = ['green', 'blue', 'purple', 'red', 'darkgoldenrod']
    if N > len(light_colors):
        raise ValueError(f'N={N} is not supported: add more colours!!')
    colors = light_colors[:N] + dark_colors[:N]

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.Normalize(vmin=0, vmax=len(colors)-1)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    fig.suptitle(f'{N} species, $\\theta$={theta}')
    with open(filepath, 'r') as file:
        lines = file.readlines()[1:]  # Skip the header
    total_frames = len(lines)
    step, lattice_str = lines[0].split('\t')
    lattice = parse_lattice(lattice_str)
    img = ax.imshow(lattice, cmap=cmap, norm=norm)
    ax.set_title(f'Step: {step}')
    ax.invert_yaxis()  # Invert the y-axis
    # plt.show(); return
    pbar = tqdm(total=total_frames)
    def update_with_progress(frame, lines, img, ax):
        pbar.update()
        return update(frame, lines, img, ax)
    ani = animation.FuncAnimation(fig, update_with_progress, frames=range(total_frames), fargs=(lines, img, ax), blit=True)
    # ffmpegwriter = animation.FFMpegWriter(fps=30, bitrate=-1)
    # ani.save(f'src/randomMatrix2D/plots/latticeAnim/N_{N}_theta_{theta}_nutrient.mp4', writer=ffmpegwriter)
    ani.save(f'src/randomMatrix2D/plots/latticeAnim/N_{N}_theta_{theta}_nutrient.gif', writer='ffmpeg', fps=30)
    pbar.close()


def main():
    N = 50  # Number of species
    L = 512
    theta = 0.1
    filepath = f"src/randomMatrix2D/outputs/latticeTimeseries/N_{N}_L_{L}_theta_{theta}.tsv"

    # Create a colormap from the rainbow, with 'N' discrete colors
    white_colors = np.array([[1, 1, 1, 0]] * N)
    rainbow_colors = plt.cm.rainbow(np.linspace(0, 1, N))
    combined_colors = np.vstack((white_colors, rainbow_colors))
    cmap = mcolors.ListedColormap(combined_colors)
    norm = mcolors.Normalize(vmin=0, vmax=2*N-1)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    fig.suptitle(f'{N} species, $\\theta$={theta}')
    with open(filepath, 'r') as file:
        lines = file.readlines()[1:]  # Skip the header
    total_frames = len(lines)
    step, lattice_str = lines[0].split('\t')
    lattice = parse_lattice(lattice_str)
    img = ax.imshow(lattice, cmap=cmap, norm=norm)
    ax.set_title(f'Step: {step}')
    ax.invert_yaxis()  # Invert the y-axis
    # plt.show(); return
    pbar = tqdm(total=total_frames)
    def update_with_progress(frame, lines, img, ax):
        pbar.update()
        return update(frame, lines, img, ax)
    ani = animation.FuncAnimation(fig, update_with_progress, frames=range(total_frames), fargs=(lines, img, ax), blit=True)
    # ffmpegwriter = animation.FFMpegWriter(fps=30, bitrate=-1)
    # ani.save(f'src/randomMatrix2D/plots/latticeAnim/N_{N}_theta_{theta}.mp4', writer=ffmpegwriter)
    ani.save(f'src/randomMatrix2D/plots/latticeAnim/N_{N}_theta_{theta}.gif', writer='ffmpeg', fps=30)
    pbar.close()


if __name__ == "__main__":
    # viz_with_nutrient()
    main()
