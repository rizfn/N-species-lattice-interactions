import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import os
from numba import njit


@njit
def run_simulation(t, x, dt, dx, D_N, D_S, boundary_N, g, N_species, save_interval):
    """Run the main simulation loop with numba acceleration."""
    
    # Initialize arrays
    species = np.ones((N_species, len(x)))
    # Normalize initial conditions so sum of species at each site = 1
    for j in range(len(x)):
        site_total = np.sum(species[:, j])
        if site_total > 0:
            species[:, j] = species[:, j] / site_total
    
    nutrients = np.zeros(len(x))
    nutrients[0] = boundary_N  # Boundary condition at x=0
    nutrients[-1] = 0  # Boundary condition at x=L
    
    # Storage for animation data
    n_frames = len(t) // save_interval + 1
    species_history = np.zeros((n_frames, N_species, len(x)))
    nutrients_history = np.zeros((n_frames, len(x)))
    
    frame_idx = 0
    species_history[0] = species.copy()
    nutrients_history[0] = nutrients.copy()

    for time_idx in range(len(t)):
        
        # Second-order derivative for diffusion (Laplacian) - nutrients
        d2nutrients_dx2 = np.zeros_like(nutrients)
        d2nutrients_dx2[1:-1] = (nutrients[2:] - 2*nutrients[1:-1] + nutrients[:-2]) / dx**2
        
        # Calculate R_i(N) for each species at each position
        R_values = np.zeros((N_species, len(x)))
        for i in range(N_species):
            R_values[i] = (g[i] * nutrients) / (10**((g[i] - 1) / 0.3) + nutrients)
        
        # Calculate local dilution term: sum_j S_j * R_j(N) at each position
        local_dilution = np.sum(R_values * species, axis=0)  # Sum over species at each position
        
        # Nutrient consumption by all species using R_i(N)
        nutrient_consumption = np.sum(R_values * species, axis=0)
        
        # Update nutrients: diffusion - consumption
        nutrients = nutrients + (D_N * d2nutrients_dx2 - nutrient_consumption) * dt
        
        # Update each species
        for i in range(N_species):
            # Species diffusion with proper zero-flux boundary conditions
            d2species_dx2 = np.zeros_like(species[i])
            
            # Interior points (standard second derivative)
            d2species_dx2[1:-1] = (species[i, 2:] - 2*species[i, 1:-1] + species[i, :-2]) / dx**2
            
            # Left boundary (x=0): zero-flux condition dS/dx = 0
            d2species_dx2[0] = 2 * (species[i, 1] - species[i, 0]) / dx**2
            
            # Right boundary (x=L): zero-flux condition dS/dx = 0  
            d2species_dx2[-1] = 2 * (species[i, -2] - species[i, -1]) / dx**2
            
            # Growth term with local dilution: R_i(N) * S_i - S_i * local_dilution
            growth = R_values[i] * species[i] - species[i] * local_dilution
            
            # Update species: diffusion + growth
            species[i] = species[i] + (D_S * d2species_dx2 + growth) * dt
        
        # Maintain boundary conditions
        nutrients[0] = boundary_N  # Dirichlet BC at left (fixed source)
        nutrients[-1] = 0  # Dirichlet BC at right (sink)
        
        # Save data for animation every save_interval steps
        if (time_idx + 1) % save_interval == 0 and frame_idx < n_frames - 1:
            frame_idx = frame_idx + 1
            species_history[frame_idx] = species.copy()
            nutrients_history[frame_idx] = nutrients.copy()

    return species_history[:frame_idx+1], nutrients_history[:frame_idx+1], species, nutrients


def main():
    N_species = 4
    D_N = 0.1
    D_S = 0.001
    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 500, 1000000)  # More time steps for stability
    dt = t[1] - t[0]  # time step
    dx = x[1] - x[0]  # spatial step
    boundary_N = 1  # Boundary condition for nutrients at x=0
    
    # Check stability condition
    stability_factor = D_N * dt / dx**2
    print(f"Stability factor: {stability_factor} (should be ≤ 0.5)")

    # g = np.linspace(0.01, 1, N_species)**0.5  # Growth parameters for R_i(N) 
    g = np.array([0.2, 0.5, 0.7, 0.8])
    # g = np.random.rand(N_species)  # Growth parameters for R_i(N) function

    # Storage for animation data (every 1 time unit)
    save_interval = int(1.0 / dt)  # Save every 1 time unit
    
    print("Running simulation...")
    species_history, nutrients_history, final_species, final_nutrients = run_simulation(
        t, x, dt, dx, D_N, D_S, boundary_N, g, N_species, save_interval
    )
    
    # Create time points array for animation
    time_points = np.arange(len(species_history)) * dt * save_interval

    # Create animation
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Generate colors from rainbow colormap
    colors = plt.cm.rainbow(np.linspace(0, 0.9, N_species))
    
    # Initialize plots
    nutrient_line, = ax.plot([], [], 'k--', alpha=0.7, label='Nutrients')
    
    species_lines = []
    for i in range(N_species):
        line, = ax.plot([], [], color=colors[i], label=f'Species {i+1}')
        species_lines.append(line)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(np.max(species_history), np.max(nutrients_history)) * 1.1) 
    ax.set_xlabel('Position')
    ax.set_ylabel('Density')
    ax.set_title('')  # Start with empty title
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    def animate(frame):
        nutrient_line.set_data(x, nutrients_history[frame])
        for i in range(N_species):
            species_lines[i].set_data(x, species_history[frame, i])
        ax.set_title(f'Time: {time_points[frame]:8.1f}')

        return [nutrient_line] + species_lines
    
    anim = animation.FuncAnimation(fig, animate, frames=len(time_points), 
                                 interval=50, blit=True, repeat=True)
    
    plt.tight_layout()
    os.makedirs(f"src/quantizedSpeciesPDE/plots/nutrientSink1D/N{N_species}_DN{D_N}_DS{D_S}", exist_ok=True)
    with tqdm(total=len(time_points), desc="Saving animation") as pbar:
        def progress_callback(current_frame, total_frames):
            pbar.update(1)

        anim.save(f"src/quantizedSpeciesPDE/plots/nutrientSink1D/N{N_species}_DN{D_N}_DS{D_S}/animation.mp4",
                  fps=30, writer='ffmpeg', bitrate=1800,
                  progress_callback=progress_callback)
    
    # Plot R_i(N) vs N for all species
    N_range = np.linspace(0, boundary_N, 1000)  # Range of nutrient concentrations
    plt.figure(figsize=(10, 6))
    for i in range(N_species):
        R_curve = (g[i] * N_range) / (10**((g[i] - 1) / 0.3) + N_range)
        plt.plot(N_range, R_curve, color=colors[i], label=f'Species {i+1} (g={g[i]:.3f})')
    plt.xlabel('Nutrient Concentration (N)')
    plt.ylabel('Growth Rate R_i(N)')
    plt.title('Growth Rate Functions for All Species')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"src/quantizedSpeciesPDE/plots/nutrientSink1D/N{N_species}_DN{D_N}_DS{D_S}/growth_functions.png")

    # Also save final state plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, final_nutrients, 'k--', alpha=0.7, label='Nutrients')
    
    for i in range(N_species):
        plt.plot(x, final_species[i], color=colors[i], label=f'Species {i+1}')
    
    # Calculate which species has max growth rate at each position
    R_final = np.zeros((N_species, len(x)))
    for i in range(N_species):
        R_final[i] = (g[i] * final_nutrients) / (10**((g[i] - 1) / 0.3) + final_nutrients)
    
    max_species_idx = np.argmax(R_final, axis=0)
    
    # Create horizontal stripe showing dominant species
    y_bottom, y_top = 1.0, 1.05
    
    # Group contiguous regions of the same dominant species
    current_species = max_species_idx[0]
    start_x = x[0]
    
    for i in range(1, len(x)):
        if max_species_idx[i] != current_species:
            # End current region and start new one
            species_color = colors[current_species]
            plt.fill_between([start_x, x[i-1]], y_bottom, y_top, 
                            color=species_color, alpha=0.6)
            current_species = max_species_idx[i]
            start_x = x[i-1]
    
    # Don't forget the last region
    species_color = colors[current_species]
    plt.fill_between([start_x, x[-1]], y_bottom, y_top, 
                    color=species_color, alpha=0.6)
    
    plt.xlabel('Position')
    plt.ylabel('Density')
    plt.title('Final State: Species and Nutrients')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"src/quantizedSpeciesPDE/plots/nutrientSink1D/N{N_species}_DN{D_N}_DS{D_S}/final_state.png")


if __name__ == "__main__":
    main()


