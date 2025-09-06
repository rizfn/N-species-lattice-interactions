import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from numba import njit


def main():
    N_species = 4
    D_N = 1
    D_S = 0.0
    x = np.linspace(0, 1, 40)
    t = np.linspace(0, 150, 500000)  # More time steps for stability
    dt = t[1] - t[0]  # time step
    dx = x[1] - x[0]  # spatial step
    gamma = 0.5
    boundary_N = 0.5  # Boundary condition for nutrients at x=0
    
    # Check stability condition
    stability_factor = D_N * dt / dx**2
    print(f"Stability factor: {stability_factor} (should be ≤ 0.5)")
    
    # g = np.random.rand(N_species)  # Growth parameters for R_i(N) function
    g = np.linspace(0.2, 0.8, N_species)  # Growth parameters for R_i(N) function

    species = np.ones((N_species, len(x)))
    nutrients = np.zeros((len(x)))
    nutrients[0] = boundary_N  # Boundary condition at x=0
    
    # Storage for animation data (every 1 time unit)
    save_interval = int(1.0 / dt)  # Save every 1 time unit
    n_frames = len(t) // save_interval + 1
    species_history = np.zeros((n_frames, N_species, len(x)))
    nutrients_history = np.zeros((n_frames, len(x)))
    time_points = []
    
    frame_idx = 0
    species_history[0] = species.copy()
    nutrients_history[0] = nutrients.copy()
    time_points.append(0)

    for time_idx, time in enumerate(tqdm(t, desc="Running simulation")):
        
        # Second-order derivative for diffusion (Laplacian) - nutrients
        d2nutrients_dx2 = np.zeros_like(nutrients)
        d2nutrients_dx2[1:-1] = (nutrients[2:] - 2*nutrients[1:-1] + nutrients[:-2]) / dx**2
        
        # Calculate R_i(N) for each species at each position
        R_values = np.zeros((N_species, len(x)))
        for i in range(N_species):
            R_values[i] = (g[i] * nutrients) / (10**((g[i] - 1) / 0.3) + nutrients)
        
        # Nutrient consumption by all species using R_i(N)
        nutrient_consumption = np.sum(R_values * species, axis=0)
        
        # Update nutrients: diffusion - consumption
        nutrients += (D_N * d2nutrients_dx2 - nutrient_consumption) * dt
        
        # Update each species
        for i in range(N_species):
            # Species diffusion
            d2species_dx2 = np.zeros_like(species[i])
            d2species_dx2[1:-1] = (species[i, 2:] - 2*species[i, 1:-1] + species[i, :-2]) / dx**2
            
            # Growth term using R_i(N)
            growth = R_values[i] * species[i] - gamma * species[i]
            
            # Update species: diffusion + growth
            species[i] += (D_S * d2species_dx2 + growth) * dt
        
        # Maintain boundary conditions
        nutrients[0] = boundary_N  # Dirichlet BC at left (fixed source)
        nutrients[-1] = nutrients[-2]  # Neumann BC at right (zero flux)
        
        # Species boundary conditions (zero flux for all species)
        for i in range(N_species):
            species[i, 0] = species[i, 1]    # Left boundary
            species[i, -1] = species[i, -2]  # Right boundary

        # Save data for animation every 1 time unit
        if (time_idx + 1) % save_interval == 0 and frame_idx < n_frames - 1:
            frame_idx += 1
            species_history[frame_idx] = species.copy()
            nutrients_history[frame_idx] = nutrients.copy()
            time_points.append(time)

    # Create animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Initialize plots
    nutrient_line, = ax1.plot([], [], 'k-', label='Nutrients')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, np.max(nutrients_history) * 1.1)
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Density')
    ax1.set_title('Nutrients')
    ax1.legend()
    
    species_lines = []
    for i in range(N_species):
        line, = ax2.plot([], [], label=f'Species {i+1}')
        species_lines.append(line)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, np.max(species_history) * 1.1)
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Density')
    ax2.set_title('Species')
    ax2.legend()
    
    # Time text
    time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=12)
    
    def animate(frame):
        # Update nutrient plot
        nutrient_line.set_data(x, nutrients_history[frame])
        
        # Update species plots
        for i in range(N_species):
            species_lines[i].set_data(x, species_history[frame, i])
        
        # Update time text
        time_text.set_text(f'Time: {time_points[frame]:.1f}')
        
        return [nutrient_line] + species_lines + [time_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(time_points), 
                                 interval=50, blit=True, repeat=True)
    
    # Save animation with progress bar
    plt.tight_layout()
    with tqdm(total=len(time_points), desc="Saving animation") as pbar:
        def progress_callback(current_frame, total_frames):
            pbar.update(1)
        
        anim.save("src/quantizedSpeciesPDE/plots/stavans_animation.mp4", 
                  fps=20, writer='ffmpeg', bitrate=1800, 
                  progress_callback=progress_callback)
    
    # Plot R_i(N) vs N for all species
    N_range = np.linspace(0, boundary_N, 1000)  # Range of nutrient concentrations
    plt.figure(figsize=(10, 6))
    for i in range(N_species):
        R_curve = (g[i] * N_range) / (10**((g[i] - 1) / 0.3) + N_range)
        plt.plot(N_range, R_curve, label=f'Species {i+1} (g={g[i]:.3f})')
    plt.xlabel('Nutrient Concentration (N)')
    plt.ylabel('Growth Rate R_i(N)')
    plt.title('Growth Rate Functions for All Species')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("src/quantizedSpeciesPDE/plots/growth_functions.png")
    
    # Also save final state plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, nutrients, 'k-', label='Nutrients')
    plt.xlabel('Position')
    plt.ylabel('Density')
    plt.title('Nutrients')
    
    plt.subplot(1, 2, 2)
    for i in range(N_species):
        plt.plot(x, species[i], label=f'Species {i+1}')
    plt.xlabel('Position')
    plt.ylabel('Density')
    plt.title('Species')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("src/quantizedSpeciesPDE/plots/stavans_final.png")

if __name__ == "__main__":
    main()

