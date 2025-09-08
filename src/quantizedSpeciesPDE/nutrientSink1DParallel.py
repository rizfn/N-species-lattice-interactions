import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from numba import njit
from multiprocessing import Pool, cpu_count


@njit
def run_simulation_final_only(t, x, dt, dx, D_N, D_S, boundary_N, g, N_species):
    """Run simulation and return only final state for parameter sweeps."""
    
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

    return species, nutrients


def simulate_parameter_set(params):
    """Wrapper function for parallel simulation."""
    D_N, D_S, x, t, dt, dx, boundary_N, g, N_species = params
    
    final_species, final_nutrients = run_simulation_final_only(
        t, x, dt, dx, D_N, D_S, boundary_N, g, N_species
    )
    
    # Calculate majority species at each position
    majority_indices = np.argmax(final_species, axis=0)
    
    # Count number of majority switches
    switches = 0
    for i in range(1, len(majority_indices)):
        if majority_indices[i] != majority_indices[i-1]:
            switches += 1
    
    # Calculate average concentration of majority species
    majority_concentrations = np.array([final_species[majority_indices[i], i] 
                                       for i in range(len(majority_indices))])
    avg_majority_concentration = np.mean(majority_concentrations)
    
    return D_N, D_S, switches, avg_majority_concentration


def main():
    N_species = 20
    boundary_N = 1  # Boundary condition for nutrients at x=0
    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 1000, 20000000)  # Shorter time for parameter sweep
    dt = t[1] - t[0]  # time step
    dx = x[1] - x[0]  # spatial step
    
    # Parameter ranges (log scale)
    D_N_values = np.geomspace(0.0001, 1, 10)
    D_S_values = np.geomspace(0.0001, 1, 10)
    
    g = np.array([0.2, 0.5, 0.7, 0.8])
    
    # Create parameter combinations
    param_combinations = []
    for D_N in D_N_values:
        for D_S in D_S_values:
            # Check stability condition
            stability_factor = max(D_N, D_S) * dt / dx**2
            if stability_factor <= 0.5:  # Only run stable simulations
                param_combinations.append((D_N, D_S, x, t, dt, dx, boundary_N, g, N_species))
    
    print(f"Running {len(param_combinations)} parameter combinations...")
    print(f"Using {cpu_count()} CPU cores")
    
    # Run simulations in parallel
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(simulate_parameter_set, param_combinations), 
                           total=len(param_combinations), 
                           desc="Running simulations"))
    
    # Extract results
    D_N_results = [r[0] for r in results]
    D_S_results = [r[1] for r in results]
    switches_results = [r[2] for r in results]
    concentration_results = [r[3] for r in results]
    
    # Create meshgrids for plotting
    switches_grid = np.full((len(D_N_values), len(D_S_values)), np.nan)
    concentration_grid = np.full((len(D_N_values), len(D_S_values)), np.nan)
    
    for D_N_val, D_S_val, switches, concentration in results:
        i = np.argmin(np.abs(D_N_values - D_N_val))
        j = np.argmin(np.abs(D_S_values - D_S_val))
        switches_grid[i, j] = switches
        concentration_grid[i, j] = concentration
    
    # Create output directory
    output_dir = f"src/quantizedSpeciesPDE/plots/nutrientSink1D/parameter_sweep_N{N_species}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot heatmap for number of majority switches
    plt.figure(figsize=(10, 8))
    im1 = plt.imshow(switches_grid, aspect='auto', origin='lower', 
                     extent=[np.log10(D_S_values[0]), np.log10(D_S_values[-1]), 
                             np.log10(D_N_values[0]), np.log10(D_N_values[-1])],
                     cmap='viridis')
    plt.colorbar(im1, label='Number of Majority Switches')
    plt.xlabel('log₁₀(Species Diffusion Coefficient D_S)')
    plt.ylabel('log₁₀(Nutrient Diffusion Coefficient D_N)')
    plt.title('Number of Majority Species Switches Across Space')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/majority_switches_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot heatmap for average majority concentration
    plt.figure(figsize=(10, 8))
    im2 = plt.imshow(concentration_grid, aspect='auto', origin='lower',
                     extent=[np.log10(D_S_values[0]), np.log10(D_S_values[-1]), 
                             np.log10(D_N_values[0]), np.log10(D_N_values[-1])],
                     cmap='plasma')
    plt.colorbar(im2, label='Average Majority Concentration')
    plt.xlabel('log₁₀(Species Diffusion Coefficient D_S)')
    plt.ylabel('log₁₀(Nutrient Diffusion Coefficient D_N)')
    plt.title('Average Concentration of Majority Species')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/majority_concentration_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmaps saved to {output_dir}/")
    print(f"Max switches: {np.nanmax(switches_grid)}")
    print(f"Max concentration: {np.nanmax(concentration_grid):.3f}")
    print(f"Min concentration: {np.nanmin(concentration_grid):.3f}")


if __name__ == "__main__":
    main()


