import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from numba import njit
from multiprocessing import Pool, cpu_count


@njit
def solve_tridiagonal(a, b, c, d):
    """Solve tridiagonal system Ax = d where A has diagonals a, b, c."""
    n = len(d)
    # Forward elimination
    for i in range(1, n):
        w = a[i-1] / b[i-1]
        b[i] = b[i] - w * c[i-1]
        d[i] = d[i] - w * d[i-1]
    
    # Back substitution
    x = np.zeros(n)
    x[n-1] = d[n-1] / b[n-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    
    return x


@njit
def run_simulation_final_only(t, x, dt, dx, D_N, D_S, boundary_N, g, N_species, mu):
    """Run simulation and return only final state for parameter sweeps."""
    
    n_points = len(x)
    
    # Pre-compute diffusion coefficients
    alpha_N = D_N * dt / dx**2
    alpha_S = D_S * dt / dx**2
    
    # Initialize arrays
    species = np.ones((N_species, n_points))
    # Normalize initial conditions so sum of species at each site = 1
    for j in range(n_points):
        site_total = np.sum(species[:, j])
        if site_total > 0:
            species[:, j] = species[:, j] / site_total
    
    nutrients = np.zeros(n_points)
    nutrients[0] = boundary_N  # Boundary condition at x=0
    nutrients[-1] = 0  # Boundary condition at x=L

    for time_idx in range(len(t)):
        
        # Calculate R_i(N) for each species at each position using current nutrients
        R_values = np.zeros((N_species, n_points))
        for i in range(N_species):
            R_values[i] = (g[i] * nutrients) / (10**((g[i] - 1) / 0.3) + nutrients)
        
        # Calculate reaction terms
        local_dilution = np.sum(R_values * species, axis=0)  # Sum over species at each position
        nutrient_consumption = np.sum(R_values * species, axis=0)
        
        # Update nutrients semi-implicitly: (I - dt*D_N*∇²)N^{n+1} = N^n - dt*consumption
        rhs_nutrients = nutrients - dt * nutrient_consumption
        
        # Set up tridiagonal system for nutrients (Dirichlet boundaries)
        a_N = np.full(n_points-1, -alpha_N)  # Lower diagonal
        b_N = np.full(n_points, 1 + 2*alpha_N)  # Main diagonal
        c_N = np.full(n_points-1, -alpha_N)  # Upper diagonal
        
        # Apply Dirichlet boundary conditions
        b_N[0] = 1.0
        c_N[0] = 0.0
        rhs_nutrients[0] = boundary_N
        
        b_N[-1] = 1.0
        a_N[-1] = 0.0
        rhs_nutrients[-1] = 0.0
        
        # Solve for new nutrients
        nutrients = solve_tridiagonal(a_N, b_N, c_N, rhs_nutrients)
        
        # Update each species semi-implicitly with mutation
        for i in range(N_species):
            # Mutation terms (no periodic boundary conditions)
            mutation_in = np.zeros(n_points)
            if i > 0:  # Can receive from species i-1
                mutation_in = mutation_in + 0.5 * mu * species[i-1]
            if i < N_species - 1:  # Can receive from species i+1
                mutation_in = mutation_in + 0.5 * mu * species[i+1]
            
            # Mutation out depends on how many neighbors this species has
            mutation_out_rate = 0.0
            if i > 0:  # Can mutate to species i-1
                mutation_out_rate = mutation_out_rate + 0.5 * mu
            if i < N_species - 1:  # Can mutate to species i+1
                mutation_out_rate = mutation_out_rate + 0.5 * mu
            
            mutation_out = mutation_out_rate * species[i]
            
            # Growth term using current values: R_i(N) * S_i - S_i * local_dilution
            growth = R_values[i] * species[i] - species[i] * local_dilution
            
            # Total reaction term: growth + mutation_in - mutation_out
            reaction = growth + mutation_in - mutation_out
            
            # (I - dt*D_S*∇²)S_i^{n+1} = S_i^n + dt*reaction
            rhs_species = species[i] + dt * reaction
            
            # Set up tridiagonal system for species (Neumann boundaries)
            a_S = np.full(n_points-1, -alpha_S)  # Lower diagonal
            b_S = np.full(n_points, 1 + 2*alpha_S)  # Main diagonal
            c_S = np.full(n_points-1, -alpha_S)  # Upper diagonal
            
            # Apply zero-flux boundary conditions (Neumann)
            b_S[0] = 1 + alpha_S  # Only right neighbor
            b_S[-1] = 1 + alpha_S  # Only left neighbor
            
            # Solve for new species
            species[i] = solve_tridiagonal(a_S, b_S, c_S, rhs_species)

    return species, nutrients


def simulate_parameter_set(params):
    """Wrapper function for parallel simulation."""
    D_N, D_S, x, t, dt, dx, boundary_N, g, N_species, mu = params
    
    final_species, final_nutrients = run_simulation_final_only(
        t, x, dt, dx, D_N, D_S, boundary_N, g, N_species, mu
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
    N_species = 50
    boundary_N = 1  # Boundary condition for nutrients at x=0
    mu = 0.01  # Mutation rate
    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 100000, 2000000)
    dt = t[1] - t[0]  # time step
    dx = x[1] - x[0]  # spatial step
    
    # Parameter ranges (log scale)
    D_N_values = np.geomspace(0.01, 1, 8)
    D_S_values = np.geomspace(0.00001, 1, 12)
    
    g = np.linspace(0.1, 0.8, N_species)
    
    # Create parameter combinations (no stability check needed for semi-implicit)
    param_combinations = []
    for D_N in D_N_values:
        for D_S in D_S_values:
            param_combinations.append((D_N, D_S, x, t, dt, dx, boundary_N, g, N_species, mu))
    
    print(f"Running {len(param_combinations)} parameter combinations...")
    print(f"Using {cpu_count()} CPU cores")
    print(f"Mutation rate: {mu}")
    
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
    output_dir = f"src/quantizedSpeciesPDE/convertBetweenSpecies/plots/constantMutations/parameter_sweep_mutations_mu{mu}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create combined plot with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot heatmap for number of majority switches
    im1 = ax1.imshow(switches_grid, aspect='auto', origin='lower', 
                     extent=[np.log10(D_S_values[0]), np.log10(D_S_values[-1]), 
                             np.log10(D_N_values[0]), np.log10(D_N_values[-1])],
                     cmap='viridis')
    cbar1 = plt.colorbar(im1, ax=ax1, label='Number of Majority Switches')
    ax1.set_xlabel('log₁₀(Species Diffusion Coefficient D_S)')
    ax1.set_ylabel('log₁₀(Nutrient Diffusion Coefficient D_N)')
    ax1.set_title(f'Number of Majority Species Switches Across Space (μ={mu})')
    
    # Plot heatmap for average majority concentration
    im2 = ax2.imshow(concentration_grid, aspect='auto', origin='lower',
                     extent=[np.log10(D_S_values[0]), np.log10(D_S_values[-1]), 
                             np.log10(D_N_values[0]), np.log10(D_N_values[-1])],
                     cmap='plasma')
    cbar2 = plt.colorbar(im2, ax=ax2, label='Average Majority Concentration')
    ax2.set_xlabel('log₁₀(Species Diffusion Coefficient D_S)')
    ax2.set_ylabel('log₁₀(Nutrient Diffusion Coefficient D_N)')
    ax2.set_title(f'Average Concentration of Majority Species (μ={mu})')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/N_{N_species}_switches_conc.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmaps saved to {output_dir}/")
    print(f"Max switches: {np.nanmax(switches_grid)}")
    print(f"Max concentration: {np.nanmax(concentration_grid):.3f}")
    print(f"Min concentration: {np.nanmin(concentration_grid):.3f}")


if __name__ == "__main__":
    main()


