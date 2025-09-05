#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>
#include <filesystem>

#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

static auto _ = []()
{std::ios_base::sync_with_stdio(false);std::cin.tie(nullptr);std::cout.tie(nullptr);return 0; }();

// Constants
constexpr int N = 3; // number of species/nutrients
constexpr int STEPS_PER_LATTICEPOINT = 10000;
constexpr double THETA = 0.01; // death probability
constexpr int L = 256;         // lattice size
constexpr float D = 0.25f;      // diffusion constant
constexpr float DRIFT = 0.5f;   // vertical drift parameter (0=no drift, 1=full downward flow)
constexpr float S = 1.0f;      // boundary condition (nutrient 0 supply rate)
constexpr int FINAL_STEPS_TO_RECORD = 500;
constexpr int RECORDING_SKIP = 10;
constexpr int EMPTY = 0;

std::random_device rd;
std::mt19937 gen(rd());

std::pair<int, int> get_chemotactic_neighbour(std::pair<int, int> c, int species_id, 
                                             const std::vector<std::vector<std::vector<float>>> &chemical_lattice,
                                             std::mt19937 &gen)
{
    // Fixed array of 4 possible moves: up, down, left, right
    std::pair<int, int> neighbors[4] = {
        {c.first - 1, c.second},           // up
        {c.first + 1, c.second},           // down  
        {c.first, (c.second - 1 + L) % L}, // left (periodic)
        {c.first, (c.second + 1) % L}      // right (periodic)
    };
    
    float weights[4];
    
    // Calculate weights for each direction
    for (int i = 0; i < 4; ++i)
    {
        int new_i = neighbors[i].first;
        int new_j = neighbors[i].second;
        
        // Set weight to 0 if out of vertical bounds, otherwise use chemical concentration
        if (new_i < 0 || new_i >= L)
        {
            weights[i] = 0.0f;
        }
        else
        {
            weights[i] = chemical_lattice[species_id][new_i][new_j];
        }
    }
    
    // Calculate total weight
    float total_weight = weights[0] + weights[1] + weights[2] + weights[3];
    
    // If no nutrients anywhere, move randomly among valid neighbors
    if (total_weight == 0.0f)
    {
        // Count valid neighbors and select randomly
        int valid_count = 0;
        int valid_indices[4];
        
        for (int i = 0; i < 4; ++i)
        {
            if (neighbors[i].first >= 0 && neighbors[i].first < L)
            {
                valid_indices[valid_count] = i;
                valid_count++;
            }
        }
        
        std::uniform_int_distribution<> dis(0, valid_count - 1);
        int selected_idx = valid_indices[dis(gen)];
        return neighbors[selected_idx];
    }
    
    // Select neighbor based on weighted probabilities
    std::uniform_real_distribution<> dis(0.0, total_weight);
    float random_value = dis(gen);
    
    float cumulative_weight = 0.0f;
    for (int i = 0; i < 4; ++i)
    {
        cumulative_weight += weights[i];
        if (random_value <= cumulative_weight)
        {
            return neighbors[i];
        }
    }
    
    // Fallback - return right (should not reach here)
    return neighbors[3];
}

std::vector<std::vector<int>> init_bacteria_lattice()
{
    std::uniform_int_distribution<> dis_site(0, N); // 0=empty, 1-5=bacteria species
    std::vector<std::vector<int>> bacteria_lattice(L, std::vector<int>(L));
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            bacteria_lattice[i][j] = dis_site(gen);
        }
    }
    return bacteria_lattice;
}

std::vector<std::vector<std::vector<float>>> init_chemical_lattice()
{
    std::vector<std::vector<std::vector<float>>> chemical_lattice(N, std::vector<std::vector<float>>(L, std::vector<float>(L, 0.0f)));
    // Initialize with some base concentrations
    for (int k = 0; k < N; ++k)
    {
        for (int i = 0; i < L; ++i)
        {
            for (int j = 0; j < L; ++j)
            {
                chemical_lattice[k][i][j] = 0.1f;
            }
        }
    }
    return chemical_lattice;
}

void supply_nutrient_from_top(std::vector<std::vector<std::vector<float>>> &chemical_lattice)
{
    // Constantly supply nutrient 0 from the top row (i=0)
    for (int j = 0; j < L; ++j)
    {
        chemical_lattice[0][0][j] = S;
    }
}

void diffuse_chemicals(std::vector<std::vector<std::vector<float>>> &chemical_lattice)
{
    // Step 1: Isotropic diffusion
    std::vector<std::vector<std::vector<float>>> new_chemical_lattice(N, std::vector<std::vector<float>>(L, std::vector<float>(L, 0.0f)));

    for (int k = 0; k < N; ++k)
    {
        for (int i = 0; i < L; ++i)
        {
            for (int j = 0; j < L; ++j)
            {
                // j-direction (horizontal/x): periodic boundary conditions
                float left = chemical_lattice[k][i][(j - 1 + L) % L];
                float right = chemical_lattice[k][i][(j + 1) % L];
                
                // i-direction (vertical/y): no-flux boundary conditions
                float up = (i > 0) ? chemical_lattice[k][i - 1][j] : chemical_lattice[k][i][j];
                float down = (i < L - 1) ? chemical_lattice[k][i + 1][j] : chemical_lattice[k][i][j];

                // Standard isotropic diffusion
                new_chemical_lattice[k][i][j] = (1 - D) * chemical_lattice[k][i][j] + D * (up + down + left + right) / 4.0f;
            }
        }
    }
    
    // Step 2: Apply downward drift (chemicals flow OUT at bottom)
    for (int k = 0; k < N; ++k)
    {
        for (int i = L - 1; i >= 0; --i) // Process ALL rows including bottom
        {
            for (int j = 0; j < L; ++j)
            {
                // Calculate how much chemical to move down
                float available_chemical = new_chemical_lattice[k][i][j];
                float drift_amount = std::min(available_chemical, DRIFT);
                
                // Remove drift_amount from current cell
                new_chemical_lattice[k][i][j] -= drift_amount;
                
                // Only add to cell below if we're not at the bottom boundary
                if (i < L - 1)
                {
                    new_chemical_lattice[k][i + 1][j] += drift_amount;
                }
                // If i == L-1 (bottom row), drift_amount flows out of system (lost)
            }
        }
    }
    
    chemical_lattice = new_chemical_lattice;
}

void update_bacteria(std::vector<std::vector<int>> &bacteria_lattice,
                     std::vector<std::vector<std::vector<float>>> &chemical_lattice)
{
    std::uniform_int_distribution<> dis_l(0, L - 1);
    std::uniform_real_distribution<> dis_real(0.0, 1.0);

    // Select a random site
    std::pair<int, int> site = {dis_l(gen), dis_l(gen)};
    int site_value = bacteria_lattice[site.first][site.second];

    if (site_value != EMPTY)
    {
        int species_id = site_value - 1;

        // Check for death
        if (dis_real(gen) < THETA)
        {
            bacteria_lattice[site.first][site.second] = EMPTY;
        }
        else
        {
            // Move to a neighbor using chemotaxis
            std::pair<int, int> new_site = get_chemotactic_neighbour(site, species_id, chemical_lattice, gen);
            int new_site_value = bacteria_lattice[new_site.first][new_site.second];

            // Move the bacteria
            bacteria_lattice[new_site.first][new_site.second] = site_value;
            bacteria_lattice[site.first][site.second] = new_site_value;

            // Check if new site is empty for potential replication
            if (new_site_value == EMPTY)
            {
                // Check if bacteria can consume its required nutrient (species i eats nutrient i)
                float nutrient_concentration = chemical_lattice[species_id][new_site.first][new_site.second];
                
                if (nutrient_concentration > 0.0f && dis_real(gen) < nutrient_concentration)
                {
                    // Consume the nutrient and replicate (no secretion when reproducing)
                    chemical_lattice[species_id][new_site.first][new_site.second] = 0.0f;
                    bacteria_lattice[site.first][site.second] = site_value; // replicate
                }
                else
                {
                    // No reproduction, but can still consume and secrete for energy
                    float consumed_amount = chemical_lattice[species_id][new_site.first][new_site.second];
                    chemical_lattice[species_id][new_site.first][new_site.second] = 0.0f;
                    
                    // Secretion: species i secretes nutrient i+1 (if i < N-1) with same amount consumed
                    if (species_id < N - 1 && consumed_amount > 0.0f)
                    {
                        chemical_lattice[species_id + 1][new_site.first][new_site.second] += consumed_amount;
                    }
                }
            }
            else
            {
                // Moving to occupied site - consume and secrete for energy only
                float consumed_amount = chemical_lattice[species_id][new_site.first][new_site.second];
                chemical_lattice[species_id][new_site.first][new_site.second] = 0.0f;
                
                // Secretion: species i secretes nutrient i+1 (if i < N-1) with same amount consumed
                if (species_id < N - 1 && consumed_amount > 0.0f)
                {
                    chemical_lattice[species_id + 1][new_site.first][new_site.second] += consumed_amount;
                }
            }
        }
    }
}

void run(std::ofstream &file)
{
    std::vector<std::vector<int>> bacteria_lattice = init_bacteria_lattice();
    std::vector<std::vector<std::vector<float>>> chemical_lattice = init_chemical_lattice();

    const int recording_step = STEPS_PER_LATTICEPOINT - FINAL_STEPS_TO_RECORD;

    for (long long step = 0; step <= STEPS_PER_LATTICEPOINT; ++step)
    {
        // Supply nutrient 0 from top
        supply_nutrient_from_top(chemical_lattice);

        // Update bacteria L*L times per step
        for (int i = 0; i < L * L; ++i)
        {
            update_bacteria(bacteria_lattice, chemical_lattice);
        }

        // Diffuse chemicals
        diffuse_chemicals(chemical_lattice);

        // Record lattice snapshots in final steps
        if (step >= recording_step and step % RECORDING_SKIP == 0)
        {
            file << step << "\t[";
            for (const auto &row : bacteria_lattice)
            {
                file << "[";
                for (int cell : row)
                {
                    file << cell;
                    if (&cell != &row.back())
                    {
                        file << ",";
                    }
                }
                file << "]";
                if (&row != &bacteria_lattice.back())
                {
                    file << ",";
                }
            }
            file << "]\n";
        }

        std::cout << "Progress: " << std::fixed << std::setprecision(2)
                  << static_cast<double>(step) / STEPS_PER_LATTICEPOINT * 100 << "%\r" << std::flush;
    }
}

int main(int argc, char *argv[])
{
    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::ostringstream filePathStream;
    filePathStream << exeDir << "/outputs/chainChemotaxisFlowLattice/N_" << N << "_L_" << L
                   << "_theta_" << THETA << "_D_" << D << "_v_" << DRIFT << "_S_" << S << ".tsv";
    std::string filePath = filePathStream.str();

    // Create output directory if it doesn't exist
    std::filesystem::create_directories(std::filesystem::path(filePath).parent_path());

    std::ofstream file;
    file.open(filePath);
    file << "step\tlattice\n";
    run(file);
    file.close();

    return 0;
}