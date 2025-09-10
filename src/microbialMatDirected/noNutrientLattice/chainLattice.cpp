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
constexpr int N = 4; // number of species/nutrients
constexpr int STEPS_PER_LATTICEPOINT = 10000;
constexpr double THETA = 0.01; // death probability
constexpr int L = 256;         // lattice size
constexpr int FINAL_STEPS_TO_RECORD = 500;
constexpr int RECORDING_SKIP = 10;

// States: 0=empty, 1-N=nutrients, N+1-2N=bacteria
constexpr int EMPTY = 0;

std::random_device rd;
std::mt19937 gen(rd());

std::pair<int, int> get_random_neighbour(std::pair<int, int> c)
{
    std::uniform_int_distribution<> dis(0, 1);
    int coord_changing = dis(gen);
    int change = 2 * dis(gen) - 1;
    
    if (coord_changing == 0) // i-direction (rows/y): no wrapping, open boundaries
    {
        int new_i = c.first + change;
        if (new_i >= 0 && new_i < L)
        {
            c.first = new_i;
        }
        // If out of bounds, don't move
    }
    else // j-direction (columns/x): periodic wrapping
    {
        c.second = (c.second + change + L) % L;
    }
    return c;
}

std::vector<std::vector<int>> init_lattice()
{
    std::uniform_int_distribution<> dis_site(0, 2 * N); // 0=empty, 1-N=nutrients, N+1-2N=bacteria
    std::vector<std::vector<int>> lattice(L, std::vector<int>(L));
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            lattice[i][j] = dis_site(gen);
        }
    }
    return lattice;
}

void supply_nutrient_from_top(std::vector<std::vector<int>> &lattice)
{
    // Supply nutrient 0 (nutrient 1 in 1-indexed) from the top row (i=0)
    for (int j = 0; j < L; ++j)
    {
        int current_value = lattice[0][j];
        // Only replace if it's empty
        if (current_value == EMPTY)
        {
            lattice[0][j] = 1; // nutrient 0 (1-indexed)
        }
    }
}

void update_bacteria(std::vector<std::vector<int>> &lattice)
{
    std::uniform_int_distribution<> dis_l(0, L - 1);
    std::uniform_real_distribution<> dis_real(0.0, 1.0);

    // Select a random site
    std::pair<int, int> site = {dis_l(gen), dis_l(gen)};
    int site_value = lattice[site.first][site.second];

    // Check if site contains a bacterium (N+1 to 2N)
    if (site_value > N && site_value <= 2 * N)
    {
        int species_id = site_value - N - 1; // Convert to 0-indexed species

        // Check for death
        if (dis_real(gen) < THETA)
        {
            lattice[site.first][site.second] = EMPTY;
        }
        else
        {
            // Move to a neighbor
            std::pair<int, int> new_site = get_random_neighbour(site);
            int new_site_value = lattice[new_site.first][new_site.second];

            // Move the bacterium
            lattice[new_site.first][new_site.second] = site_value;
            
            // Check what was at the new site and respond accordingly
            if (new_site_value == species_id + 1) // Consumed its required nutrient (1-indexed)
            {
                // Reproduce behind (at original site)
                lattice[site.first][site.second] = site_value;
            }
            else if (new_site_value == EMPTY)
            {
                // Leave nutrient behind (species i produces nutrient i+1, but last species produces nothing)
                if (species_id < N - 1)
                {
                    lattice[site.first][site.second] = species_id + 2; // nutrient i+1 (1-indexed)
                }
                else
                {
                    lattice[site.first][site.second] = EMPTY; // Last species doesn't produce nutrient
                }
            }
            else if (new_site_value > N && new_site_value <= 2 * N)
            {
                // Swapped with another bacterium
                lattice[site.first][site.second] = new_site_value;
            }
            else
            {
                // Moved onto a different nutrient or other state - just leave empty behind
                lattice[site.first][site.second] = EMPTY;
            }
        }
    }
}

void run(std::ofstream &file)
{
    std::vector<std::vector<int>> lattice = init_lattice();

    const int recording_step = STEPS_PER_LATTICEPOINT - FINAL_STEPS_TO_RECORD;

    for (long long step = 0; step <= STEPS_PER_LATTICEPOINT; ++step)
    {
        // Supply nutrient from top
        supply_nutrient_from_top(lattice);

        // Update bacteria L*L times per step
        for (int i = 0; i < L * L; ++i)
        {
            update_bacteria(lattice);
        }

        // Record lattice snapshots in final steps
        if (step >= recording_step && step % RECORDING_SKIP == 0)
        {
            file << step << "\t[";
            for (const auto &row : lattice)
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
                if (&row != &lattice.back())
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
    filePathStream << exeDir << "/outputs/lattice2D/N_" << N << "_L_" << L
                   << "_theta_" << THETA << ".tsv";
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