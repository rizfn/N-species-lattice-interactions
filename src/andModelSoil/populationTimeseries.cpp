#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <thread>
#include <filesystem>

#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

static auto _ = []()
{std::ios_base::sync_with_stdio(false);std::cin.tie(nullptr);std::cout.tie(nullptr);return 0; }();

constexpr int DEFAULT_STEPS_PER_LATTICEPOINT = 10000;
constexpr double DEFAULT_THETA = 0.1;
constexpr int DEFAULT_L = 256;          // side length of the square lattice
constexpr int DEFAULT_N_SPECIES = 200;  // number of species
constexpr int DEFAULT_N_CHEMICALS = 50; // number of chemicals
constexpr int DEFAULT_N_RUNS = 4;       // number of parallel runs
constexpr int DEFAULT_K_IN = 2;            // number of necessary resources  (NOTE: LIMITS ON K DUE TO MUTUALLY EXCLUSIVE INTAKE/OUTTAKE CHEMICALS)
constexpr int DEFAULT_K_OUT = 4;           // number of secreted resources  (NOTE: LIMITS ON K DUE TO MUTUALLY EXCLUSIVE INTAKE/OUTTAKE CHEMICALS)
constexpr double DEFAULT_D = 0.1;      // diffusion constant
constexpr double DEFAULT_SIGMA = 1.0;  // soil filling rate

constexpr int RECORDING_SKIP = 10;

constexpr int EMPTY = 0;
constexpr int SOIL = 1;

std::vector<int> init_bacteria(int N_SPECIES)
{
    std::vector<int> vec(N_SPECIES);
    for (int i = 0; i < N_SPECIES; ++i)
    {
        vec[i] = i + 2; // Bacteria indexed from 2
    }
    return vec;
}

std::pair<int, int> get_random_neighbour(std::pair<int, int> c, int L, std::mt19937 &gen)
{
    std::uniform_int_distribution<> dis(0, 1);
    // choose a random coordinate to change
    int coord_changing = dis(gen);
    // choose a random direction to change the coordinate
    int change = 2 * dis(gen) - 1;
    // change the coordinate
    c.first = (coord_changing == 0) ? (c.first + change + L) % L : c.first;
    c.second = (coord_changing == 1) ? (c.second + change + L) % L : c.second;

    return c;
}

std::vector<std::vector<int>> init_bacteria_lattice(int L, int N_SPECIES, std::mt19937 &gen)
{
    std::uniform_int_distribution<> dis_bacteria(2, N_SPECIES + 1); // Bacteria indexed from 2
    std::vector<std::vector<int>> bacteria_lattice(L, std::vector<int>(L));

    int total_sites = L * L;
    int soil_sites = total_sites / 3; // Approximately one-third of the sites as soil
    int bacteria_sites = total_sites - soil_sites;

    // Create a vector with the desired number of soil and bacteria sites
    std::vector<int> sites(total_sites, 0);
    std::fill(sites.begin(), sites.begin() + soil_sites, SOIL);
    std::generate(sites.begin() + soil_sites, sites.end(), [&]() { return dis_bacteria(gen); });

    // Shuffle the sites to randomize their positions
    std::shuffle(sites.begin(), sites.end(), gen);

    // Fill the lattice with the shuffled sites
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            bacteria_lattice[i][j] = sites[i * L + j];
        }
    }

    return bacteria_lattice;
}

std::vector<std::vector<std::vector<float>>> init_chemical_lattice(int L, int N_CHEMICALS)
{
    std::vector<std::vector<std::vector<float>>> chemical_lattice(N_CHEMICALS, std::vector<std::vector<float>>(L, std::vector<float>(L)));
    for (int k = 0; k < N_CHEMICALS; ++k)
    {
        for (int i = 0; i < L; ++i)
        {
            for (int j = 0; j < L; ++j)
            {
                chemical_lattice[k][i][j] = 1.0f; // Initialize with float value 1.0
            }
        }
    }
    return chemical_lattice;
}

std::pair<std::map<int, std::vector<int>>, std::map<int, std::vector<int>>> init_J_S_maps(int N_SPECIES, int N_CHEMICALS, int K_IN, int K_OUT, std::mt19937 &gen)
{
    std::map<int, std::vector<int>> J;
    std::map<int, std::vector<int>> S;

    for (int i = 0; i < N_SPECIES; ++i)
    {
        std::vector<int> possible_indices(N_CHEMICALS);
        std::iota(possible_indices.begin(), possible_indices.end(), 0); // Fill with 0, 1, ..., N_CHEMICALS-1

        std::shuffle(possible_indices.begin(), possible_indices.end(), gen); // Shuffle the indices

        J[i] = std::vector<int>(possible_indices.begin(), possible_indices.begin() + K_IN);
        S[i] = std::vector<int>(possible_indices.begin() + K_IN, possible_indices.begin() + K_IN + K_OUT);
    }

    return {J, S};
}

void diffuseChemicals(std::vector<std::vector<std::vector<float>>> &chemicalLattice, int L, float D)
{
    // Create a new lattice to store the updated chemical concentrations
    std::vector<std::vector<std::vector<float>>> newChemicalLattice(chemicalLattice.size(), std::vector<std::vector<float>>(L, std::vector<float>(L, 0.0f)));

    for (int k = 0; k < chemicalLattice.size(); ++k) // Iterate over each chemical lattice
    {
        for (int i = 0; i < L; ++i) // Iterate over each point in the lattice
        {
            for (int j = 0; j < L; ++j)
            {
                // Get the concentrations from the rolled copies
                float up = chemicalLattice[k][(i - 1 + L) % L][j];
                float down = chemicalLattice[k][(i + 1) % L][j];
                float left = chemicalLattice[k][i][(j - 1 + L) % L];
                float right = chemicalLattice[k][i][(j + 1) % L];

                // Apply the diffusion formula
                newChemicalLattice[k][i][j] = (1 - D) * chemicalLattice[k][i][j] + D * (up + down + left + right) / 4.0f;
            }
        }
    }
    // Update the original lattice with the new values
    chemicalLattice = newChemicalLattice;
}

void update_bacteria(std::vector<std::vector<int>> &bacteria_lattice,
                     std::vector<std::vector<std::vector<float>>> &chemical_lattice,
                     std::map<int, std::vector<int>> &J,
                     std::map<int, std::vector<int>> &S,
                     const std::vector<int> &BACTERIA,
                     int L,
                     double THETA,
                     double SIGMA,
                     std::uniform_int_distribution<> &dis_l,
                     std::uniform_real_distribution<> &dis_real,
                     std::mt19937 &gen)
{
    // select a random site
    std::pair<int, int> site = {dis_l(gen), dis_l(gen)};

    int site_value = bacteria_lattice[site.first][site.second];

    if (site_value == EMPTY)
    {
        std::pair<int, int> new_site = get_random_neighbour(site, L, gen);
        int new_site_value = bacteria_lattice[new_site.first][new_site.second];
        if (new_site_value == SOIL && dis_real(gen) < SIGMA)
        {
            bacteria_lattice[site.first][site.second] = SOIL;
        }
    }

    else if (site_value != SOIL)
    {
        int i = site_value - 2; // bacteria index

        // check for death
        if (dis_real(gen) < THETA)
        {
            bacteria_lattice[site.first][site.second] = EMPTY;
        }
        else
        {
            // move into a neighbour
            std::pair<int, int> new_site = get_random_neighbour(site, L, gen);
            // check the value of the new site
            int new_site_value = bacteria_lattice[new_site.first][new_site.second];
            // move the bacteria
            bacteria_lattice[new_site.first][new_site.second] = site_value;
            bacteria_lattice[site.first][site.second] = new_site_value;

            // Secretion: set chemicals to 1 for all chemicals that bacteria i can secrete
            for (int chemical : S[i])
            {
                chemical_lattice[chemical][new_site.first][new_site.second] = 1.0f;
            }

            // check if the new site is soil
            if (new_site_value == SOIL)
            {
                // Check if the new site has all the required chemicals
                float min_chemical_concentration = 1.0f; // Start with the maximum possible value
                for (int chemical : J[i])
                {
                    float concentration = chemical_lattice[chemical][new_site.first][new_site.second];
                    if (concentration == 0.0f)
                    {
                        min_chemical_concentration = 0.0f;
                        break;
                    }
                    min_chemical_concentration = std::min(min_chemical_concentration, concentration);
                }

                // Set the replication probability to the minimum chemical concentration
                float replicationProbability = min_chemical_concentration;

                // Consume the chemicals
                for (int chemical : J[i])
                {
                    chemical_lattice[chemical][new_site.first][new_site.second] -= min_chemical_concentration;
                }

                if (dis_real(gen) < replicationProbability)
                {
                    bacteria_lattice[site.first][site.second] = BACTERIA[i];
                }
            }
        }
    }
}

void save_matrix(const std::map<int, std::vector<int>> &matrix, const std::string &filename)
{
    std::ofstream file(filename);
    for (const auto &pair : matrix)
    {
        file << pair.first;
        for (int value : pair.second)
        {
            file << "," << value;
        }
        file << "\n";
    }
    file.close();
}

void run(int L, int N_SPECIES, int N_CHEMICALS, double THETA, double SIGMA, int K_IN, int K_OUT, int STEPS_PER_LATTICEPOINT, double D, const std::string &dirPath, int run_idx)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_l(0, L - 1);
    std::uniform_real_distribution<> dis_real(0.0, 1.0);

    std::vector<std::vector<int>> bacteria_lattice = init_bacteria_lattice(L, N_SPECIES, gen);
    std::vector<std::vector<std::vector<float>>> chemical_lattice = init_chemical_lattice(L, N_CHEMICALS);
    auto [J, S] = init_J_S_maps(N_SPECIES, N_CHEMICALS, K_IN, K_OUT, gen);
    std::vector<int> BACTERIA = init_bacteria(N_SPECIES);

    // Save J and S matrices
    std::ostringstream jFilePathStream, sFilePathStream;
    jFilePathStream << dirPath << "/J_" << run_idx << ".csv";
    sFilePathStream << dirPath << "/S_" << run_idx << ".csv";
    save_matrix(J, jFilePathStream.str());
    save_matrix(S, sFilePathStream.str());

    // Open timeseries file
    std::ostringstream timeseriesFilePathStream;
    timeseriesFilePathStream << dirPath << "/timeseries_" << run_idx << ".csv";
    std::ofstream timeseriesFile(timeseriesFilePathStream.str());
    timeseriesFile << "step,empty,soil,";
    for (int i = 2; i <= N_SPECIES + 1; ++i)
    {
        timeseriesFile << "bacteria" << i - 1;
        if (i != N_SPECIES + 1)
        {
            timeseriesFile << ",";
        }
    }
    timeseriesFile << "\n";

    int i = 0; // indexing for recording steps

    for (long long step = 0; step <= STEPS_PER_LATTICEPOINT; ++step)
    {
        for (int i = 0; i < L * L; ++i)
        {
            update_bacteria(bacteria_lattice, chemical_lattice, J, S, BACTERIA, L, THETA, SIGMA, dis_l, dis_real, gen);
        }
        diffuseChemicals(chemical_lattice, L, D);

        if (step % RECORDING_SKIP == 0)
        {
            std::vector<int> counts(N_SPECIES + 2, 0);
            for (const auto &row : bacteria_lattice)
            {
                for (int cell : row)
                {
                    ++counts[cell];
                }
            }
            timeseriesFile << step;
            for (int j = 0; j < counts.size(); ++j)
            {
                timeseriesFile << "," << static_cast<double>(counts[j]) / (L * L);
            }
            timeseriesFile << "\n";
        }
    }

    timeseriesFile.close();
}


int main(int argc, char *argv[])
{
    int N_SPECIES = DEFAULT_N_SPECIES;
    int N_CHEMICALS = DEFAULT_N_CHEMICALS;
    int L = DEFAULT_L;
    double THETA = DEFAULT_THETA;
    int K_IN = DEFAULT_K_IN;
    int K_OUT = DEFAULT_K_OUT;
    int STEPS_PER_LATTICEPOINT = DEFAULT_STEPS_PER_LATTICEPOINT;
    int N_runs = DEFAULT_N_RUNS;
    double D = DEFAULT_D;
    double SIGMA = DEFAULT_SIGMA;

    if (argc > 1)
        N_SPECIES = std::stoi(argv[1]);
    if (argc > 2)
        N_CHEMICALS = std::stoi(argv[2]);
    if (argc > 3)
        L = std::stoi(argv[3]);
    if (argc > 4)
        THETA = std::stod(argv[4]);
    if (argc > 5)
        K_IN = std::stoi(argv[5]);
    if (argc > 6)
        K_OUT = std::stoi(argv[6]);
    if (argc > 7)
        STEPS_PER_LATTICEPOINT = std::stoi(argv[7]);
    if (argc > 8)
        N_runs = std::stoi(argv[8]);
    if (argc > 9)
        D = std::stod(argv[9]);
    if (argc > 10)
        SIGMA = std::stod(argv[10]);

    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();

    std::vector<std::thread> threads;

    for (int run_idx = 0; run_idx < N_runs; ++run_idx)
    {
        threads.push_back(std::thread([run_idx, N_SPECIES, N_CHEMICALS, L, THETA, SIGMA, D, K_IN, K_OUT, STEPS_PER_LATTICEPOINT, exeDir]()
                                      {
                std::ostringstream dirPathStream;
                dirPathStream << exeDir << "/outputs/timeseries/N_" << N_SPECIES << "-" << N_CHEMICALS << "_L_" << L << "_theta_" << THETA << "_K_IN_" << K_IN << "_K_OUT_" << K_OUT << "_D_" << D << "_sigma_" << SIGMA;
                std::string dirPath = dirPathStream.str();
                std::filesystem::create_directories(dirPath);

                run(L, N_SPECIES, N_CHEMICALS, THETA, SIGMA, K_IN, K_OUT, STEPS_PER_LATTICEPOINT, D, dirPath, run_idx); }));
    }

    for (std::thread &t : threads)
    {
        t.join();
    }

    return 0;
}