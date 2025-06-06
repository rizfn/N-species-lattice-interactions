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
#include <regex>

#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

constexpr int EMPTY = 0;
constexpr int SOIL = 1;

constexpr int DEFAULT_STEPS_PER_LATTICEPOINT = 500;
constexpr double DEFAULT_THETA = 0.1;
constexpr int DEFAULT_L = 256;
constexpr int DEFAULT_N_SPECIES = 200;
constexpr int DEFAULT_N_CHEMICALS = 50;
constexpr int DEFAULT_K_IN = 2;
constexpr int DEFAULT_K_OUT = 4;
constexpr double DEFAULT_D = 0.1;
constexpr double DEFAULT_SIGMA = 1.0;

std::vector<std::vector<int>> load_lattice(const std::string &filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open lattice file: " + filepath);
    }

    std::vector<std::vector<int>> bacteria_lattice;
    std::string line;
    while (std::getline(file, line))
    {
        if (line.find("\t[") != std::string::npos)
        {
            std::string lattice_data = line.substr(line.find("\t[") + 1);
            std::stringstream ss(lattice_data);
            std::vector<int> row;
            char ch;
            int value;
            while (ss >> ch)
            {
                if (ch == '[' || ch == ',' || ch == ']')
                    continue;
                ss.putback(ch);
                ss >> value;
                row.push_back(value);
            }
            bacteria_lattice.push_back(row);
        }
    }
    file.close();
    return bacteria_lattice;
}

std::map<int, std::vector<int>> load_matrix(const std::string &filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open matrix file: " + filepath);
    }

    std::map<int, std::vector<int>> matrix;
    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        int key;
        char comma;
        ss >> key;
        std::vector<int> values;
        int value;
        while (ss >> comma >> value)
        {
            values.push_back(value);
        }
        matrix[key] = values;
    }
    file.close();
    return matrix;
}

std::vector<int> get_surviving_species(const std::vector<std::vector<int>> &bacteria_lattice)
{
    std::map<int, int> species_count;
    for (const auto &row : bacteria_lattice)
    {
        for (int cell : row)
        {
            if (cell > 1)
            { // Exclude EMPTY (0) and SOIL (1)
                species_count[cell]++;
            }
        }
    }
    std::vector<int> surviving_species;
    for (const auto &[species, count] : species_count)
    {
        surviving_species.push_back(species);
    }
    return surviving_species;
}

std::vector<int> init_bacteria(int N_SPECIES)
{
    std::vector<int> vec(N_SPECIES);
    for (int i = 0; i < N_SPECIES; ++i)
    {
        vec[i] = i + 2; // Bacteria indexed from 2
    }
    return vec;
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
                chemical_lattice[k][i][j] = 0.5f; // Initialize with float value 1.0
            }
        }
    }
    return chemical_lattice;
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

void run_simulation_with_species_removed(
    const std::vector<std::vector<int>> &initial_bacteria_lattice,
    const std::vector<std::vector<std::vector<float>>> &initial_chemical_lattice,
    const std::map<int, std::vector<int>> &J,
    const std::map<int, std::vector<int>> &S,
    const std::vector<int> &BACTERIA,
    int L, double THETA, double SIGMA, int K_IN, int K_OUT,
    int STEPS_PER_LATTICEPOINT, double D,
    const std::string &dirPath,
    const std::string &latticeFile, int run_idx, int species_to_remove)
{
    // Create a copy of the lattice and remove the specified species
    std::vector<std::vector<int>> bacteria_lattice = initial_bacteria_lattice;
    for (auto &row : bacteria_lattice)
    {
        for (auto &cell : row)
        {
            if (cell == species_to_remove)
            {
                cell = EMPTY;
            }
        }
    }

    std::vector<std::vector<std::vector<float>>> chemical_lattice = initial_chemical_lattice;

    for (long long step = 0; step <= STEPS_PER_LATTICEPOINT; ++step)
    {
        for (int i = 0; i < L * L; ++i)
        {
            update_bacteria(bacteria_lattice, chemical_lattice, J, S, BACTERIA, L, THETA, SIGMA, dis_l, dis_real, gen);
        }
        diffuseChemicals(chemical_lattice, L, D);

        latticeFile << step << "\t["; // Use \t as separator
        for (const auto &row : bacteria_lattice)
        {
            latticeFile << "[";
            for (int cell : row)
            {
                latticeFile << cell;
                if (&cell != &row.back()) // Check if it's not the last element in the row
                {
                    latticeFile << ",";
                }
            }
            latticeFile << "]";
            if (&row != &bacteria_lattice.back()) // Check if it's not the last row
            {
                latticeFile << ",";
            }
        }
        latticeFile << "]\n";
    }

    latticeFile.close();
}

void continue_simulation(int N_SPECIES, int N_CHEMICALS, int L, double THETA, double SIGMA, int K_IN, int K_OUT, double D, const std::string &dirPath)
{
    // Construct file paths
    std::ostringstream latticeFilePathStream, jFilePathStream, sFilePathStream;
    latticeFilePathStream << dirPath << "/lattice_0.tsv";
    jFilePathStream << dirPath << "/J_0.csv";
    sFilePathStream << dirPath << "/S_0.csv";

    std::string lattice_filepath = latticeFilePathStream.str();
    std::string J_filepath = jFilePathStream.str();
    std::string S_filepath = sFilePathStream.str();

    // Load the lattice and matrices
    std::vector<std::vector<int>> bacteria_lattice = load_lattice(lattice_filepath);
    std::map<int, std::vector<int>> J = load_matrix(J_filepath);
    std::map<int, std::vector<int>> S = load_matrix(S_filepath);

    const std::vector<int> BACTERIA = init_bacteria(N_SPECIES);
    // temp: todo: instead of regenning new chem lattice, load (or make a smart one based on secretions)
    std::vector<std::vector<std::vector<float>>> chemical_lattice = init_chemical_lattice(L, N_CHEMICALS)

        // Get surviving species
        std::vector<int>
            surviving_species = get_surviving_species(bacteria_lattice);

    // Run simulations for each surviving species with that species removed
    std::vector<std::thread> threads;
    for (int i = 0; i < surviving_species.size(); ++i)
    {
        int species_to_remove = surviving_species[i];
        threads.push_back(std::thread(
            run_simulation_with_species_removed,
            bacteria_lattice, J, S, BACTERIA, L, THETA, SIGMA, K_IN, K_OUT,
            DEFAULT_STEPS_PER_LATTICEPOINT, D, dirPath, i, species_to_remove));
    }

    // Join all threads
    for (auto &t : threads)
    {
        t.join();
    }
}

int main(int argc, char *argv[])
{
    // Default parameters
    int N_SPECIES = DEFAULT_N_SPECIES;
    int N_CHEMICALS = DEFAULT_N_CHEMICALS;
    int L = DEFAULT_L;
    double THETA = DEFAULT_THETA;
    int K_IN = DEFAULT_K_IN;
    int K_OUT = DEFAULT_K_OUT;
    double D = DEFAULT_D;
    double SIGMA = DEFAULT_SIGMA;

    // Override defaults with command-line arguments
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
        D = std::stod(argv[7]);
    if (argc > 8)
        SIGMA = std::stod(argv[8]);



    // Construct directory path
    std::ostringstream dirPathStream;
    dirPathStream << "outputs/lattice2D/N_" << N_SPECIES << "-" << N_CHEMICALS
                  << "_L_" << L << "_theta_" << THETA << "_K_IN_" << K_IN
                  << "_K_OUT_" << K_OUT << "_D_" << D << "_sigma_" << SIGMA;
    std::string dirPath = dirPathStream.str();

    try
    {
        continue_simulation(N_SPECIES, N_CHEMICALS, L, THETA, SIGMA, K_IN, K_OUT, D, dirPath);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}