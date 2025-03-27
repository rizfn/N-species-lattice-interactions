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

constexpr int DEFAULT_STEPS_PER_LATTICEPOINT = 10000;
constexpr double DEFAULT_THETA = 0.1;
constexpr int DEFAULT_L = 256;
constexpr int DEFAULT_N_SPECIES = 200;
constexpr int DEFAULT_N_CHEMICALS = 50;
constexpr int DEFAULT_K_IN = 2;
constexpr int DEFAULT_K_OUT = 4;
constexpr double DEFAULT_D = 0.1;
constexpr double DEFAULT_SIGMA = 1.0;

std::vector<std::vector<int>> load_lattice(const std::string &filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open lattice file: " + filepath);
    }

    std::vector<std::vector<int>> lattice;
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("\t[") != std::string::npos) {
            std::string lattice_data = line.substr(line.find("\t[") + 1);
            std::stringstream ss(lattice_data);
            std::vector<int> row;
            char ch;
            int value;
            while (ss >> ch) {
                if (ch == '[' || ch == ',' || ch == ']') continue;
                ss.putback(ch);
                ss >> value;
                row.push_back(value);
            }
            lattice.push_back(row);
        }
    }
    file.close();
    return lattice;
}

std::map<int, std::vector<int>> load_matrix(const std::string &filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open matrix file: " + filepath);
    }

    std::map<int, std::vector<int>> matrix;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        int key;
        char comma;
        ss >> key;
        std::vector<int> values;
        int value;
        while (ss >> comma >> value) {
            values.push_back(value);
        }
        matrix[key] = values;
    }
    file.close();
    return matrix;
}

std::vector<int> get_surviving_species(const std::vector<std::vector<int>> &lattice) {
    std::map<int, int> species_count;
    for (const auto &row : lattice) {
        for (int cell : row) {
            if (cell > 1) { // Exclude EMPTY (0) and SOIL (1)
                species_count[cell]++;
            }
        }
    }
    std::vector<int> surviving_species;
    for (const auto &[species, count] : species_count) {
        surviving_species.push_back(species);
    }
    return surviving_species;
}

void run_simulation_with_species_removed(
    const std::vector<std::vector<int>> &initial_lattice,
    const std::map<int, std::vector<int>> &J,
    const std::map<int, std::vector<int>> &S,
    int L, double THETA, double SIGMA, int K_IN, int K_OUT,
    int STEPS_PER_LATTICEPOINT, double D,
    const std::string &dirPath, int run_idx, int species_to_remove
) {
    // Create a copy of the lattice and remove the specified species
    std::vector<std::vector<int>> lattice = initial_lattice;
    for (auto &row : lattice) {
        for (auto &cell : row) {
            if (cell == species_to_remove) {
                cell = EMPTY;
            }
        }
    }

    // Run the simulation (placeholder for actual simulation logic)
    std::cout << "Running simulation for species " << species_to_remove << " removed, run index " << run_idx << std::endl;
}

void continue_simulation(int N_SPECIES, int N_CHEMICALS, int L, double THETA, double SIGMA, int K_IN, int K_OUT, double D, const std::string &dirPath) {
    // Construct file paths
    std::ostringstream latticeFilePathStream, jFilePathStream, sFilePathStream;
    latticeFilePathStream << dirPath << "/lattice_0.tsv";
    jFilePathStream << dirPath << "/J_0.csv";
    sFilePathStream << dirPath << "/S_0.csv";

    std::string lattice_filepath = latticeFilePathStream.str();
    std::string J_filepath = jFilePathStream.str();
    std::string S_filepath = sFilePathStream.str();

    // Load the lattice and matrices
    std::vector<std::vector<int>> lattice = load_lattice(lattice_filepath);
    std::map<int, std::vector<int>> J = load_matrix(J_filepath);
    std::map<int, std::vector<int>> S = load_matrix(S_filepath);

    // Get surviving species
    std::vector<int> surviving_species = get_surviving_species(lattice);

    // Run simulations for each surviving species with that species removed
    std::vector<std::thread> threads;
    for (int i = 0; i < surviving_species.size(); ++i) {
        int species_to_remove = surviving_species[i];
        threads.push_back(std::thread(
            run_simulation_with_species_removed,
            lattice, J, S, L, THETA, SIGMA, K_IN, K_OUT,
            DEFAULT_STEPS_PER_LATTICEPOINT, D, dirPath, i, species_to_remove
        ));
    }

    // Join all threads
    for (auto &t : threads) {
        t.join();
    }
}

int main(int argc, char *argv[]) {
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
    if (argc > 1) N_SPECIES = std::stoi(argv[1]);
    if (argc > 2) N_CHEMICALS = std::stoi(argv[2]);
    if (argc > 3) L = std::stoi(argv[3]);
    if (argc > 4) THETA = std::stod(argv[4]);
    if (argc > 5) K_IN = std::stoi(argv[5]);
    if (argc > 6) K_OUT = std::stoi(argv[6]);
    if (argc > 7) D = std::stod(argv[7]);
    if (argc > 8) SIGMA = std::stod(argv[8]);

    // Construct directory path
    std::ostringstream dirPathStream;
    dirPathStream << "outputs/lattice2D/N_" << N_SPECIES << "-" << N_CHEMICALS
                  << "_L_" << L << "_theta_" << THETA << "_K_IN_" << K_IN
                  << "_K_OUT_" << K_OUT << "_D_" << D << "_sigma_" << SIGMA;
    std::string dirPath = dirPathStream.str();

    try {
        continue_simulation(N_SPECIES, N_CHEMICALS, L, THETA, SIGMA, K_IN, K_OUT, D, dirPath);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}