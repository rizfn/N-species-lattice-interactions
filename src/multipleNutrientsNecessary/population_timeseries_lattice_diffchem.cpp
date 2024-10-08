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

constexpr int DEFAULT_STEPS_PER_LATTICEPOINT = 100000;
constexpr double DEFAULT_THETA = 0.1;
constexpr int DEFAULT_L = 1024;          // side length of the square lattice
constexpr int DEFAULT_N_SPECIES = 200;   // number of species
constexpr int DEFAULT_N_CHEMICALS = 50; // number of chemicals
constexpr int DEFAULT_N_RUNS = 4;       // number of parallel runs
constexpr int DEFAULT_K = 2;            // number of necessary resources  (NOTE: LIMITS ON K DUE TO MUTUALLY EXCLUSIVE INTAKE/OUTTAKE CHEMICALS)

constexpr int EMPTY = 0;

std::vector<int> init_bacteria(int N_SPECIES)
{
    std::vector<int> vec(N_SPECIES);
    for (int i = 0; i < N_SPECIES; ++i)
    {
        vec[i] = i + 1;
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
    std::uniform_int_distribution<> dis_site(0, N_SPECIES);
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

std::vector<std::vector<std::vector<bool>>> init_chemical_lattice(int L, int N_CHEMICALS)
{
    std::vector<std::vector<std::vector<bool>>> chemical_lattice(N_CHEMICALS, std::vector<std::vector<bool>>(L, std::vector<bool>(L)));
    for (int k = 0; k < N_CHEMICALS; ++k)
    {
        for (int i = 0; i < L; ++i)
        {
            for (int j = 0; j < L; ++j)
            {
                chemical_lattice[k][i][j] = 1;
            }
        }
    }
    return chemical_lattice;
}

std::pair<std::map<int, std::vector<int>>, std::map<int, std::vector<int>>> init_J_S_maps(int N_SPECIES, int N_CHEMICALS, int K, std::mt19937 &gen)
{
    std::map<int, std::vector<int>> J;
    std::map<int, std::vector<int>> S;

    for (int i = 0; i < N_SPECIES; ++i)
    {
        std::vector<int> possible_indices(N_CHEMICALS);
        std::iota(possible_indices.begin(), possible_indices.end(), 0); // Fill with 0, 1, ..., N_CHEMICALS-1

        std::shuffle(possible_indices.begin(), possible_indices.end(), gen); // Shuffle the indices

        J[i] = std::vector<int>(possible_indices.begin(), possible_indices.begin() + K);
        S[i] = std::vector<int>(possible_indices.begin() + K, possible_indices.begin() + 2 * K);
    }

    return {J, S};
}

void update(std::vector<std::vector<int>> &bacteria_lattice,
            std::vector<std::vector<std::vector<bool>>> &chemical_lattice,
            std::map<int, std::vector<int>> &J,
            std::map<int, std::vector<int>> &S,
            const std::vector<int> &BACTERIA,
            int L,
            double THETA,
            std::uniform_int_distribution<> &dis_l,
            std::uniform_real_distribution<> &dis_real,
            std::mt19937 &gen)
{
    // select a random site
    std::pair<int, int> site = {dis_l(gen), dis_l(gen)};

    int site_value = bacteria_lattice[site.first][site.second];

    if (site_value != EMPTY)
    {
        int i = site_value - 1; // bacteria index

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

            // Secretion: set chemicals to true for all chemicals that bacteria i can secrete
            for (int chemical : S[i])
            {
                chemical_lattice[chemical][new_site.first][new_site.second] = true;
            }

            // check if the new site is empty
            if (new_site_value == EMPTY)
            {
                bool canReproduce = true;
                // Check if all required chemicals are available
                for (int chemical : J[i])
                {
                    if (!chemical_lattice[chemical][new_site.first][new_site.second])
                    {
                        canReproduce = false;
                        break;
                    }
                }
                if (canReproduce)
                {
                    // Consume the required chemicals
                    for (int chemical : J[i])
                    {
                        chemical_lattice[chemical][new_site.first][new_site.second] = false;
                    }
                    // Replicate the bacteria
                    bacteria_lattice[site.first][site.second] = BACTERIA[i];
                }
            }
        }
    }
}

void run(std::ofstream &file, int L, int N_SPECIES, int N_CHEMICALS, double THETA, int K, int STEPS_PER_LATTICEPOINT)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_l(0, L - 1);
    std::uniform_real_distribution<> dis_real(0.0, 1.0);

    std::vector<std::vector<int>> bacteria_lattice = init_bacteria_lattice(L, N_SPECIES, gen);
    std::vector<std::vector<std::vector<bool>>> chemical_lattice = init_chemical_lattice(L, N_CHEMICALS);
    auto [J, S] = init_J_S_maps(N_SPECIES, N_CHEMICALS, K, gen);
    std::vector<int> BACTERIA = init_bacteria(N_SPECIES);

    int i = 0; // indexing for recording steps

    for (long long step = 0; step <= STEPS_PER_LATTICEPOINT; ++step)
    {
        for (int i = 0; i < L * L; ++i)
        {
            update(bacteria_lattice, chemical_lattice, J, S, BACTERIA, L, THETA, dis_l, dis_real, gen);
        }
        std::vector<int> counts(N_SPECIES + 1, 0);
        for (const auto &row : bacteria_lattice)
        {
            for (int cell : row)
            {
                ++counts[cell];
            }
        }
        file << step;
        for (int j = 0; j < counts.size(); ++j)
        {
            file << "," << static_cast<double>(counts[j]) / (L * L);
        }
        file << "\n";
    }
}

int main(int argc, char *argv[])
{
    int N_SPECIES = DEFAULT_N_SPECIES;
    int N_CHEMICALS = DEFAULT_N_CHEMICALS;
    int L = DEFAULT_L;
    double THETA = DEFAULT_THETA;
    int K = DEFAULT_K;
    int STEPS_PER_LATTICEPOINT = DEFAULT_STEPS_PER_LATTICEPOINT;
    int N_runs = DEFAULT_N_RUNS;

    if (argc > 1)
        N_SPECIES = std::stoi(argv[1]);
    if (argc > 2)
        N_CHEMICALS = std::stoi(argv[2]);
    if (argc > 3)
        L = std::stoi(argv[3]);
    if (argc > 4)
        THETA = std::stod(argv[4]);
    if (argc > 5)
        K = std::stoi(argv[5]);
    if (argc > 6)
        STEPS_PER_LATTICEPOINT = std::stoi(argv[6]);
    if (argc > 7)
        N_runs = std::stoi(argv[7]);

    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();

    std::vector<std::thread> threads;

    for (int run_idx = 0; run_idx < N_runs; ++run_idx)
    {
        threads.push_back(std::thread([run_idx, N_SPECIES, N_CHEMICALS, L, THETA, K, STEPS_PER_LATTICEPOINT, exeDir]()
                                      {
                std::ostringstream filePathStream;

                filePathStream << exeDir << "/outputs/populationTimeseriesLatticeDiffchem/N_" << N_SPECIES << "-" << N_CHEMICALS << "_L_" << L << "_theta_" << THETA << "_K_" << K << "_" << run_idx << ".csv";
                std::string filePath = filePathStream.str();

                std::ofstream file;
                file.open(filePath);
                file << "step,empty,";
                for (int i = 1; i <= N_SPECIES; ++i)
                {
                    file << "bacteria" << i;
                    if (i != N_SPECIES)
                    {
                        file << ",";
                    }
                }
                file << "\n";
                run(file, L, N_SPECIES, N_CHEMICALS, THETA, K, STEPS_PER_LATTICEPOINT);
                file.close(); }));
    }

    for (std::thread &t : threads)
    {
        t.join();
    }

    return 0;
}