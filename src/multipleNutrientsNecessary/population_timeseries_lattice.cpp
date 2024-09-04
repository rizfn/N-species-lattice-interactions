#include <iostream>
#include <vector>
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

constexpr int DEFAULT_STEPS_PER_LATTICEPOINT = 1000;
constexpr double DEFAULT_THETA = 0.1;
constexpr int DEFAULT_L = 512; // side length of the square lattice
constexpr int DEFAULT_N = 3;      // number of species
constexpr int DEFAULT_N_RUNS = 4;  // number of parallel runs
constexpr int DEFAULT_K = 2;        // number of necessary resources

constexpr int EMPTY = 0;

std::vector<int> init_bacteria(int N)
{
    std::vector<int> vec(N);
    for (int i = 0; i < N; ++i)
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

std::vector<std::vector<int>> init_bacteria_lattice(int L, int N, std::mt19937 &gen)
{
    std::uniform_int_distribution<> dis_site(0, N);
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

std::vector<std::vector<std::vector<bool>>> init_chemical_lattice(int L, int N)
{
    std::vector<std::vector<std::vector<bool>>> chemical_lattice(N, std::vector<std::vector<bool>>(L, std::vector<bool>(L)));
    for (int k = 0; k < N; ++k)
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

std::vector<std::vector<bool>> init_Jmatrix(int N, int K, std::mt19937 &gen)
{
    std::vector<std::vector<bool>> J(N, std::vector<bool>(N, false));
    for (int i = 0; i < N; ++i)
    {
        std::vector<int> possible_indices(N);
        std::iota(possible_indices.begin(), possible_indices.end(), 0); // Fill with 0, 1, ..., N-1
        possible_indices.erase(possible_indices.begin() + i);           // Remove the diagonal element

        std::shuffle(possible_indices.begin(), possible_indices.end(), gen); // Shuffle the indices

        for (int j = 0; j < K; ++j)
        {
            J[i][possible_indices[j]] = true;
        }
    }
    return J;
}

void update(std::vector<std::vector<int>> &bacteria_lattice,
            std::vector<std::vector<std::vector<bool>>> &chemical_lattice,
            std::vector<std::vector<bool>> &J,
            const std::vector<int> &BACTERIA,
            int L,
            int N,
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
            chemical_lattice[i][new_site.first][new_site.second] = true;
            // check if the new site is a empty
            if (new_site_value == EMPTY)
            {
                std::vector<int> requiredChemicals;
                for (int j = 0; j < N; ++j)
                {
                    if (J[i][j])
                    {
                        requiredChemicals.push_back(j);
                    }
                }
                bool canReproduce = true;
                for (int chemical : requiredChemicals)
                {
                    if (!chemical_lattice[chemical][new_site.first][new_site.second])
                    {
                        canReproduce = false;
                        break;
                    }
                }
                if (canReproduce)
                {
                    for (int chemical : requiredChemicals)
                    {
                        chemical_lattice[chemical][new_site.first][new_site.second] = false;
                    }
                    bacteria_lattice[site.first][site.second] = BACTERIA[i];
                }
            }
        }
    }
}

void run(std::ofstream &file, int L, int N, double THETA, int K, int STEPS_PER_LATTICEPOINT)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_l(0, L - 1);
    std::uniform_real_distribution<> dis_real(0.0, 1.0);

    std::vector<std::vector<int>> bacteria_lattice = init_bacteria_lattice(L, N, gen);
    std::vector<std::vector<std::vector<bool>>> chemical_lattice = init_chemical_lattice(L, N);
    std::vector<std::vector<bool>> J = init_Jmatrix(N, K, gen);
    std::vector<int> BACTERIA = init_bacteria(N);

    int i = 0; // indexing for recording steps

    for (long long step = 0; step <= STEPS_PER_LATTICEPOINT; ++step)
    {
        for (int i = 0; i < L * L; ++i)
        {
            update(bacteria_lattice, chemical_lattice, J, BACTERIA, L, N, THETA, dis_l, dis_real, gen);
        }
        std::vector<int> counts(N + 1, 0);
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
    int N = DEFAULT_N;
    int L = DEFAULT_L;
    double THETA = DEFAULT_THETA;
    int K = DEFAULT_K;
    int STEPS_PER_LATTICEPOINT = DEFAULT_STEPS_PER_LATTICEPOINT;
    int N_runs = DEFAULT_N_RUNS;

    if (argc > 1)
        N = std::stoi(argv[1]);
    if (argc > 2)
        L = std::stoi(argv[2]);
    if (argc > 3)
        THETA = std::stod(argv[3]);
    if (argc > 4)
        K = std::stoi(argv[4]);
    if (argc > 5)
        STEPS_PER_LATTICEPOINT = std::stoi(argv[5]);
    if (argc > 6)
        N_runs = std::stoi(argv[6]);

    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();

    std::vector<std::thread> threads;

    for (int run_idx = 0; run_idx < N_runs; ++run_idx)
    {
        threads.push_back(std::thread([run_idx, N, L, THETA, K, STEPS_PER_LATTICEPOINT, exeDir]()
                                      {
                std::ostringstream filePathStream;

                filePathStream << exeDir << "/outputs/populationTimeseriesLattice/N_" << N << "_L_" << L << "_theta_" << THETA << "_K_" << K << "_" << run_idx << ".csv";
                std::string filePath = filePathStream.str();

                std::ofstream file;
                file.open(filePath);
                file << "step,empty,";
                for (int i = 1; i <= N; ++i)
                {
                    file << "bacteria" << i;
                    if (i != N)
                    {
                        file << ",";
                    }
                }
                file << "\n";
                run(file, L, N, THETA, K, STEPS_PER_LATTICEPOINT);
                file.close(); }));
    }

    for (std::thread &t : threads)
    {
        t.join();
    }

    return 0;
}