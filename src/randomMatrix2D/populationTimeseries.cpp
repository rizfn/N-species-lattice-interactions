#include <iostream>
#include <vector>
#include <array>
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

std::random_device rd;
std::mt19937 gen(rd());

// Define a struct for coordinates
struct Coordinate
{
    int x;
    int y;
};

// Define constants
constexpr int STEPS_PER_LATTICEPOINT = 10000;
constexpr double THETA = 0.2;
constexpr int L = 512; // side length of the square lattice
constexpr int N = 50;  // number of species

constexpr std::array<int, N> RESOURCES = []
{
    std::array<int, N> arr{};
    for (int i = 0; i < N; ++i)
    {
        arr[i] = i;
    }
    return arr;
}();
constexpr std::array<int, N> BACTERIA = []
{
    std::array<int, N> arr{};
    for (int i = 0; i < N; ++i)
    {
        arr[i] = i + N;
    }
    return arr;
}();

// Define distributions
std::uniform_int_distribution<> dis(0, 1);
std::uniform_int_distribution<> dis_site(0, 2 * N - 1);
std::uniform_int_distribution<> dis_l(0, L - 1);
std::uniform_real_distribution<> dis_real(0.0, 1.0);

Coordinate get_random_neighbour(Coordinate c)
{
    // choose a random coordinate to change
    int coord_changing = dis(gen);
    // choose a random direction to change the coordinate
    int change = 2 * dis(gen) - 1;
    // change the coordinate
    c.x = (coord_changing == 0) ? (c.x + change + L) % L : c.x;
    c.y = (coord_changing == 1) ? (c.y + change + L) % L : c.y;

    return c;
}

std::vector<std::vector<int>> init_lattice()
{
    std::vector<std::vector<int>> soil_lattice(L, std::vector<int>(L)); // Changed to 2D vector
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            soil_lattice[i][j] = dis_site(gen);
        }
    }
    return soil_lattice;
}

std::vector<std::vector<float>> init_Jmatrix()
{
    std::vector<std::vector<float>> J(N, std::vector<float>(N, 0.0));
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (i != j)
            {
                J[i][j] = dis_real(gen);
            }
        }
    }
    return J;
}

void update(std::vector<std::vector<int>> &soil_lattice, std::vector<std::vector<float>> &J)
{
    // select a random site
    Coordinate site = {dis_l(gen), dis_l(gen)};

    for (int i = 0; i < N; ++i)
    {
        if (soil_lattice[site.x][site.y] == BACTERIA[i])
        { // bacteria of species i
            // check for death
            if (dis_real(gen) < THETA)
            {
                soil_lattice[site.x][site.y] = RESOURCES[i];
            }
            else
            {
                // move into a neighbour
                Coordinate new_site = get_random_neighbour(site);
                // check the value of the new site
                int new_site_value = soil_lattice[new_site.x][new_site.y];
                // move the bacteria
                soil_lattice[new_site.x][new_site.y] = BACTERIA[i];
                soil_lattice[site.x][site.y] = RESOURCES[i];
                // check if the new site is a nutrient that this bacteria can consume
                if (std::find(RESOURCES.begin(), RESOURCES.end(), new_site_value) != RESOURCES.end())
                {
                    // check if the bacteria can consume the nutrient
                    if (dis_real(gen) < J[i][new_site_value])
                    {
                        soil_lattice[site.x][site.y] = BACTERIA[i];
                    }
                }
                // check if the new site is a bacteria
                else
                {
                    // keep both with worms (undo the nutrient space in original site)
                    soil_lattice[site.x][site.y] = new_site_value;
                }
            }
        }
    }
}

void run(std::ofstream &file)
{
    std::vector<std::vector<int>> soil_lattice = init_lattice();

    std::vector<std::vector<float>> J = init_Jmatrix();

    int i = 0; // indexing for recording steps

    for (long long step = 0; step <= STEPS_PER_LATTICEPOINT; ++step)
    {
        for (int i = 0; i < L * L; ++i)
        {
            update(soil_lattice, J);
        }
        std::vector<int> counts(2 * N, 0);
        for (const auto &row : soil_lattice)
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

        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / STEPS_PER_LATTICEPOINT * 100 << "%\r" << std::flush;
    }
}

int main(int argc, char *argv[])
{
    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::ostringstream filePathStream;
    filePathStream << exeDir << "/outputs/populationTimeseries/N_" << N << "_L_" << L << "_theta_" << THETA << ".csv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    file << "step,";
    for (int i = 1; i <= N; ++i)
    {
        file << "resource" << i << ",";
    }
    for (int i = 1; i <= N; ++i)
    {
        file << "bacteria" << i;
        if (i != N)
        {
            file << ",";
        }
    }
    file << "\n";
    run(file);
    file.close();

    return 0;
}