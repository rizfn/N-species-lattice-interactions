#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <fstream>
#include <cmath>
#include <map>
#include <string>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <functional>
#include <filesystem>

constexpr int DEFAULT_N_SPECIES = 200;
constexpr int DEFAULT_N_CHEMICALS = 50;
constexpr int DEFAULT_K = 2;
constexpr int DEFAULT_N_STEPS = 1000000;
constexpr int DEFAULT_N_SIMS = 2;
constexpr double DEFAULT_THETA = 0.01;

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

void set_random_ICs(int N_SPECIES, int N_CHEMICALS, int n_records, std::vector<std::vector<double>> &Os, std::vector<std::vector<double>> &Ns, std::mt19937 &gen)
{
    std::uniform_real_distribution<> dis(0.0, 1.0);

    Os.assign(N_SPECIES, std::vector<double>(n_records, 0.0));
    Ns.assign(N_CHEMICALS, std::vector<double>(n_records, 0.0));

    std::vector<double> randomICs(N_SPECIES + 1);
    for (auto &val : randomICs)
        val = dis(gen);
    double sum = std::accumulate(randomICs.begin(), randomICs.end(), 0.0);
    for (int i = 0; i < N_SPECIES; ++i)
    {
        Os[i][0] = randomICs[i + 1] / sum;
    }
    for (int i = 0; i < N_CHEMICALS; ++i)
    {
        Ns[i][0] = 1.0;
    }
}

void ode_derivatives(const std::vector<double> &O_vals,
                     const std::vector<double> &N_vals,
                     const std::map<int, std::vector<int>> &J,
                     const std::map<int, std::vector<int>> &S,
                     double THETA,
                     int K,
                     std::vector<double> &d_Os,
                     std::vector<double> &d_Ns)
{
    std::fill(d_Os.begin(), d_Os.end(), 0.0);
    std::fill(d_Ns.begin(), d_Ns.end(), 0.0);

    double empty_frac = 1.0 - std::accumulate(O_vals.begin(), O_vals.end(), 0.0);

    for (int i = 0; i < O_vals.size(); ++i)
    {
        double nutrient_sum_O = 0.0;
        for (int j : J.at(i))
        {
            nutrient_sum_O += N_vals[j];
        }

        nutrient_sum_O /= K;

        d_Os[i] = O_vals[i] * empty_frac * nutrient_sum_O - THETA * O_vals[i];
        for (int j : J.at(i))
        {
            d_Ns[j] -= O_vals[i] * empty_frac * nutrient_sum_O;
        }
        for (int j : S.at(i))
        {
            d_Ns[j] += O_vals[i] * (1.0 - N_vals[j]);
        }
    }
}

void ode_integrate_rk4(int N_SPECIES,
                       int N_CHEMICALS,
                       const std::map<int, std::vector<int>> &J,
                       const std::map<int, std::vector<int>> &S,
                       double THETA,
                       int K,
                       int stoptime,
                       int nsteps,
                       int dataskip,
                       std::vector<double> &T,
                       std::vector<std::vector<double>> &Os,
                       std::vector<std::vector<double>> &Ns,
                       std::mt19937 &gen)
{
    double dt = static_cast<double>(stoptime) / nsteps;
    int n_records = nsteps / dataskip + 1;

    set_random_ICs(N_SPECIES, N_CHEMICALS, n_records, Os, Ns, gen);
    T.assign(n_records, 0.0);

    std::vector<double> current_Os(N_SPECIES), current_Ns(N_CHEMICALS);
    for (int i = 0; i < N_SPECIES; ++i)
        current_Os[i] = Os[i][0];
    for (int i = 0; i < N_CHEMICALS; ++i)
        current_Ns[i] = Ns[i][0];

    std::vector<double> k1_Os(N_SPECIES), k1_Ns(N_CHEMICALS);
    std::vector<double> k2_Os(N_SPECIES), k2_Ns(N_CHEMICALS);
    std::vector<double> k3_Os(N_SPECIES), k3_Ns(N_CHEMICALS);
    std::vector<double> k4_Os(N_SPECIES), k4_Ns(N_CHEMICALS);

    int record_idx = 1;

    for (int i = 0; i < nsteps; ++i)
    {
        ode_derivatives(current_Os, current_Ns, J, S, THETA, K, k1_Os, k1_Ns);

        std::vector<double> O_temp(N_SPECIES), N_temp(N_CHEMICALS);
        for (int j = 0; j < N_SPECIES; ++j)
            O_temp[j] = current_Os[j] + 0.5 * dt * k1_Os[j];
        for (int j = 0; j < N_CHEMICALS; ++j)
            N_temp[j] = current_Ns[j] + 0.5 * dt * k1_Ns[j];
        ode_derivatives(O_temp, N_temp, J, S, THETA, K, k2_Os, k2_Ns);

        for (int j = 0; j < N_SPECIES; ++j)
            O_temp[j] = current_Os[j] + 0.5 * dt * k2_Os[j];
        for (int j = 0; j < N_CHEMICALS; ++j)
            N_temp[j] = current_Ns[j] + 0.5 * dt * k2_Ns[j];
        ode_derivatives(O_temp, N_temp, J, S, THETA, K, k3_Os, k3_Ns);

        for (int j = 0; j < N_SPECIES; ++j)
            O_temp[j] = current_Os[j] + dt * k3_Os[j];
        for (int j = 0; j < N_CHEMICALS; ++j)
            N_temp[j] = current_Ns[j] + dt * k3_Ns[j];
        ode_derivatives(O_temp, N_temp, J, S, THETA, K, k4_Os, k4_Ns);

        for (int j = 0; j < N_SPECIES; ++j)
            current_Os[j] += (dt / 6.0) * (k1_Os[j] + 2.0 * k2_Os[j] + 2.0 * k3_Os[j] + k4_Os[j]);
        for (int j = 0; j < N_CHEMICALS; ++j)
            current_Ns[j] += (dt / 6.0) * (k1_Ns[j] + 2.0 * k2_Ns[j] + 2.0 * k3_Ns[j] + k4_Ns[j]);

        if (i % dataskip == 0)
        {
            for (int j = 0; j < N_SPECIES; ++j)
                Os[j][record_idx] = current_Os[j];
            for (int j = 0; j < N_CHEMICALS; ++j)
                Ns[j][record_idx] = current_Ns[j];
            T[record_idx] = T[record_idx - 1] + dataskip * dt;
            record_idx++;
        }
    }
}

void run_simulation(std::ofstream &file, int N_SPECIES, int N_CHEMICALS, double THETA, int K, int N_STEPS, int dataskip, const std::map<int, std::vector<int>> &J, const std::map<int, std::vector<int>> &S)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<double> T;
    std::vector<std::vector<double>> Os, Ns;
    ode_integrate_rk4(N_SPECIES, N_CHEMICALS, J, S, THETA, K, N_STEPS, N_STEPS, dataskip, T, Os, Ns, gen);

    if (file.is_open())
    {
        file << "step,";
        for (int i = 0; i < N_SPECIES; ++i)
        {
            file << "bacteria" << i + 1 << ",";
        }
        for (int i = 0; i < N_CHEMICALS; ++i)
        {
            file << "chemical" << i + 1 << ",";
        }
        file << "\n";
        for (int i = 0; i < T.size(); ++i)
        {
            file << T[i] << ",";
            for (int j = 0; j < N_SPECIES; ++j)
            {
                file << Os[j][i] << ",";
            }
            for (int j = 0; j < N_CHEMICALS; ++j)
            {
                file << Ns[j][i] << ",";
            }
            file << "\n";
        }
    }
}

int main(int argc, char *argv[])
{
    int N_SPECIES = DEFAULT_N_SPECIES;
    int N_CHEMICALS = DEFAULT_N_CHEMICALS;
    double THETA = DEFAULT_THETA;
    int K = DEFAULT_K;
    int N_STEPS = DEFAULT_N_STEPS;
    int N_sims = DEFAULT_N_SIMS;
    int dataskip = N_STEPS / 10000;

    if (argc > 1)
        N_SPECIES = std::stoi(argv[1]);
    if (argc > 2)
        N_CHEMICALS = std::stoi(argv[2]);
    if (argc > 3)
        THETA = std::stod(argv[3]);
    if (argc > 4)
        K = std::stoi(argv[4]);
    if (argc > 5)
        N_STEPS = std::stoi(argv[5]);
    if (argc > 6)
        N_sims = std::stoi(argv[6]);

    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();

    std::random_device rd;
    std::mt19937 gen(rd());
    auto [J, S] = init_J_S_maps(N_SPECIES, N_CHEMICALS, K, gen);

    std::vector<std::thread> threads;
    for (int sim = 0; sim < N_sims; ++sim)
    {
        threads.push_back(std::thread([sim, N_SPECIES, N_CHEMICALS, THETA, K, N_STEPS, dataskip, J, S, exeDir]()
        {
            std::ostringstream filePathStream;
            filePathStream << exeDir << "/sum_div_K/N_" << N_SPECIES << "-" << N_CHEMICALS << "_theta_" << THETA << "_K_" << K << "_" << std::setw(2) << std::setfill('0') << sim << ".csv";
            std::string filePath = filePathStream.str();

            std::ofstream file;
            file.open(filePath);
            run_simulation(file, N_SPECIES, N_CHEMICALS, THETA, K, N_STEPS, dataskip, J, S);
            file.close();
        }));
    }

    for (std::thread &th : threads)
    {
        th.join();
    }

    return 0;
}