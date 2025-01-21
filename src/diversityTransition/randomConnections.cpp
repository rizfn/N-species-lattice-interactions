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

constexpr int DEFAULT_N_RESOURCES = 10;
constexpr int DEFAULT_N_CHEMICALS = 100;
constexpr int DEFAULT_N_STEPS = 10000000;
constexpr int DEFAULT_STOPTIME = 100000;
constexpr int DEFAULT_N_SIMS = 4;
constexpr double DEFAULT_GROWTH_RATE = 10;
constexpr double REACTION_RATE_MIN = 0.5;
constexpr double REACTION_RATE_MAX = 1.0;
constexpr double SPARSITY = 0.8;

void set_random_ICs(int N_RESOURCES, int N_CHEMICALS, std::vector<double> &S_vals, std::vector<double> &X_vals, std::mt19937 &gen)
{
    std::uniform_real_distribution<> dis(0.0, 1.0);

    S_vals.assign(N_RESOURCES, 0.0);
    X_vals.assign(N_CHEMICALS, 0.0);

    for (int i = 0; i < N_RESOURCES; ++i)
    {
        S_vals[i] = dis(gen);
    }

    std::vector<double> randomICs(N_CHEMICALS);
    for (auto &val : randomICs)
        val = dis(gen);
    double sum = std::accumulate(randomICs.begin(), randomICs.end(), 0.0);
    for (int i = 0; i < N_CHEMICALS; ++i)
    {
        X_vals[i] = randomICs[i] / sum;
    }
}

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<double>>> init_connectivity_and_reaction_rates(int N_RESOURCES, int N_CHEMICALS, double sparsity, std::mt19937 &gen)
{
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_int_distribution<> int_dis(0, N_RESOURCES - 1);

    std::vector<std::vector<int>> connectivity(N_CHEMICALS, std::vector<int>(N_CHEMICALS, -1));
    std::vector<std::vector<double>> reaction_rates(N_CHEMICALS, std::vector<double>(N_CHEMICALS, 0.0));

    for (int i = 0; i < N_CHEMICALS; ++i)
    {
        for (int j = 0; j < N_CHEMICALS; ++j)
        {
            if (dis(gen) < sparsity)
            {
                connectivity[i][j] = int_dis(gen);
                reaction_rates[i][j] = REACTION_RATE_MIN + (REACTION_RATE_MAX - REACTION_RATE_MIN) * dis(gen);
            }
        }
    }

    return {connectivity, reaction_rates};
}

void ode_derivatives(const std::vector<double> &S_vals,
                     const std::vector<double> &X_vals,
                     const std::vector<std::vector<int>> &connectivity,
                     const std::vector<std::vector<double>> &reaction_rates,
                     double growth_rate,
                     std::vector<double> &d_S,
                     std::vector<double> &d_X)
{
    std::fill(d_S.begin(), d_S.end(), growth_rate * (1.0 - std::accumulate(S_vals.begin(), S_vals.end(), 0.0)));
    std::fill(d_X.begin(), d_X.end(), 0.0);

    for (int j = 0; j < X_vals.size(); ++j)
    {
        for (int k = 0; k < X_vals.size(); ++k)
        {
            int resource_index = connectivity[j][k];
            if (resource_index >= 0)
            {
                double reaction_term = reaction_rates[j][k] * X_vals[j] * X_vals[k] * S_vals[resource_index];
                d_S[resource_index] -= reaction_term;
                d_X[k] += reaction_term;
            }
        }
    }
}

void ode_integrate_rk4(int N_RESOURCES,
                       int N_CHEMICALS,
                       const std::vector<std::vector<int>> &connectivity,
                       const std::vector<std::vector<double>> &reaction_rates,
                       double growth_rate,
                       int stoptime,
                       int nsteps,
                       int dataskip,
                       std::ofstream &file,
                       std::mt19937 &gen)
{
    double dt = static_cast<double>(stoptime) / nsteps;

    std::vector<double> current_S(N_RESOURCES), current_X(N_CHEMICALS);
    set_random_ICs(N_RESOURCES, N_CHEMICALS, current_S, current_X, gen);

    std::vector<double> k1_S(N_RESOURCES), k1_X(N_CHEMICALS);
    std::vector<double> k2_S(N_RESOURCES), k2_X(N_CHEMICALS);
    std::vector<double> k3_S(N_RESOURCES), k3_X(N_CHEMICALS);
    std::vector<double> k4_S(N_RESOURCES), k4_X(N_CHEMICALS);

    if (file.is_open())
    {
        file << "time,";
        for (int i = 0; i < N_RESOURCES; ++i)
        {
            file << "resource" << i + 1 << ",";
        }
        for (int i = 0; i < N_CHEMICALS; ++i)
        {
            file << "chemical" << i + 1 << ",";
        }
        file << "\n";
    }

    for (int i = 0; i < nsteps; ++i)
    {
        ode_derivatives(current_S, current_X, connectivity, reaction_rates, growth_rate, k1_S, k1_X);

        std::vector<double> S_temp(N_RESOURCES), X_temp(N_CHEMICALS);
        for (int j = 0; j < N_RESOURCES; ++j)
            S_temp[j] = current_S[j] + 0.5 * dt * k1_S[j];
        for (int j = 0; j < N_CHEMICALS; ++j)
            X_temp[j] = current_X[j] + 0.5 * dt * k1_X[j];
        ode_derivatives(S_temp, X_temp, connectivity, reaction_rates, growth_rate, k2_S, k2_X);

        for (int j = 0; j < N_RESOURCES; ++j)
            S_temp[j] = current_S[j] + 0.5 * dt * k2_S[j];
        for (int j = 0; j < N_CHEMICALS; ++j)
            X_temp[j] = current_X[j] + 0.5 * dt * k2_X[j];
        ode_derivatives(S_temp, X_temp, connectivity, reaction_rates, growth_rate, k3_S, k3_X);

        for (int j = 0; j < N_RESOURCES; ++j)
            S_temp[j] = current_S[j] + dt * k3_S[j];
        for (int j = 0; j < N_CHEMICALS; ++j)
            X_temp[j] = current_X[j] + dt * k3_X[j];
        ode_derivatives(S_temp, X_temp, connectivity, reaction_rates, growth_rate, k4_S, k4_X);

        for (int j = 0; j < N_RESOURCES; ++j)
        {
            current_S[j] += (dt / 6.0) * (k1_S[j] + 2.0 * k2_S[j] + 2.0 * k3_S[j] + k4_S[j]);
        }
        for (int j = 0; j < N_CHEMICALS; ++j)
        {
            current_X[j] += (dt / 6.0) * (k1_X[j] + 2.0 * k2_X[j] + 2.0 * k3_X[j] + k4_X[j]);
        }

        double sum_X = std::accumulate(current_X.begin(), current_X.end(), 0.0);
        for (int j = 0; j < N_CHEMICALS; ++j)
            current_X[j] /= sum_X;

        if (i % dataskip == 0 && file.is_open())
        {
            double current_time = i * dt;
            file << current_time << ",";
            for (int j = 0; j < N_RESOURCES; ++j)
            {
                file << current_S[j] << ",";
            }
            for (int j = 0; j < N_CHEMICALS; ++j)
            {
                file << current_X[j] << ",";
            }
            file << "\n";
        }
    }
}

void run_simulation(std::ofstream &file, int N_RESOURCES, int N_CHEMICALS, double growth_rate, int stoptime, int N_STEPS, int dataskip, const std::vector<std::vector<int>> &connectivity, const std::vector<std::vector<double>> &reaction_rates)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    ode_integrate_rk4(N_RESOURCES, N_CHEMICALS, connectivity, reaction_rates, growth_rate, stoptime, N_STEPS, dataskip, file, gen);
}

int main(int argc, char *argv[])
{
    int N_RESOURCES = DEFAULT_N_RESOURCES;
    int N_CHEMICALS = DEFAULT_N_CHEMICALS;
    double growth_rate = DEFAULT_GROWTH_RATE;
    int N_STEPS = DEFAULT_N_STEPS;
    int N_sims = DEFAULT_N_SIMS;
    int stoptime = DEFAULT_STOPTIME;
    int dataskip = N_STEPS / 10000;

    if (argc > 1)
        N_RESOURCES = std::stoi(argv[1]);
    if (argc > 2)
        N_CHEMICALS = std::stoi(argv[2]);
    if (argc > 3)
        growth_rate = std::stod(argv[3]);
    if (argc > 4)
        stoptime = std::stoi(argv[4]);
    if (argc > 5)
        N_STEPS = std::stoi(argv[5]);
    if (argc > 6)
        N_sims = std::stoi(argv[6]);

    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();

    std::random_device rd;
    std::mt19937 gen(rd());
    auto [connectivity, reaction_rates] = init_connectivity_and_reaction_rates(N_RESOURCES, N_CHEMICALS, SPARSITY, gen);

    std::vector<std::thread> threads;
    for (int sim = 0; sim < N_sims; ++sim)
    {
        threads.push_back(std::thread([sim, N_RESOURCES, N_CHEMICALS, growth_rate, stoptime, N_STEPS, dataskip, connectivity, reaction_rates, exeDir]()
        {
            std::ostringstream filePathStream;
            filePathStream << exeDir << "/outputs/randomConnections/N_" << N_RESOURCES << "-" << N_CHEMICALS << "_gamma_" << growth_rate << "_sparsity_" << SPARSITY << "_" << std::setw(2) << std::setfill('0') << sim << ".csv";
            std::string filePath = filePathStream.str();

            std::ofstream file;
            file.open(filePath);
            run_simulation(file, N_RESOURCES, N_CHEMICALS, growth_rate, stoptime, N_STEPS, dataskip, connectivity, reaction_rates);
            file.close();
        }));
    }

    for (std::thread &th : threads)
    {
        th.join();
    }

    return 0;
}