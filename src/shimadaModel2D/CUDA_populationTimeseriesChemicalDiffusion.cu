#include <cuda.h>
#include <curand_kernel.h>
#include <random>
#include <vector>
#include <array>
#include <numeric>
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>

// Define constants
constexpr int DEFAULT_N_SPECIES = 200;
constexpr int DEFAULT_N_CHEMICALS = 50;
constexpr double DEFAULT_THETA = 0.1;
constexpr int DEFAULT_L = 1024;
constexpr int DEFAULT_K = 2;
constexpr int DEFAULT_STEPS_PER_LATTICEPOINT = 100000;
constexpr float DEFAULT_D = 1.0f;
constexpr int BLOCK_LENGTH = 4;
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

std::vector<int> init_bacteria_lattice(int L, int N_SPECIES, std::mt19937 &gen)
{
    std::uniform_int_distribution<> dis_site(0, N_SPECIES);
    std::vector<int> bacteria_lattice(L * L);
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            bacteria_lattice[i * L + j] = dis_site(gen);
        }
    }
    return bacteria_lattice;
}

std::vector<float> init_chemical_lattice(int L, int N_CHEMICALS)
{
    std::vector<float> chemical_lattice(N_CHEMICALS * L * L, 1.0f);
    return chemical_lattice;
}

std::pair<std::vector<int>, std::vector<int>> init_J_S_maps(int N_SPECIES, int N_CHEMICALS, int K, std::mt19937 &gen)
{
    std::vector<int> J(N_SPECIES * K);
    std::vector<int> S(N_SPECIES * K);

    for (int i = 0; i < N_SPECIES; ++i)
    {
        std::vector<int> possible_indices(N_CHEMICALS);
        std::iota(possible_indices.begin(), possible_indices.end(), 0); // Fill with 0, 1, ..., N_CHEMICALS-1

        std::shuffle(possible_indices.begin(), possible_indices.end(), gen); // Shuffle the indices

        for (int j = 0; j < K; ++j)
        {
            J[i * K + j] = possible_indices[j];
            S[i * K + j] = possible_indices[K + j];
        }
    }

    return {J, S};
}

__global__ void initCurand(curandState *state, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the unique index for the thread
    int index = idx + idy * blockDim.x * gridDim.x;

    // Initialize the RNG state for this thread
    curand_init(seed, index, 0, &state[index]);
}

__device__ void getRandomNeighbor(int x, int y, int L, curandState *localState, int *nbr_x, int *nbr_y)
{
    if (curand_uniform(localState) < 0.5f)
    {
        *nbr_x = (x + (curand_uniform(localState) < 0.5f ? -1 : 1) + L) % L;
        *nbr_y = y;
    }
    else
    {
        *nbr_x = x;
        *nbr_y = (y + (curand_uniform(localState) < 0.5f ? -1 : 1) + L) % L;
    }
}

__global__ void updateKernel(int *d_bacteria_lattice, float *d_chemical_lattice, curandState *state, int *d_J, int *d_S, int L, double THETA, int K, int offsetX, int offsetY)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the unique index for the thread
    int index = idx + idy * blockDim.x * gridDim.x;

    // Initialize the RNG
    curandState localState = state[index]; // Copy the state to local memory for efficiency

    int square_x = (blockIdx.x * BLOCK_LENGTH + offsetX * BLOCK_LENGTH / 2 + threadIdx.x) % L;
    int square_y = (blockIdx.y * BLOCK_LENGTH + offsetY * BLOCK_LENGTH / 2 + threadIdx.y) % L;

    // Select a random site in the 4x4 square
    int site_x = square_x + curand(&localState) % BLOCK_LENGTH / 2;
    int site_y = square_y + curand(&localState) % BLOCK_LENGTH / 2;

    // Get the value at the selected site
    int site_value = d_bacteria_lattice[site_x * L + site_y];

    if (site_value != EMPTY)
    {
        int i = site_value - 1; // bacteria index

        // check for death
        if (curand_uniform(&localState) < THETA)
        {
            d_bacteria_lattice[site_x * L + site_y] = EMPTY;
        }
        else
        {
            // move into a neighbour
            int new_site_x, new_site_y;
            getRandomNeighbor(site_x, site_y, L, &localState, &new_site_x, &new_site_y);
            int new_site_value = d_bacteria_lattice[new_site_x * L + new_site_y];

            // move the bacteria
            d_bacteria_lattice[new_site_x * L + new_site_y] = site_value;
            d_bacteria_lattice[site_x * L + site_y] = new_site_value;

            // Secretion: set chemicals to true for all chemicals that bacteria i can secrete
            for (int j = 0; j < K; ++j)
            {
                int chemical = d_S[i * K + j];
                d_chemical_lattice[chemical * L * L + new_site_x * L + new_site_y] = true;
            }

            // check if the new site is empty
            if (new_site_value == EMPTY)
            {
                // Sum the chemical concentrations at the new site to get the replication probability
                float replicationProbability = 0.0f;
                for (int j = 0; j < K; ++j)
                {
                    int chemical = d_J[i * K + j];
                    replicationProbability += d_chemical_lattice[chemical * L * L + new_site_x * L + new_site_y];
                    d_chemical_lattice[chemical * L * L + new_site_x * L + new_site_y] = 0.0f;
                }
                replicationProbability = replicationProbability / K; // normalization so it's not exactly "or"
                if (curand_uniform(&localState) < replicationProbability)
                {
                    d_bacteria_lattice[site_x * L + site_y] = site_value;
                }
            }
        }
    }

    // Update the state
    state[index] = localState;
}

__global__ void diffuseKernel(float *d_chemical_lattice, int N_CHEMICALS, int L, float D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Apply the diffusion formula
    for (int k = 0; k < N_CHEMICALS; ++k)
    {
        float up = d_chemical_lattice[k * L * L + ((idx - 1 + L) % L) * L + idy];
        float down = d_chemical_lattice[k * L * L + ((idx + 1) % L) * L + idy];
        float left = d_chemical_lattice[k * L * L + idx * L + ((idy - 1 + L) % L)];
        float right = d_chemical_lattice[k * L * L + idx * L + ((idy + 1) % L)];

        d_chemical_lattice[k * L * L + idx * L + idy] = (1 - D) * d_chemical_lattice[k * L * L + idx * L + idy] + D * (up + down + left + right) / 4.0f;
    }
}

__global__ void countLattice(int *d_lattice, int *d_counts, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        int value = d_lattice[idx];
        atomicAdd(&d_counts[value], 1);
    }
}

void run(int N_STEPS, std::ofstream &file, int L, int N_SPECIES, int N_CHEMICALS, double THETA, int K, float D)
{
    // Initialize the lattice
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<int> bacteria_lattice = init_bacteria_lattice(L, N_SPECIES, gen);
    std::vector<float> chemical_lattice = init_chemical_lattice(L, N_CHEMICALS);
    auto [J, S] = init_J_S_maps(N_SPECIES, N_CHEMICALS, K, gen);
    std::vector<int> BACTERIA = init_bacteria(N_SPECIES);

    // Initialize CUDA
    cudaSetDevice(0);

    // Allocate memory on the GPU
    int *d_bacteria_lattice;
    float *d_chemical_lattice;
    int *d_J, *d_S;
    curandState *d_state;
    cudaMalloc(&d_bacteria_lattice, L * L * sizeof(int));
    cudaMalloc(&d_chemical_lattice, N_CHEMICALS * L * L * sizeof(float));
    cudaMalloc(&d_J, N_SPECIES * K * sizeof(int));
    cudaMalloc(&d_S, N_SPECIES * K * sizeof(int));
    cudaMalloc(&d_state, L * L * sizeof(curandState));

    // Initialize the RNG states
    initCurand<<<L / BLOCK_LENGTH, L / BLOCK_LENGTH>>>(d_state, time(0));

    // Copy the lattice data to the GPU
    cudaMemcpy(d_bacteria_lattice, bacteria_lattice.data(), L * L * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_chemical_lattice, chemical_lattice.data(), N_CHEMICALS * L * L * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_J, J.data(), N_SPECIES * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_S, S.data(), N_SPECIES * K * sizeof(int), cudaMemcpyHostToDevice);

    // Define the block and grid sizes
    dim3 blockSize(1, 1);
    dim3 gridSize(L / BLOCK_LENGTH, L / BLOCK_LENGTH);

    // Initialize counts to 0
    int *d_countArray;
    cudaMalloc(&d_countArray, N_SPECIES * sizeof(int));
    cudaMemset(d_countArray, 0, N_SPECIES * sizeof(int));

    int threadsPerBlockCounting = 256;
    int blocksPerGridCounting = (L * L + threadsPerBlockCounting - 1) / threadsPerBlockCounting;

    // Launch the CUDA kernel for each of the A, B, C, and D squares
    for (int step = 0; step < N_STEPS; ++step)
    {
        for (int i = 0; i < BLOCK_LENGTH / 2 * BLOCK_LENGTH / 2; ++i) // 1 iteration per square in subblock
        {
            updateKernel<<<gridSize, blockSize>>>(d_bacteria_lattice, d_chemical_lattice, d_state, d_J, d_S, L, THETA, K, 0, 0); // A squares
            cudaDeviceSynchronize();
            updateKernel<<<gridSize, blockSize>>>(d_bacteria_lattice, d_chemical_lattice, d_state, d_J, d_S, L, THETA, K, 1, 0); // B squares
            cudaDeviceSynchronize();
            updateKernel<<<gridSize, blockSize>>>(d_bacteria_lattice, d_chemical_lattice, d_state, d_J, d_S, L, THETA, K, 0, 1); // C squares
            cudaDeviceSynchronize();
            updateKernel<<<gridSize, blockSize>>>(d_bacteria_lattice, d_chemical_lattice, d_state, d_J, d_S, L, THETA, K, 1, 1); // D squares
            cudaDeviceSynchronize();
        }

        diffuseKernel<<<gridSize, blockSize>>>(d_chemical_lattice, N_CHEMICALS, L, D);
        cudaDeviceSynchronize();

        // Count the bacteria population
        cudaMemset(d_countArray, 0, N_SPECIES * sizeof(int));
        countLattice<<<blocksPerGridCounting, threadsPerBlockCounting>>>(d_bacteria_lattice, d_countArray, L * L);
        cudaDeviceSynchronize();

        // Copy the counts back to the host
        std::vector<int> counts(N_SPECIES);
        cudaMemcpy(counts.data(), d_countArray, N_SPECIES * sizeof(int), cudaMemcpyDeviceToHost);

        // Write the counts to the file
        file << step;
        for (int i = 0; i < N_SPECIES; ++i)
        {
            file << "," << counts[i];
        }
        file << "\n";

        std::cout << "Progress: " << std::fixed << std::setprecision(2) << static_cast<double>(step) / N_STEPS * 100 << "%\r" << std::flush;
    }

    // Free the memory allocated on the GPU
    cudaFree(d_bacteria_lattice);
    cudaFree(d_chemical_lattice);
    cudaFree(d_state);
    cudaFree(d_countArray);
}

int main(int argc, char *argv[])
{
    int N_SPECIES = DEFAULT_N_SPECIES;
    int N_CHEMICALS = DEFAULT_N_CHEMICALS;
    int L = DEFAULT_L;
    double THETA = DEFAULT_THETA;
    int K = DEFAULT_K;
    int STEPS_PER_LATTICEPOINT = DEFAULT_STEPS_PER_LATTICEPOINT;
    float D = DEFAULT_D;

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
        D = std::stof(argv[7]);

    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();

    std::ostringstream filename;
    filename << exeDir << "/outputs/timeseriesChemicalDiffusionCUDA/N_" << N_SPECIES << "-" << N_CHEMICALS << "_L_" << L << "_theta_" << THETA << "_K_" << K << "_D_" << D << ".csv";
    std::ofstream file(filename.str());

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

    run(STEPS_PER_LATTICEPOINT, file, L, N_SPECIES, N_CHEMICALS, THETA, K, D);
    file.close();

    return 0;
}