#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

constexpr int L = 100;  // Size of the lattice
constexpr float DIFFUSION_COEFF = 0.1;  // Diffusion coefficient
constexpr int N_STEPS = 100;  // Number of timesteps
constexpr float DT = 1;  // Timestep for the diffusion
constexpr float GAMMA = 0.01;  // Flow-in rate from the edges
constexpr int N_RESOURCES = 4; // Number of resources

__global__ void initialize_lattice(float *lattice, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idy * L + idx;

    if (idx < L && idy < L) {
        curandState localState = states[id];
        for (int r = 0; r < N_RESOURCES; ++r) {
            lattice[id * N_RESOURCES + r] = curand_uniform(&localState);
        }
        states[id] = localState;
    }
}

__global__ void diffuse(float *lattice, float *new_lattice) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idy * L + idx;

    if (idx < L && idy < L) {
        for (int r = 0; r < N_RESOURCES; ++r) {
            float value = lattice[id * N_RESOURCES + r];
            float laplacian = 0.0f;

            if (idx > 0) laplacian += lattice[(id - 1) * N_RESOURCES + r];
            if (idx < L - 1) laplacian += lattice[(id + 1) * N_RESOURCES + r];
            if (idy > 0) laplacian += lattice[(id - L) * N_RESOURCES + r];
            if (idy < L - 1) laplacian += lattice[(id + L) * N_RESOURCES + r];

            laplacian -= 4.0f * value;

            new_lattice[id * N_RESOURCES + r] = value + DIFFUSION_COEFF * laplacian * DT;

            // Apply flow-in from the edges
            if (idx == 0 || idx == L-1 || idy == 0 || idy == L-1) {
                new_lattice[id * N_RESOURCES + r] += GAMMA * (1.0f - value) * DT;
            }
        }
    }
}

__global__ void init_curand(curandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idy * L + idx;

    if (idx < L && idy < L) {
        curand_init(seed, id, 0, &states[id]);
    }
}

void save_lattice(const std::vector<float> &lattice, const std::string &filename) {
    std::ofstream file(filename);
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            for (int r = 0; r < N_RESOURCES; ++r) {
                file << lattice[(i * L + j) * N_RESOURCES + r] << " ";
            }
            file << "\t";
        }
        file << "\n";
    }
    file.close();
}

int main() {
    float *d_lattice, *d_new_lattice;
    curandState *d_states;

    cudaMalloc(&d_lattice, L * L * N_RESOURCES * sizeof(float));
    cudaMalloc(&d_new_lattice, L * L * N_RESOURCES * sizeof(float));
    cudaMalloc(&d_states, L * L * sizeof(curandState));

    dim3 blockSize(16, 16);
    dim3 gridSize((L + blockSize.x - 1) / blockSize.x, (L + blockSize.y - 1) / blockSize.y);

    init_curand<<<gridSize, blockSize>>>(d_states, time(0));
    cudaDeviceSynchronize();

    initialize_lattice<<<gridSize, blockSize>>>(d_lattice, d_states);
    cudaDeviceSynchronize();

    std::vector<float> h_lattice(L * L * N_RESOURCES);

    for (int t = 0; t < N_STEPS; ++t) {
        diffuse<<<gridSize, blockSize>>>(d_lattice, d_new_lattice);
        cudaDeviceSynchronize();

        // Swap the pointers (d_lattice and d_new_lattice)
        float *temp = d_lattice;
        d_lattice = d_new_lattice;
        d_new_lattice = temp;

        cudaMemcpy(h_lattice.data(), d_lattice, L * L * N_RESOURCES * sizeof(float), cudaMemcpyDeviceToHost);
        save_lattice(h_lattice, "src/diversityTransitionCML/outputs/lattice_" + std::to_string(t) + ".txt");
    }

    cudaFree(d_lattice);
    cudaFree(d_new_lattice);
    cudaFree(d_states);

    return 0;
}