#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>

constexpr int L = 100;  // Define the size of the lattice
constexpr int N_RESOURCES = 1;  // Number of resources
constexpr int N_CHEMICALS = 2;  // Number of chemicals
constexpr float GAMMA = 0;  // Growth rate at the borders
constexpr float DIFFUSION_COEFF = 0;  // Diffusion coefficient
constexpr float FINAL_TIME = 10.0;  // Final time for the simulation
constexpr int N_STEPS = 1000;  // Number of timesteps
constexpr float DT = FINAL_TIME / N_STEPS;  // Timestep for the ODE integration
constexpr float SPARSITY = 0.9;  // Sparsity of the reaction rates

__global__ void initialize_lattice(float *resources, float *chemicals, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idy * L + idx;

    if (idx < L && idy < L) {
        curandState localState = states[id];
        for (int i = 0; i < N_RESOURCES; i++) {
            resources[id * N_RESOURCES + i] = 0;
        }
        for (int i = 0; i < N_CHEMICALS; i++) {
            chemicals[id * N_CHEMICALS + i] = curand_uniform(&localState);
        }
        states[id] = localState;
    }
}

__global__ void initialize_reaction_rates_and_connectivity(float *reaction_rates, int *connectivity, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idy * N_CHEMICALS + idx;

    if (idx < N_CHEMICALS && idy < N_CHEMICALS) {
        curandState localState = states[id];
        reaction_rates[id] = (curand_uniform(&localState) < SPARSITY) ? 0.0f : curand_uniform(&localState);
        connectivity[id] = static_cast<int>(curand(&localState) % N_RESOURCES);
        states[id] = localState;
    }
}

__global__ void apply_growth_rate(float *resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idy * L + idx;

    if (idx < L && idy < L) {
        if (idx == 0 || idx == L-1 || idy == 0 || idy == L-1) {
            for (int i = 0; i < N_RESOURCES; i++) {
                resources[id * N_RESOURCES + i] += GAMMA * DT * (1.0f - resources[id * N_RESOURCES + i]);
                if (resources[id * N_RESOURCES + i] > 1.0f) {
                    resources[id * N_RESOURCES + i] = 1.0f;
                }
            }
        }
    }
}

__global__ void update_lattice(float *resources, float *chemicals, float *reaction_rates, int *connectivity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idy * L + idx;

    if (idx < L && idy < L) {
        float dS[N_RESOURCES] = {0};
        float dX[N_CHEMICALS] = {0};

        for (int j = 0; j < N_CHEMICALS; j++) {
            for (int k = 0; k < N_CHEMICALS; k++) {
                int resource_index = connectivity[j * N_CHEMICALS + k];
                if (reaction_rates[j * N_CHEMICALS + k] > 0.0f) {
                    float reaction_rate = reaction_rates[j * N_CHEMICALS + k];
                    float reaction_term = reaction_rate * chemicals[id * N_CHEMICALS + j] * chemicals[id * N_CHEMICALS + k] * resources[id * N_RESOURCES + resource_index] * DT;
                    dS[resource_index] -= reaction_term;
                    dX[k] += reaction_term;
                }
            }
        }

        for (int i = 0; i < N_RESOURCES; i++) {
            resources[id * N_RESOURCES + i] += dS[i];
            if (resources[id * N_RESOURCES + i] < 0.0f) {
                resources[id * N_RESOURCES + i] = 0.0f;
            }
        }
        for (int i = 0; i < N_CHEMICALS; i++) {
            chemicals[id * N_CHEMICALS + i] += dX[i];
        }
    }
}

__global__ void sum_lattice(float *data, float *sum, int num_elements) {
    extern __shared__ float shared_data[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    shared_data[tid] = (idx < num_elements) ? data[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum, shared_data[0]);
    }
}

__global__ void normalize_chemicals(float *chemicals, float total_chemical_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idy * L + idx;

    if (idx < L && idy < L) {
        for (int i = 0; i < N_CHEMICALS; i++) {
            chemicals[id * N_CHEMICALS + i] /= total_chemical_sum;
        }
    }
}

__global__ void diffuse(float *resources, float *chemicals) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idy * L + idx;

    if (idx < L && idy < L) {
        float new_resources[N_RESOURCES] = {0};
        float new_chemicals[N_CHEMICALS] = {0};

        for (int i = 0; i < N_RESOURCES; i++) {
            new_resources[i] = resources[id * N_RESOURCES + i];
        }
        for (int i = 0; i < N_CHEMICALS; i++) {
            new_chemicals[i] = chemicals[id * N_CHEMICALS + i];
        }

        for (int i = 0; i < N_RESOURCES; i++) {
            new_resources[i] += DIFFUSION_COEFF * (
                resources[(id - 1 + L) % L * L + idx] +
                resources[(id + 1) % L * L + idx] +
                resources[idy * L + (idx - 1 + L) % L] +
                resources[idy * L + (idx + 1) % L] -
                4 * resources[id]
            ) * DT;
        }

        for (int i = 0; i < N_CHEMICALS; i++) {
            new_chemicals[i] += DIFFUSION_COEFF * (
                chemicals[(id - 1 + L) % L * L + idx] +
                chemicals[(id + 1) % L * L + idx] +
                chemicals[idy * L + (idx - 1 + L) % L] +
                chemicals[idy * L + (idx + 1) % L] -
                4 * chemicals[id]
            ) * DT;
        }

        for (int i = 0; i < N_RESOURCES; i++) {
            resources[id * N_RESOURCES + i] = new_resources[i];
        }
        for (int i = 0; i < N_CHEMICALS; i++) {
            chemicals[id * N_CHEMICALS + i] = new_chemicals[i];
        }
    }
}

void run_simulation(std::ofstream &file) {
    float *d_resources, *d_chemicals, *d_reaction_rates;
    int *d_connectivity;
    curandState *d_states;
    float *d_resource_sum, *d_chemical_sum, *d_total_chemical_sum;

    cudaMalloc(&d_resources, L * L * N_RESOURCES * sizeof(float));
    cudaMalloc(&d_chemicals, L * L * N_CHEMICALS * sizeof(float));
    cudaMalloc(&d_reaction_rates, N_CHEMICALS * N_CHEMICALS * sizeof(float));
    cudaMalloc(&d_connectivity, N_CHEMICALS * N_CHEMICALS * sizeof(int));
    cudaMalloc(&d_states, L * L * sizeof(curandState));
    cudaMalloc(&d_resource_sum, N_RESOURCES * sizeof(float));
    cudaMalloc(&d_chemical_sum, N_CHEMICALS * sizeof(float));
    cudaMalloc(&d_total_chemical_sum, sizeof(float));

    dim3 blockSize(16, 16);
    dim3 gridSize((L + blockSize.x - 1) / blockSize.x, (L + blockSize.y - 1) / blockSize.y);
    dim3 gridSizeChem((N_CHEMICALS + blockSize.x - 1) / blockSize.x, (N_CHEMICALS + blockSize.y - 1) / blockSize.y);

    initialize_lattice<<<gridSize, blockSize>>>(d_resources, d_chemicals, d_states);
    cudaDeviceSynchronize();

    initialize_reaction_rates_and_connectivity<<<gridSizeChem, blockSize>>>(d_reaction_rates, d_connectivity, d_states);
    cudaDeviceSynchronize();

    file << "time";
    for (int i = 0; i < N_RESOURCES; i++) {
        file << ",resource" << i;
    }
    for (int i = 0; i < N_CHEMICALS; i++) {
        file << ",chemical" << i;
    }
    file << "\n";

    for (int t = 0; t < N_STEPS; t++) {
        apply_growth_rate<<<gridSize, blockSize>>>(d_resources);
        cudaDeviceSynchronize();

        update_lattice<<<gridSize, blockSize>>>(d_resources, d_chemicals, d_reaction_rates, d_connectivity);
        cudaDeviceSynchronize();

        cudaMemset(d_total_chemical_sum, 0, sizeof(float));
        for (int i = 0; i < N_CHEMICALS; i++) {
            sum_lattice<<<gridSize, blockSize, blockSize.x * sizeof(float)>>>(d_chemicals + i * L * L, d_total_chemical_sum, L * L);
        }
        cudaDeviceSynchronize();

        float total_chemical_sum;
        cudaMemcpy(&total_chemical_sum, d_total_chemical_sum, sizeof(float), cudaMemcpyDeviceToHost);

        normalize_chemicals<<<gridSize, blockSize>>>(d_chemicals, total_chemical_sum);
        cudaDeviceSynchronize();

        diffuse<<<gridSize, blockSize>>>(d_resources, d_chemicals);
        cudaDeviceSynchronize();

        cudaMemset(d_resource_sum, 0, N_RESOURCES * sizeof(float));
        cudaMemset(d_chemical_sum, 0, N_CHEMICALS * sizeof(float));

        for (int i = 0; i < N_RESOURCES; i++) {
            sum_lattice<<<gridSize, blockSize, blockSize.x * sizeof(float)>>>(d_resources + i * L * L, d_resource_sum + i, L * L);
        }
        for (int i = 0; i < N_CHEMICALS; i++) {
            sum_lattice<<<gridSize, blockSize, blockSize.x * sizeof(float)>>>(d_chemicals + i * L * L, d_chemical_sum + i, L * L);
        }
        cudaDeviceSynchronize();

        std::vector<float> resource_sum(N_RESOURCES);
        std::vector<float> chemical_sum(N_CHEMICALS);

        cudaMemcpy(resource_sum.data(), d_resource_sum, N_RESOURCES * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(chemical_sum.data(), d_chemical_sum, N_CHEMICALS * sizeof(float), cudaMemcpyDeviceToHost);

        file << t * DT;
        for (int i = 0; i < N_RESOURCES; i++) {
            file << "," << resource_sum[i];
        }
        for (int i = 0; i < N_CHEMICALS; i++) {
            file << "," << chemical_sum[i];
        }
        file << "\n";
    }

    cudaFree(d_resources);
    cudaFree(d_chemicals);
    cudaFree(d_reaction_rates);
    cudaFree(d_connectivity);
    cudaFree(d_states);
    cudaFree(d_resource_sum);
    cudaFree(d_chemical_sum);
    cudaFree(d_total_chemical_sum);
}

int main(int argc, char *argv[]) {
    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::ostringstream filePathStream;
    filePathStream << exeDir << "/outputs/timeseries2D/N_" << N_RESOURCES << "-" << N_CHEMICALS << "_L_" << L << "_gamma_" << GAMMA << "_D_" << DIFFUSION_COEFF << ".csv";
    std::string filePath = filePathStream.str();

    std::ofstream file;
    file.open(filePath);
    run_simulation(file);
    file.close();

    return 0;
}