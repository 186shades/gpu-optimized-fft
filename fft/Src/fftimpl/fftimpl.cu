#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <chrono>
#include <iostream>

#define BLOCK_SIZE 256;
#define BASE_SIZE 1024;
#define PADDING_SIZE 4;
#define WARP_SIZE 32;


// Log functions
inline int log2i(int n) {
    int r = 0;
    while ((n >>= 1) != 0) r++;
    return r;
}

// Bit-reversal kernel
__global__ void bit_reverse_kernel(float2* d_data, int N, int logN)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned int reversed = __brev(tid) >> (32 - logN);
        if (reversed > (unsigned)tid) {
            float2 temp = d_data[tid];
            d_data[tid] = d_data[reversed];
            d_data[reversed] = temp;
        }
    }
}

// simple FFT kernel for 1 stage
__global__ void fft_stage_kernel(float2* d_data, int N, int s)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int halfSize = 1 << (s - 1);
    int fftSize = 1 << s; 

    if (tid < N / 2) {
        int group = tid / halfSize;
        int j = tid % halfSize;
        int k = group * fftSize;

        float angle = -2.0f * (float)M_PI * j / (float)fftSize;
        float2 w = make_float2(cosf(angle), sinf(angle));

        int index1 = k + j;
        int index2 = k + j + halfSize;

        float2 u = d_data[index1];
        float2 t = d_data[index2];

        float2 temp;
        temp.x = w.x * t.x - w.y * t.y;
        temp.y = w.x * t.y + w.y * t.x;

        d_data[index1].x = u.x + temp.x;
        d_data[index1].y = u.y + temp.y;
        d_data[index2].x = u.x - temp.x;
        d_data[index2].y = u.y - temp.y;
    }
}

// CPU-based FFT
void fft_cpu(float2* data, int N)
{
    int logN = log2i(N);
    for (int i = 0; i < N; ++i) {
        unsigned int reversed = 0;
        unsigned int x = i;
        for (int b = 0; b < logN; b++) {
            reversed = (reversed << 1) | (x & 1);
            x >>= 1;
        }
        if (reversed > (unsigned)i) {
            float2 temp = data[i];
            data[i] = data[reversed];
            data[reversed] = temp;
        }
    }

    // Iterative FFT
    for (int s = 1; s <= logN; s++) {
        int m = 1 << s;
        int m2 = m >> 1;
        float theta = -2.0f * (float)M_PI / m;
        for (int k = 0; k < N; k += m) {
            for (int j = 0; j < m2; j++) {
                float angle = theta * j;
                float2 w;
                w.x = cosf(angle);
                w.y = sinf(angle);

                float2 u = data[k + j];
                float2 t = data[k + j + m2];

                float2 temp;
                temp.x = w.x * t.x - w.y * t.y;
                temp.y = w.x * t.y + w.y * t.x;

                data[k + j].x = u.x + temp.x;
                data[k + j].y = u.y + temp.y;
                data[k + j + m2].x = u.x - temp.x;
                data[k + j + m2].y = u.y - temp.y;
            }
        }
    }
}

// Optimized (Shared Memory) kernel for all stages
__global__ void fft_kernel_sm_all_stages(float2* d_data, int N, int logN)
{
    extern __shared__ float2 s_data[];

    int tid = threadIdx.x;
    s_data[tid] = d_data[tid];

    for (int s = 1; s <= logN; s++) {
        int m = 1 << s;
        int m2 = m >> 1;

        __syncthreads();
        if (tid < N / 2) {
            int group = tid / m2;
            int j = tid % m2;
            int k = group * m;

            float angle = -2.0f * (float)M_PI * (float)j / (float)m;
            float2 w = make_float2(cosf(angle), sinf(angle));

            int i1 = k + j;
            int i2 = k + j + m2;

            float2 u = s_data[i1];
            float2 t = s_data[i2];

            float2 temp;
            temp.x = w.x * t.x - w.y * t.y;
            temp.y = w.x * t.y + w.y * t.x;

            s_data[i1].x = u.x + temp.x;
            s_data[i1].y = u.y + temp.y;
            s_data[i2].x = u.x - temp.x;
            s_data[i2].y = u.y - temp.y;
        }
    }

    __syncthreads();
    d_data[tid] = s_data[tid];
}

// Optimized (Shared Memory) Kernel for all stages (padding, instructions, sync optimization)
__global__ void fft_kernel_sm_all_stages_padding(float2* d_data, int N, int logN)
{
    extern __shared__ float s_mem[];

    const int tid = threadIdx.x;
    const int warpSize = WARP_SIZE;
    const int paddingPerWarp = PADDING_SIZE;

    const int numWarps = N / warpSize;
    const int paddedLength = N + numWarps * paddingPerWarp;

    float* s_real = s_mem;               
    float* s_imag = s_mem + paddedLength;

    int warpId_t = tid / warpSize;
    int pTid = tid + warpId_t * paddingPerWarp;

    float2 val = d_data[tid];
    s_real[pTid] = val.x;
    s_imag[pTid] = val.y;

    for (int s = 1; s <= logN; s++) {
        int m = 1 << s;
        int m2 = m >> 1;

        float angleBase = -2.0f * (float)M_PI / (float)m;

        __syncthreads();
        if (tid < N / 2) {
            int group = tid / m2;
            int j = tid % m2;
            int k = group * m;
            float angle = angleBase * j;
            float2 w = make_float2(__cosf(angle), __sinf(angle));

            int i1 = k + j;
            int i2 = k + j + m2;

            int warpId_i1 = i1 / warpSize;
            int pI1 = i1 + warpId_i1 * paddingPerWarp;
            int warpId_i2 = i2 / warpSize;
            int pI2 = i2 + warpId_i2 * paddingPerWarp;

            float ur = s_real[pI1];
            float ui = s_imag[pI1];
            float tr = s_real[pI2];
            float ti = s_imag[pI2];

            float temp_r = w.x * tr - w.y * ti;
            float temp_i = w.x * ti + w.y * tr;

            s_real[pI1] = ur + temp_r;
            s_imag[pI1] = ui + temp_i;
            s_real[pI2] = ur - temp_r;
            s_imag[pI2] = ui - temp_i;
        }
    }

    __syncthreads();

    val.x = s_real[pTid];
    val.y = s_imag[pTid];
    d_data[tid] = val;
}

// Run CPU FFT
double run_cpu_fft(float2* h_data, int N)
{
    auto cpu_start = std::chrono::high_resolution_clock::now();
    fft_cpu(h_data, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    return cpu_duration.count() * 1000.0;
}

// Runs simple GPU FFT
double run_simple_gpu_fft(float2* h_data, int N)
{
    int logN = log2i(N);

    float2* d_data;
    cudaMalloc((void**)&d_data, N * sizeof(float2));
    cudaMemcpy(d_data, h_data, N * sizeof(float2), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blockSize = BLOCK_SIZE;
    dim3 blockDim(blockSize);
    dim3 gridDimBitRev((N + blockSize - 1) / blockSize);
    int halfN = N / 2;
    dim3 gridDimStage((halfN + blockSize - 1) / blockSize);

    cudaEventRecord(start);

    bit_reverse_kernel << <gridDimBitRev, blockDim >> > (d_data, N, logN);

    for (int s = 1; s <= logN; s++) {
        fft_stage_kernel << <gridDimStage, blockDim >> > (d_data, N, s);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    cudaMemcpy(h_data, d_data, N * sizeof(float2), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return gpu_time_ms;
}

// Run optimized GPU FFT (Shared Memory)
double run_optimized_gpu_fft(float2* h_data, int N)
{
    int logN = log2i(N);
    int baseSize = BASE_SIZE;
    int logBaseSize = log2i(baseSize);
    int numSegments = (N + baseSize - 1) / baseSize;

    float2* d_data;
    cudaMalloc((void**)&d_data, N * sizeof(float2));
    cudaMemcpy(d_data, h_data, N * sizeof(float2), cudaMemcpyHostToDevice);

    cudaStream_t* streams = new cudaStream_t[numSegments];
    for (int i = 0; i < numSegments; i++)
        cudaStreamCreate(&streams[i]);

    cudaEvent_t ostart, ostop;
    cudaEventCreate(&ostart);
    cudaEventCreate(&ostop);

    int blockSize = BLOCK_SIZE;
    dim3 blockDim(blockSize);
    dim3 gridDimBitRev((N + blockSize - 1) / blockSize);

    cudaEventRecord(ostart);

    bit_reverse_kernel << <gridDimBitRev, blockDim >> > (d_data, N, logN);

    for (int seg = 0; seg < numSegments; seg++) {
        float2* segment_ptr = d_data + seg * baseSize;
        fft_kernel_sm_all_stages << <1, baseSize, baseSize * sizeof(float2), streams[seg] >> > (segment_ptr, baseSize, logBaseSize);
    }

    for (int i = 0; i < numSegments; i++)
        cudaStreamSynchronize(streams[i]);

    if (N > baseSize) {
        for (int s = logBaseSize + 1; s <= logN; s++) {
            int m = 1 << s;
            int totalPairs = N / 2;
            int gridSize = (totalPairs + blockSize - 1) / blockSize;
            fft_stage_kernel << <gridSize, blockSize >> > (d_data, N, s);
        }
    }

    cudaEventRecord(ostop);
    cudaEventSynchronize(ostop);

    for (int i = 0; i < numSegments; i++)
        cudaStreamDestroy(streams[i]);


    float opt_gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&opt_gpu_time_ms, ostart, ostop);

    cudaMemcpy(h_data, d_data, N * sizeof(float2), cudaMemcpyDeviceToHost);

    delete[] streams;
    cudaFree(d_data);
    cudaEventDestroy(ostart);
    cudaEventDestroy(ostop);

    return opt_gpu_time_ms;
}

// Run ultra optimized GPU FFT (all final optimizations)
double run_ultra_optimized_gpu_fft(float2* h_data, int N)
{
    int logN = log2i(N);
    int baseSize = BASE_SIZE;
    int logBaseSize = log2i(baseSize);
    int numSegments = (N + baseSize - 1) / baseSize;
    const int warpSize = WARP_SIZE;
    const int paddingPerWarp = PADDING_SIZE;

    float2* d_data;
    cudaMalloc((void**)&d_data, N * sizeof(float2));
    cudaMemcpy(d_data, h_data, N * sizeof(float2), cudaMemcpyHostToDevice);

    cudaStream_t* streams = new cudaStream_t[numSegments];
    for (int i = 0; i < numSegments; i++)
        cudaStreamCreate(&streams[i]);

    cudaEvent_t ostart, ostop;
    cudaEventCreate(&ostart);
    cudaEventCreate(&ostop);

    int blockSize = BLOCK_SIZE;
    dim3 blockDim(blockSize);
    dim3 gridDimBitRev((N + blockSize - 1) / blockSize);

    cudaEventRecord(ostart);

    bit_reverse_kernel << <gridDimBitRev, blockDim >> > (d_data, N, logN);

    for (int seg = 0; seg < numSegments; seg++) {
        float2* segment_ptr = d_data + seg * baseSize;

        fft_kernel_sm_all_stages_padding << <1, baseSize, (baseSize + (paddingPerWarp * baseSize / warpSize) + paddingPerWarp) * sizeof(float2), streams[seg] >> > (segment_ptr, baseSize, logBaseSize);
    }

    for (int i = 0; i < numSegments; i++)
        cudaStreamSynchronize(streams[i]);

    if (N > baseSize) {
        for (int s = logBaseSize + 1; s <= logN; s++) {
            int m = 1 << s;
            int totalPairs = N / 2;
            int gridSize = (totalPairs + blockSize - 1) / blockSize;
            fft_stage_kernel << <gridSize, blockSize >> > (d_data, N, s);
        }
    }

    cudaEventRecord(ostop);
    cudaEventSynchronize(ostop);

    for (int i = 0; i < numSegments; i++)
        cudaStreamDestroy(streams[i]);


    float opt_gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&opt_gpu_time_ms, ostart, ostop);

    cudaMemcpy(h_data, d_data, N * sizeof(float2), cudaMemcpyDeviceToHost);

    delete[] streams;
    cudaFree(d_data);
    cudaEventDestroy(ostart);
    cudaEventDestroy(ostop);

    return opt_gpu_time_ms;
}


int main()
{
    // Size of N should be power of two
    int N = 4096;
    double tolerance = 1e-5;
    std::cout << "Starting computation for N: " << N << "\n";

    float2* h_data_cpu = (float2*)malloc(N * sizeof(float2));
    float2* h_data_simple_gpu = (float2*)malloc(N * sizeof(float2));
    float2* h_data_optimized_gpu = (float2*)malloc(N * sizeof(float2));
    float2* h_data_ultra_optimized_gpu = (float2*)malloc(N * sizeof(float2));

    for (int i = 0; i < N; ++i) {
        float val = sinf(2 * M_PI * i / N);
        h_data_cpu[i].x = val; h_data_cpu[i].y = 0.0f;
        h_data_simple_gpu[i].x = val; h_data_simple_gpu[i].y = 0.0f;
        h_data_optimized_gpu[i].x = val; h_data_optimized_gpu[i].y = 0.0f;
        h_data_ultra_optimized_gpu[i].x = val; h_data_ultra_optimized_gpu[i].y = 0.0f;
    }

    // Run CPU FFT
    double cpu_time_ms = run_cpu_fft(h_data_cpu, N);

    // Run simple GPU FFT
    double simple_gpu_time_ms = run_simple_gpu_fft(h_data_simple_gpu, N);

    // Run optimized GPU FFT
    double optimized_gpu_time_ms = run_optimized_gpu_fft(h_data_optimized_gpu, N);

    // Run ultra optimized GPU FFT
    double ultra_optimized_gpu_time_ms = run_ultra_optimized_gpu_fft(h_data_ultra_optimized_gpu, N);

    double max_diff_simple = 0.0;
    for (int i = 0; i < N; ++i) {
        double diff_x = fabs(h_data_simple_gpu[i].x - h_data_cpu[i].x);
        double diff_y = fabs(h_data_simple_gpu[i].y - h_data_cpu[i].y);
        if (diff_x > max_diff_simple) max_diff_simple = diff_x;
        if (diff_y > max_diff_simple) max_diff_simple = diff_y;
    }

    double max_diff_optimized = 0.0;
    for (int i = 0; i < N; ++i) {
        double diff_x = fabs(h_data_optimized_gpu[i].x - h_data_cpu[i].x);
        double diff_y = fabs(h_data_optimized_gpu[i].y - h_data_cpu[i].y);
        if (diff_x > max_diff_optimized) max_diff_optimized = diff_x;
        if (diff_y > max_diff_optimized) max_diff_optimized = diff_y;
    }

    double max_diff_ultra_optimized = 0.0;
    for (int i = 0; i < N; ++i) {
        double diff_x = fabs(h_data_ultra_optimized_gpu[i].x - h_data_cpu[i].x);
        double diff_y = fabs(h_data_ultra_optimized_gpu[i].y - h_data_cpu[i].y);
        if (diff_x > max_diff_ultra_optimized) max_diff_ultra_optimized = diff_x;
        if (diff_y > max_diff_ultra_optimized) max_diff_ultra_optimized = diff_y;
    }

    std::cout << "CPU Time: " << cpu_time_ms << " ms\n";
    std::cout << "Simple GPU Time: " << simple_gpu_time_ms << " ms\n";
    std::cout << "Optimized GPU Time: " << optimized_gpu_time_ms << " ms\n";
    std::cout << "Ultra Optimized GPU Time: " << ultra_optimized_gpu_time_ms << " ms\n";
    std::cout << "Max diff (Simple GPU vs CPU): " << max_diff_simple << "\n";
    std::cout << "Max diff (Optimized GPU vs CPU): " << max_diff_optimized << "\n";
    std::cout << "Max diff (Ultra Optimized GPU vs CPU): " << max_diff_ultra_optimized << "\n\n";

    free(h_data_cpu);
    free(h_data_simple_gpu);
    free(h_data_optimized_gpu);
    free(h_data_ultra_optimized_gpu);


    return 0;
}