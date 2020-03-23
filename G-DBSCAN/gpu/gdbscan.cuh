#include <cmath>
#include <thrust/device_vector.h>

// TODO:
// - construct Va and Ea, DO NOT use temp_adj. AVOID data transfer to GPU.
//   - Va: thrust::exclusive_scan
//   - Ea: one thread each node
// - ID core and non-core: one thread each node
// - BFS. no clue ... on thread each node? then ima hving a n^2 bfs???
//
namespace GDBSCAN {
    int const blocksize = 512;

    __device__ float square_dist(const float &x1, const float &y1, const float &x2, const float &y2) {
        return std::pow(x1 - x2, 2.f) + std::pow(y1 - y2, 2.f);
    }

    // Kernel to calculate Va
    __global__ void
    calc_va(float *x, float *y, uint64_t *Va, const float &rad_sq, const uint64_t &num_nodes) {
        int const u = threadIdx.x + blockIdx.x * blockDim.x;
        Va[u] = 0;
        for (auto v = 0u; v < num_nodes; ++v) {
            if (u != v && square_dist(x[u], y[u], x[v], y[v]) <= rad_sq)
                ++Va[u];
        }
    }

    void insert_edge(float *x, float *y, uint64_t *Va, const float &rad_sq, const uint64_t &num_nodes) {
        const auto num_blocks = std::ceil(num_nodes / static_cast< float >( blocksize ));
        float *dev_x, *dev_y;
        uint64_t *dev_Va;
        cudaMalloc((void **) &dev_x, sizeof(x));
        cudaMalloc((void **) &dev_y, sizeof(y));
        cudaMalloc((void **) &dev_Va, sizeof(Va));
        cudaMemcpy(dev_x, x, sizeof(x), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_y, y, sizeof(y), cudaMemcpyHostToDevice);

        calc_va << < num_blocks, blocksize >> > (dev_x, dev_y, dev_Va, rad_sq, num_nodes);

        cudaMemcpy(Va, dev_Va, sizeof(Va), cudaMemcpyDeviceToHost);
        cudaFree(dev_x);
        cudaFree(dev_y);
        cudaFree(dev_Va);
    }
}