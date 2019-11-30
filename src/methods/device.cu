#include "device.cuh"

namespace gsg {

namespace cuda {

    void bfs(const graph& input, uint32_t searched_vertex, uint32_t initial_vertex, bool verbose);

    void floyd_warshall(const graph& input_host, uint block_size, bool verbose)
    {
        int num_gpus;
        HANDLE_ERROR(cudaGetDeviceCount(&num_gpus));

        auto n_vertex = input_host.size;

        dim3 dim_grid((n_vertex - 1) / block_size + 1, (n_vertex - 1) / block_size + 1, 1);
        dim3 dim_block(block_size, block_size, 1);
    }

}

}
