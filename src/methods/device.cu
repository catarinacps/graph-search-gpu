#include "device.cuh"

namespace gsg {

namespace cuda {

    static __global__ void fw_kernel(int const u, size_t pitch, int const n_vertex, int* const graph)
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (y < n_vertex && x < n_vertex) {
            int indexYX = y * pitch + x;
            int indexUX = u * pitch + x;

            int new_path = graph[y * pitch + u] + graph[indexUX];
            int old_path = graph[indexYX];

            if (old_path > new_path)
                graph[indexYX] = new_path;
        }
    }

    bool bfs(const graph& input, uint32_t searched_vertex, uint32_t initial_vertex, bool verbose)
    {
        return true;
    }

    bool floyd_warshall(const graph& input_host, uint block_size, bool verbose)
    {
        // int num_gpus;
        // HANDLE_ERROR(cudaGetDeviceCount(&num_gpus));
        HANDLE_ERROR(cudaSetDevice(0));

        auto n_vertex = input_host.size;

        dim3 dim_grid((n_vertex - 1) / block_size + 1, (n_vertex - 1) / block_size + 1, 1);
        dim3 dim_block(block_size, block_size, 1);

        int* d_matrix;
        auto pitch = move_to_device(input_host, &d_matrix);

        auto initial_time = get_time();

        cudaFuncSetCacheConfig(fw_kernel, cudaFuncCachePreferL1);

        for (int vertex = 0; vertex < n_vertex; ++vertex) {
            fw_kernel<<<dim_grid, dim_block>>>(vertex, pitch / sizeof(int), n_vertex, d_matrix);
        }

        HANDLE_ERROR(cudaGetLastError());
        HANDLE_ERROR(cudaDeviceSynchronize());

        fmt::print("time: {}", get_time() - initial_time);

        graph ret_graph(input_host.size);
        move_from_device(ret_graph, d_matrix, pitch);

        cudaFree(d_matrix);

        return true;
    }
}

}
