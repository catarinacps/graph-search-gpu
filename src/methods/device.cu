#include "device.cuh"

namespace gsg {

namespace cuda {

    static __global__ void fw_kernel(
        int const u,
        size_t pitch,
        int const n_vertex,
        int* const graph)
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

    static __global__ void bfs_kernel_1(
        node* g_graph_nodes,
        int* g_graph_edges,
        bool* g_graph_mask,
        bool* g_updating_graph_mask,
        bool* g_graph_visited,
        int* g_cost,
        int no_of_nodes)
    {
        int tid = blockIdx.x * 512 + threadIdx.x;
        if (tid < no_of_nodes && g_graph_mask[tid]) {
            g_graph_mask[tid] = false;
            for (int i = g_graph_nodes[tid].first; i < (g_graph_nodes[tid].second + g_graph_nodes[tid].first); i++) {
                int id = g_graph_edges[i];
                if (!g_graph_visited[id]) {
                    g_cost[id] = g_cost[tid] + 1;
                    g_updating_graph_mask[id] = true;
                }
            }
        }
    }

    static __global__ void bfs_kernel_2(
        bool* g_graph_mask,
        bool* g_updating_graph_mask,
        bool* g_graph_visited,
        bool* g_over,
        int no_of_nodes)
    {
        int tid = blockIdx.x * 512 + threadIdx.x;
        if (tid < no_of_nodes && g_updating_graph_mask[tid]) {

            g_graph_mask[tid] = true;
            g_graph_visited[tid] = true;
            *g_over = true;
            g_updating_graph_mask[tid] = false;
        }
    }

    bool bfs(const graph& input, uint searched_vertex, uint initial_vertex, uint block_size, bool verbose)
    {
        HANDLE_ERROR(cudaSetDevice(0));

        int num_blocks = (int)ceil(input.size / (double)block_size);

        node* h_graph_nodes = (node*)calloc(input.size, sizeof(node));
        bool* h_graph_mask = (bool*)calloc(input.size, sizeof(bool));
        bool* h_updating_graph_mask = (bool*)calloc(input.size, sizeof(bool));
        bool* h_graph_visited = (bool*)calloc(input.size, sizeof(bool));
        int* h_graph_edges = (int*)calloc(2 * input.num_edges, sizeof(int));

        uint edge_index = 0;
        for (uint i = 0; i < input.size; i++) {
            uint connected_edges = 0;
            for (uint j = 0; j < input.size; j++) {
                if (input.matrix[i][j] != 0) {
                    connected_edges++;
                    h_graph_edges[edge_index++] = j;
                }
            }

            h_graph_nodes[i].second = connected_edges;
            h_graph_nodes[i].first = i;
        }

        node* d_graph_nodes;
        HANDLE_ERROR(cudaMalloc((void**)&d_graph_nodes, sizeof(node) * input.size));
        HANDLE_ERROR(cudaMemcpy(d_graph_nodes, h_graph_nodes, sizeof(node) * input.size, cudaMemcpyHostToDevice));

        int* d_graph_edges;
        HANDLE_ERROR(cudaMalloc((void**)&d_graph_edges, sizeof(int) * input.num_edges));
        HANDLE_ERROR(cudaMemcpy(d_graph_edges, h_graph_edges, sizeof(int) * input.num_edges, cudaMemcpyHostToDevice));

        bool* d_graph_mask;
        HANDLE_ERROR(cudaMalloc((void**)&d_graph_mask, sizeof(bool) * input.size));
        HANDLE_ERROR(cudaMemcpy(d_graph_mask, h_graph_mask, sizeof(bool) * input.size, cudaMemcpyHostToDevice));

        bool* d_updating_graph_mask;
        HANDLE_ERROR(cudaMalloc((void**)&d_updating_graph_mask, sizeof(bool) * input.size));
        HANDLE_ERROR(cudaMemcpy(d_updating_graph_mask, h_updating_graph_mask, sizeof(bool) * input.size, cudaMemcpyHostToDevice));

        bool* d_graph_visited;
        HANDLE_ERROR(cudaMalloc((void**)&d_graph_visited, sizeof(bool) * input.size));
        HANDLE_ERROR(cudaMemcpy(d_graph_visited, h_graph_visited, sizeof(bool) * input.size, cudaMemcpyHostToDevice));

        int* h_cost = (int*)malloc(sizeof(int) * input.size);
        for (int i = 0; i < input.size; i++)
            h_cost[i] = -1;
        h_cost[initial_vertex] = 0;

        int* d_cost;
        HANDLE_ERROR(cudaMalloc((void**)&d_cost, sizeof(int) * input.size));
        HANDLE_ERROR(cudaMemcpy(d_cost, h_cost, sizeof(int) * input.size, cudaMemcpyHostToDevice));

        bool* d_over;
        HANDLE_ERROR(cudaMalloc((void**)&d_over, sizeof(bool)));

        dim3 grid(num_blocks, 1, 1);
        dim3 threads(block_size, 1, 1);

        auto initial_time = get_time();

        int k = 0;
        bool stop;

        do {
            stop = false;
            HANDLE_ERROR(cudaMemcpy(d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice));
            bfs_kernel_1<<<grid, threads, 0>>>(d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, input.size);

            bfs_kernel_2<<<grid, threads, 0>>>(d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, input.size);

            HANDLE_ERROR(cudaMemcpy(&stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost));
            k++;
        } while (stop);

        HANDLE_ERROR(cudaGetLastError());
        HANDLE_ERROR(cudaDeviceSynchronize());

        fmt::print("time: {}", get_time() - initial_time);

        free(h_graph_nodes);
        free(h_graph_edges);
        free(h_graph_mask);
        free(h_updating_graph_mask);
        free(h_graph_visited);
        free(h_cost);
        HANDLE_ERROR(cudaFree(d_graph_nodes));
        HANDLE_ERROR(cudaFree(d_graph_edges));
        HANDLE_ERROR(cudaFree(d_graph_mask));
        HANDLE_ERROR(cudaFree(d_updating_graph_mask));
        HANDLE_ERROR(cudaFree(d_graph_visited));
        HANDLE_ERROR(cudaFree(d_cost));

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
