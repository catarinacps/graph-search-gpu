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

    static __global__ void bfs_kernel(
        node* Va, // nodes
        int* Ea, // edges
        bool* Fa, // frontier
        bool* Xa, // visited
        int* Ca, // cost
        int num_nodes,
        bool* done)
    {
        int id = threadIdx.x + blockIdx.x * blockDim.x;

        if (id >= num_nodes) {
            *done = false;
        } else if (Fa[id] == true && Xa[id] == false) {
            printf("%d ", id); //This printf gives the order of vertices in BFS
            Fa[id] = false;
            Xa[id] = true;
            __syncthreads();
            int start = Va[id].first;
            int end = start + Va[id].second;
            for (int i = start; i < end; i++) {
                int nid = Ea[i];

                if (Xa[nid] == false) {
                    Ca[nid] = Ca[id] + 1;
                    Fa[nid] = true;
                    *done = false;
                }
            }
        }
    }

    bool bfs(const graph& input, uint searched_vertex, uint initial_vertex, uint block_size, bool verbose)
    {
        HANDLE_ERROR(cudaSetDevice(0));

        int num_blocks = (int)ceil(input.size / (double)block_size);

        node* h_nodes = (node*)calloc(input.size, sizeof(node));
        int* h_edges = (int*)calloc(input.num_edges, sizeof(int));
        bool* h_frontier = (bool*)calloc(input.size, sizeof(bool));
        bool* h_visited = (bool*)calloc(input.size, sizeof(bool));
        int* h_cost = (int*)calloc(input.size, sizeof(int));

        h_frontier[initial_vertex] = true;

        node* Va;
        HANDLE_ERROR(cudaMalloc((void**)&Va, sizeof(node) * input.size));
        HANDLE_ERROR(cudaMemcpy(Va, h_nodes, sizeof(node) * input.size, cudaMemcpyHostToDevice));
        int* Ea;
        HANDLE_ERROR(cudaMalloc((void**)&Ea, sizeof(node) * input.size));
        HANDLE_ERROR(cudaMemcpy(Ea, h_edges, sizeof(node) * input.size, cudaMemcpyHostToDevice));
        bool* Fa;
        HANDLE_ERROR(cudaMalloc((void**)&Fa, sizeof(bool) * input.size));
        HANDLE_ERROR(cudaMemcpy(Fa, h_frontier, sizeof(bool) * input.size, cudaMemcpyHostToDevice));
        bool* Xa;
        HANDLE_ERROR(cudaMalloc((void**)&Xa, sizeof(bool) * input.size));
        HANDLE_ERROR(cudaMemcpy(Xa, h_visited, sizeof(bool) * input.size, cudaMemcpyHostToDevice));
        int* Ca;
        HANDLE_ERROR(cudaMalloc((void**)&Ca, sizeof(int) * input.size));
        HANDLE_ERROR(cudaMemcpy(Ca, h_cost, sizeof(int) * input.size, cudaMemcpyHostToDevice));

        uint edge_index = 0;
        for (uint i = 0; i < input.size; i++) {
            uint connected_edges = 0;
            for (uint j = 0; j < input.size; j++) {
                if (input.matrix[i][j] != 0) {
                    connected_edges++;
                    h_edges[edge_index++] = j;
                }
            }

            h_nodes[i].first = i;
            h_nodes[i].second = connected_edges;
            h_frontier[i] = false;
            h_visited[i] = false;
        }

        bool* d_over;
        HANDLE_ERROR(cudaMalloc((void**)&d_over, sizeof(bool)));

        dim3 grid(num_blocks, 1, 1);
        dim3 threads(block_size, 1, 1);

        auto initial_time = get_time();

        int k = 0;
        bool stop;

        do {
            stop = true;

            HANDLE_ERROR(cudaMemcpy(d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice));
            bfs_kernel<<<grid, threads>>>(Va, Ea, Fa, Xa, Ca, input.size, d_over);
            HANDLE_ERROR(cudaMemcpy(&stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost));

            k++;
        } while (!stop);

        HANDLE_ERROR(cudaGetLastError());
        HANDLE_ERROR(cudaDeviceSynchronize());

        fmt::print("time: {}", get_time() - initial_time);

        free(h_nodes);
        free(h_edges);
        free(h_frontier);
        free(h_visited);
        free(h_cost);
        HANDLE_ERROR(cudaFree(Va));
        HANDLE_ERROR(cudaFree(Ea));
        HANDLE_ERROR(cudaFree(Fa));
        HANDLE_ERROR(cudaFree(Xa));
        HANDLE_ERROR(cudaFree(Ca));
        HANDLE_ERROR(cudaFree(d_over));

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
