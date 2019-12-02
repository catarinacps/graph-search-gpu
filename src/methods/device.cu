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
        uint* Ea, // edges
        bool* Fa, // frontier
        bool* Xa, // visited
        int* Ca, // cost
        uint* num_nodes,
        bool* done)
    {
        int id = threadIdx.x + blockIdx.x * blockDim.x;

        if (id >= *num_nodes) {
            return;
        } else if (Fa[id] == true && Xa[id] == false) {
            Fa[id] = false;
            Xa[id] = true;
            __syncthreads();
            uint start = Va[id].first;
            uint end = start + Va[id].second;
            for (uint i = start; i < end; i++) {
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

        uint num_blocks = (uint)ceil(input.size / (double)block_size);

        node* h_nodes = (node*)calloc(input.size, sizeof(node));
        uint* h_edges = (uint*)calloc(2 * input.num_edges, sizeof(uint));
        bool* h_frontier = (bool*)calloc(input.size, sizeof(bool));
        bool* h_visited = (bool*)calloc(input.size, sizeof(bool));
        int* h_cost = (int*)calloc(input.size, sizeof(int));

        h_frontier[initial_vertex] = true;

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

        for (uint i = 0; i < input.size; i++)
            if (input.matrix[initial_vertex][i] != 0)
                h_frontier[i] = true;

        node* Va;
        HANDLE_ERROR(cudaMalloc((void**)&Va, sizeof(node) * input.size));
        HANDLE_ERROR(cudaMemcpy(Va, h_nodes, sizeof(node) * input.size, cudaMemcpyHostToDevice));
        uint* Ea;
        HANDLE_ERROR(cudaMalloc((void**)&Ea, sizeof(uint) * 2 * input.num_edges));
        HANDLE_ERROR(cudaMemcpy(Ea, h_edges, sizeof(uint) * 2 * input.num_edges, cudaMemcpyHostToDevice));
        bool* Fa;
        HANDLE_ERROR(cudaMalloc((void**)&Fa, sizeof(bool) * input.size));
        HANDLE_ERROR(cudaMemcpy(Fa, h_frontier, sizeof(bool) * input.size, cudaMemcpyHostToDevice));
        bool* Xa;
        HANDLE_ERROR(cudaMalloc((void**)&Xa, sizeof(bool) * input.size));
        HANDLE_ERROR(cudaMemcpy(Xa, h_visited, sizeof(bool) * input.size, cudaMemcpyHostToDevice));
        int* Ca;
        HANDLE_ERROR(cudaMalloc((void**)&Ca, sizeof(int) * input.size));
        HANDLE_ERROR(cudaMemcpy(Ca, h_cost, sizeof(int) * input.size, cudaMemcpyHostToDevice));

        bool* d_over;
        HANDLE_ERROR(cudaMalloc((void**)&d_over, sizeof(bool)));

        uint* d_num_nodes;
        HANDLE_ERROR(cudaMalloc((void**)&d_num_nodes, sizeof(uint)));
        HANDLE_ERROR(cudaMemcpy(d_num_nodes, &input.size, sizeof(uint), cudaMemcpyHostToDevice));

        auto initial_time = get_time();

        int k = 0;
        bool stop;

        do {
            stop = true;

            HANDLE_ERROR(cudaMemcpy(d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice));
            bfs_kernel<<<num_blocks, block_size>>>(Va, Ea, Fa, Xa, Ca, d_num_nodes, d_over);
            HANDLE_ERROR(cudaMemcpy(&stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost));

            k++;
        } while (!stop);

        HANDLE_ERROR(cudaGetLastError());
        HANDLE_ERROR(cudaDeviceSynchronize());

        fmt::print("time: {}", get_time() - initial_time);

        HANDLE_ERROR(cudaMemcpy(h_cost, Ca, sizeof(int) * input.size, cudaMemcpyDeviceToHost));

        if (verbose) {
            fmt::print("\nnumber of times the kernel is called : {}\n", k);

            fmt::print("\ncost:\n");
            for (uint i = 0; i < input.size; i++)
                fmt::print("{} ", h_cost[i]);
        }

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
