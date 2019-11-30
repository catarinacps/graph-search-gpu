#include "host.h"

namespace gsg {

namespace cpu {

    bool bfs(const graph& input, uint32_t searched_vertex, uint32_t initial_vertex, bool verbose)
    {
        std::vector<bool> visited(input.size, false);
        std::queue<uint32_t> order;

        order.push(initial_vertex);
        visited[initial_vertex] = true;

        auto initial_time = get_time();

        while (!order.empty()) {
            auto current_vertex = order.front();
            order.pop();

            if (verbose)
                fmt::print("visited vertex {}...\n", current_vertex);

            if (current_vertex == searched_vertex) {
                if (verbose)
                    fmt::print("and found the objetive ({})\n", searched_vertex);

                fmt::print("time: {}", get_time() - initial_time);

                return true;
            }

            for (uint32_t vertex = 0, edge; vertex < input.size; ++vertex) {
                edge = input.matrix[current_vertex][vertex];

                if (vertex != current_vertex and edge != 0) {
                    if (!visited[vertex]) {
                        visited[vertex] = true;
                        order.push(vertex);
                    }
                }
            }
        }

        if (verbose)
            fmt::print("\ndid not found {}\n", searched_vertex);

        fmt::print("time: {}", get_time() - initial_time);

        return false;
    }

    bool floyd_warshall(const graph& input, bool verbose)
    {
        gsg::graph distances(input);

        auto initial_time = gsg::get_time();

        for (uint k = 0; k < input.size; k++) {
            for (uint i = 0; i < input.size; i++) {
                for (uint j = 0; j < input.size; j++) {
                    if (distances.matrix[i][k] + distances.matrix[k][j] < distances.matrix[i][j]) {
                        if (verbose)
                            fmt::print("k ({}) is in the shortest path between i ({}) and j ({})\n", k, i, j);
                        distances.matrix[i][j] = distances.matrix[i][k] + distances.matrix[k][j];
                    }
                }
            }
        }

        fmt::print("time: {}", get_time() - initial_time);

        return true;
    }
}

}
