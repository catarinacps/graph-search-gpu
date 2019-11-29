#include "methods.h"

namespace gsg {

bool bfs(const graph& input, uint32_t searched_vertex, uint32_t initial_vertex, bool verbose)
{
    std::vector<bool> visited(input.num_nodes, false);
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

            fmt::print("\ntime: {}", get_time() - initial_time);

            return true;
        }

        for (uint32_t vertex = 0, edge; vertex < input.num_nodes; ++vertex) {
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
        fmt::print("\ndid not found {}", searched_vertex);

    fmt::print("\ntime: {}", get_time() - initial_time);

    return false;
}

}
