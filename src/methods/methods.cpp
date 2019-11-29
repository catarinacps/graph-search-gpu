#include "methods.h"

namespace gsg {

bool bfs(const graph& input, int element, bool verbose)
{
    std::vector<bool> visited(input.num_nodes, false);
    std::queue<int> order;

    order.push(0);
    visited[0] = true;

    auto initial_time = get_time();

    while (!order.empty()) {
        auto current_node = order.front();
        order.pop();

        if (verbose)
            fmt::print("visited node {}...\n", current_node);

        if (current_node == element) {
            if (verbose)
                fmt::print("and found the objetive ({})\n", element);

            fmt::print("\ntime: {}", get_time() - initial_time);

            return true;
        }

        for (uint32_t vertex = 0, edge; vertex < input.num_nodes; ++vertex) {
            edge = input.matrix[current_node][vertex];

            if (vertex != current_node and edge != 0) {
                if (!visited[vertex]) {
                    visited[vertex] = true;
                    order.push(vertex);
                }
            }
        }
    }

    if (verbose)
        fmt::print("\ndid not found {}", element);

    fmt::print("\ntime: {}", get_time() - initial_time);

    return false;
}

}
