#pragma once

#include <memory>
#include <vector>

namespace gsg {

class graph {
    uint32_t num_nodes;

public:
    std::vector<std::vector<int>> matrix;

    graph(uint32_t num_nodes_p)
        : num_nodes(num_nodes_p)
        , matrix(num_nodes_p, std::vector<int>(num_nodes_p, 0))
    {
    }
};
}
