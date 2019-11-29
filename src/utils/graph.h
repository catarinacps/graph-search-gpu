#pragma once

#include <memory>
#include <vector>

namespace gsg {

struct graph {
public:
    const uint32_t num_nodes;
    std::vector<int*> matrix;

    graph(uint32_t num_nodes_p)
        : num_nodes(num_nodes_p)
        , matrix(num_nodes_p)
    {
        auto temp_array = (int*)calloc(this->num_nodes * this->num_nodes, sizeof(int));

        for (uint i = 0; i < this->num_nodes; i++)
            matrix[i] = &temp_array[i * this->num_nodes];
    }

    ~graph()
    {
        delete &matrix[0][0];
    }
};

}
