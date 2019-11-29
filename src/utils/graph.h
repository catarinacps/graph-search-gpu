#pragma once

#include <memory>
#include <vector>
#include <cstring>

namespace gsg {

struct graph {
private:
    int* data;

public:
    const uint32_t size;
    std::vector<int*> matrix;

    graph(uint32_t num_nodes_p)
        : data((int*)calloc(num_nodes_p * num_nodes_p, sizeof(int)))
        , size(num_nodes_p)
        , matrix(num_nodes_p)
    {
        for (uint i = 0; i < this->size; i++)
            matrix[i] = &data[i * this->size];
    }

    graph(const graph& copy)
        : data((int*)calloc(copy.size * copy.size, sizeof(int)))
        , size(copy.size)
        , matrix(copy.size)
    {
        for (uint i = 0; i < this->size; i++)
            std::memcpy(matrix[i], copy.matrix[i], this->size * sizeof(int));
    }

    ~graph()
    {
        delete data;
    }
};

}
