#include "graph.h"

namespace gsg {

graph::graph(uint32_t num_nodes_p)
    : data((int*)calloc(num_nodes_p * num_nodes_p, sizeof(int)))
    , size(num_nodes_p)
    , num_edges(0)
    , matrix(num_nodes_p)
{
    for (uint i = 0; i < this->size; i++)
        matrix[i] = &data[i * this->size];
}

graph::graph(const graph& copy)
    : data((int*)calloc(copy.size * copy.size, sizeof(int)))
    , size(copy.size)
    , num_edges(copy.num_edges)
    , matrix(copy.size)
{
    std::memcpy(data, copy.data, copy.size * copy.size * sizeof(int));
    for (uint i = 0; i < this->size; i++)
        matrix[i] = &data[i * this->size];
}

graph::~graph()
{
    delete data;
}

}
