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
    uint32_t num_edges;
    std::vector<int*> matrix;

    /** Simple graph constructor.
     *
     * @param num_nodes_p the desired graph cardinality
     */
    graph(uint32_t num_nodes_p);

    /** Simple graph copy constructor.
     *
     * @param copy the graph to be copied
     */
    graph(const graph& copy);

    /* Default move constructor. */
    graph(graph&& move) = default;

    /* No empty constructor! */
    graph() = delete;

    /* Frees the memory on the member data. */
    ~graph();
};

}
