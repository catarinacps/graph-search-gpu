#include "parse.h"

namespace gsg {

std::unique_ptr<graph> parse_file(const std::string& path)
{
    std::ifstream file(path);
    uint32_t num_nodes;

    file >> num_nodes;

    auto read_graph = std::make_unique<graph>(num_nodes);

    int pos_a, pos_b, weight;

    while (file >> pos_a >> pos_b >> weight) {
        read_graph->matrix[pos_a][pos_b] = weight;
        read_graph->matrix[pos_b][pos_a] = weight;
        read_graph->num_edges++;
    }

    return read_graph;
}

}
