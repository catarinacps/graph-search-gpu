#pragma once

#include <fmt/core.h>
#include <memory>

#include "helpers/error.cuh"
#include "helpers/memory.cuh"
#include "utils/graph.h"

namespace gsg::cuda {

void bfs(const graph& input_host, uint32_t searched_vertex, uint32_t initial_vertex, bool verbose);

void floyd_warshall(graph& input_host, uint block_size, bool verbose);

}
