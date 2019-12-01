#pragma once

#include <fmt/core.h>
#include <memory>
#include <utility>

#include "helpers/error.cuh"
#include "helpers/memory.cuh"
#include "utils/graph.h"
#include "utils/utils.h"

namespace gsg {

namespace cuda {

    // pair containing vertex num and num of connected edges
    using node = std::pair<int, int>;

    bool bfs(const graph& input, uint searched_vertex, uint initial_vertex, uint block_size, bool verbose);

    bool floyd_warshall(const graph& input_host, uint block_size, bool verbose);
}

}
