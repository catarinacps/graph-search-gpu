#pragma once

#include <fmt/core.h>
#include <memory>

#include "helpers/error.cuh"
#include "helpers/memory.cuh"
#include "utils/graph.h"
#include "utils/utils.h"

namespace gsg {

namespace cuda {

    bool bfs(const graph& input_host, uint32_t searched_vertex, uint32_t initial_vertex, bool verbose);

    bool floyd_warshall(const graph& input_host, uint block_size, bool verbose);
}

}
