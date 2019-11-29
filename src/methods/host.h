#pragma once

#include <queue>
#include <vector>

#include <fmt/core.h>

#include "utils/graph.h"
#include "utils/utils.h"

namespace gsg {

bool bfs(const graph& input, uint32_t searched_vertex, uint32_t initial_vertex, bool verbose = false);

bool floyd_warshall(const graph& input, bool verbose = false);

}
