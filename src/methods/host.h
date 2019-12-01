#pragma once

#include <queue>
#include <vector>

#include <fmt/core.h>

#include "utils/graph.h"
#include "utils/utils.h"

namespace gsg::cpu {

bool bfs(const graph& input, uint searched_vertex, uint initial_vertex, bool verbose = false);

bool floyd_warshall(const graph& input, bool verbose = false);

}
