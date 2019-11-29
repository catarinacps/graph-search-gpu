#pragma once

#include <vector>
#include <queue>

#include <fmt/core.h>

#include "utils/graph.h"
#include "utils/utils.h"

namespace gsg {

bool bfs(const graph& input, int element, bool verbose = false);

}
