#pragma once

#include <stdio.h>
#include <sys/time.h>

#include <fmt/core.h>

#include "utils/graph.h"

namespace gsg {

double get_time(void);

void print_graph(const graph& input);

}
