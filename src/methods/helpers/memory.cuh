#pragma once

#include <memory>

#include "error.cuh"
#include "utils/graph.h"

namespace gsg::cuda {

size_t move_to_device(const graph& input, int** device);

void move_from_device(graph& input, int* device, size_t pitch);

}
