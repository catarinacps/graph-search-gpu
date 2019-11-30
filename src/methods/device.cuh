#pragma once

#include <fmt/core.h>
#include <memory>

#include "utils/graph.h"

#define HANDLE_ERROR(error)                                     \
    {                                                           \
        if (error != cudaSuccess) {                             \
            fprintf(stderr, "%s in %s at line %d\n",            \
                cudaGetErrorString(error), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    }

namespace gsg {

namespace cuda {

    void bfs(const graph& input_host, uint32_t searched_vertex, uint32_t initial_vertex, bool verbose);

    void floyd_warshall(const graph& input_host, bool verbose);

}
}
