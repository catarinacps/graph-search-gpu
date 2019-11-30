#pragma once

#define HANDLE_ERROR(error)                                     \
    {                                                           \
        if (error != cudaSuccess) {                             \
            fprintf(stderr, "%s in %s at line %d\n",            \
                cudaGetErrorString(error), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    }
