#include "memory.cuh"

namespace gsg::cuda {

size_t move_to_device(const graph& input, int** device)
{
    size_t height = input.size;
    size_t width = input.size * sizeof(int);
    size_t pitch;

    HANDLE_ERROR(cudaMallocPitch(device, &pitch, width, height));

    HANDLE_ERROR(cudaMemcpy2D(*device, pitch, &input.matrix[0][0], width, width, height, cudaMemcpyHostToDevice));

    return pitch;
}

void move_from_device(graph& input, int* device, size_t pitch)
{
    size_t height = input.size;
    size_t width = height * sizeof(int);

    HANDLE_ERROR(cudaMemcpy2D(&input.matrix[0][0], width, device, pitch, width, height, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(device));
}
}
