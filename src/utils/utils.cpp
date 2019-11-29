#include "utils.h"

namespace gsg {

double get_time(void)
{
    struct timeval tr;
    gettimeofday(&tr, NULL);
    return (double)tr.tv_sec + (double)tr.tv_usec / 1000000;
}

void print_graph(const graph& input)
{
    for (uint i = 0; i < input.size; i++) {
        for (uint j = 0; j < input.size; j++) {
            if (input.matrix[i][j] == 0)
                fmt::print("INF\t");
            else
                fmt::print("{}\t", input.matrix[i][j]);
        }
        fmt::print("\n");
    }
}

}
