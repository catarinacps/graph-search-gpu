#include <iostream>
#include <string>

#include <clipp/clipp.h>
#include <fmt/core.h>

#include "utils/utils.h"
#include "utils/parse.h"
#include "utils/graph.h"
#include "methods/host.h"

using namespace clipp;

enum class method {
    bfs,
    floyd,
    help
};

enum class imp {
    cpu,
    cuda
};

int main(int argc, char* argv[])
{
    uint32_t searched_vertex = 5000, initial_vertex = 0;
    auto selected_m = method::help;
    auto selected_i = imp::cpu;
    bool verbose = false;
    std::string path_to_instance;

    auto first_opt = (
        option("-v", "--verbose").set(verbose) % "show detailed output");

    auto implementations = "possible implementations" % (
        command("cpu").set(selected_i, imp::cpu) % "use the implementation in CPU host code" |
        command("cuda").set(selected_i, imp::cuda) % "use the implementation in CUDA device code");

    auto methods = (
        (command("bfs").set(selected_m, method::bfs) % "use the BFS method to find the element" |
         command("floyd").set(selected_m, method::floyd) % "use the floyd-warshall method to find the element") % "possible methods:",
        (option("-s", "--search") & value("VERTEX", searched_vertex)) % "the vertex being searched for (default: 5000)",
        (option("-i", "--initial") & value("INITIAL", initial_vertex)) % "where to start from (default: 0)");

    auto cli = (
        command("help").set(selected_m, method::help) |
        (first_opt,
         implementations,
         methods & value("instance", path_to_instance) % "path to the graph instance"));

    if (parse(argc, argv, cli)) {
        switch (selected_m) {
        case method::bfs: {
            auto graph = gsg::parse_file(path_to_instance);

            bool ret;
            if (selected_i == imp::cpu) {
                ret = gsg::cpu::bfs(*graph, searched_vertex, initial_vertex, verbose);
            } else {
            }

            if (verbose) {
                if (ret) {
                    fmt::print("\n\nall is well, and the vertex is found\n");
                } else {
                    fmt::print("\n\nsad because no vertex\n");
                }
            }

            std::cout << std::endl;
            return ret ? 0 : 1;
        } break;
        case method::floyd: {
            auto graph = gsg::parse_file(path_to_instance);

            bool ret;
            if (selected_i == imp::cpu) {
                ret = gsg::cpu::floyd_warshall(*graph, verbose);
            } else {
            }

            if (verbose) {
                if (ret) {
                    fmt::print("\n\nall is well, and the vertex is found\n");
                } else {
                    fmt::print("\n\nsad because no vertex\n");
                }
            }

            std::cout << std::endl;
            return ret ? 0 : 1;
        } break;
        case method::help: {
            std::cout << make_man_page(cli, argv[0])
                .prepend_section("DESCRIPTION",
                                 "        "
                                 "finds an element in a graph using different methods\n"
                                 "        "
                                 "these methods are intended as a way of comparing the\n"
                                 "        "
                                 "different effectiveness in visiting a whole graph");
        } break;
        }
    } else {
        fmt::print("Please follow the following usage lines:\n");
        fmt::print(usage_lines(cli, argv[0]).str());
    }

    std::cout << std::endl;
    return 0;
}
