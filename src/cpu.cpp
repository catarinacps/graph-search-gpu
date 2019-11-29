#include <iostream>
#include <string>

#include <clipp/clipp.h>
#include <fmt/core.h>

#include "utils/parse.h"
#include "utils/graph.h"
#include "methods/methods.h"

using namespace clipp;

enum class method {
    bfs,
    floyd,
    help
};

int main(int argc, char* argv[])
{
    uint32_t searched_vertex = 5000, initial_vertex = 0;
    auto selected = method::help;
    bool verbose = false;
    std::string path_to_instance;

    auto first_opt = (
        option("-v", "--verbose").set(verbose) % "show detailed output");

    auto methods = (
        (command("bfs").set(selected, method::bfs) % "use the BFS method to find the element" |
         command("floyd").set(selected, method::floyd) % "use the floyd method to find the element") % "possible methods:",
        (option("-v", "--vertex") & value("VERTEX=5000", searched_vertex)) % "the vertex being searched for (default: 5000)",
        (option("-i", "--initial") & value("INITIAL=0", initial_vertex)) % "where to start from (default: 0)");

    auto cli = (
        command("help").set(selected, method::help) |
        (first_opt,
         methods & value("instance", path_to_instance) % "path to the graph instance"));

    if (parse(argc, argv, cli)) {
        switch (selected) {
        case method::bfs: {
            auto graph = gsg::parse_file(path_to_instance);

            bool ret = gsg::bfs(*graph.get(), searched_vertex, initial_vertex, verbose);

            if (verbose) {
                if (ret) {
                    fmt::print("all is well, and the element is found\n");
                } else {
                    fmt::print("sad because no element\n");
                }
            }
        } break;
        case method::floyd: {
            fmt::print("hiya!");
        } break;
        case method::help: {
            std::cout << make_man_page(cli, argv[0]).prepend_section("DESCRIPTION", "        finds a element in a graph");
        } break;
        }
    } else {
        fmt::print("Please follow the following usage lines:\n");
        std::cout << usage_lines(cli, argv[0]) << std::endl;
    }
}
