#include "parse.h"

cxxopts::ParseResult parse_options(int argc, char* argv[])
{
    try {
        cxxopts::Options options(argv[0], " - search a graph");

        options
            .positional_help("[optional arguments]")
            .show_positional_help();

        options
            .add_options()
            ("e, element",
             "the element being searched",
             cxxopts::value<int>()->default_value("5000"));

        options
            .add_options("GPU options")
            ("b, block",
             "the block size used",
             cxxopts::value<int>());

        options.parse_positional({"method", "instance"});

        return options.parse(argc, argv);
    } catch (const cxxopts::OptionException& e) {
        printf("oh well");
    }
}

void parse_file(const std::string& path)
{
    return;
}
