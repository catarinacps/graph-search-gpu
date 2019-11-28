#pragma once

#include <fstream>
#include <string>
#include <memory>

#include <cxxopts/cxxopts.hpp>

std::unique_ptr< parse_file(const std::string& path);

cxxopts::ParseResult parse_options(int argc, char* argv[]);
