#pragma once

#include <fstream>
#include <memory>
#include <string>

#include "graph.h"

std::unique_ptr<gsg::graph> parse_file(const std::string& path);
