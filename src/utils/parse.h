#pragma once

#include <fstream>
#include <memory>
#include <string>

#include "graph.h"

namespace gsg {

std::unique_ptr<graph> parse_file(const std::string& path);

}
