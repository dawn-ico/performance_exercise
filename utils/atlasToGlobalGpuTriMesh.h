#pragma once

#include <atlas/mesh.h>

#include "driver-includes/cuda_utils.hpp"

dawn::GlobalGpuTriMesh atlasToGlobalGpuTriMesh(const atlas::Mesh& mesh);