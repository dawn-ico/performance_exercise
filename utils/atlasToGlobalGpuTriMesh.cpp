#include "atlasToGlobalGpuTriMesh.h"

#include "driver-includes/unstructured_interface.hpp"
#include "interface/atlas_interface.hpp"

#include <unordered_map>
#include <utility>

static std::map<std::vector<dawn::LocationType>, size_t> sizeCatalogue() {
  static std::map<std::vector<dawn::LocationType>, size_t> sizeMap({
      {{dawn::LocationType::Edges, dawn::LocationType::Cells}, 2},
      {{dawn::LocationType::Edges, dawn::LocationType::Vertices}, 2},
      {{dawn::LocationType::Cells, dawn::LocationType::Edges}, 3},
      {{dawn::LocationType::Cells, dawn::LocationType::Vertices}, 3},
      {{dawn::LocationType::Vertices, dawn::LocationType::Edges}, 6},
      {{dawn::LocationType::Vertices, dawn::LocationType::Cells}, 6},
      {{dawn::LocationType::Edges, dawn::LocationType::Cells, dawn::LocationType::Vertices}, 4},
      {{dawn::LocationType::Edges, dawn::LocationType::Cells, dawn::LocationType::Edges}, 4},
  });
  return sizeMap;
}

static void addNbhListToGlobalMesh(const atlas::Mesh& atlasMesh, dawn::GlobalGpuTriMesh& globalMesh,
                                   const std::vector<dawn::LocationType>& chain, size_t size) {

  auto denseSize = [&atlasMesh](dawn::LocationType denseLoc) {
    switch(denseLoc) {
    case dawn::LocationType::Edges:
      return atlasMesh.edges().size();
    case dawn::LocationType::Cells:
      return atlasMesh.cells().size();
    case dawn::LocationType::Vertices:
      return atlasMesh.nodes().size();
    default:
      assert(false);
      return -1;
    }
  };

  gpuErrchk(cudaMalloc((void**)&globalMesh.NeighborTables[chain],
                       sizeof(int) * denseSize(chain[0]) * size));
  dawn::generateNbhTable<atlasInterface::atlasTag>(atlasMesh, chain, denseSize(chain[0]), size,
                                                   globalMesh.NeighborTables.at(chain));
}

dawn::GlobalGpuTriMesh atlasToGlobalGpuTriMesh(const atlas::Mesh& mesh) {
  dawn::GlobalGpuTriMesh retMesh = {.NumEdges = mesh.edges().size(),
                                    .NumCells = mesh.cells().size(),
                                    .NumVertices = mesh.nodes().size()};
  for(const auto [chain, size] : sizeCatalogue()) {
    addNbhListToGlobalMesh(mesh, retMesh, chain, size);
  }
  return retMesh;
}