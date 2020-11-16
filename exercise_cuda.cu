#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#define GRIDTOOLS_DAWN_NO_INCLUDE
#include "driver-includes/math.hpp"
#include "driver-includes/timer_cuda.hpp"
using namespace gridtools::dawn;
#define BLOCK_SIZE 16

// To enable k-loop parallelization, put to 1
#define LEVELS_PER_THREAD 80

static const int C_E_SIZE = 3;
static const int E_C_SIZE = 2;
static const int E_V_SIZE = 2;
static const int V_E_SIZE = 6;

//
// START OF KERNELS
//

// curl_vec = sum_over(Vertex > Edge, vec * geofac_curl)
__global__ void kernel1(
    int NumVertices, int NumEdges, int kSize, const int *veTable,
    const ::dawn::float_type *__restrict__ vec,
    const ::dawn::float_type *__restrict__ geofac_curl,
    ::dawn::float_type *__restrict__ curl_vec)
{
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int kidx = blockIdx.y * blockDim.y + threadIdx.y;
  int klo = kidx * LEVELS_PER_THREAD + 0;
  int khi = (kidx + 1) * LEVELS_PER_THREAD + 0;
  if (pidx >= NumVertices)
    return;

  for (int kIter = klo; kIter < khi; kIter++)
  {
    if (kIter >= kSize + 0)
      return;

    ::dawn::float_type curl_vec_red = (::dawn::float_type)0;
    for (int nbhIter = 0; nbhIter < V_E_SIZE; nbhIter++)
    {
      int nbhIdx = veTable[pidx * V_E_SIZE + nbhIter];
      if (nbhIdx == DEVICE_MISSING_VALUE)
        continue;

      curl_vec_red += (vec[kIter * NumEdges + nbhIdx] *
                       geofac_curl[nbhIter * NumVertices + pidx]);
    }
    curl_vec[kIter * NumVertices + pidx] = curl_vec_red;
  }
}

// div_vec = sum_over(Cell > Edge, vec * geofac_div)
__global__ void kernel2(
    int NumCells, int NumEdges, int kSize, const int *ceTable,
    const ::dawn::float_type *__restrict__ vec,
    const ::dawn::float_type *__restrict__ geofac_div,
    ::dawn::float_type *__restrict__ div_vec)
{
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int kidx = blockIdx.y * blockDim.y + threadIdx.y;
  int klo = kidx * LEVELS_PER_THREAD + 0;
  int khi = (kidx + 1) * LEVELS_PER_THREAD + 0;
  if (pidx >= NumCells)
    return;

  for (int kIter = klo; kIter < khi; kIter++)
  {
    if (kIter >= kSize + 0)
      return;

    ::dawn::float_type div_vec_red = (::dawn::float_type)0;
    for (int nbhIter = 0; nbhIter < C_E_SIZE; nbhIter++)
    {
      int nbhIdx = ceTable[pidx * C_E_SIZE + nbhIter];
      if (nbhIdx == DEVICE_MISSING_VALUE)
        continue;

      div_vec_red += (vec[kIter * NumEdges + nbhIdx] *
                      geofac_div[nbhIter * NumCells + pidx]);
    }
    div_vec[kIter * NumCells + pidx] = div_vec_red;
  }
}

// nabla2t1_vec = sum_over(Edge > Vertex, curl_vec, weights=[-1.0, 1])
__global__ void kernel3(
    int NumEdges, int NumVertices, int kSize, const int *evTable,
    const ::dawn::float_type *__restrict__ curl_vec,
    ::dawn::float_type *__restrict__ nabla2t1_vec)
{
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int kidx = blockIdx.y * blockDim.y + threadIdx.y;
  int klo = kidx * LEVELS_PER_THREAD + 0;
  int khi = (kidx + 1) * LEVELS_PER_THREAD + 0;
  if (pidx >= NumEdges)
    return;

  for (int kIter = klo; kIter < khi; kIter++)
  {
    if (kIter >= kSize + 0)
      return;

    ::dawn::float_type nabla2t1_vec_red = (::dawn::float_type)0;
    ::dawn::float_type nabla2t1_vec_weights[2] = {(-(::dawn::float_type)1.0), (int)1};
    for (int nbhIter = 0; nbhIter < E_V_SIZE; nbhIter++)
    {
      int nbhIdx = evTable[pidx * E_V_SIZE + nbhIter];
      if (nbhIdx == DEVICE_MISSING_VALUE)
        continue;

      nabla2t1_vec_red +=
          nabla2t1_vec_weights[nbhIter] * curl_vec[kIter * NumVertices + nbhIdx];
    }
    nabla2t1_vec[kIter * NumEdges + pidx] = nabla2t1_vec_red;
  }
}

// nabla2t1_vec = tangent_orientation * nabla2t1_vec / primal_edge_length
__global__ void kernel4(
    int NumEdges, int kSize,
    const ::dawn::float_type *__restrict__ primal_edge_length,
    const ::dawn::float_type *__restrict__ tangent_orientation,
    ::dawn::float_type *__restrict__ nabla2t1_vec)
{
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int kidx = blockIdx.y * blockDim.y + threadIdx.y;
  int klo = kidx * LEVELS_PER_THREAD + 0;
  int khi = (kidx + 1) * LEVELS_PER_THREAD + 0;
  if (pidx >= NumEdges)
    return;

  for (int kIter = klo; kIter < khi; kIter++)
  {
    if (kIter >= kSize + 0)
      return;

    nabla2t1_vec[kIter * NumEdges + pidx] =
        ((tangent_orientation[pidx] *
          nabla2t1_vec[kIter * NumEdges + pidx]) /
         primal_edge_length[pidx]);
  }
}

// nabla2t2_vec = sum_over(Edge > Cell, div_vec, weights=[-1.0, 1])
__global__ void kernel5(
    int NumEdges, int NumCells, int kSize, const int *ecTable,
    const ::dawn::float_type *__restrict__ div_vec,
    ::dawn::float_type *__restrict__ nabla2t2_vec)
{
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int kidx = blockIdx.y * blockDim.y + threadIdx.y;
  int klo = kidx * LEVELS_PER_THREAD + 0;
  int khi = (kidx + 1) * LEVELS_PER_THREAD + 0;
  if (pidx >= NumEdges)
    return;

  for (int kIter = klo; kIter < khi; kIter++)
  {
    if (kIter >= kSize + 0)
      return;

    ::dawn::float_type nabla2t2_vec_red = (::dawn::float_type)0;
    ::dawn::float_type nabla2t2_vec_weights[2] = {(-(::dawn::float_type)1.0), (int)1};
    for (int nbhIter = 0; nbhIter < E_C_SIZE; nbhIter++)
    {
      int nbhIdx = ecTable[pidx * E_C_SIZE + nbhIter];
      if (nbhIdx == DEVICE_MISSING_VALUE)
        continue;

      nabla2t2_vec_red +=
          nabla2t2_vec_weights[nbhIter] * div_vec[kIter * NumCells + nbhIdx];
    }
    nabla2t2_vec[kIter * NumEdges + pidx] = nabla2t2_vec_red;
  }
}

// nabla2t2_vec = tangent_orientation * nabla2t2_vec / dual_edge_length
__global__ void kernel6(
    int NumEdges, int kSize,
    const ::dawn::float_type *__restrict__ dual_edge_length,
    const ::dawn::float_type *__restrict__ tangent_orientation,
    ::dawn::float_type *__restrict__ nabla2t2_vec)
{
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int kidx = blockIdx.y * blockDim.y + threadIdx.y;
  int klo = kidx * LEVELS_PER_THREAD + 0;
  int khi = (kidx + 1) * LEVELS_PER_THREAD + 0;
  if (pidx >= NumEdges)
    return;

  for (int kIter = klo; kIter < khi; kIter++)
  {
    if (kIter >= kSize + 0)
      return;

    nabla2t2_vec[kIter * NumEdges + pidx] =
        ((tangent_orientation[pidx] *
          nabla2t2_vec[kIter * NumEdges + pidx]) /
         dual_edge_length[pidx]);
  }
}

// nabla2_vec = nabla2t2_vec - nabla2t1_vec
__global__ void kernel7(
    int NumEdges, int kSize, ::dawn::float_type *__restrict__ nabla2_vec,
    const ::dawn::float_type *__restrict__ nabla2t1_vec,
    const ::dawn::float_type *__restrict__ nabla2t2_vec)
{
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int kidx = blockIdx.y * blockDim.y + threadIdx.y;
  int klo = kidx * LEVELS_PER_THREAD + 0;
  int khi = (kidx + 1) * LEVELS_PER_THREAD + 0;
  if (pidx >= NumEdges)
    return;

  for (int kIter = klo; kIter < khi; kIter++)
  {
    if (kIter >= kSize + 0)
      return;

    nabla2_vec[kIter * NumEdges + pidx] =
        (nabla2t2_vec[kIter * NumEdges + pidx] -
         nabla2t1_vec[kIter * NumEdges + pidx]);
  }
}

// interp_out = sum_over(Edge > Cell, interp_in * interp_coeff)
__global__ void kernel8(
    int NumEdges, int NumCells, int kSize, const int *ecTable,
    const ::dawn::float_type *__restrict__ interp_in,
    const ::dawn::float_type *__restrict__ interp_coeff,
    ::dawn::float_type *__restrict__ interp_out)
{
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int kidx = blockIdx.y * blockDim.y + threadIdx.y;
  int klo = kidx * LEVELS_PER_THREAD + 0;
  int khi = (kidx + 1) * LEVELS_PER_THREAD + 0;
  if (pidx >= NumEdges)
    return;

  for (int kIter = klo; kIter < khi; kIter++)
  {
    if (kIter >= kSize + 0)
      return;

    ::dawn::float_type interp_out_red = (::dawn::float_type)0;
    for (int nbhIter = 0; nbhIter < E_C_SIZE; nbhIter++)
    {
      int nbhIdx = ecTable[pidx * E_C_SIZE + nbhIter];
      if (nbhIdx == DEVICE_MISSING_VALUE)
        continue;

      interp_out_red += (interp_in[kIter * NumCells + nbhIdx] *
                         interp_coeff[nbhIter * NumEdges + pidx]);
    }
    interp_out[kIter * NumEdges + pidx] = interp_out_red;
  }
}

// STUBS

// nabla2t1_vec = sum_over(Edge > Vertex, curl_vec, weights=[-1.0, 1])
// nabla2t1_vec = tangent_orientation * nabla2t1_vec / primal_edge_length
__global__ void kernel_3_and_4(
    int NumEdges, int NumVertices, int kSize, const int *evTable,
    const ::dawn::float_type *__restrict__ curl_vec,
    const ::dawn::float_type *__restrict__ primal_edge_length,
    const ::dawn::float_type *__restrict__ tangent_orientation,
    ::dawn::float_type *__restrict__ nabla2t1_vec)
{
  // Add implementation
}

// nabla2t2_vec = sum_over(Edge > Cell, div_vec, weights=[-1.0, 1])
// nabla2t2_vec = tangent_orientation * nabla2t2_vec / dual_edge_length
__global__ void kernel_5_and_6(
    int NumEdges, int NumCells, int kSize, const int *ecTable,
    const ::dawn::float_type *__restrict__ div_vec,
    const ::dawn::float_type *__restrict__ dual_edge_length,
    const ::dawn::float_type *__restrict__ tangent_orientation,
    ::dawn::float_type *__restrict__ nabla2t2_vec)
{
  // Add implementation
}

// nabla2_vec = nabla2t2_vec - nabla2t1_vec
// interp_out = sum_over(Edge > Cell, interp_in * interp_coeff)
__global__ void kernel_7_and_8(
    int NumEdges, int NumCells, int kSize, const int *ecTable,
    ::dawn::float_type *__restrict__ nabla2_vec,
    const ::dawn::float_type *__restrict__ nabla2t1_vec,
    const ::dawn::float_type *__restrict__ nabla2t2_vec,
    const ::dawn::float_type *__restrict__ interp_in,
    const ::dawn::float_type *__restrict__ interp_coeff,
    ::dawn::float_type *__restrict__ interp_out)
{
  // Add implementation
}

// nabla2t1_vec = sum_over(Edge > Vertex, curl_vec, weights=[-1.0, 1])
// nabla2t1_vec = tangent_orientation * nabla2t1_vec / primal_edge_length
// nabla2t2_vec = sum_over(Edge > Cell, div_vec, weights=[-1.0, 1])
// nabla2t2_vec = tangent_orientation * nabla2t2_vec / dual_edge_length
__global__ void kernel_34_and_56(
    int NumEdges, int NumVertices, int NumCells,
    int kSize, const int *evTable, const int *ecTable,
    const ::dawn::float_type *__restrict__ curl_vec,
    const ::dawn::float_type *__restrict__ primal_edge_length,
    const ::dawn::float_type *__restrict__ tangent_orientation,
    ::dawn::float_type *__restrict__ nabla2t1_vec,
    const ::dawn::float_type *__restrict__ div_vec,
    const ::dawn::float_type *__restrict__ dual_edge_length,
    ::dawn::float_type *__restrict__ nabla2t2_vec)
{
  // Add implementation
}

// nabla2t1_vec = sum_over(Edge > Vertex, curl_vec, weights=[-1.0, 1])
// nabla2t1_vec = tangent_orientation * nabla2t1_vec / primal_edge_length
// nabla2t2_vec = sum_over(Edge > Cell, div_vec, weights=[-1.0, 1])
// nabla2t2_vec = tangent_orientation * nabla2t2_vec / dual_edge_length
// nabla2_vec = nabla2t2_vec - nabla2t1_vec
// interp_out = sum_over(Edge > Cell, interp_in * interp_coeff)
__global__ void kernel_3456_and_78(
    int NumEdges, int NumVertices, int NumCells, int kSize,
    const int *evTable, const int *ecTable,
    const ::dawn::float_type *__restrict__ curl_vec,
    const ::dawn::float_type *__restrict__ primal_edge_length,
    const ::dawn::float_type *__restrict__ tangent_orientation,
    const ::dawn::float_type *__restrict__ div_vec,
    const ::dawn::float_type *__restrict__ dual_edge_length,
    ::dawn::float_type *__restrict__ nabla2_vec,
    const ::dawn::float_type *__restrict__ interp_in,
    const ::dawn::float_type *__restrict__ interp_coeff,
    ::dawn::float_type *__restrict__ interp_out)
{
  // Add implementation
}

// nabla2t1_vec = sum_over(Edge > Vertex, curl_vec, weights=[-1.0, 1])
// nabla2t1_vec = tangent_orientation * nabla2t1_vec / primal_edge_length
// nabla2t2_vec = sum_over(Edge > Cell, div_vec, weights=[-1.0, 1])
// nabla2t2_vec = tangent_orientation * nabla2t2_vec / dual_edge_length
// nabla2_vec = nabla2t2_vec - nabla2t1_vec
// interp_out = sum_over(Edge > Cell, interp_in * interp_coeff)
__global__ void kernel_fuse_e_c(
    int NumEdges, int NumVertices, int NumCells, int kSize,
    const int *evTable, const int *ecTable,
    const ::dawn::float_type *__restrict__ curl_vec,
    const ::dawn::float_type *__restrict__ primal_edge_length,
    const ::dawn::float_type *__restrict__ tangent_orientation,
    const ::dawn::float_type *__restrict__ div_vec,
    const ::dawn::float_type *__restrict__ dual_edge_length,
    ::dawn::float_type *__restrict__ nabla2_vec,
    const ::dawn::float_type *__restrict__ interp_in,
    const ::dawn::float_type *__restrict__ interp_coeff,
    ::dawn::float_type *__restrict__ interp_out)
{
  // Add implementation
}

// curl_vec = sum_over(Vertex > Edge, vec * geofac_curl)
// nabla2t1_vec = sum_over(Edge > Vertex, curl_vec, weights=[-1.0, 1])
// nabla2t1_vec = tangent_orientation * nabla2t1_vec / primal_edge_length
// nabla2t2_vec = sum_over(Edge > Cell, div_vec, weights=[-1.0, 1])
// nabla2t2_vec = tangent_orientation * nabla2t2_vec / dual_edge_length
// nabla2_vec = nabla2t2_vec - nabla2t1_vec
// interp_out = sum_over(Edge > Cell, interp_in * interp_coeff)
__global__ void kernel_inline_curl_vec(
    int NumEdges, int NumVertices, int NumCells, int kSize,
    const int *veTable, const int *evTable, const int *ecTable,
    const ::dawn::float_type *__restrict__ vec,
    const ::dawn::float_type *__restrict__ geofac_curl,
    const ::dawn::float_type *__restrict__ primal_edge_length,
    const ::dawn::float_type *__restrict__ tangent_orientation,
    const ::dawn::float_type *__restrict__ div_vec,
    const ::dawn::float_type *__restrict__ dual_edge_length,
    ::dawn::float_type *__restrict__ nabla2_vec,
    const ::dawn::float_type *__restrict__ interp_in,
    const ::dawn::float_type *__restrict__ interp_coeff,
    ::dawn::float_type *__restrict__ interp_out)
{
  // Add implementation
}

// curl_vec = sum_over(Vertex > Edge, vec * geofac_curl)
// div_vec = sum_over(Cell > Edge, vec * geofac_div)
// nabla2t1_vec = sum_over(Edge > Vertex, curl_vec, weights=[-1.0, 1])
// nabla2t1_vec = tangent_orientation * nabla2t1_vec / primal_edge_length
// nabla2t2_vec = sum_over(Edge > Cell, div_vec, weights=[-1.0, 1])
// nabla2t2_vec = tangent_orientation * nabla2t2_vec / dual_edge_length
// nabla2_vec = nabla2t2_vec - nabla2t1_vec
// interp_out = sum_over(Edge > Cell, interp_in * interp_coeff)
__global__ void kernel_inline_div_vec(
    int NumEdges, int NumVertices, int NumCells, int kSize,
    const int *veTable, const int *evTable, const int *ecTable,
    const ::dawn::float_type *__restrict__ vec,
    const ::dawn::float_type *__restrict__ geofac_curl,
    const ::dawn::float_type *__restrict__ primal_edge_length,
    const ::dawn::float_type *__restrict__ tangent_orientation,
    const ::dawn::float_type *__restrict__ geofac_div,
    const ::dawn::float_type *__restrict__ dual_edge_length,
    ::dawn::float_type *__restrict__ nabla2_vec,
    const ::dawn::float_type *__restrict__ interp_in,
    const ::dawn::float_type *__restrict__ interp_coeff,
    ::dawn::float_type *__restrict__ interp_out)
{
  // Add implementation
}

//
//  END OF KERNELS
//

template <typename LibTag>
class exercise
{

public:
  struct sbase : public timer_cuda
  {

    sbase(std::string name) : timer_cuda(name) {}

    double get_time() { return total_time(); }
  };

  struct GpuTriMesh
  {
    int NumVertices;
    int NumEdges;
    int NumCells;
    dawn::unstructured_domain Domain;
    int *ceTable;
    int *ecTable;
    int *evTable;
    int *veTable;

    GpuTriMesh(const dawn::mesh_t<LibTag> &mesh)
    {
      NumVertices = mesh.nodes().size();
      NumCells = mesh.cells().size();
      NumEdges = mesh.edges().size();
      gpuErrchk(cudaMalloc((void **)&ceTable,
                           sizeof(int) * mesh.cells().size() * C_E_SIZE));
      dawn::generateNbhTable<LibTag>(
          mesh, {dawn::LocationType::Cells, dawn::LocationType::Edges},
          mesh.cells().size(), C_E_SIZE, ceTable);
      gpuErrchk(cudaMalloc((void **)&ecTable,
                           sizeof(int) * mesh.edges().size() * E_C_SIZE));
      dawn::generateNbhTable<LibTag>(
          mesh, {dawn::LocationType::Edges, dawn::LocationType::Cells},
          mesh.edges().size(), E_C_SIZE, ecTable);
      gpuErrchk(cudaMalloc((void **)&evTable,
                           sizeof(int) * mesh.edges().size() * E_V_SIZE));
      dawn::generateNbhTable<LibTag>(
          mesh, {dawn::LocationType::Edges, dawn::LocationType::Vertices},
          mesh.edges().size(), E_V_SIZE, evTable);
      gpuErrchk(cudaMalloc((void **)&veTable,
                           sizeof(int) * mesh.nodes().size() * V_E_SIZE));
      dawn::generateNbhTable<LibTag>(
          mesh, {dawn::LocationType::Vertices, dawn::LocationType::Edges},
          mesh.nodes().size(), V_E_SIZE, veTable);
    }

    GpuTriMesh(const dawn::GlobalGpuTriMesh *mesh)
    {
      NumVertices = mesh->NumVertices;
      NumCells = mesh->NumCells;
      NumEdges = mesh->NumEdges;
      Domain = mesh->Domain;
      ceTable = mesh->NeighborTables.at(
          {dawn::LocationType::Cells, dawn::LocationType::Edges});
      ecTable = mesh->NeighborTables.at(
          {dawn::LocationType::Edges, dawn::LocationType::Cells});
      evTable = mesh->NeighborTables.at(
          {dawn::LocationType::Edges, dawn::LocationType::Vertices});
      veTable = mesh->NeighborTables.at(
          {dawn::LocationType::Vertices, dawn::LocationType::Edges});
    }
  };

  struct stencil_151 : public sbase
  {
  private:
    ::dawn::float_type *vec_;
    ::dawn::float_type *nabla2_vec_;
    ::dawn::float_type *primal_edge_length_;
    ::dawn::float_type *dual_edge_length_;
    ::dawn::float_type *tangent_orientation_;
    ::dawn::float_type *geofac_curl_;
    ::dawn::float_type *geofac_div_;
    ::dawn::float_type *interp_in_;
    ::dawn::float_type *interp_coeff_;
    ::dawn::float_type *interp_out_;
    ::dawn::float_type *div_vec_;
    ::dawn::float_type *curl_vec_;
    ::dawn::float_type *nabla2t1_vec_;
    ::dawn::float_type *nabla2t2_vec_;
    int kSize_ = 0;
    GpuTriMesh mesh_;

  public:
    stencil_151(
        const dawn::mesh_t<LibTag> &mesh, int kSize,
        dawn::edge_field_t<LibTag, ::dawn::float_type> &vec,
        dawn::edge_field_t<LibTag, ::dawn::float_type> &nabla2_vec,
        dawn::edge_field_t<LibTag, ::dawn::float_type> &primal_edge_length,
        dawn::edge_field_t<LibTag, ::dawn::float_type> &dual_edge_length,
        dawn::edge_field_t<LibTag, ::dawn::float_type> &tangent_orientation,
        dawn::sparse_vertex_field_t<LibTag, ::dawn::float_type> &geofac_curl,
        dawn::sparse_cell_field_t<LibTag, ::dawn::float_type> &geofac_div,
        dawn::cell_field_t<LibTag, ::dawn::float_type> &interp_in,
        dawn::sparse_edge_field_t<LibTag, ::dawn::float_type> &interp_coeff,
        dawn::edge_field_t<LibTag, ::dawn::float_type> &interp_out,
        dawn::cell_field_t<LibTag, ::dawn::float_type> &div_vec,
        dawn::vertex_field_t<LibTag, ::dawn::float_type> &curl_vec,
        dawn::edge_field_t<LibTag, ::dawn::float_type> &nabla2t1_vec,
        dawn::edge_field_t<LibTag, ::dawn::float_type> &nabla2t2_vec)
        : sbase("stencil_151"), mesh_(mesh), kSize_(kSize)
    {
      copy_memory(vec.data(), nabla2_vec.data(), primal_edge_length.data(),
                  dual_edge_length.data(), tangent_orientation.data(),
                  geofac_curl.data(), geofac_div.data(), interp_in.data(),
                  interp_coeff.data(), interp_out.data(), div_vec.data(),
                  curl_vec.data(), nabla2t1_vec.data(), nabla2t2_vec.data(),
                  true);
    }

    dim3 grid(int kSize, int elSize)
    {
      int dK = (kSize + LEVELS_PER_THREAD - 1) / LEVELS_PER_THREAD;
      return dim3((elSize + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (dK + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    }

    stencil_151(const dawn::GlobalGpuTriMesh *mesh, int kSize)
        : sbase("stencil_151"), mesh_(mesh), kSize_(kSize) {}

    void run()
    {
      dim3 dB(BLOCK_SIZE, BLOCK_SIZE, 1);
      sbase::start();

      int denseSize1 = mesh_.NumVertices;
      dim3 dG1 = grid(kSize_, denseSize1);
      kernel1<<<dG1, dB>>>(denseSize1, mesh_.NumEdges, kSize_, mesh_.veTable, vec_,
                           geofac_curl_, curl_vec_);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      int denseSize2 = mesh_.NumCells;
      dim3 dG2 = grid(kSize_, denseSize2);
      kernel2<<<dG2, dB>>>(denseSize2, mesh_.NumEdges, kSize_, mesh_.ceTable, vec_,
                           geofac_div_, div_vec_);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      int denseSize3 = mesh_.NumEdges;
      dim3 dG3 = grid(kSize_, denseSize3);
      kernel3<<<dG3, dB>>>(denseSize3, mesh_.NumVertices, kSize_, mesh_.evTable,
                           curl_vec_, nabla2t1_vec_);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      int denseSize4 = mesh_.NumEdges;
      dim3 dG4 = grid(kSize_, denseSize4);
      kernel4<<<dG4, dB>>>(
          denseSize4, kSize_, primal_edge_length_, tangent_orientation_,
          nabla2t1_vec_);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      // int denseSize34 = mesh_.NumEdges;
      // dim3 dG34 = grid(kSize_, denseSize34);
      // kernel_3_and_4<<<dG34, dB>>>(denseSize34, mesh_.NumVertices, kSize_, mesh_.evTable,
      //                              curl_vec_, primal_edge_length_, tangent_orientation_, nabla2t1_vec_);
      // gpuErrchk(cudaPeekAtLastError());
      // gpuErrchk(cudaDeviceSynchronize());

      int denseSize5 = mesh_.NumEdges;
      dim3 dG5 = grid(kSize_, denseSize5);
      kernel5<<<dG5, dB>>>(denseSize5, mesh_.NumCells, kSize_, mesh_.ecTable,
                           div_vec_, nabla2t2_vec_);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      int denseSize6 = mesh_.NumEdges;
      dim3 dG6 = grid(kSize_, denseSize6);
      kernel6<<<dG6, dB>>>(
          denseSize6, kSize_, dual_edge_length_, tangent_orientation_,
          nabla2t2_vec_);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      // int denseSize56 = mesh_.NumEdges;
      // dim3 dG56 = grid(kSize_, denseSize56);
      // kernel_5_and_6<<<dG56, dB>>>(denseSize56, mesh_.NumCells, kSize_, mesh_.ecTable,
      //                              div_vec_, dual_edge_length_, tangent_orientation_, nabla2t2_vec_);
      // gpuErrchk(cudaPeekAtLastError());
      // gpuErrchk(cudaDeviceSynchronize());

      // int denseSize3456 = mesh_.NumEdges;
      // dim3 dG3456 = grid(kSize_, denseSize3456);
      // kernel_34_and_56<<<dG3456, dB>>>(denseSize3456, mesh_.NumVertices, mesh_.NumCells,
      //                                  kSize_, mesh_.evTable, mesh_.ecTable,
      //                                  curl_vec_, primal_edge_length_, tangent_orientation_, nabla2t1_vec_,
      //                                  div_vec_, dual_edge_length_, nabla2t2_vec_);
      // gpuErrchk(cudaPeekAtLastError());
      // gpuErrchk(cudaDeviceSynchronize());

      int denseSize7 = mesh_.NumEdges;
      dim3 dG7 = grid(kSize_, denseSize7);
      kernel7<<<dG7, dB>>>(
          denseSize7, kSize_, nabla2_vec_, nabla2t1_vec_, nabla2t2_vec_);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      int denseSize8 = mesh_.NumEdges;
      dim3 dG8 = grid(kSize_, denseSize8);
      kernel8<<<dG8, dB>>>(denseSize8, mesh_.NumCells, kSize_, mesh_.ecTable,
                           interp_in_, interp_coeff_, interp_out_);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      // int denseSize78 = mesh_.NumEdges;
      // dim3 dG78 = grid(kSize_, denseSize78);
      // kernel_7_and_8<<<dG78, dB>>>(
      //     denseSize78, mesh_.NumCells, kSize_, mesh_.ecTable,
      //     nabla2_vec_, nabla2t1_vec_, nabla2t2_vec_,
      //     interp_in_, interp_coeff_, interp_out_);
      // gpuErrchk(cudaPeekAtLastError());
      // gpuErrchk(cudaDeviceSynchronize());

      // int denseSize345678 = mesh_.NumEdges;
      // dim3 dG345678 = grid(kSize_, denseSize345678);
      // kernel_3456_and_78<<<dG345678, dB>>>(denseSize345678, mesh_.NumVertices, mesh_.NumCells,
      //                                      kSize_, mesh_.evTable, mesh_.ecTable,
      //                                      curl_vec_, primal_edge_length_, tangent_orientation_,
      //                                      div_vec_, dual_edge_length_, nabla2_vec_,
      //                                      interp_in_, interp_coeff_, interp_out_);
      // gpuErrchk(cudaPeekAtLastError());
      // gpuErrchk(cudaDeviceSynchronize());

      // int denseSize_fuse_e_c = mesh_.NumEdges;
      // dim3 dG_fuse_e_c = grid(kSize_, denseSize_fuse_e_c);
      // kernel_fuse_e_c<<<dG_fuse_e_c, dB>>>(denseSize_fuse_e_c, mesh_.NumVertices, mesh_.NumCells,
      //                                      kSize_, mesh_.evTable, mesh_.ecTable,
      //                                      curl_vec_, primal_edge_length_, tangent_orientation_,
      //                                      div_vec_, dual_edge_length_, nabla2_vec_,
      //                                      interp_in_, interp_coeff_, interp_out_);
      // gpuErrchk(cudaPeekAtLastError());
      // gpuErrchk(cudaDeviceSynchronize());

      // int denseSize_inline_curl_vec = mesh_.NumEdges;
      // dim3 dG_inline_curl_vec = grid(kSize_, denseSize_inline_curl_vec);
      // kernel_inline_curl_vec<<<dG_inline_curl_vec, dB>>>(denseSize_inline_curl_vec, mesh_.NumVertices, mesh_.NumCells,
      //                                                    kSize_, mesh_.evTable, mesh_.ecTable,
      //                                                    vec_, geofac_curl_, primal_edge_length_, tangent_orientation_,
      //                                                    div_vec_, dual_edge_length_, nabla2_vec_,
      //                                                    interp_in_, interp_coeff_, interp_out_);
      // gpuErrchk(cudaPeekAtLastError());
      // gpuErrchk(cudaDeviceSynchronize());

      // int denseSize_inline_div_vec = mesh_.NumEdges;
      // dim3 dG_inline_div_vec = grid(kSize_, denseSize_inline_div_vec);
      // kernel_inline_div_vec<<<dG_inline_div_vec, dB>>>(denseSize_inline_div_vec, mesh_.NumVertices, mesh_.NumCells,
      //                                                  kSize_, mesh_.veTable, mesh_.evTable, mesh_.ecTable,
      //                                                  vec_, geofac_curl_, primal_edge_length_, tangent_orientation_,
      //                                                  geofac_div_, dual_edge_length_, nabla2_vec_,
      //                                                  interp_in_, interp_coeff_, interp_out_);
      // gpuErrchk(cudaPeekAtLastError());
      // gpuErrchk(cudaDeviceSynchronize());

      sbase::pause();
    }

    void CopyResultToHost(::dawn::float_type *nabla2_vec,
                          ::dawn::float_type *interp_out,
                          ::dawn::float_type *div_vec,
                          ::dawn::float_type *curl_vec,
                          ::dawn::float_type *nabla2t1_vec,
                          ::dawn::float_type *nabla2t2_vec, bool do_reshape)
    {
      if (do_reshape)
      {
        ::dawn::float_type *host_buf =
            new ::dawn::float_type[mesh_.NumEdges * kSize_];
        gpuErrchk(
            cudaMemcpy((::dawn::float_type *)host_buf, nabla2_vec_,
                       mesh_.NumEdges * kSize_ * sizeof(::dawn::float_type),
                       cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, nabla2_vec, kSize_, mesh_.NumEdges);
        delete[] host_buf;
      }
      else
      {
        gpuErrchk(
            cudaMemcpy(nabla2_vec, nabla2_vec_,
                       mesh_.NumEdges * kSize_ * sizeof(::dawn::float_type),
                       cudaMemcpyDeviceToHost));
      }
      if (do_reshape)
      {
        ::dawn::float_type *host_buf =
            new ::dawn::float_type[mesh_.NumEdges * kSize_];
        gpuErrchk(
            cudaMemcpy((::dawn::float_type *)host_buf, interp_out_,
                       mesh_.NumEdges * kSize_ * sizeof(::dawn::float_type),
                       cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, interp_out, kSize_, mesh_.NumEdges);
        delete[] host_buf;
      }
      else
      {
        gpuErrchk(
            cudaMemcpy(interp_out, interp_out_,
                       mesh_.NumEdges * kSize_ * sizeof(::dawn::float_type),
                       cudaMemcpyDeviceToHost));
      }
      if (do_reshape)
      {
        ::dawn::float_type *host_buf =
            new ::dawn::float_type[mesh_.NumCells * kSize_];
        gpuErrchk(
            cudaMemcpy((::dawn::float_type *)host_buf, div_vec_,
                       mesh_.NumCells * kSize_ * sizeof(::dawn::float_type),
                       cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, div_vec, kSize_, mesh_.NumCells);
        delete[] host_buf;
      }
      else
      {
        gpuErrchk(
            cudaMemcpy(div_vec, div_vec_,
                       mesh_.NumCells * kSize_ * sizeof(::dawn::float_type),
                       cudaMemcpyDeviceToHost));
      }
      if (do_reshape)
      {
        ::dawn::float_type *host_buf =
            new ::dawn::float_type[mesh_.NumVertices * kSize_];
        gpuErrchk(
            cudaMemcpy((::dawn::float_type *)host_buf, curl_vec_,
                       mesh_.NumVertices * kSize_ * sizeof(::dawn::float_type),
                       cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, curl_vec, kSize_, mesh_.NumVertices);
        delete[] host_buf;
      }
      else
      {
        gpuErrchk(
            cudaMemcpy(curl_vec, curl_vec_,
                       mesh_.NumVertices * kSize_ * sizeof(::dawn::float_type),
                       cudaMemcpyDeviceToHost));
      }
      if (do_reshape)
      {
        ::dawn::float_type *host_buf =
            new ::dawn::float_type[mesh_.NumEdges * kSize_];
        gpuErrchk(
            cudaMemcpy((::dawn::float_type *)host_buf, nabla2t1_vec_,
                       mesh_.NumEdges * kSize_ * sizeof(::dawn::float_type),
                       cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, nabla2t1_vec, kSize_, mesh_.NumEdges);
        delete[] host_buf;
      }
      else
      {
        gpuErrchk(
            cudaMemcpy(nabla2t1_vec, nabla2t1_vec_,
                       mesh_.NumEdges * kSize_ * sizeof(::dawn::float_type),
                       cudaMemcpyDeviceToHost));
      }
      if (do_reshape)
      {
        ::dawn::float_type *host_buf =
            new ::dawn::float_type[mesh_.NumEdges * kSize_];
        gpuErrchk(
            cudaMemcpy((::dawn::float_type *)host_buf, nabla2t2_vec_,
                       mesh_.NumEdges * kSize_ * sizeof(::dawn::float_type),
                       cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, nabla2t2_vec, kSize_, mesh_.NumEdges);
        delete[] host_buf;
      }
      else
      {
        gpuErrchk(
            cudaMemcpy(nabla2t2_vec, nabla2t2_vec_,
                       mesh_.NumEdges * kSize_ * sizeof(::dawn::float_type),
                       cudaMemcpyDeviceToHost));
      }
    }

    void CopyResultToHost(
        dawn::edge_field_t<LibTag, ::dawn::float_type> &nabla2_vec,
        dawn::edge_field_t<LibTag, ::dawn::float_type> &interp_out,
        dawn::cell_field_t<LibTag, ::dawn::float_type> &div_vec,
        dawn::vertex_field_t<LibTag, ::dawn::float_type> &curl_vec,
        dawn::edge_field_t<LibTag, ::dawn::float_type> &nabla2t1_vec,
        dawn::edge_field_t<LibTag, ::dawn::float_type> &nabla2t2_vec,
        bool do_reshape)
    {
      if (do_reshape)
      {
        ::dawn::float_type *host_buf =
            new ::dawn::float_type[nabla2_vec.numElements()];
        gpuErrchk(
            cudaMemcpy((::dawn::float_type *)host_buf, nabla2_vec_,
                       nabla2_vec.numElements() * sizeof(::dawn::float_type),
                       cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, nabla2_vec.data(), kSize_, mesh_.NumEdges);
        delete[] host_buf;
      }
      else
      {
        gpuErrchk(
            cudaMemcpy(nabla2_vec.data(), nabla2_vec_,
                       nabla2_vec.numElements() * sizeof(::dawn::float_type),
                       cudaMemcpyDeviceToHost));
      }
      if (do_reshape)
      {
        ::dawn::float_type *host_buf =
            new ::dawn::float_type[interp_out.numElements()];
        gpuErrchk(
            cudaMemcpy((::dawn::float_type *)host_buf, interp_out_,
                       interp_out.numElements() * sizeof(::dawn::float_type),
                       cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, interp_out.data(), kSize_, mesh_.NumEdges);
        delete[] host_buf;
      }
      else
      {
        gpuErrchk(
            cudaMemcpy(interp_out.data(), interp_out_,
                       interp_out.numElements() * sizeof(::dawn::float_type),
                       cudaMemcpyDeviceToHost));
      }
      if (do_reshape)
      {
        ::dawn::float_type *host_buf =
            new ::dawn::float_type[div_vec.numElements()];
        gpuErrchk(cudaMemcpy((::dawn::float_type *)host_buf, div_vec_,
                             div_vec.numElements() * sizeof(::dawn::float_type),
                             cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, div_vec.data(), kSize_, mesh_.NumCells);
        delete[] host_buf;
      }
      else
      {
        gpuErrchk(cudaMemcpy(div_vec.data(), div_vec_,
                             div_vec.numElements() * sizeof(::dawn::float_type),
                             cudaMemcpyDeviceToHost));
      }
      if (do_reshape)
      {
        ::dawn::float_type *host_buf =
            new ::dawn::float_type[curl_vec.numElements()];
        gpuErrchk(
            cudaMemcpy((::dawn::float_type *)host_buf, curl_vec_,
                       curl_vec.numElements() * sizeof(::dawn::float_type),
                       cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, curl_vec.data(), kSize_,
                           mesh_.NumVertices);
        delete[] host_buf;
      }
      else
      {
        gpuErrchk(
            cudaMemcpy(curl_vec.data(), curl_vec_,
                       curl_vec.numElements() * sizeof(::dawn::float_type),
                       cudaMemcpyDeviceToHost));
      }
      if (do_reshape)
      {
        ::dawn::float_type *host_buf =
            new ::dawn::float_type[nabla2t1_vec.numElements()];
        gpuErrchk(cudaMemcpy((::dawn::float_type *)host_buf, nabla2t1_vec_,
                             nabla2t1_vec.numElements() *
                                 sizeof(::dawn::float_type),
                             cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, nabla2t1_vec.data(), kSize_,
                           mesh_.NumEdges);
        delete[] host_buf;
      }
      else
      {
        gpuErrchk(cudaMemcpy(nabla2t1_vec.data(), nabla2t1_vec_,
                             nabla2t1_vec.numElements() *
                                 sizeof(::dawn::float_type),
                             cudaMemcpyDeviceToHost));
      }
      if (do_reshape)
      {
        ::dawn::float_type *host_buf =
            new ::dawn::float_type[nabla2t2_vec.numElements()];
        gpuErrchk(cudaMemcpy((::dawn::float_type *)host_buf, nabla2t2_vec_,
                             nabla2t2_vec.numElements() *
                                 sizeof(::dawn::float_type),
                             cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, nabla2t2_vec.data(), kSize_,
                           mesh_.NumEdges);
        delete[] host_buf;
      }
      else
      {
        gpuErrchk(cudaMemcpy(nabla2t2_vec.data(), nabla2t2_vec_,
                             nabla2t2_vec.numElements() *
                                 sizeof(::dawn::float_type),
                             cudaMemcpyDeviceToHost));
      }
    }

    void copy_memory(::dawn::float_type *vec, ::dawn::float_type *nabla2_vec,
                     ::dawn::float_type *primal_edge_length,
                     ::dawn::float_type *dual_edge_length,
                     ::dawn::float_type *tangent_orientation,
                     ::dawn::float_type *geofac_curl,
                     ::dawn::float_type *geofac_div,
                     ::dawn::float_type *interp_in,
                     ::dawn::float_type *interp_coeff,
                     ::dawn::float_type *interp_out,
                     ::dawn::float_type *div_vec, ::dawn::float_type *curl_vec,
                     ::dawn::float_type *nabla2t1_vec,
                     ::dawn::float_type *nabla2t2_vec, bool do_reshape)
    {
      dawn::initField(vec, &vec_, mesh_.NumEdges, kSize_, do_reshape);
      dawn::initField(nabla2_vec, &nabla2_vec_, mesh_.NumEdges, kSize_,
                      do_reshape);
      dawn::initField(primal_edge_length, &primal_edge_length_, mesh_.NumEdges,
                      1, do_reshape);
      dawn::initField(dual_edge_length, &dual_edge_length_, mesh_.NumEdges, 1,
                      do_reshape);
      dawn::initField(tangent_orientation, &tangent_orientation_,
                      mesh_.NumEdges, 1, do_reshape);
      dawn::initSparseField(geofac_curl, &geofac_curl_, mesh_.NumVertices,
                            V_E_SIZE, 1, do_reshape);
      dawn::initSparseField(geofac_div, &geofac_div_, mesh_.NumCells, C_E_SIZE,
                            1, do_reshape);
      dawn::initField(interp_in, &interp_in_, mesh_.NumCells, kSize_,
                      do_reshape);
      dawn::initSparseField(interp_coeff, &interp_coeff_, mesh_.NumEdges,
                            E_C_SIZE, 1, do_reshape);
      dawn::initField(interp_out, &interp_out_, mesh_.NumEdges, kSize_,
                      do_reshape);
      dawn::initField(div_vec, &div_vec_, mesh_.NumCells, kSize_, do_reshape);
      dawn::initField(curl_vec, &curl_vec_, mesh_.NumVertices, kSize_,
                      do_reshape);
      dawn::initField(nabla2t1_vec, &nabla2t1_vec_, mesh_.NumEdges, kSize_,
                      do_reshape);
      dawn::initField(nabla2t2_vec, &nabla2t2_vec_, mesh_.NumEdges, kSize_,
                      do_reshape);
    }

    void copy_pointers(
        ::dawn::float_type *vec, ::dawn::float_type *nabla2_vec,
        ::dawn::float_type *primal_edge_length,
        ::dawn::float_type *dual_edge_length,
        ::dawn::float_type *tangent_orientation,
        ::dawn::float_type *geofac_curl, ::dawn::float_type *geofac_div,
        ::dawn::float_type *interp_in, ::dawn::float_type *interp_coeff,
        ::dawn::float_type *interp_out, ::dawn::float_type *div_vec,
        ::dawn::float_type *curl_vec, ::dawn::float_type *nabla2t1_vec,
        ::dawn::float_type *nabla2t2_vec)
    {
      vec_ = vec;
      nabla2_vec_ = nabla2_vec;
      primal_edge_length_ = primal_edge_length;
      dual_edge_length_ = dual_edge_length;
      tangent_orientation_ = tangent_orientation;
      geofac_curl_ = geofac_curl;
      geofac_div_ = geofac_div;
      interp_in_ = interp_in;
      interp_coeff_ = interp_coeff;
      interp_out_ = interp_out;
      div_vec_ = div_vec;
      curl_vec_ = curl_vec;
      nabla2t1_vec_ = nabla2t1_vec;
      nabla2t2_vec_ = nabla2t2_vec;
    }
  };
};
extern "C"
{
  double run_exercise_from_c_host(
      dawn::GlobalGpuTriMesh *mesh, int k_size, ::dawn::float_type *vec,
      ::dawn::float_type *nabla2_vec, ::dawn::float_type *primal_edge_length,
      ::dawn::float_type *dual_edge_length,
      ::dawn::float_type *tangent_orientation, ::dawn::float_type *geofac_curl,
      ::dawn::float_type *geofac_div, ::dawn::float_type *interp_in,
      ::dawn::float_type *interp_coeff, ::dawn::float_type *interp_out,
      ::dawn::float_type *div_vec, ::dawn::float_type *curl_vec,
      ::dawn::float_type *nabla2t1_vec, ::dawn::float_type *nabla2t2_vec)
  {
    exercise<dawn::NoLibTag>::stencil_151 s(mesh,
                                            k_size);
    s.copy_memory(vec, nabla2_vec, primal_edge_length, dual_edge_length,
                  tangent_orientation, geofac_curl, geofac_div, interp_in,
                  interp_coeff, interp_out, div_vec, curl_vec, nabla2t1_vec,
                  nabla2t2_vec, true);
    s.run();
    double time = s.get_time();
    s.reset();
    s.CopyResultToHost(nabla2_vec, interp_out, div_vec, curl_vec, nabla2t1_vec,
                       nabla2t2_vec, true);
    return time;
  }
  double run_exercise_from_fort_host(
      dawn::GlobalGpuTriMesh *mesh, int k_size, ::dawn::float_type *vec,
      ::dawn::float_type *nabla2_vec, ::dawn::float_type *primal_edge_length,
      ::dawn::float_type *dual_edge_length,
      ::dawn::float_type *tangent_orientation, ::dawn::float_type *geofac_curl,
      ::dawn::float_type *geofac_div, ::dawn::float_type *interp_in,
      ::dawn::float_type *interp_coeff, ::dawn::float_type *interp_out,
      ::dawn::float_type *div_vec, ::dawn::float_type *curl_vec,
      ::dawn::float_type *nabla2t1_vec, ::dawn::float_type *nabla2t2_vec)
  {
    exercise<dawn::NoLibTag>::stencil_151 s(mesh,
                                            k_size);
    s.copy_memory(vec, nabla2_vec, primal_edge_length, dual_edge_length,
                  tangent_orientation, geofac_curl, geofac_div, interp_in,
                  interp_coeff, interp_out, div_vec, curl_vec, nabla2t1_vec,
                  nabla2t2_vec, false);
    s.run();
    double time = s.get_time();
    s.reset();
    s.CopyResultToHost(nabla2_vec, interp_out, div_vec, curl_vec, nabla2t1_vec,
                       nabla2t2_vec, false);
    return time;
  }
  double run_exercise(
      dawn::GlobalGpuTriMesh *mesh, int k_size, ::dawn::float_type *vec,
      ::dawn::float_type *nabla2_vec, ::dawn::float_type *primal_edge_length,
      ::dawn::float_type *dual_edge_length,
      ::dawn::float_type *tangent_orientation, ::dawn::float_type *geofac_curl,
      ::dawn::float_type *geofac_div, ::dawn::float_type *interp_in,
      ::dawn::float_type *interp_coeff, ::dawn::float_type *interp_out,
      ::dawn::float_type *div_vec, ::dawn::float_type *curl_vec,
      ::dawn::float_type *nabla2t1_vec, ::dawn::float_type *nabla2t2_vec)
  {
    exercise<dawn::NoLibTag>::stencil_151 s(mesh,
                                            k_size);
    s.copy_pointers(vec, nabla2_vec, primal_edge_length, dual_edge_length,
                    tangent_orientation, geofac_curl, geofac_div, interp_in,
                    interp_coeff, interp_out, div_vec, curl_vec, nabla2t1_vec,
                    nabla2t2_vec);
    s.run();
    double time = s.get_time();
    s.reset();
    return time;
  }
}
