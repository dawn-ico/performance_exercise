
// atlas functions
#include <atlas/array.h>
#include <atlas/grid.h>
#include <atlas/mesh.h>
#include <atlas/mesh/actions/BuildEdges.h>
#include <atlas/output/Gmsh.h>
#include <atlas/util/CoordinateEnums.h>

// atlas interface for dawn generated code
#include "interface/atlas_interface.hpp"

// driver includes
#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"

// atlas utilities
#include "atlas_utils/utils/AtlasCartesianWrapper.h"
#include "atlas_utils/utils/AtlasFromNetcdf.h"
#include "atlas_utils/utils/GenerateRectAtlasMesh.h"
#include "atlas_utils/utils/GenerateStrIndxAtlasMesh.h"

#include <cmath>
#include <cstdio>
#include <fenv.h>
#include <optional>
#include <vector>

#include "utils/UnstructuredVerifier.h"

#include "utils/atlasToGlobalGpuTriMesh.h"
#include "utils/getAtlasFields.h"

#include "exercise_cuda.h"

template <typename T>
static int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

int main()
{
    // Setup a 32 by 32 grid of quads and generate a mesh out of it
    const int nx = 320, ny = 320;
    std::cout << "Setting up " << nx << "x" << ny << " grid\n";
    auto mesh = AtlasMeshRect(nx, ny);
    const int k_size = 80;

    // wrapper with various atlas helper functions
    AtlasToCartesian wrapper(mesh, true);

    const int edgesPerVertex = 6;
    const int edgesPerCell = 3;
    const int cellsPerEdge = 2;

    std::cout << "Making fields\n";

    auto [interp_in_F, interp_in] = MakeAtlasField("interp_in", mesh.cells().size(), k_size);

    auto [vec_F, vec] = MakeAtlasField("vec", mesh.edges().size(), k_size);

    auto [lapVecSol_F, lapVecSol] = MakeAtlasField("lapVecSol", mesh.edges().size(), k_size);

    auto [nabla2_vec_F, nabla2_vec] =
        MakeAtlasField("nabla2_vec", mesh.edges().size(), k_size);

    auto [interp_out_F, interp_out] = MakeAtlasField("interp_out", mesh.edges().size(), k_size);

    auto [interp_out_sol_F, interp_out_sol] = MakeAtlasField("interp_out_sol", mesh.edges().size(), k_size);

    // term 1 and term 2 of nabla for debugging
    auto [nabla2t1_vec_F, nabla2t1_vec] = MakeAtlasField("nabla2t1_vec", mesh.edges().size(), k_size);
    auto [nabla2t2_vec_F, nabla2t2_vec] = MakeAtlasField("nabla2t2_vec", mesh.edges().size(), k_size);

    // rotation (more commonly curl) of vec_e on vertices
    auto [rot_vec_F, rot_vec] = MakeAtlasField("rot_vec", mesh.nodes().size(), k_size);

    // divergence of vec_e on cells
    auto [div_vec_F, div_vec] = MakeAtlasField("div_vec", mesh.cells().size(), k_size);

    auto [interp_coeff_F, interp_coeff] =
        MakeAtlasSparseField("interp_coeff", mesh.edges().size(), cellsPerEdge, 1);

    auto [geofac_rot_F, geofac_rot] =
        MakeAtlasSparseField("geofac_rot", mesh.nodes().size(), edgesPerVertex, 1);

    auto [edge_orientation_vertex_F, edge_orientation_vertex] =
        MakeAtlasSparseField("edge_orientation_vertex", mesh.nodes().size(), edgesPerVertex, 1);

    auto [geofac_div_F, geofac_div] =
        MakeAtlasSparseField("geofac_div", mesh.cells().size(), edgesPerCell, 1);

    auto [edge_orientation_cell_F, edge_orientation_cell] =
        MakeAtlasSparseField("edge_orientation_cell", mesh.cells().size(), edgesPerCell, 1);

    auto [tangent_orientation_F, tangent_orientation] =
        MakeAtlasField("tangent_orientation", mesh.edges().size(), 1);
    auto [primal_edge_length_F, primal_edge_length] =
        MakeAtlasField("primal_edge_length", mesh.edges().size(), 1);
    auto [dual_edge_length_F, dual_edge_length] =
        MakeAtlasField("dual_edge_length", mesh.edges().size(), 1);
    auto [primal_normal_x_F, primal_normal_x] =
        MakeAtlasField("primal_normal_x", mesh.edges().size(), 1);
    auto [primal_normal_y_F, primal_normal_y] =
        MakeAtlasField("primal_normal_y", mesh.edges().size(), 1);
    auto [dual_normal_x_F, dual_normal_x] = MakeAtlasField("dual_normal_x", mesh.edges().size(), 1);
    auto [dual_normal_y_F, dual_normal_y] = MakeAtlasField("dual_normal_y", mesh.edges().size(), 1);
    auto [cell_area_F, cell_area] = MakeAtlasField("cell_area", mesh.cells().size(), 1);
    auto [dual_cell_area_F, dual_cell_area] =
        MakeAtlasField("dual_cell_area", mesh.nodes().size(), 1);

    std::cout << "Filling input fields\n";

    for (int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++)
    {
        primal_edge_length(edgeIdx) = wrapper.edgeLength(mesh, edgeIdx);
        dual_edge_length(edgeIdx) = wrapper.dualEdgeLength(mesh, edgeIdx);
        tangent_orientation(edgeIdx) = wrapper.tangentOrientation(mesh, edgeIdx);
        auto [nx, ny] = wrapper.primalNormal(mesh, edgeIdx);
        primal_normal_x(edgeIdx) = nx;
        primal_normal_y(edgeIdx) = ny;
        // The primal normal, dual normal
        // forms a left-handed coordinate system
        dual_normal_x(edgeIdx) = ny;
        dual_normal_y(edgeIdx) = -nx;
    }

    for (int cellIdx = 0; cellIdx < mesh.cells().size(); cellIdx++)
    {
        cell_area(cellIdx) = wrapper.cellArea(mesh, cellIdx);
    }

    for (int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++)
    {
        dual_cell_area(nodeIdx) = wrapper.dualCellArea(mesh, nodeIdx);
    }

    auto sphericalHarmonic = [](double x, double y) -> std::tuple<double, double> {
        return {0.25 * sqrt(105. / (2 * M_PI)) * cos(2 * x) * cos(y) * cos(y) * sin(y),
                0.5 * sqrt(15. / (2 * M_PI)) * cos(x) * cos(y) * sin(y)};
    };

    auto analyticalLaplacian = [](double x, double y) -> std::tuple<double, double> {
        double c1 = 0.25 * sqrt(105. / (2 * M_PI));
        double c2 = 0.5 * sqrt(15. / (2 * M_PI));
        return {-4 * c1 * cos(2 * x) * cos(y) * cos(y) * sin(y), -4 * c2 * cos(x) * sin(y) * cos(y)};
    };

    for (int level = 0; level < k_size; level++)
    {
        for (int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++)
        {
            auto [xm, ym] = wrapper.edgeMidpoint(mesh, edgeIdx);
            auto [u, v] = sphericalHarmonic(xm, ym);
            auto [lu, lv] = analyticalLaplacian(xm, ym);
            vec(edgeIdx, level) = primal_normal_x(edgeIdx) * u + primal_normal_y(edgeIdx) * v;
            lapVecSol(edgeIdx, level) = primal_normal_x(edgeIdx) * lu + primal_normal_y(edgeIdx) * lv;
        }
    }

    //===------------------------------------------------------------------------------------------===//
    // Init geometrical factors (sparse fields)
    //===------------------------------------------------------------------------------------------===//

    // init edge orientations for vertices and cells
    auto dot = [](const Vector &v1, const Vector &v2) {
        return std::get<0>(v1) * std::get<0>(v2) + std::get<1>(v1) * std::get<1>(v2);
    };

    for (int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++)
    {
        const auto &nodeEdgeConnectivity = mesh.nodes().edge_connectivity();
        const auto &edgeNodeConnectivity = mesh.edges().node_connectivity();

        const int missingVal = nodeEdgeConnectivity.missing_value();
        int numNbh = nodeEdgeConnectivity.cols(nodeIdx);

        // arbitrary val at boundary
        bool anyMissing = false;
        for (int nbhIdx = 0; nbhIdx < numNbh; nbhIdx++)
        {
            anyMissing |= nodeEdgeConnectivity(nodeIdx, nbhIdx) == missingVal;
        }
        if (numNbh != 6 || anyMissing)
        {
            for (int nbhIdx = 0; nbhIdx < numNbh; nbhIdx++)
            {
                edge_orientation_vertex(nodeIdx, nbhIdx) = -1;
            }
            continue;
        }

        for (int nbhIdx = 0; nbhIdx < numNbh; nbhIdx++)
        {
            int edgeIdx = nodeEdgeConnectivity(nodeIdx, nbhIdx);

            int n0 = edgeNodeConnectivity(edgeIdx, 0);
            int n1 = edgeNodeConnectivity(edgeIdx, 1);

            int centerIdx = (n0 == nodeIdx) ? n0 : n1;
            int farIdx = (n0 == nodeIdx) ? n1 : n0;

            auto [xLo, yLo] = wrapper.nodeLocation(centerIdx);
            auto [xHi, yHi] = wrapper.nodeLocation(farIdx);

            Vector edge = {xHi - xLo, yHi - yLo};
            Vector dualNormal = {dual_normal_x(edgeIdx), dual_normal_y(edgeIdx)};

            double dbg = dot(edge, dualNormal);
            int systemSign = sgn(dot(edge, dualNormal)); // geometrical factor "corrects" normal such
                                                         // that the resulting system is left handed
            edge_orientation_vertex(nodeIdx, nbhIdx) = systemSign;
        }
    }

    for (int cellIdx = 0; cellIdx < mesh.cells().size(); cellIdx++)
    {
        const atlas::mesh::HybridElements::Connectivity &cellEdgeConnectivity =
            mesh.cells().edge_connectivity();
        auto [xm, ym] = wrapper.cellCircumcenter(mesh, cellIdx);

        const int missingVal = cellEdgeConnectivity.missing_value();
        int numNbh = cellEdgeConnectivity.cols(cellIdx);
        assert(numNbh == edgesPerCell);

        for (int nbhIdx = 0; nbhIdx < numNbh; nbhIdx++)
        {
            int edgeIdx = cellEdgeConnectivity(cellIdx, nbhIdx);
            auto [emX, emY] = wrapper.edgeMidpoint(mesh, edgeIdx);
            Vector toOutsdie{emX - xm, emY - ym};
            Vector primal = {primal_normal_x(edgeIdx), primal_normal_y(edgeIdx)};
            edge_orientation_cell(cellIdx, nbhIdx) = sgn(dot(toOutsdie, primal));
        }
    }

    for (int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++)
    {
        const atlas::mesh::Nodes::Connectivity &nodeEdgeConnectivity = mesh.nodes().edge_connectivity();

        int numNbh = nodeEdgeConnectivity.cols(nodeIdx);

        for (int nbhIdx = 0; nbhIdx < numNbh; nbhIdx++)
        {
            int edgeIdx = nodeEdgeConnectivity(nodeIdx, nbhIdx);
            geofac_rot(nodeIdx, nbhIdx) = (dual_cell_area(nodeIdx) == 0.)
                                              ? 0
                                              : dual_edge_length(edgeIdx) *
                                                    edge_orientation_vertex(nodeIdx, nbhIdx) /
                                                    dual_cell_area(nodeIdx);
        }
    }

    for (int cellIdx = 0; cellIdx < mesh.cells().size(); cellIdx++)
    {
        const atlas::mesh::HybridElements::Connectivity &cellEdgeConnectivity =
            mesh.cells().edge_connectivity();

        int numNbh = cellEdgeConnectivity.cols(cellIdx);
        assert(numNbh == edgesPerCell);

        for (int nbhIdx = 0; nbhIdx < numNbh; nbhIdx++)
        {
            int edgeIdx = cellEdgeConnectivity(cellIdx, nbhIdx);
            geofac_div(cellIdx, nbhIdx) =
                primal_edge_length(edgeIdx) * edge_orientation_cell(cellIdx, nbhIdx) / cell_area(cellIdx);
        }
    }

    //===------------------------------------------------------------------------------------------===//
    // Init fields interpolation
    //===------------------------------------------------------------------------------------------===//
    for (int level = 0; level < k_size; level++)
    {
        for (int cellIdx = 0; cellIdx < mesh.cells().size(); cellIdx++)
        {
            // init input
            interp_in(cellIdx, level) = 3.0;
        }

        for (int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++)
        {
            // init output
            interp_out_sol(edgeIdx, level) = 3.0;
        }
    }
    for (int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++)
    {
        const atlas::mesh::HybridElements::Connectivity &edgeCellConn =
            mesh.edges().cell_connectivity();
        const int missingVal = edgeCellConn.missing_value();

        int numNbh = edgeCellConn.cols(edgeIdx);
        int actualNbhs = 0;
        for (int nbhIdx = 0; nbhIdx < numNbh; nbhIdx++)
        {
            if (edgeCellConn(edgeIdx, nbhIdx) != missingVal)
                actualNbhs++;
        }

        for (int nbhIdx = 0; nbhIdx < numNbh; nbhIdx++)
        {
            // init coeff
            if (edgeCellConn(edgeIdx, nbhIdx) != missingVal)
                interp_coeff(edgeIdx, nbhIdx) = 1.0 / (double)actualNbhs;
        }
    }

    auto globalMesh = atlasToGlobalGpuTriMesh(mesh);

    // Allocate and init fields on GPU
    double *vec_gpu;
    double *nabla2_vec_gpu;
    double *primal_edge_length_gpu;
    double *dual_edge_length_gpu;
    double *tangent_orientation_gpu;
    double *geofac_rot_gpu;
    double *geofac_div_gpu;
    double *interp_in_gpu;
    double *interp_coeff_gpu;
    double *interp_out_gpu;
    double *div_vec_gpu;
    double *rot_vec_gpu;
    double *nabla2t1_vec_gpu;
    double *nabla2t2_vec_gpu;

    std::cout
        << "Uploading to GPU\n";

    bool do_reshape = true;
    dawn::initField(vec.data(), &vec_gpu, globalMesh.NumEdges, k_size, do_reshape);
    dawn::initField(nabla2_vec.data(), &nabla2_vec_gpu, globalMesh.NumEdges, k_size,
                    do_reshape);
    dawn::initField(primal_edge_length.data(), &primal_edge_length_gpu, globalMesh.NumEdges,
                    1, do_reshape);
    dawn::initField(dual_edge_length.data(), &dual_edge_length_gpu, globalMesh.NumEdges, 1,
                    do_reshape);
    dawn::initField(tangent_orientation.data(), &tangent_orientation_gpu,
                    globalMesh.NumEdges, 1, do_reshape);
    dawn::initSparseField(geofac_rot.data(), &geofac_rot_gpu, globalMesh.NumVertices,
                          edgesPerVertex, 1, do_reshape);
    dawn::initSparseField(geofac_div.data(), &geofac_div_gpu, globalMesh.NumCells, edgesPerCell,
                          1, do_reshape);
    dawn::initField(interp_in.data(), &interp_in_gpu, globalMesh.NumCells, k_size,
                    do_reshape);
    dawn::initSparseField(interp_coeff.data(), &interp_coeff_gpu, globalMesh.NumEdges,
                          cellsPerEdge, 1, do_reshape);
    dawn::initField(interp_out.data(), &interp_out_gpu, globalMesh.NumEdges, k_size,
                    do_reshape);
    dawn::initField(div_vec.data(), &div_vec_gpu, globalMesh.NumCells, k_size, do_reshape);
    dawn::initField(rot_vec.data(), &rot_vec_gpu, globalMesh.NumVertices, k_size,
                    do_reshape);
    dawn::initField(nabla2t1_vec.data(), &nabla2t1_vec_gpu, globalMesh.NumEdges, k_size,
                    do_reshape);
    dawn::initField(nabla2t2_vec.data(), &nabla2t2_vec_gpu, globalMesh.NumEdges, k_size,
                    do_reshape);

    std::cout << "Verifying stencil correctness\n";
    //===------------------------------------------------------------------------------------------===//
    // verification
    //===------------------------------------------------------------------------------------------===//
    // Run the stencil (cuda backend)

    run_exercise_from_c_host(
        &globalMesh, k_size, vec.data(), nabla2_vec.data(), primal_edge_length.data(),
        dual_edge_length.data(), tangent_orientation.data(), geofac_rot.data(), geofac_div.data(),
        interp_in.data(), interp_coeff.data(), interp_out.data(),
        div_vec.data(), rot_vec.data(), nabla2t1_vec.data(), nabla2t2_vec.data());

    UnstructuredVerifier verif;
    {
        auto lapl_vec_gpu_v = atlas::array::make_view<double, 2>(nabla2_vec_F);
        auto lapl_vec_sol_v = atlas::array::make_view<double, 2>(lapVecSol_F);

        auto [Linf_gpu, L1_gpu, L2_gpu] = UnstructuredVerifier::getErrorNorms(
            wrapper.innerEdges(mesh), k_size, lapl_vec_gpu_v, lapl_vec_sol_v);

        // constant error (not debendent on nx)
        if (log(Linf_gpu) >= 1)
        {
            std::cerr << "L-inf norm too high, verification failed for nabla2\n";
            return 1;
        }
        else if (log(L1_gpu) >= 0)
        {
            std::cerr << "L1 norm too high, verification failed for nabla2\n";
            return 1;
        }
        else if (log(L2_gpu) >= 0)
        {
            std::cerr << "L2 norm too high, verification failed for nabla2\n";
            return 1;
        }
    }
    {
        auto interp_out_gpu_v = atlas::array::make_view<double, 2>(interp_out_F);
        auto interp_out_sol_v = atlas::array::make_view<double, 2>(interp_out_sol_F);

        auto [Linf_gpu, L1_gpu, L2_gpu] = UnstructuredVerifier::getErrorNorms(
            wrapper.innerEdges(mesh), k_size, interp_out_gpu_v, interp_out_sol_v);

        // constant error (not debendent on nx)
        if (log(Linf_gpu) >= 1)
        {
            std::cerr << "L-inf norm too high, verification failed for interpolation\n";
            return 1;
        }
        else if (log(L1_gpu) >= 0)
        {
            std::cerr << "L1 norm too high, verification failed for interpolation\n";
            return 1;
        }
        else if (log(L2_gpu) >= 0)
        {
            std::cerr << "L2 norm too high, verification failed for interpolation\n";
            return 1;
        }
    }

    //===------------------------------------------------------------------------------------------===//
    // timing
    //===------------------------------------------------------------------------------------------===//
    const int nruns = 100;
    std::vector<double> times(nruns);
    for (int i = 0; i < nruns; i++)
    {
        times[i] = run_exercise(
            &globalMesh, k_size, vec_gpu, nabla2_vec_gpu, primal_edge_length_gpu,
            dual_edge_length_gpu, tangent_orientation_gpu, geofac_rot_gpu, geofac_div_gpu,
            interp_in_gpu, interp_coeff_gpu, interp_out_gpu,
            div_vec_gpu, rot_vec_gpu, nabla2t1_vec_gpu, nabla2t2_vec_gpu);
    }

    auto mean = [](const std::vector<double> &times) {
        double avg = 0.;
        for (auto time : times)
        {
            avg += time;
        }
        return avg / times.size();
    };
    auto standard_deviation = [&](const std::vector<double> &times) {
        auto avg = mean(times);
        double sd = 0.;
        for (auto time : times)
        {
            sd += (time - avg) * (time - avg);
        }
        return sqrt(1. / (times.size() - 1) * sd);
    };

    std::cout << "Avg timing: " << std::scientific << mean(times) << " s, std dev " << std::scientific << standard_deviation(times) << "\n";

    return 0;
}
