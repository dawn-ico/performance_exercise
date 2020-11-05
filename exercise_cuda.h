#pragma once
#include "driver-includes/defs.hpp"
#include "driver-includes/cuda_utils.hpp"
extern "C" {
double run_exercise_from_c_host(dawn::GlobalGpuTriMesh *mesh, int k_size, ::dawn::float_type *vec, ::dawn::float_type *nabla2_vec, ::dawn::float_type *primal_edge_length, ::dawn::float_type *dual_edge_length, ::dawn::float_type *tangent_orientation, ::dawn::float_type *geofac_curl, ::dawn::float_type *geofac_div, ::dawn::float_type *interp_in, ::dawn::float_type *interp_coeff, ::dawn::float_type *interp_out, ::dawn::float_type *div_vec, ::dawn::float_type *curl_vec, ::dawn::float_type *__tmp_nabl_149, ::dawn::float_type *__tmp_nabl_150) 
;
double run_exercise_from_fort_host(dawn::GlobalGpuTriMesh *mesh, int k_size, ::dawn::float_type *vec, ::dawn::float_type *nabla2_vec, ::dawn::float_type *primal_edge_length, ::dawn::float_type *dual_edge_length, ::dawn::float_type *tangent_orientation, ::dawn::float_type *geofac_curl, ::dawn::float_type *geofac_div, ::dawn::float_type *interp_in, ::dawn::float_type *interp_coeff, ::dawn::float_type *interp_out, ::dawn::float_type *div_vec, ::dawn::float_type *curl_vec, ::dawn::float_type *__tmp_nabl_149, ::dawn::float_type *__tmp_nabl_150) 
;
double run_exercise(dawn::GlobalGpuTriMesh *mesh, int k_size, ::dawn::float_type *vec, ::dawn::float_type *nabla2_vec, ::dawn::float_type *primal_edge_length, ::dawn::float_type *dual_edge_length, ::dawn::float_type *tangent_orientation, ::dawn::float_type *geofac_curl, ::dawn::float_type *geofac_div, ::dawn::float_type *interp_in, ::dawn::float_type *interp_coeff, ::dawn::float_type *interp_out, ::dawn::float_type *div_vec, ::dawn::float_type *curl_vec, ::dawn::float_type *__tmp_nabl_149, ::dawn::float_type *__tmp_nabl_150) 
;
}
