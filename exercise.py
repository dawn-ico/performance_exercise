
from dusk.script import *


@stencil
def exercise(
    vec: Field[Edge, K],
    nabla2_vec: Field[Edge, K],
    primal_edge_length: Field[Edge],
    dual_edge_length: Field[Edge],
    tangent_orientation: Field[Edge],
    geofac_curl: Field[Vertex > Edge],
    geofac_div: Field[Cell > Edge],
    interp_in: Field[Cell, K],
    interp_coeff: Field[Edge > Cell],
    interp_out: Field[Edge, K]
) -> None:

    div_vec: Field[Cell, K]
    curl_vec: Field[Vertex, K]
    nabla2t1_vec: Field[Edge, K]
    nabla2t2_vec: Field[Edge, K]

    with levels_upward:

        ### Laplacian computation ###

        # compute curl (on vertices)
        curl_vec = sum_over(Vertex > Edge, vec * geofac_curl)

        # compute divergence (on cells)
        div_vec = sum_over(Cell > Edge, vec * geofac_div)

        # first term of of nabla2 (gradient of curl)
        nabla2t1_vec = sum_over(Edge > Vertex, curl_vec, weights=[-1.0, 1])
        nabla2t1_vec = tangent_orientation * nabla2t1_vec / primal_edge_length

        # second term of of nabla2 (gradient of divergence)
        nabla2t2_vec = sum_over(Edge > Cell, div_vec, weights=[-1.0, 1])
        nabla2t2_vec = tangent_orientation * nabla2t2_vec / dual_edge_length

        # finalize nabla2 (difference between the two gradients)
        nabla2_vec = nabla2t2_vec - nabla2t1_vec

        ### Simple interpolation computation ###

        interp_out = sum_over(Edge > Cell, interp_in * interp_coeff)
