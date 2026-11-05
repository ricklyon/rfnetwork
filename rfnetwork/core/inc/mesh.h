#ifndef MESH_H
#define MESH_H

#include <vector>
#include <cstdint>

std::vector<double> blend_cell_widths(double a, double b, double d, int n_min, double tol);

std::vector<double> remove_interior_edges(const double * edges_p, int n_edges, double tolerance, double zero);

void is_point_in_surface(
    const double * points_p,
    int n_points,
    const double * vertices_p,
    const int * vertex_count,
    const double * edges_p,
    const int * edge_count,
    int n_faces,
    int axis,
    int c1_axis,
    int c2_axis,
    double plane_pos,
    double tolerance,
    double d_min,
    double zero_tol,
    uint8_t * in_surface
);

#endif /* MESH_H */
