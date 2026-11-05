/**
 * @file mesh.cpp
 * @brief Mesh generation helper functions.
 *
 * @author Rick Lyon
 * @date 2026-11-05
 *
 */

#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <cstdint>
#include <utility>
#include <algorithm>
#include <stdexcept>

#include "mesh.h"

#define GROWTH_RATE_X0 1.2
#define GROWTH_RATE_TARGET 1.5
#define SOLVE_MAX_ITER 200

#define SQR(x) ((x) * (x))

static inline double dot3(const double * a, const double * b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static inline double norm3(const double * a)
{
    return std::sqrt(dot3(a, a));
}

static inline double sign(double x)
{
    return (x > 0) - (x < 0);
}

/** same behavior as np.clip, returns hi if lo > hi */
static inline double clip(double x, double lo, double hi)
{
    return std::min(std::max(x, lo), hi);
}

/**
 * @brief Solve for the positive growth rate m that satisfies a * sum(m^r, r=1..k) = D.
 *
 * The function is monotonically increasing for m > 0 and negative at m = 0, so there is a single positive
 * root. Newton's method is used with a bisection fallback to keep the iterate inside the bracketing interval.
 */
static double solve_growth_rate(double a, int k, double D)
{
    // f(0) = -D < 0, and f(D/a) >= a * (D/a) - D = 0, so the root lies in (0, D/a]
    double lo = 0.0;
    double hi = D / a;
    double m = GROWTH_RATE_X0;

    if (m <= lo || m >= hi)
    {
        m = 0.5 * (lo + hi);
    }

    for (int it = 0; it < SOLVE_MAX_ITER; ++it)
    {
        // evaluate f(m) and f'(m)
        double f = 0.0;
        double df = 0.0;
        double m_pow = 1.0;
        for (int r = 1; r <= k; ++r)
        {
            df += r * m_pow;
            m_pow *= m;
            f += m_pow;
        }
        f = a * f - D;
        df = a * df;

        if (f == 0.0)
        {
            return m;
        }

        // update bracket
        if (f < 0.0)
        {
            lo = m;
        }
        else
        {
            hi = m;
        }

        double m_next = m - f / df;

        // fall back to bisection if the newton step leaves the bracket
        if (!(m_next > lo && m_next < hi))
        {
            m_next = 0.5 * (lo + hi);
        }

        if (std::abs(m_next - m) <= 1e-15 * std::max(1.0, std::abs(m)))
        {
            return m_next;
        }

        m = m_next;
    }

    return m;
}

/**
 * @brief Return a list of cell widths that divide a cell with width d into graded sub-cells.
 *
 * a is the width of the previous cell, b is the width of the next cell. Sub-cells are graded from the smaller
 * of a and b towards the larger, choosing the number of sub-cells so the growth rate is as near to 1.5 as
 * possible.
 *
 * @param a width of the previous cell
 * @param b width of the next cell
 * @param d width of the cell to divide
 * @param n_min minimum number of sub-cells (used for linear grading only)
 * @param tol tolerance used to check that the sub-cells sum to d
 */
std::vector<double> blend_cell_widths(double a, double b, double d, int n_min, double tol)
{
    // if d is less than or equal to a, no grading is possible, return d as the cell width
    if (std::abs(d - a) < tol)
    {
        return std::vector<double>{d};
    }

    // swap a and b if a is larger than b. Always start with the smaller cell and increase up to b
    bool flip_a_b = false;
    if (b < a)
    {
        std::swap(a, b);
        flip_a_b = true;
    }

    if (!(a > 0.0))
    {
        throw std::runtime_error("Mesh did not converge. Adjacent cell width must be positive.");
    }

    // number of cells can be no larger than d/a (with all cells are of width a)
    int n_max = (int) (d / a) + 1;

    // if cells are nearly equal, use symmetrical grading
    bool symmetric_grading = ((b / a) < 1.2) && (d > (2 * a));

    // number of sub-cells to test
    std::vector<int> n_test;
    if (symmetric_grading)
    {
        for (int n = 2; n <= n_max; n += 2)
        {
            n_test.push_back(n);
        }
    }
    else
    {
        for (int n = n_min; n <= n_max; ++n)
        {
            n_test.push_back(n);
        }
    }

    if (n_test.empty())
    {
        throw std::runtime_error("Mesh did not converge.");
    }

    // track the sub-cells with the lowest combined difference from the desired growth rate of 1.5
    std::vector<double> cells_best;
    double err_best = 0.0;

    for (size_t i = 0; i < n_test.size(); ++i)
    {
        int n = n_test[i];
        std::vector<double> cells;
        double m;

        // if using symmetrical grading, solve for the growth rate that reaches half of the cell width,
        // and create the full subcell vector
        if (symmetric_grading)
        {
            int n_half = n / 2;
            m = solve_growth_rate(a, n_half, d / 2);

            cells.resize(2 * n_half);
            for (int r = 1; r <= n_half; ++r)
            {
                double w = a * std::pow(m, r);
                cells[r - 1] = w;
                cells[2 * n_half - r] = w;
            }
        }
        // if using linear grading, create the list of cells with the growth rate m
        else
        {
            m = solve_growth_rate(a, n, d);

            cells.resize(n);
            for (int r = 1; r <= n; ++r)
            {
                cells[r - 1] = a * std::pow(m, r);
            }
        }

        // growth rates decrease as n increases, find the growth rate (for both internal sub-cells and between
        // the last sub-cell and the next cell b) nearest to 1.5
        double err = std::abs(m - GROWTH_RATE_TARGET) + std::abs(b / cells.back() - 1.0);
        if (i == 0 || err < err_best)
        {
            err_best = err;
            cells_best = cells;
        }
    }

    // check that sub-cell widths add up the desired total width
    double total = 0.0;
    for (double w : cells_best)
    {
        total += w;
    }

    if (!(std::abs(total - d) < tol))
    {
        throw std::runtime_error("Mesh did not converge.");
    }

    if (flip_a_b)
    {
        std::reverse(cells_best.begin(), cells_best.end());
    }

    return cells_best;
}

/**
 * @brief Remove interior edges of a 2D surface that are shared (fully or partially) with other faces.
 *
 * Each edge AB is compared against all other edges. Colinear edges that overlap AB are removed from AB, and the
 * remaining non-overlapping segments of AB are returned as new edges.
 *
 * @param edges_p pointer to a row-major (n_edges x 2 x 3) array of edge endpoints
 * @param n_edges number of edges
 * @param tolerance distance and colinearity tolerance
 * @param zero tolerance for zero lengths
 * @return flattened (M x 2 x 3) array of the non-overlapping edges
 */
std::vector<double> remove_interior_edges(const double * edges_p, int n_edges, double tolerance, double zero)
{
    // remove edges with zero length
    std::vector<std::array<double, 3>> p1;
    std::vector<std::array<double, 3>> p2;

    for (int i = 0; i < n_edges; ++i)
    {
        const double * e = edges_p + (i * 6);
        double len = std::sqrt(SQR(e[3] - e[0]) + SQR(e[4] - e[1]) + SQR(e[5] - e[2]));
        if (len > tolerance)
        {
            p1.push_back({e[0], e[1], e[2]});
            p2.push_back({e[3], e[4], e[5]});
        }
    }

    int n = (int) p1.size();

    // signed distances from A to the endpoints of each edge, along the direction of AB
    std::vector<double> len_A_p1(n);
    std::vector<double> len_A_p2(n);

    // non-overlapping edges, flattened (M x 2 x 3)
    std::vector<double> nonovl_edges;

    for (int i = 0; i < n; ++i)
    {
        // two end points of current edge
        const std::array<double, 3> & A = p1[i];
        const std::array<double, 3> & B = p2[i];

        double v_AB[3] = {B[0] - A[0], B[1] - A[1], B[2] - A[2]};
        double len_AB = std::max(norm3(v_AB), 1e-12);

        // indices of edges that overlap with AB (excluding itself)
        std::vector<int> ovl_idx;

        for (int j = 0; j < n; ++j)
        {
            // vectors from the base point of the current edge to the end points of edge j
            double v_A_p1[3] = {p1[j][0] - A[0], p1[j][1] - A[1], p1[j][2] - A[2]};
            double v_A_p2[3] = {p2[j][0] - A[0], p2[j][1] - A[1], p2[j][2] - A[2]};

            double l1 = std::max(norm3(v_A_p1), zero);
            double l2 = std::max(norm3(v_A_p2), zero);

            // dot product normalized by the lengths to get cos(theta)
            double p1_dot = dot3(v_A_p1, v_AB) / (l1 * len_AB);
            double p2_dot = dot3(v_A_p2, v_AB) / (l2 * len_AB);

            // if the edges share a point with A, set to 1 to indicate the edge is in the same direction as AB.
            if (l1 <= zero * 2)
            {
                p1_dot = 1;
            }
            if (l2 <= zero * 2)
            {
                p2_dot = 1;
            }

            // set lengths to negative if lines in opposite direction
            l1 = sign(p1_dot) * l1;
            l2 = sign(p2_dot) * l2;

            len_A_p1[j] = l1;
            len_A_p2[j] = l2;

            if (j == i)
            {
                continue;
            }

            bool is_edge_colinear = (
                (std::abs(std::abs(p1_dot) - 1) < tolerance) && (std::abs(std::abs(p2_dot) - 1) < tolerance)
            );

            bool does_edge_overlap = is_edge_colinear && (
                ((l1 + tolerance > 0) && ((l1 - tolerance) < len_AB)) || // if p1 falls on AB
                ((l2 + tolerance > 0) && ((l2 - tolerance) < len_AB)) || // if p2 falls on AB
                (sign(l2) != sign(l1)) // p1 is on opposite side of A as p2, or vice versa
            );

            if (does_edge_overlap)
            {
                ovl_idx.push_back(j);
            }
        }

        // if no other edge overlaps with AB, add the full AB edge as a non-overlapping edge
        if (ovl_idx.empty())
        {
            nonovl_edges.insert(nonovl_edges.end(), A.begin(), A.end());
            nonovl_edges.insert(nonovl_edges.end(), B.begin(), B.end());
            continue;
        }

        // distances from point A that define colinear line segments that overlap with AB.
        std::vector<std::pair<double, double>> segments;
        for (int j : ovl_idx)
        {
            double l1 = len_A_p1[j];
            double l2 = len_A_p2[j];

            // if P1 is on A, overlapping segment is A->P2
            if (std::abs(l1) < tolerance)
            {
                segments.push_back({0, l2});
            }
            else if (std::abs(l2) < tolerance)
            {
                segments.push_back({0, l1});
            }
            // if A->P1 and A->P2 are in the same direction as AB, and both are smaller in length than AB, the
            // overlapping segment is fully contained in A and B
            else if ((l1 > 0) && (l1 < len_AB) && (l2 < len_AB))
            {
                if (l1 < l2)
                {
                    segments.push_back({l1, l2});
                }
                else
                {
                    segments.push_back({l2, l1});
                }
            }
            // if A->P1 is in the opposite direction, P2 is before or past B, and the overlapping segment is A->P2
            else if (l1 < 0)
            {
                segments.push_back({0, l2});
            }
            // A->P2 is in the opposite direction, P1 is before or past B, and the overlapping segment is A->P1
            else if (l2 < 0)
            {
                segments.push_back({0, l1});
            }
            // both A->P1 and A->P2 are in the same direction, but one is longer than AB.
            // if A->P1 is smaller than A->P2, the overlapping segment is P1->B
            else if (l1 < l2)
            {
                segments.push_back({l1, len_AB});
            }
            // A->P2 is smaller than A->P1, then A->P1 is longer than AB, overlapping segment is P2->B
            else
            {
                segments.push_back({l2, len_AB});
            }
        }

        // start with the full segment A->B, and remove the sections in segments. Each segment might
        // split the non-overlapping parts into two different parts.
        std::vector<std::pair<double, double>> nonovl_segments = {{0, len_AB}};

        for (const auto & seg : segments)
        {
            // iterate over the segments that existed before this pass, new segments are appended to the end
            size_t n_seg = nonovl_segments.size();
            for (size_t k = 0; k < n_seg; ++k)
            {
                std::pair<double, double> nseg = nonovl_segments[k];

                // clip endpoints of segment to the non-overlapping segment
                double s0 = clip(seg.first, nseg.first, nseg.second);
                double s1 = clip(seg.second, nseg.first, nseg.second);

                // start points are the same, non-overlapping part is from end point of segment to the end
                if (std::abs(s0 - nseg.first) < tolerance)
                {
                    nonovl_segments[k] = {s1, nseg.second};
                }
                // endpoints are the same, non-overlapping part is up to the start point of segment
                else if (std::abs(s1 - nseg.second) < tolerance)
                {
                    nonovl_segments[k] = {nseg.first, s0};
                }
                // segment is between non-overlapping endpoints, split into two segments
                else
                {
                    nonovl_segments[k] = {nseg.first, s0};
                    nonovl_segments.push_back({s1, nseg.second});
                }
            }
        }

        // unit vector along the line from A->B
        double v[3] = {v_AB[0] / len_AB, v_AB[1] / len_AB, v_AB[2] / len_AB};

        // create edges from segments
        for (const auto & nseg : nonovl_segments)
        {
            // remove segments with 0 length.
            if ((nseg.second - nseg.first) < tolerance)
            {
                continue;
            }

            for (int c = 0; c < 3; ++c)
            {
                nonovl_edges.push_back(A[c] + v[c] * nseg.first);
            }
            for (int c = 0; c < 3; ++c)
            {
                nonovl_edges.push_back(A[c] + v[c] * nseg.second);
            }
        }
    }

    return nonovl_edges;
}

/**
 * @brief Determine if each point is inside a 2D surface on a cardinal plane.
 *
 * Each face of the surface should be a simple polygon (no cutouts). Lines are drawn from each polygon vertex to
 * the test point. A point is inside a face if, for every vertex, the angles between the vertex->point line and the
 * two edges meeting at the vertex sum to the vertex angle.
 *
 * @param points_p row-major (n_points x 3) array of test points
 * @param n_points number of test points
 * @param vertices_p row-major (sum(vertex_count) x 3) array of face vertices, grouped by face
 * @param vertex_count number of vertices in each face
 * @param edges_p row-major (sum(edge_count) x 2 x 3) array of face edges, grouped by face
 * @param edge_count number of edges in each face
 * @param n_faces number of faces
 * @param axis axis normal to the surface
 * @param c1_axis first axis on the surface plane
 * @param c2_axis second axis on the surface plane
 * @param plane_pos position of the surface along axis
 * @param tolerance angle tolerance in radians
 * @param d_min points within d_min / 3 of a vertex are included in the surface. Ignored if negative.
 * @param zero_tol tolerance for zero values
 * @param in_surface output array of length n_points, set to 1 if the point is in the surface
 */
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
)
{
    // per-vertex data needed to evaluate each test point
    struct VertexAngle
    {
        double v[3];
        double v_p1[3];
        double v_p2[3];
        double len_v_p1;
        double len_v_p2;
        double vertex_ang;
    };

    // vertex data for all faces, and the start index in vertex_data of each face
    std::vector<VertexAngle> vertex_data;
    std::vector<int> face_start(n_faces + 1, 0);

    int vertex_offset = 0;
    int edge_offset = 0;

    for (int f = 0; f < n_faces; ++f)
    {
        face_start[f] = (int) vertex_data.size();

        const double * face_vertices = vertices_p + (vertex_offset * 3);
        const double * face_edges = edges_p + (edge_offset * 6);
        int n_face_edges = edge_count[f];

        for (int i = 0; i < vertex_count[f]; ++i)
        {
            const double * v = face_vertices + (i * 3);
            double vx = v[c1_axis];
            double vy = v[c2_axis];

            // get the two edges that meet this vertex. If either end point is on the vertex, mark the edge.
            std::vector<int> edge_on_vertex_idx;
            for (int e = 0; e < n_face_edges; ++e)
            {
                for (int p = 0; p < 2; ++p)
                {
                    const double * ep = face_edges + (e * 6) + (p * 3);
                    if ((std::abs(ep[c1_axis] - vx) < zero_tol) && (std::abs(ep[c2_axis] - vy) < zero_tol))
                    {
                        edge_on_vertex_idx.push_back(e);
                        break;
                    }
                }
            }

            if (edge_on_vertex_idx.size() != 2)
            {
                throw std::runtime_error(std::to_string(edge_on_vertex_idx.size()));
            }

            VertexAngle va;
            std::copy(v, v + 3, va.v);

            // get the far endpoint of each edge, the other endpoint is the vertex
            double * v_p[2] = {va.v_p1, va.v_p2};
            for (int k = 0; k < 2; ++k)
            {
                const double * edge = face_edges + (edge_on_vertex_idx[k] * 6);
                double d0[3] = {v[0] - edge[0], v[1] - edge[1], v[2] - edge[2]};
                double d1[3] = {v[0] - edge[3], v[1] - edge[4], v[2] - edge[5]};
                const double * endpoint = (norm3(d1) > norm3(d0)) ? (edge + 3) : edge;

                // vector from vertex to the endpoint of the line attached to vertex
                for (int c = 0; c < 3; ++c)
                {
                    v_p[k][c] = endpoint[c] - v[c];
                }
            }

            // angle between the two lines
            va.len_v_p1 = std::max(norm3(va.v_p1), zero_tol);
            va.len_v_p2 = std::max(norm3(va.v_p2), zero_tol);
            va.vertex_ang = std::acos(clip(dot3(va.v_p1, va.v_p2) / (va.len_v_p1 * va.len_v_p2), -1, 1));

            vertex_data.push_back(va);
        }

        vertex_offset += vertex_count[f];
        edge_offset += n_face_edges;
    }
    face_start[n_faces] = (int) vertex_data.size();

    for (int p = 0; p < n_points; ++p)
    {
        const double * tp = points_p + (p * 3);
        in_surface[p] = 0;

        // check that the point is in the same plane as the surface
        if (!(std::abs(tp[axis] - plane_pos) < zero_tol))
        {
            continue;
        }

        // point is in the surface if it is in any face
        for (int f = 0; f < n_faces; ++f)
        {
            bool in_face = true;

            // point is in the face if it is within the edge lines from all vertices
            for (int i = face_start[f]; i < face_start[f + 1]; ++i)
            {
                const VertexAngle & va = vertex_data[i];

                // compute angle between the test point and the two edges
                double v_tp[3] = {tp[0] - va.v[0], tp[1] - va.v[1], tp[2] - va.v[2]};
                double norm_tp_v = norm3(v_tp);
                double len_tp_v = std::max(norm_tp_v, zero_tol);

                double e1_ang = std::acos(clip(dot3(v_tp, va.v_p1) / (va.len_v_p1 * len_tp_v), -1, 1));
                double e2_ang = std::acos(clip(dot3(v_tp, va.v_p2) / (va.len_v_p2 * len_tp_v), -1, 1));

                // if the total angle is the same as the vertex angle, the point is between the two edge lines.
                bool within_vertex = std::abs((e1_ang + e2_ang) - va.vertex_ang) < tolerance;

                // include point in surface if it is very close to a vertex. These can suffer from numerical errors
                // when computing angles.
                if ((d_min >= 0) && (norm_tp_v < (d_min / 3)))
                {
                    within_vertex = true;
                }

                if (!within_vertex)
                {
                    in_face = false;
                    break;
                }
            }

            if (in_face)
            {
                in_surface[p] = 1;
                break;
            }
        }
    }
}
