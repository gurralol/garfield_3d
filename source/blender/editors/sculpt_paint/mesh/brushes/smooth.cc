/* SPDX-FileCopyrightText: 2024 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include <cfloat>
#include <cmath>

#include "editors/sculpt_paint/mesh/brushes/brushes.hh"

#include "DNA_brush_types.h"
#include "DNA_mesh_types.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"

#include "BKE_mesh.hh"
#include "BKE_object_types.hh"
#include "BKE_paint.hh"
#include "BKE_paint_bvh.hh"
#include "BKE_subdiv_ccg.hh"

#include "BLI_array.hh"
#include "BLI_enumerable_thread_specific.hh"
#include "BLI_math_vector.hh"
#include "BLI_task.hh"

#include "editors/sculpt_paint/mesh/mesh_brush_common.hh"
#include "editors/sculpt_paint/mesh/sculpt_automask.hh"
#include "editors/sculpt_paint/mesh/sculpt_boundary.hh"
#include "editors/sculpt_paint/mesh/sculpt_intern.hh"
#include "editors/sculpt_paint/mesh/sculpt_smooth.hh"

#include "bmesh.hh"

namespace blender::ed::sculpt_paint::brushes {

inline namespace smooth_cc {

constexpr int smooth_iteration_count = 4;
constexpr float preserve_form_alpha = 0.75f;
constexpr float preserve_form_beta = 0.25f;
constexpr float preserve_form_blur_radius_factor = 0.15f;
constexpr int max_preserve_form_blur_iterations = 12;

static float iteration_budget_from_strength(const float strength, const int max_iterations)
{
  BLI_assert_msg(strength >= 0.0f,
                 "The smooth brush expects a non-negative strength to behave properly");
  return std::min(strength, 1.0f) * max_iterations;
}

static Vector<float> iteration_strengths_from_budget(const float budget, const int max_iterations)
{
  const float clamped_budget = std::clamp(budget, 0.0f, float(max_iterations));
  const int count = int(clamped_budget);
  const float last = clamped_budget - float(count);
  Vector<float> result;
  result.append_n_times(1.0f, count);
  if (last > 0.0f || result.is_empty()) {
    result.append(last);
  }
  return result;
}

static Vector<float> iteration_strengths(const float strength)
{
  return iteration_strengths_from_budget(
      iteration_budget_from_strength(strength, smooth_iteration_count), smooth_iteration_count);
}

static int preserve_form_blur_iterations(const float radius, const float avg_edge_length)
{
  const float safe_edge_length = std::max(avg_edge_length, FLT_EPSILON);
  return std::clamp(
      int(std::round((radius * preserve_form_blur_radius_factor) / safe_edge_length)),
      1,
      max_preserve_form_blur_iterations);
}

struct LocalData {
  Vector<float3> positions;
  Vector<float> factors;
  Vector<float> distances;
  Vector<int> neighbor_offsets;
  Vector<int> neighbor_data;
  Vector<float3> new_positions;
  Vector<float3> average_positions;
  Vector<float3> laplacian_disp;
  Vector<float3> translations;
  Vector<float3> blur_buffer;
  Vector<int> vert_indices;
  Vector<Vector<SubdivCCGCoord>> grid_neighbors;
  Vector<Vector<BMVert *>> bmesh_neighbors;
};

static float average_neighbor_edge_length_mesh(const Span<float3> positions,
                                               const Span<int> verts,
                                               const GroupedSpan<int> neighbors,
                                               const Span<float> factors)
{
  BLI_assert(verts.size() == neighbors.size());
  BLI_assert(verts.size() == factors.size());

  double total_length = 0.0;
  int total_edges = 0;
  for (const int i : verts.index_range()) {
    if (factors[i] == 0.0f) {
      continue;
    }
    const float3 &position = positions[verts[i]];
    for (const int neighbor : neighbors[i]) {
      total_length += math::distance(position, positions[neighbor]);
      total_edges++;
    }
  }
  return total_edges > 0 ? float(total_length / total_edges) : 0.0f;
}

static float average_neighbor_edge_length_grids(const CCGKey &key,
                                                const Span<float3> all_positions,
                                                const Span<float3> positions,
                                                const Span<Vector<SubdivCCGCoord>> neighbors,
                                                const Span<float> factors)
{
  BLI_assert(positions.size() == neighbors.size());
  BLI_assert(positions.size() == factors.size());

  double total_length = 0.0;
  int total_edges = 0;
  for (const int i : positions.index_range()) {
    if (factors[i] == 0.0f) {
      continue;
    }
    const float3 &position = positions[i];
    for (const SubdivCCGCoord &neighbor : neighbors[i]) {
      total_length += math::distance(position, all_positions[neighbor.to_index(key)]);
      total_edges++;
    }
  }
  return total_edges > 0 ? float(total_length / total_edges) : 0.0f;
}

static float average_neighbor_edge_length_bmesh(const Span<float3> positions,
                                                const Span<Vector<BMVert *>> neighbors,
                                                const Span<float> factors)
{
  BLI_assert(positions.size() == neighbors.size());
  BLI_assert(positions.size() == factors.size());

  double total_length = 0.0;
  int total_edges = 0;
  for (const int i : positions.index_range()) {
    if (factors[i] == 0.0f) {
      continue;
    }
    const float3 &position = positions[i];
    for (const BMVert *neighbor : neighbors[i]) {
      total_length += math::distance(position, float3(neighbor->co));
      total_edges++;
    }
  }
  return total_edges > 0 ? float(total_length / total_edges) : 0.0f;
}

BLI_NOINLINE static void apply_positions_faces(const Sculpt &sd,
                                               const bke::pbvh::MeshNode &node,
                                               Object &object,
                                               LocalData &tls,
                                               const Span<float> factors,
                                               const Span<float3> new_positions,
                                               const PositionDeformData &position_data)
{
  SculptSession &ss = *object.runtime->sculpt_session;

  const Span<int> verts = node.verts();

  tls.translations.resize(verts.size());
  const MutableSpan<float3> translations = tls.translations;
  translations_from_new_positions(new_positions, verts, position_data.eval, translations);
  scale_translations(translations, factors);

  clip_and_lock_translations(sd, ss, position_data.eval, verts, translations);
  position_data.deform(translations, verts);
}

BLI_NOINLINE static void do_smooth_brush_mesh(const Depsgraph &depsgraph,
                                              const Sculpt &sd,
                                              const Brush &brush,
                                              Object &object,
                                              const IndexMask &node_mask,
                                              const float brush_strength)
{
  const SculptSession &ss = *object.runtime->sculpt_session;
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  MutableSpan<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  const OffsetIndices faces = mesh.faces();
  const Span<int> corner_verts = mesh.corner_verts();
  const GroupedSpan<int> vert_to_face_map = mesh.vert_to_face_map();
  const MeshAttributeData attribute_data(mesh);

  const PositionDeformData position_data(depsgraph, object);
  const Span<float3> vert_normals = bke::pbvh::vert_normals_eval(depsgraph, object);

  Array<int> node_offset_data;
  const OffsetIndices<int> node_vert_offsets = create_node_vert_offsets(
      nodes, node_mask, node_offset_data);
  Array<float3> new_positions(node_vert_offsets.total_size());
  Array<float> all_factors(node_vert_offsets.total_size());
  Array<float> all_distances(node_vert_offsets.total_size());

  threading::EnumerableThreadSpecific<LocalData> all_tls;

  /* Calculate the new positions into a separate array in a separate loop because multiple loops
   * are updated in parallel. Without this there would be non-threadsafe access to changing
   * positions in other bke::pbvh::Tree nodes. */
  for (const float strength : iteration_strengths(brush_strength)) {
    node_mask.foreach_index(
        [&](const int i, const int pos) {
          LocalData &tls = all_tls.local();
          const Span<int> verts = nodes[i].verts();
          const MutableSpan<float> node_factors = all_factors.as_mutable_span().slice(
              node_vert_offsets[pos]);
          calc_factors_common_mesh_indexed(
              depsgraph,
              brush,
              object,
              attribute_data,
              position_data.eval,
              vert_normals,
              nodes[i],
              node_factors,
              all_distances.as_mutable_span().slice(node_vert_offsets[pos]));
          scale_factors(node_factors, strength);
          const GroupedSpan<int> neighbors = calc_vert_neighbors_interior(
              faces,
              corner_verts,
              vert_to_face_map,
              ss.boundary_info_cache->verts,
              ss.boundary_info_cache->edges,
              attribute_data.hide_poly,
              verts,
              node_factors,
              tls.neighbor_offsets,
              tls.neighbor_data);
          smooth::neighbor_data_average_mesh_check_loose(
              position_data.eval,
              verts,
              neighbors,
              new_positions.as_mutable_span().slice(node_vert_offsets[pos]));
        },
        exec_mode::grain_size(1));

    node_mask.foreach_index(
        [&](const int i, const int pos) {
          LocalData &tls = all_tls.local();
          apply_positions_faces(sd,
                                nodes[i],
                                object,
                                tls,
                                all_factors.as_mutable_span().slice(node_vert_offsets[pos]),
                                new_positions.as_span().slice(node_vert_offsets[pos]),
                                position_data);
        },
        exec_mode::grain_size(1));
  }
}

static void calc_grids(const Depsgraph &depsgraph,
                       const Sculpt &sd,
                       const OffsetIndices<int> faces,
                       const Span<int> corner_verts,
                       const BitSpan boundary_verts,
                       const Set<OrderedEdge> &boundary_edges,
                       Object &object,
                       const Brush &brush,
                       const float strength,
                       const bke::pbvh::GridsNode &node,
                       LocalData &tls)
{
  SculptSession &ss = *object.runtime->sculpt_session;
  SubdivCCG &subdiv_ccg = *ss.subdiv_ccg;

  const Span<int> grids = node.grids();
  const MutableSpan positions = gather_grids_positions(subdiv_ccg, grids, tls.positions);

  calc_factors_common_grids(depsgraph, brush, object, positions, node, tls.factors, tls.distances);

  scale_factors(tls.factors, strength);

  tls.new_positions.resize(positions.size());
  const MutableSpan<float3> new_positions = tls.new_positions;
  smooth::neighbor_position_average_interior_grids(faces,
                                                   corner_verts,
                                                   boundary_verts,
                                                   boundary_edges,
                                                   subdiv_ccg,
                                                   grids,
                                                   tls.factors,
                                                   new_positions);

  tls.translations.resize(positions.size());
  const MutableSpan<float3> translations = tls.translations;
  translations_from_new_positions(new_positions, positions, translations);
  scale_translations(translations, tls.factors);

  clip_and_lock_translations(sd, ss, positions, translations);
  apply_translations(translations, grids, subdiv_ccg);
}

static void calc_bmesh(const Depsgraph &depsgraph,
                       const Sculpt &sd,
                       Object &object,
                       const Brush &brush,
                       const float strength,
                       bke::pbvh::BMeshNode &node,
                       LocalData &tls)
{
  SculptSession &ss = *object.runtime->sculpt_session;

  const Set<BMVert *, 0> &verts = BKE_pbvh_bmesh_node_unique_verts(&node);
  const MutableSpan positions = gather_bmesh_positions(verts, tls.positions);

  calc_factors_common_bmesh(depsgraph, brush, object, positions, node, tls.factors, tls.distances);

  scale_factors(tls.factors, strength);

  tls.new_positions.resize(verts.size());
  const MutableSpan<float3> new_positions = tls.new_positions;
  smooth::neighbor_position_average_interior_bmesh(verts, tls.factors, new_positions);

  tls.translations.resize(verts.size());
  const MutableSpan<float3> translations = tls.translations;
  translations_from_new_positions(new_positions, positions, translations);
  scale_translations(translations, tls.factors);

  clip_and_lock_translations(sd, ss, positions, translations);
  apply_translations(translations, verts);
}

BLI_NOINLINE static void do_smooth_brush_preserve_form_mesh(const Depsgraph &depsgraph,
                                                            const Sculpt &sd,
                                                            const Brush &brush,
                                                            Object &object,
                                                            const IndexMask &node_mask,
                                                            const float brush_strength)
{
  SculptSession &ss = *object.runtime->sculpt_session;
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  MutableSpan<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  const OffsetIndices faces = mesh.faces();
  const Span<int> corner_verts = mesh.corner_verts();
  const GroupedSpan<int> vert_to_face_map = mesh.vert_to_face_map();
  const MeshAttributeData attribute_data(mesh);

  const PositionDeformData position_data(depsgraph, object);
  const Span<float3> vert_normals = bke::pbvh::vert_normals_eval(depsgraph, object);

  Array<int> node_offset_data;
  const OffsetIndices<int> node_vert_offsets = create_node_vert_offsets(
      nodes, node_mask, node_offset_data);
  Array<float> all_factors(node_vert_offsets.total_size());
  Array<float> all_distances(node_vert_offsets.total_size());
  Array<int> node_blur_iterations(node_mask.size(), 1);

  threading::EnumerableThreadSpecific<LocalData> all_tls;
  MutableSpan<float3> all_laplacian_disp = ss.cache->surface_smooth_laplacian_disp;

  node_mask.foreach_index(
      [&](const int i, const int pos) {
        LocalData &tls = all_tls.local();
        const Span<int> verts = nodes[i].verts();
        const MutableSpan<float> node_factors = all_factors.as_mutable_span().slice(
            node_vert_offsets[pos]);
        calc_factors_common_mesh_indexed(depsgraph,
                                         brush,
                                         object,
                                         attribute_data,
                                         position_data.eval,
                                         vert_normals,
                                         nodes[i],
                                         node_factors,
                                         all_distances.as_mutable_span().slice(
                                             node_vert_offsets[pos]));
        scale_factors(node_factors, brush_strength);

        const GroupedSpan<int> neighbors = calc_vert_neighbors_interior(faces,
                                                                        corner_verts,
                                                                        vert_to_face_map,
                                                                        ss.boundary_info_cache->verts,
                                                                        ss.boundary_info_cache->edges,
                                                                        attribute_data.hide_poly,
                                                                        verts,
                                                                        node_factors,
                                                                        tls.neighbor_offsets,
                                                                        tls.neighbor_data);
        const float avg_edge_length = average_neighbor_edge_length_mesh(
            position_data.eval, verts, neighbors, node_factors);
        node_blur_iterations[pos] = preserve_form_blur_iterations(ss.cache->radius, avg_edge_length);
      },
      exec_mode::grain_size(1));

  node_mask.foreach_index(
      [&](const int i, const int pos) {
        LocalData &tls = all_tls.local();
        const Span<int> verts = nodes[i].verts();
        const Span<float> node_factors = all_factors.as_span().slice(node_vert_offsets[pos]);
        const MutableSpan<float3> positions = gather_data_mesh(position_data.eval, verts, tls.positions);
        const GroupedSpan<int> neighbors = calc_vert_neighbors_interior(faces,
                                                                        corner_verts,
                                                                        vert_to_face_map,
                                                                        ss.boundary_info_cache->verts,
                                                                        ss.boundary_info_cache->edges,
                                                                        attribute_data.hide_poly,
                                                                        verts,
                                                                        node_factors,
                                                                        tls.neighbor_offsets,
                                                                        tls.neighbor_data);

        tls.new_positions.resize(verts.size());
        const MutableSpan<float3> orig_positions = tls.new_positions;
        for (const int i : positions.index_range()) {
          orig_positions[i] = positions[i];
        }

        tls.average_positions.resize(verts.size());
        tls.blur_buffer.resize(verts.size());
        smooth::blur_positions_mesh(position_data.eval,
                                    verts,
                                    neighbors,
                                    node_factors,
                                    node_blur_iterations[pos],
                                    positions,
                                    tls.average_positions,
                                    tls.blur_buffer);

        tls.laplacian_disp.resize(verts.size());
        const MutableSpan<float3> laplacian_disp = tls.laplacian_disp;
        tls.translations.resize(verts.size());
        const MutableSpan<float3> translations = tls.translations;
        smooth::surface_smooth_laplacian_step(
            positions, orig_positions, tls.average_positions, preserve_form_alpha, laplacian_disp, translations);
        scale_translations(translations, node_factors);

        scatter_data_mesh(laplacian_disp.as_span(), verts, all_laplacian_disp);

        clip_and_lock_translations(sd, ss, position_data.eval, verts, translations);
        position_data.deform(translations, verts);
      },
      exec_mode::grain_size(1));

  node_mask.foreach_index(
      [&](const int i, const int pos) {
        LocalData &tls = all_tls.local();
        const Span<int> verts = nodes[i].verts();
        const Span<float> node_factors = all_factors.as_span().slice(node_vert_offsets[pos]);
        const MutableSpan<float3> laplacian_disp = gather_data_mesh(
            all_laplacian_disp.as_span(), verts, tls.laplacian_disp);
        const GroupedSpan<int> neighbors = calc_vert_neighbors_interior(faces,
                                                                        corner_verts,
                                                                        vert_to_face_map,
                                                                        ss.boundary_info_cache->verts,
                                                                        ss.boundary_info_cache->edges,
                                                                        attribute_data.hide_poly,
                                                                        verts,
                                                                        node_factors,
                                                                        tls.neighbor_offsets,
                                                                        tls.neighbor_data);

        tls.average_positions.resize(verts.size());
        const MutableSpan<float3> average_laplacian_disps = tls.average_positions;
        smooth::neighbor_data_average_mesh_check_loose(
            all_laplacian_disp.as_span(), verts, neighbors, average_laplacian_disps);

        tls.translations.resize(verts.size());
        const MutableSpan<float3> translations = tls.translations;
        smooth::surface_smooth_displace_step(
            laplacian_disp, average_laplacian_disps, preserve_form_beta, translations);
        scale_translations(translations, node_factors);

        clip_and_lock_translations(sd, ss, position_data.eval, verts, translations);
        position_data.deform(translations, verts);
      },
      exec_mode::grain_size(1));
}

static void calc_grids_preserve_form(const Depsgraph &depsgraph,
                                     const Sculpt &sd,
                                     const OffsetIndices<int> faces,
                                     const Span<int> corner_verts,
                                     const BitSpan boundary_verts,
                                     const Set<OrderedEdge> &boundary_edges,
                                     Object &object,
                                     const Brush &brush,
                                     const float strength,
                                     const bke::pbvh::GridsNode &node,
                                     const MutableSpan<float3> all_laplacian_disp,
                                     LocalData &tls)
{
  SculptSession &ss = *object.runtime->sculpt_session;
  SubdivCCG &subdiv_ccg = *ss.subdiv_ccg;

  const Span<int> grids = node.grids();
  const MutableSpan<float3> positions = gather_grids_positions(subdiv_ccg, grids, tls.positions);

  calc_factors_common_grids(depsgraph, brush, object, positions, node, tls.factors, tls.distances);
  scale_factors(tls.factors, strength);

  const CCGKey key = BKE_subdiv_ccg_key_top_level(subdiv_ccg);
  tls.grid_neighbors.resize(positions.size());
  calc_vert_neighbors_interior(
      faces,
      corner_verts,
      boundary_verts,
      boundary_edges,
      subdiv_ccg,
      grids,
      tls.grid_neighbors.as_mutable_span());
  const float avg_edge_length = average_neighbor_edge_length_grids(
      key, subdiv_ccg.positions.as_span(), positions, tls.grid_neighbors.as_span(), tls.factors);
  const int blur_iterations = preserve_form_blur_iterations(ss.cache->radius, avg_edge_length);

  tls.vert_indices.resize(positions.size());
  for (const int i : grids.index_range()) {
    const IndexRange grid_range = bke::ccg::grid_range(key, grids[i]);
    const int node_start = i * key.grid_area;
    for (const int offset : IndexRange(key.grid_area)) {
      tls.vert_indices[node_start + offset] = grid_range[offset];
    }
  }

  tls.new_positions.resize(positions.size());
  const MutableSpan<float3> orig_positions = tls.new_positions;
  for (const int i : positions.index_range()) {
    orig_positions[i] = positions[i];
  }

  tls.average_positions.resize(positions.size());
  tls.blur_buffer.resize(positions.size());
  smooth::blur_positions_grids(key,
                               subdiv_ccg.positions.as_span(),
                               tls.vert_indices,
                               tls.grid_neighbors.as_span(),
                               tls.factors,
                               blur_iterations,
                               positions,
                               tls.average_positions,
                               tls.blur_buffer);

  tls.laplacian_disp.resize(positions.size());
  const MutableSpan<float3> laplacian_disp = tls.laplacian_disp;
  tls.translations.resize(positions.size());
  const MutableSpan<float3> translations = tls.translations;
  smooth::surface_smooth_laplacian_step(
      positions, orig_positions, tls.average_positions, preserve_form_alpha, laplacian_disp, translations);
  scale_translations(translations, tls.factors);

  scatter_data_grids(subdiv_ccg, laplacian_disp.as_span(), grids, all_laplacian_disp);

  clip_and_lock_translations(sd, ss, positions, translations);
  apply_translations(translations, grids, subdiv_ccg);

  const MutableSpan<float3> gathered_laplacian_disp = gather_data_grids(
      subdiv_ccg, all_laplacian_disp.as_span(), grids, tls.laplacian_disp);
  tls.average_positions.resize(positions.size());
  const MutableSpan<float3> average_laplacian_disps = tls.average_positions;
  smooth::average_data_grids(
      subdiv_ccg, all_laplacian_disp.as_span(), grids, average_laplacian_disps);

  tls.translations.resize(positions.size());
  const MutableSpan<float3> displace_translations = tls.translations;
  smooth::surface_smooth_displace_step(
      gathered_laplacian_disp, average_laplacian_disps, preserve_form_beta, displace_translations);
  scale_translations(displace_translations, tls.factors);

  clip_and_lock_translations(sd, ss, positions, displace_translations);
  apply_translations(displace_translations, grids, subdiv_ccg);
}

static void calc_bmesh_preserve_form(const Depsgraph &depsgraph,
                                     const Sculpt &sd,
                                     Object &object,
                                     const Brush &brush,
                                     const float strength,
                                     const MutableSpan<float3> all_laplacian_disp,
                                     bke::pbvh::BMeshNode &node,
                                     LocalData &tls)
{
  SculptSession &ss = *object.runtime->sculpt_session;

  const Set<BMVert *, 0> &verts = BKE_pbvh_bmesh_node_unique_verts(&node);
  const MutableSpan<float3> positions = gather_bmesh_positions(verts, tls.positions);

  calc_factors_common_bmesh(depsgraph, brush, object, positions, node, tls.factors, tls.distances);
  scale_factors(tls.factors, strength);

  tls.bmesh_neighbors.resize(verts.size());
  calc_vert_neighbors_interior(verts, tls.bmesh_neighbors.as_mutable_span());
  const float avg_edge_length = average_neighbor_edge_length_bmesh(
      positions, tls.bmesh_neighbors.as_span(), tls.factors);
  const int blur_iterations = preserve_form_blur_iterations(ss.cache->radius, avg_edge_length);

  tls.vert_indices.resize(verts.size());
  int i = 0;
  for (const BMVert *vert : verts) {
    tls.vert_indices[i] = BM_elem_index_get(vert);
    i++;
  }

  tls.new_positions.resize(verts.size());
  const MutableSpan<float3> orig_positions = tls.new_positions;
  for (const int i : positions.index_range()) {
    orig_positions[i] = positions[i];
  }

  tls.average_positions.resize(verts.size());
  tls.blur_buffer.resize(verts.size());
  smooth::blur_positions_bmesh(tls.vert_indices,
                               tls.bmesh_neighbors.as_span(),
                               tls.factors,
                               blur_iterations,
                               positions,
                               tls.average_positions,
                               tls.blur_buffer);

  tls.laplacian_disp.resize(verts.size());
  const MutableSpan<float3> laplacian_disp = tls.laplacian_disp;
  tls.translations.resize(verts.size());
  const MutableSpan<float3> translations = tls.translations;
  smooth::surface_smooth_laplacian_step(
      positions, orig_positions, tls.average_positions, preserve_form_alpha, laplacian_disp, translations);
  scale_translations(translations, tls.factors);

  scatter_data_bmesh(laplacian_disp.as_span(), verts, all_laplacian_disp);

  clip_and_lock_translations(sd, ss, positions, translations);
  apply_translations(translations, verts);

  const MutableSpan<float3> gathered_laplacian_disp = gather_data_bmesh(
      all_laplacian_disp.as_span(), verts, tls.laplacian_disp);
  tls.average_positions.resize(verts.size());
  const MutableSpan<float3> average_laplacian_disps = tls.average_positions;
  smooth::average_data_bmesh(all_laplacian_disp.as_span(), verts, average_laplacian_disps);

  tls.translations.resize(verts.size());
  const MutableSpan<float3> displace_translations = tls.translations;
  smooth::surface_smooth_displace_step(
      gathered_laplacian_disp, average_laplacian_disps, preserve_form_beta, displace_translations);
  scale_translations(displace_translations, tls.factors);

  clip_and_lock_translations(sd, ss, positions, displace_translations);
  apply_translations(displace_translations, verts);
}

}  // namespace smooth_cc

void do_smooth_brush(const Depsgraph &depsgraph,
                     const Sculpt &sd,
                     Object &object,
                     const IndexMask &node_mask,
                     const float brush_strength)
{
  SculptSession &ss = *object.runtime->sculpt_session;
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  const Brush &brush = *BKE_paint_brush_for_read(&sd.paint);

  boundary::ensure_boundary_info(object);

  switch (pbvh.type()) {
    case bke::pbvh::Type::Mesh:
      do_smooth_brush_mesh(depsgraph, sd, brush, object, node_mask, brush_strength);
      break;
    case bke::pbvh::Type::Grids: {
      const Mesh &base_mesh = *id_cast<const Mesh *>(object.data);
      const OffsetIndices faces = base_mesh.faces();
      const Span<int> corner_verts = base_mesh.corner_verts();

      threading::EnumerableThreadSpecific<LocalData> all_tls;
      for (const float strength : iteration_strengths(brush_strength)) {
        MutableSpan<bke::pbvh::GridsNode> nodes = pbvh.nodes<bke::pbvh::GridsNode>();
        node_mask.foreach_index(
            [&](const int i) {
              LocalData &tls = all_tls.local();
              calc_grids(depsgraph,
                         sd,
                         faces,
                         corner_verts,
                         ss.boundary_info_cache->verts,
                         ss.boundary_info_cache->edges,
                         object,
                         brush,
                         strength,
                         nodes[i],
                         tls);
            },
            exec_mode::grain_size(1));
      }
      break;
    }
    case bke::pbvh::Type::BMesh: {
      vert_random_access_ensure(object);
      threading::EnumerableThreadSpecific<LocalData> all_tls;
      for (const float strength : iteration_strengths(brush_strength)) {
        MutableSpan<bke::pbvh::BMeshNode> nodes = pbvh.nodes<bke::pbvh::BMeshNode>();
        node_mask.foreach_index(
            [&](const int i) {
              LocalData &tls = all_tls.local();
              calc_bmesh(depsgraph, sd, object, brush, strength, nodes[i], tls);
            },
            exec_mode::grain_size(1));
      }
      break;
    }
  }
  pbvh.tag_positions_changed(node_mask);
  pbvh.update_bounds(depsgraph, object);
}

void do_smooth_brush_multiscale(const Depsgraph &depsgraph,
                                const Sculpt &sd,
                                Object &object,
                                const IndexMask &node_mask,
                                const float brush_strength)
{
  SculptSession &ss = *object.runtime->sculpt_session;
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  const Brush &brush = *BKE_paint_brush_for_read(&sd.paint);

  boundary::ensure_boundary_info(object);

  if (ss.cache->surface_smooth_laplacian_disp.is_empty()) {
    BLI_assert_msg(stroke_is_first_brush_step(*ss.cache),
                   "Should only be allocated on the first step");
    ss.cache->surface_smooth_laplacian_disp = Array<float3>(vertex_count_get(object), float3(0));
  }

  switch (pbvh.type()) {
    case bke::pbvh::Type::Mesh:
      do_smooth_brush_preserve_form_mesh(
          depsgraph, sd, brush, object, node_mask, brush_strength);
      break;
    case bke::pbvh::Type::Grids: {
      const Mesh &base_mesh = *id_cast<const Mesh *>(object.data);
      const OffsetIndices faces = base_mesh.faces();
      const Span<int> corner_verts = base_mesh.corner_verts();

      threading::EnumerableThreadSpecific<LocalData> all_tls;
      MutableSpan<bke::pbvh::GridsNode> nodes = pbvh.nodes<bke::pbvh::GridsNode>();
      node_mask.foreach_index(
          [&](const int i) {
            LocalData &tls = all_tls.local();
            calc_grids_preserve_form(depsgraph,
                                     sd,
                                     faces,
                                     corner_verts,
                                     ss.boundary_info_cache->verts,
                                     ss.boundary_info_cache->edges,
                                     object,
                                     brush,
                                     brush_strength,
                                     nodes[i],
                                     ss.cache->surface_smooth_laplacian_disp,
                                     tls);
          },
          exec_mode::grain_size(1));
      break;
    }
    case bke::pbvh::Type::BMesh: {
      vert_random_access_ensure(object);
      threading::EnumerableThreadSpecific<LocalData> all_tls;
      MutableSpan<bke::pbvh::BMeshNode> nodes = pbvh.nodes<bke::pbvh::BMeshNode>();
      node_mask.foreach_index(
          [&](const int i) {
            LocalData &tls = all_tls.local();
            calc_bmesh_preserve_form(depsgraph,
                                     sd,
                                     object,
                                     brush,
                                     brush_strength,
                                     ss.cache->surface_smooth_laplacian_disp,
                                     nodes[i],
                                     tls);
          },
          exec_mode::grain_size(1));
      break;
    }
  }
  pbvh.tag_positions_changed(node_mask);
  pbvh.update_bounds(depsgraph, object);
}

}  // namespace blender::ed::sculpt_paint::brushes
