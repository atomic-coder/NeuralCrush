import numpy as np
import torch
import gmsh
from scipy.spatial import Voronoi
from shapely.geometry import LineString, Polygon, MultiPolygon, box
from shapely.ops import unary_union

# =========================================================
# PHYSICS & GEOMETRY CONTROLS (in mm)
# =========================================================
BOX_SIZE = 50.0
WEB_THICKNESS = 2.0
FRAME_THICKNESS = 4.0

SPHERE_RADIUS = 25.0
SPHERE_GAP = 2.0
INITIAL_VEL_Y = -0.005  # Impactor moves DOWN at 5 mm/s (in m)

MESH_SIZE = 2.5

EXTRUSION_THICKNESS = 0.05
STEEL_RHO = 7850        # kg/m^3
TPU95A_RHO = 1210       # kg/m^3

def setup_adaptive_mesh(box_size, sphere_gap, sphere_radius, frame_thickness, web_thickness):
    """
    Replace the uniform mesh size with adaptive sizing fields.
    Call this AFTER gmsh.model.occ.synchronize() and BEFORE gmsh.model.mesh.generate(2).
    """
    
    # ── 1. Identify contact zone: top of absorber + bottom of impactor ──
    # The contact will happen around y = BOX_SIZE to y = BOX_SIZE + SPHERE_GAP
    contact_y = box_size + sphere_gap / 2.0
    
    # ── 2. Size parameters (all relative to geometry) ──
    domain_diag = (box_size**2 + (box_size + sphere_gap + sphere_radius*2)**2)**0.5
    
    size_contact = web_thickness / 2.0    # ~3 elements across thinnest web near contact
    size_absorber = web_thickness / 1.0   # 2 elements across webs away from contact
    size_impactor_coarse = 40.0            # impactor is rigid — waste no resolution
    size_far = 40.0                        # far from contact, coarse is fine
    
    # ── 3. Distance field from contact region curves ──
    # Get all boundary curves near the contact zone
    contact_curves = []
    all_curves = gmsh.model.getEntities(1)
    for dim, tag in all_curves:
        bbox = gmsh.model.getBoundingBox(dim, tag)
        # Curve is near contact zone if its y-range overlaps [BOX_SIZE - 5, BOX_SIZE + GAP + 5]
        y_min, y_max = bbox[1], bbox[4]
        if y_max > (box_size - 5.0) and y_min < (box_size + sphere_gap + 5.0):
            contact_curves.append(tag)
    
    # Distance from contact curves
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", contact_curves)
    gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 200)
    
    # Threshold: fine near contact, coarse far away
    f_contact = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_contact, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_contact, "SizeMin", size_contact)
    gmsh.model.mesh.field.setNumber(f_contact, "SizeMax", size_far)
    gmsh.model.mesh.field.setNumber(f_contact, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_contact, "DistMax", 30.0)  # transition distance
    
    # ── 4. Coarsen the impactor interior ──
    # MathEval field: large size everywhere above the gap (impactor region)
    f_impactor = gmsh.model.mesh.field.add("MathEval")
    impactor_y_start = box_size + sphere_gap
    gmsh.model.mesh.field.setString(
        f_impactor, "F", 
        f"{size_contact} + ({size_impactor_coarse} - {size_contact}) * "
        f"((y - {impactor_y_start}) / {sphere_radius})"
    )
    # This linearly increases size from contact zone up into the impactor
    
    # Restrict it to impactor surfaces only
    impactor_surfaces = []
    for dim, tag in gmsh.model.getEntities(2):
        bbox = gmsh.model.getBoundingBox(dim, tag)
        center_y = (bbox[1] + bbox[4]) / 2.0
        if center_y > impactor_y_start:
            impactor_surfaces.append(tag)
    
    f_impactor_restrict = gmsh.model.mesh.field.add("Restrict")
    gmsh.model.mesh.field.setNumber(f_impactor_restrict, "InField", f_impactor)
    gmsh.model.mesh.field.setNumbers(f_impactor_restrict, "SurfacesList", impactor_surfaces)
    
    # ── 5. Thin wall refinement for absorber webs ──
    # Distance from ALL absorber boundary curves
    absorber_curves = []
    for dim, tag in all_curves:
        bbox = gmsh.model.getBoundingBox(dim, tag)
        y_max = bbox[4]
        if y_max < box_size + sphere_gap / 2.0:  # absorber region
            absorber_curves.append(tag)
    
    f_dist_walls = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist_walls, "CurvesList", absorber_curves)
    gmsh.model.mesh.field.setNumber(f_dist_walls, "Sampling", 200)
    
    f_walls = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_walls, "InField", f_dist_walls)
    gmsh.model.mesh.field.setNumber(f_walls, "SizeMin", size_absorber)
    gmsh.model.mesh.field.setNumber(f_walls, "SizeMax", size_far)
    gmsh.model.mesh.field.setNumber(f_walls, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_walls, "DistMax", web_thickness * 2)
    
    # ── 6. Combine: minimum of all fields ──
    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [
        f_contact, f_impactor_restrict, f_walls
    ])
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)
    
    # ── 7. Disable default sizing (let our fields control everything) ──
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    
    # Global safety bounds
    gmsh.option.setNumber("Mesh.MeshSizeMin", size_contact * 0.5)
    gmsh.option.setNumber("Mesh.MeshSizeMax", size_impactor_coarse)
    
    # Delaunay handles complex size fields best
    gmsh.option.setNumber("Mesh.Algorithm", 5)

def get_skin_edges(elements, mesh_vertices):
    edges_local_idx = np.array([[0, 1], [1, 2], [2, 0]])
    all_edges = elements[:, edges_local_idx].reshape(-1, 2)
    sorted_edges = np.sort(all_edges, axis=1)
    
    void_dt = np.dtype((np.void, sorted_edges.dtype.itemsize * sorted_edges.shape[1]))
    edges_void = np.ascontiguousarray(sorted_edges).view(void_dt).reshape(-1)
    
    _, indices, counts = np.unique(edges_void, return_index=True, return_counts=True)
    skin_indices = indices[counts == 1]
    skin_edges = all_edges[skin_indices]
    
    parent_tri_indices = skin_indices // 3
    parent_tris = elements[parent_tri_indices]
    
    edge_coords = mesh_vertices[skin_edges]
    elem_coords = mesh_vertices[parent_tris]
    
    edge_centers = np.mean(edge_coords, axis=1)
    elem_centers = np.mean(elem_coords, axis=1)
    
    outward_vec = edge_centers - elem_centers
    v1 = edge_coords[:, 1] - edge_coords[:, 0]
    
    normals = np.column_stack([-v1[:, 1], v1[:, 0]])
    dot_prod = np.sum(normals * outward_vec, axis=1)
    flip_mask = dot_prod < 0
    
    if np.any(flip_mask):
        skin_edges[flip_mask, 0], skin_edges[flip_mask, 1] = \
        skin_edges[flip_mask, 1].copy(), skin_edges[flip_mask, 0].copy()
        
    return skin_edges

def build_edge_index_from_mesh(tri_elements, num_nodes):
    edges = set()
    tri_edges = [(0, 1), (0, 2), (1, 2)]
    for tri in tri_elements:
        for i, j in tri_edges:
            edge = tuple(sorted([tri[i], tri[j]]))
            edges.add(edge)
            edges.add((edge[1], edge[0]))
    return np.array(list(edges), dtype=np.int64).T

# --- ADDED: CCW WINDING ENFORCEMENT ---
def enforce_consistent_winding(mesh_vertices, elements):
    """Ensures all triangles have Counter-Clockwise (CCW) winding."""
    pts = mesh_vertices[elements]
    p1, p2, p3 = pts[:, 0, :], pts[:, 1, :], pts[:, 2, :]
    
    # Cross product
    cross_prod = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
                 (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
                 
    # Find elements with negative signed areas (Clockwise)
    cw_mask = cross_prod < 0
    
    if np.any(cw_mask):
        # Swap node index 1 and 2 to reverse the winding to CCW
        elements[cw_mask, 1], elements[cw_mask, 2] = \
            elements[cw_mask, 2].copy(), elements[cw_mask, 1].copy()
            
    return elements
# --------------------------------------

def generate_rl_environment(seeds):
    try:
        dummies = np.array([[-500, -500], [BOX_SIZE+500, -500], 
                            [BOX_SIZE+500, BOX_SIZE+500], [-500, BOX_SIZE+500]])
        all_pts = np.vstack([seeds, dummies])
        vor = Voronoi(all_pts)
        
        lines = []
        for ridge in vor.ridge_vertices:
            if -1 not in ridge:
                p1, p2 = vor.vertices[ridge[0]], vor.vertices[ridge[1]]
                lines.append(LineString([p1, p2]))

        merged_lines = unary_union(lines)
        voronoi_web = merged_lines.buffer(WEB_THICKNESS / 2.0, cap_style=2, join_style=2)
        
        target_box = box(0, 0, BOX_SIZE, BOX_SIZE)
        outer_frame = target_box.exterior.buffer(FRAME_THICKNESS / 2.0, cap_style=2, join_style=2)
        
        final_2d_shape = unary_union([voronoi_web, outer_frame]).intersection(target_box)
        final_2d_shape = final_2d_shape.simplify(0.05, preserve_topology=False)

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0) 
        gmsh.model.add("RL_Batch")
        
        polys = final_2d_shape.geoms if isinstance(final_2d_shape, MultiPolygon) else [final_2d_shape]
        
        for poly in polys:
            def draw_ring(ring):
                coords = list(ring.coords)[:-1] 
                p_tags = [gmsh.model.occ.addPoint(x, y, 0) for x, y in coords]
                l_tags = [gmsh.model.occ.addLine(p_tags[i], p_tags[(i+1)%len(p_tags)]) for i in range(len(p_tags))]
                return gmsh.model.occ.addCurveLoop(l_tags)

            ext_loop = draw_ring(poly.exterior)
            int_loops = [draw_ring(hole) for hole in poly.interiors]
            gmsh.model.occ.addPlaneSurface([ext_loop] + int_loops)

        impactor_center_y = BOX_SIZE + SPHERE_GAP + SPHERE_RADIUS
        gmsh.model.occ.addDisk(BOX_SIZE/2.0, impactor_center_y, 0, SPHERE_RADIUS, SPHERE_RADIUS)

        gmsh.model.occ.synchronize()
        gmsh.model.occ.fragment(gmsh.model.getEntities(2), [])
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", MESH_SIZE * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", MESH_SIZE)
        gmsh.model.mesh.generate(2)

        nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes(-1, -1)
        raw_pos = np.array(nodeCoords).reshape(-1, 3)[:, :2] 
        
        tag2idx = {tag: i for i, tag in enumerate(nodeTags)}
        num_nodes = len(raw_pos)

        tri_elements = []
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(2, -1)
        for i, etype in enumerate(elemTypes):
            if etype == 2:
                nodes = elemNodeTags[i].reshape(-1, 3)
                for n_tri in nodes:
                    tri_elements.append([tag2idx[n_tri[0]], tag2idx[n_tri[1]], tag2idx[n_tri[2]]])
        
        mapped_elements = np.array(tri_elements, dtype=np.int64)

        # --- ADDED: ENFORCE WINDING DIRECTION ---
        mapped_elements = enforce_consistent_winding(raw_pos, mapped_elements)
        # ----------------------------------------

        is_steel = raw_pos[:, 1] > (BOX_SIZE + (SPHERE_GAP / 2.0))
        is_constraint = raw_pos[:, 1] < (FRAME_THICKNESS / 1.0)
        num_impactors = np.sum(is_steel)

        edge_index = torch.tensor(build_edge_index_from_mesh(mapped_elements, num_nodes), dtype=torch.long)
        face_index = torch.tensor(get_skin_edges(mapped_elements, raw_pos).T, dtype=torch.long)

        node_areas = np.zeros(num_nodes, dtype=np.float64)
        pos_meters = raw_pos * 0.001
        pts = pos_meters[mapped_elements]
        p1, p2, p3 = pts[:, 0, :], pts[:, 1, :], pts[:, 2, :]
        areas = 0.5 * np.abs((p2[:, 0]-p1[:, 0])*(p3[:, 1]-p1[:, 1]) - (p3[:, 0]-p1[:, 0])*(p2[:, 1]-p1[:, 1]))
        third_areas = areas / 3.0
        np.add.at(node_areas, mapped_elements[:, 0], third_areas)
        np.add.at(node_areas, mapped_elements[:, 1], third_areas)
        np.add.at(node_areas, mapped_elements[:, 2], third_areas)
        
        masses = np.zeros(num_nodes, dtype=np.float64)
        masses[is_steel] = node_areas[is_steel] * STEEL_RHO * EXTRUSION_THICKNESS
        masses[~is_steel] = node_areas[~is_steel] * TPU95A_RHO * EXTRUSION_THICKNESS
        
        inv_mass = 1.0 / (masses + 1e-8)

        final_pos = raw_pos * 0.001
        pos_tensor = torch.tensor(final_pos, dtype=torch.float32)
        
        vel_curr = np.zeros((num_nodes, 2), dtype=np.float32)
        vel_curr[is_steel, 1] = INITIAL_VEL_Y
        
        data_dict = {
            'pos': pos_tensor,
            'mesh_pos': pos_tensor.clone(),
            'inv_mass': torch.tensor(inv_mass, dtype=torch.float32),
            'prev_velocity': torch.tensor(vel_curr, dtype=torch.float32), 
            'velocity': torch.tensor(vel_curr, dtype=torch.float32),
            'target_accel': torch.zeros((num_nodes, 2), dtype=torch.float32),
            'target_accel_next': torch.zeros((num_nodes, 2), dtype=torch.float32),
            'node_type': torch.tensor(is_steel.astype(np.int64), dtype=torch.long),
            'is_constraint': torch.tensor(is_constraint, dtype=torch.bool),
            'num_impactors': torch.tensor([num_impactors], dtype=torch.long),
            'face_index': face_index,
            'edge_index': edge_index,
            'world_edge_index': torch.empty((2, 0), dtype=torch.long),
            'elements': torch.tensor(mapped_elements, dtype=torch.long)
        }
        
        return data_dict

    except Exception as e:
        print(f"Meshing failed: {e}")
        return None
        
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()