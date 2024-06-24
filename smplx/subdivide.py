import trimesh
import torch
import numpy as np 
import json
import pickle as pkl 

from .common import safe_normalize, dot

class SMPLXSeg:
    remesh_mask = remesh_mask[smplx_face].all(axis=1)
 
def subdivide_inorder(vertices, faces, unique):
    triangles = vertices[faces]
    mid = torch.vstack([triangles[:, g, :].mean(1) for g in [[0, 1], [1, 2], [2, 0]]])

    mid = mid[unique]
    new_vertices = torch.vstack((vertices, mid))
    return new_vertices



def subdivide(vertices, faces, attributes=None, face_index=None):
    """
    Subdivide a mesh into smaller triangles.

    Note that if `face_index` is passed, only those faces will
    be subdivided and their neighbors won't be modified making
    the mesh no longer "watertight."

    Parameters
    ----------
    vertices : (n, 3) float
      Vertices in space
    faces : (n, 3) int
      Indexes of vertices which make up triangular faces
    attributes: (n, d) float
      vertices attributes
    face_index : faces to subdivide.
      if None: all faces of mesh will be subdivided
      if (n,) int array of indices: only specified faces

    Returns
    ----------
    new_vertices : (n, 3) float
      Vertices in space
    new_faces : (n, 3) int
      Remeshed faces
    """
    if face_index is None:
        face_index = np.arange(len(faces))
    else:
        face_index = np.asanyarray(face_index)

    # the (c,3) int set of vertex indices
    faces = faces[face_index]
    # the (c, 3, 3) float set of points in the triangles
    triangles = vertices[faces]
    # the 3 midpoints of each triangle edge
    # stacked to a (3 * c, 3) float
    mid = np.vstack([triangles[:, g, :].mean(axis=1) for g in [[0, 1], [1, 2], [2, 0]]])

    # for adjacent faces we are going to be generating
    # the same midpoint twice so merge them here
    mid_idx = (np.arange(len(face_index) * 3)).reshape((3, -1)).T
    unique, inverse = trimesh.grouping.unique_rows(mid)
    mid = mid[unique]
    mid_idx = inverse[mid_idx] + len(vertices)

    # the new faces with correct winding
    f = np.column_stack([faces[:, 0],
                         mid_idx[:, 0],
                         mid_idx[:, 2],
                         mid_idx[:, 0],
                         faces[:, 1],
                         mid_idx[:, 1],
                         mid_idx[:, 2],
                         mid_idx[:, 1],
                         faces[:, 2],
                         mid_idx[:, 0],
                         mid_idx[:, 1],
                         mid_idx[:, 2]]).reshape((-1, 3))
    # add the 3 new faces per old face
    new_faces = np.vstack((faces, f[len(face_index):]))
    # replace the old face with a smaller face
    new_faces[face_index] = f[:len(face_index)]

    new_vertices = np.vstack((vertices, mid))

    if attributes is not None:
        tri_att = attributes[faces]
        mid_att = np.vstack([tri_att[:, g, :].mean(axis=1) for g in [[0, 1], [1, 2], [2, 0]]])
        mid_att = mid_att[unique]
        new_attributes = np.vstack((attributes, mid_att))
        return new_vertices, new_faces, new_attributes, unique

    return new_vertices, new_faces, unique

    e1 = np.linalg.norm(cano_v[faces[:, 0]] - cano_v[faces[:, 1]], axis=1, keepdims=True)
    e2 = np.linalg.norm(cano_v[faces[:, 1]] - cano_v[faces[:, 2]], axis=1, keepdims=True)
    e3 = np.linalg.norm(cano_v[faces[:, 2]] - cano_v[faces[:, 0]], axis=1, keepdims=True)
    e = np.concatenate([e1,e2,e3], 1)  
  
    E1 = np.linalg.norm(pose_v[faces[:, 0]] - pose_v[faces[:, 1]], axis=1, keepdims=True)
    E2 = np.linalg.norm(pose_v[faces[:, 1]] - pose_v[faces[:, 2]], axis=1, keepdims=True)
    E3 = np.linalg.norm(pose_v[faces[:, 2]] - pose_v[faces[:, 0]], axis=1, keepdims=True)
    E = np.concatenate([E1,E2,E3], 1)  
  
    thresh = E.max() * 0.8 + E.min() * 0.2

    tri_mask = ((E.max(1) > thresh) | (e.max(1) > thresh) | (e.max(1) > thresh) | (e.max(1) > thresh))
    tri_mask = np.logical_not(tri_mask)
 
    return tri_mask