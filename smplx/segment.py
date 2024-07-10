import json
import pickle as pkl
import numpy as np
import torch
from matplotlib import cm as mpl_cm, colors as mpl_colors


def segm_to_vertex_colors(part_segm, n_vertices, alpha=1.0):
    vertex_labels = np.zeros(n_vertices)

    for part_idx, (k, v) in enumerate(part_segm.items()):
        vertex_labels[v] = part_idx

    cm = mpl_cm.get_cmap('jet')
    norm_gt = mpl_colors.Normalize()

    vertex_colors = np.ones((n_vertices, 4))
    vertex_colors[:, 3] = alpha
    vertex_colors[:, :3] = cm(norm_gt(vertex_labels))[:, :3]

    return vertex_colors


def index_triangles_from_vertex_mask(vertex_mask, triangles):
    """
    return the related triangles of the vertices 
    args:
        vertex_mask: (N,) a boolean mask of vertices, True for selected vertices
        triangles: (F, 3) np.ndarray, mesh faces
    return: 
        triangles: [M, 3] np.ndarray   
    """   
    tri_ids = []
    for i, f in enumerate(triangles): 
        for vid in f:
            if vertex_mask[vid]: 
                tri_ids.append(i)
                break 
    return triangles[tri_ids]  

class FlameSeg:
    def __init__(self, flame_dir, device):
        # flame: ['eye_region', 'neck', 'left_eyeball', 'right_eyeball', 'right_ear', 'right_eye_region', 'forehead', 'lips', 'nose', 'scalp', 'boundary', 'face', 'left_ear', 'left_eye_region'])
        self.segms = pkl.load(open(f"{flame_dir}/FLAME_masks.pkl", "rb"), encoding='latin1')
        self.device = device
        self.N = 5023
        self._vc = None

    def get_triangles(self, part_name):
        v_mask = np.zeros((self.N, 1))
        v_mask[self.segms[part_name]] = 1
        triangles = index_triangles_from_vertex_mask(v_mask, self.flame_faces)
        return torch.tensor(triangles, dtype=torch.long, device=self.device)

    def get_vertices(self, vertices, part_name):
        return vertices[self.segms[part_name]]

    @property 
    def vc(self):
        if self._vc is None:
            self._vc = segm_to_vertex_colors(self.segms, self.N)[:, :3]
            self._vc = torch.tensor(self._vc, dtype=torch.float, device=self.device)
        return self._vc 

class SmplxSeg:
    def __init__(self, smplx_dir, device):
        #  'leftHand', 'rightHand', 
        #  'rightUpLeg', 'leftUpLeg'
        #  'leftArm', 'rightArm'
        #  'head',  'neck' 
        #  'leftEye', 'rightEye', 'eyeballs'
        #  'leftLeg', 'rightLeg'
        #  'leftToeBase', 'rightToeBase'
        #  'leftFoot', 'rightFoot',
        #  'spine', 'spine1', 'spine2', 
        #  'leftShoulder', 'rightShoulder',    
        #  'leftHandIndex1',  , 'rightHandIndex1', 
        #  'leftForeArm', 'rightForeArm', 
        #  'hips' 
        self.smplx_segs = json.load(open(f"{smplx_dir}/smplx_vert_segementation.json", 'r'))
        self.flame_segs = pkl.load(open(f"{smplx_dir}/FLAME_masks.pkl", "rb"), encoding='latin1')
        self.flame_to_smplx_vid = np.load(f"{smplx_dir}/FLAME_SMPLX_vertex_ids.npy", allow_pickle=True)
        self.smplx_faces = np.load(f"{smplx_dir}/smplx_faces.npy")   
        self.device = device  
        self.N = 10475
        self._vc = None
    
    def mapping_smplx_to_flame(self, flame_vid):
        for key in ['right_ear', 'left_ear', 'nose', 'lips']:
            if key in flame_vid:
                return list(self.smplx_flame_vid[self.flame_segs["left_ear"]])

    def get_triangles(self, part_name):
        v_mask = np.zeros((self.N, 1))   
        v_mask[self.smplx_segs[part_name]] = 1 
        triangles = index_triangles_from_vertex_mask(v_mask, self.smplx_faces) 
        return torch.tensor(triangles, dtype=torch.long, device=self.device)

    def get_vertex_ids(self, part_name):
        return self.smplx_segs[part_name]

    def init_part_triangls(self):
        for part_name in self.smplx_segs.keys(): 
            setattr(self, part_name + '_tri', self.get_triangles(part_name))
    
    @property 
    def vc(self):
        if self._vc is None:
            self._vc = segm_to_vertex_colors(self.smplx_segs, self.N)[:, :3]
            self._vc = torch.tensor(self._vc, dtype=torch.float, device=self.device)
        return self._vc 
  
       