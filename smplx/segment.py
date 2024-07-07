import json
import numpy as np
import torch
from matplotlib import cm as mpl_cm, colors as mpl_colors


def smplx_segm_to_vertex_colors(part_segm, n_vertices, alpha=1.0):
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


class SMPLXSeg:
    def __init__(self, vseg_path, smplx_faces, device):
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
        self.smplx_segs = json.load(open(vseg_path, 'r'))
        self.smplx_faces = smplx_faces  
        self.device = device  
        self.N = 10475
        self._vc = None
        
    def get_triangles(self, part_name):
        v_mask = np.zeros((self.N, 1))   
        v_mask[self.smplx_segs[part_name]] = 1 
        triangles = index_triangles_from_vertex_mask(v_mask, self.smplx_faces) 
        return torch.tensor(triangles, dtype=torch.long, device=self.device)

    def init_part_triangls(self):
        for part_name in self.smplx_segs.keys(): 
            setattr(self, part_name + '_tri', self.get_triangles(part_name))
    
    @property 
    def vc(self):
        if self._vc is None:
            self._vc = smplx_segm_to_vertex_colors(self.smplx_segs, self.N)[:, :3]
            self._vc = torch.tensor(self._vc, dtype=torch.float, device=self.device)
        return self._vc 
  
       