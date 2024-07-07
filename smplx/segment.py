import json
import numpy as np
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

def index_triangles_from_vertices(v_ids, triangles):
    """
    return the related triangles of the vertices 
    args:
        v_ids: (N,) a list of vertex ids
        faces: (F, 3) np.ndarray, mesh faces
    return:
        # triangle_mask: (F, 3) np.ndarray, 0 or 1
        triangle_ids: a list of triangle ids
    """ 
    tri_ids = []
    for i, f in enumerate(faces): 
        for vid in f:
            if vertex_mask[vid]: 
                tri_ids.append(i)
                break 
    return triangles[tri_ids]  


class SMPLXSegs:
    def __init__(self, vseg_path, smplx_faces, device):
        #  ['rightHand', 'rightUpLeg', 'leftArm', 'head', 'leftEye', 'rightEye', 'leftLeg', 'leftToeBase', 'leftFoot', 'spine1', 'spine2', 'leftShoulder', 'rightShoulder', 'rightFoot', 'rightArm', 'leftHandIndex1', 'rightLeg', 'rightHandIndex1', 'leftForeArm', 'rightForeArm', 'neck', 'rightToeBase', 'spine', 'leftUpLeg', 'eyeballs', 'leftHand', 'hips']
        self.smplx_segs = json.load(vseg_path)
        self.smplx_faces = smplx_faces  
        self.device = device  
        self.N = 10475

        self.init_part_triangls()

        # todo add flame segs
    
    def init_part_triangls(self):
        for part_name, v_ids in self.smplx_segs.items():
            setattr(self, 'tri_'+part_name, index_triangles_from_vertices(v_ids, self.smplx_faces))
         
    def get_vertex_color():
        smplx_vc = smplx_segm_to_vertex_colors(self.smplx_segs, self.N) 

       