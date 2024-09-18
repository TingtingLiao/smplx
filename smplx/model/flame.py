from typing import Optional, Dict, Union
import os
import os.path as osp 
import pickle 
import numpy as np 
import torch
import torch.nn as nn

from ..lbs import (
    lbs, vertices2landmarks, find_dynamic_lmk_idx_and_bcoords, blend_shapes, pose_blend_shapes)
from ..vertex_ids import vertex_ids as VERTEX_IDS
from ..utils import (
    Struct, to_np, to_tensor, Tensor, Array,
    FLAMEOutput, 
    find_joint_kin_chain)
from ..vertex_joint_selector import VertexJointSelector
from .smpl import SMPL 
from ..subdivide import subdivide, subdivide_inorder
from ..segment import FlameSeg 

class FLAME(SMPL):
    NUM_JOINTS = 5
    SHAPE_SPACE_DIM = 300
    EXPRESSION_SPACE_DIM = 100
    NECK_IDX = 0

    def __init__(
        self,
        model_path: str,
        data_struct=None,  
        create_expression: bool = True,
        expression: Optional[Tensor] = None,
        create_v_offsets: bool = False,
        v_offsets: Optional[Tensor] = None,        
        create_neck_pose: bool = True,
        neck_pose: Optional[Tensor] = None,
        create_jaw_pose: bool = True,
        jaw_pose: Optional[Tensor] = None,
        create_leye_pose: bool = True,
        leye_pose: Optional[Tensor] = None,
        create_reye_pose=True,
        reye_pose: Optional[Tensor] = None,
        use_face_contour=True,
        batch_size: int = 1,
        gender: str = 'neutral',
        dtype: torch.dtype = torch.float32,
        ext='pkl',
        upsample: bool = False, 
        create_segms: bool = False,
        add_teeth: bool = False, 
        **kwargs
    ) -> None:
        ''' FLAME model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored 
            create_expression: bool, optional
                Flag for creating a member variable for the expression space
                (default = True).
            expression: torch.tensor, optional, Bx100 
                The default value for the expression member variable.
                (default = None)
            create_v_offsets: bool, optional
                Flag for creating a member variable for the vertex offsets
                (default = True)
            v_offsets: torch.tensor, optional, BxNx3
                The default value for the vertex offsets member variable.
                (default = None)                
            create_neck_pose: bool, optional
                Flag for creating a member variable for the neck pose.
                (default = False)
            neck_pose: torch.tensor, optional, Bx3
                The default value for the neck pose variable.
                (default = None)
            create_jaw_pose: bool, optional
                Flag for creating a member variable for the jaw pose.
                (default = False)
            jaw_pose: torch.tensor, optional, Bx3
                The default value for the jaw pose variable.
                (default = None)
            create_leye_pose: bool, optional
                Flag for creating a member variable for the left eye pose.
                (default = False)
            leye_pose: torch.tensor, optional, Bx10
                The default value for the left eye pose variable.
                (default = None)
            create_reye_pose: bool, optional
                Flag for creating a member variable for the right eye pose.
                (default = False)
            reye_pose: torch.tensor, optional, Bx10
                The default value for the right eye pose variable.
                (default = None)
            use_face_contour: bool, optional
                Whether to compute the keypoints that form the facial contour
            batch_size: int, optional
                The batch size used for creating the member variables
            gender: str, optional
                Which gender to load
            dtype: torch.dtype
                The data type for the created variables
        '''
        self.upsample = upsample
     
        if osp.isdir(model_path): 
            model_fn = f'FLAME_{gender.upper()}.{ext}'
            flame_path = os.path.join(model_path, model_fn)
        else: 
            flame_path = model_path
            model_path = os.path.dirname(model_path)

        assert osp.exists(flame_path), 'Path {} does not exist!'.format(flame_path)
        if ext == 'npz':
            file_data = np.load(flame_path, allow_pickle=True)
        elif ext == 'pkl':
            with open(flame_path, 'rb') as smpl_file:
                file_data = pickle.load(smpl_file, encoding='latin1')
        else:
            raise ValueError('Unknown extension: {}'.format(ext))
        data_struct = Struct(**file_data)

        super(FLAME, self).__init__( 
            data_struct=data_struct,
            dtype=dtype,
            batch_size=batch_size,
            gender=gender,
            ext=ext,
            num_betas=self.SHAPE_SPACE_DIM, 
            **kwargs)

        self.use_face_contour = use_face_contour

        self.vertex_joint_selector.extra_joints_idxs = to_tensor([], dtype=torch.long)

        if create_neck_pose:
            if neck_pose is None:
                default_neck_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_neck_pose = torch.tensor(neck_pose, dtype=dtype)
            neck_pose_param = nn.Parameter(default_neck_pose, requires_grad=True)
            self.register_parameter('neck_pose', neck_pose_param)

        if create_jaw_pose:
            if jaw_pose is None:
                default_jaw_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_jaw_pose = torch.tensor(jaw_pose, dtype=dtype)
            jaw_pose_param = nn.Parameter(default_jaw_pose, requires_grad=True)
            self.register_parameter('jaw_pose', jaw_pose_param)

        if create_leye_pose:
            if leye_pose is None:
                default_leye_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_leye_pose = torch.tensor(leye_pose, dtype=dtype)
            leye_pose_param = nn.Parameter(default_leye_pose, requires_grad=True)
            self.register_parameter('leye_pose', leye_pose_param)

        if create_reye_pose:
            if reye_pose is None:
                default_reye_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_reye_pose = torch.tensor(reye_pose, dtype=dtype)
            reye_pose_param = nn.Parameter(default_reye_pose, requires_grad=True)
            self.register_parameter('reye_pose', reye_pose_param)

        self.register_buffer("shapedirs", to_tensor(to_np(data_struct.shapedirs), dtype=self.dtype))

        if create_expression:
            if expression is None:
                default_expression = torch.zeros(
                    [batch_size, self.EXPRESSION_SPACE_DIM], dtype=dtype)
            else:
                default_expression = torch.tensor(expression, dtype=dtype)
            expression_param = nn.Parameter(default_expression, requires_grad=True)
            self.register_parameter('expression', expression_param)

        landmark_bcoord_filename = osp.join(model_path, 'landmark_embedding_with_eyes.npy')  
        landmarks_data = np.load(landmark_bcoord_filename, allow_pickle=True, encoding='latin1')[()] 
        self.register_buffer('lmk_faces_idx', torch.tensor(landmarks_data['static_lmk_faces_idx'], dtype=torch.long)) 
        self.register_buffer('lmk_bary_coords', torch.tensor(landmarks_data['static_lmk_bary_coords'], dtype=dtype))    
        self.register_buffer('dynamic_lmk_faces_idx', landmarks_data['dynamic_lmk_faces_idx'].long())
        self.register_buffer('dynamic_lmk_bary_coords', landmarks_data['dynamic_lmk_bary_coords'].to(dtype)) 
        self.register_buffer(
            "full_lmk_faces_idx", torch.tensor(landmarks_data["full_lmk_faces_idx"], dtype=torch.long),
        )
        self.register_buffer(
            "full_lmk_bary_coords", torch.tensor(landmarks_data["full_lmk_bary_coords"], dtype=self.dtype),
        )
        
        self.register_buffer('neck_kin_chain', torch.tensor(find_joint_kin_chain(1, self.parents), dtype=torch.long))
        
        if upsample:
            self.upsampling()
        
        self.N = self.v_template.shape[0]
        
        if create_v_offsets:
            if v_offsets is None:
                default_v_offsets = torch.zeros(
                    [batch_size, self.N, 3], dtype=dtype)
            else:
                default_v_offsets = torch.tensor(v_offsets, dtype=dtype)
            v_offsets_param = nn.Parameter(default_v_offsets, requires_grad=True)
            self.register_parameter('v_offsets', v_offsets_param)

        if create_segms: 
            self.segment = FlameSeg(model_path, N=self.N, faces=self.faces)

        if add_teeth:
            print('debugiing add teeth')
            self.add_teeth()
        
        # laplacian  
        from pytorch3d.structures.meshes import Meshes
        
        laplacian_matrix = Meshes(verts=[self.v_template], faces=[self.faces_tensor]).laplacian_packed().to_dense()
        self.register_buffer("laplacian_matrix", laplacian_matrix)
        
    def add_teeth(self):
        # self.teeth = teeth
        vid_lip_outside_ring_upper = self.segment.get_vertex_ids(['lip_outside_ring_upper']) 
        vid_lip_outside_ring_lower = self.segment.get_vertex_ids(['lip_outside_ring_lower'])

        v_lip_upper = self.v_template[vid_lip_outside_ring_upper]
        v_lip_lower = self.v_template[vid_lip_outside_ring_lower]

        # construct vertices for teeth
        mean_dist = (v_lip_upper - v_lip_lower).norm(dim=-1, keepdim=True).mean()
        v_teeth_middle = (v_lip_upper + v_lip_lower) / 2
        v_teeth_middle[:, 1] = v_teeth_middle[:, [1]].mean(dim=0, keepdim=True)
        # v_teeth_middle[:, 2] -= mean_dist * 2.5  # how far the teeth are from the lips
        # v_teeth_middle[:, 2] -= mean_dist * 2  # how far the teeth are from the lips
        v_teeth_middle[:, 2] -= mean_dist * 1.5  # how far the teeth are from the lips

        # upper, front
        v_teeth_upper_edge = v_teeth_middle.clone() + torch.tensor([[0, mean_dist, 0]])*0.1
        v_teeth_upper_root = v_teeth_upper_edge + torch.tensor([[0, mean_dist, 0]]) * 2  # scale the height of teeth

        # lower, front
        v_teeth_lower_edge = v_teeth_middle.clone() - torch.tensor([[0, mean_dist, 0]])*0.1
        # v_teeth_lower_edge -= torch.tensor([[0, 0, mean_dist]]) * 0.2  # slightly move the lower teeth to the back
        v_teeth_lower_edge -= torch.tensor([[0, 0, mean_dist]]) * 0.4  # slightly move the lower teeth to the back
        v_teeth_lower_root = v_teeth_lower_edge - torch.tensor([[0, mean_dist, 0]]) * 2  # scale the height of teeth

        # thickness = mean_dist * 0.5
        thickness = mean_dist * 1.
        # upper, back
        v_teeth_upper_root_back = v_teeth_upper_root.clone()
        v_teeth_upper_edge_back = v_teeth_upper_edge.clone()
        v_teeth_upper_root_back[:, 2] -= thickness  # how thick the teeth are
        v_teeth_upper_edge_back[:, 2] -= thickness  # how thick the teeth are

        # lower, back
        v_teeth_lower_root_back = v_teeth_lower_root.clone()
        v_teeth_lower_edge_back = v_teeth_lower_edge.clone()
        v_teeth_lower_root_back[:, 2] -= thickness  # how thick the teeth are
        v_teeth_lower_edge_back[:, 2] -= thickness  # how thick the teeth are
        
        # concatenate to v_template
        num_verts_orig = self.v_template.shape[0]
        v_teeth = torch.cat([
            v_teeth_upper_root,  # num_verts_orig + 0-14 
            v_teeth_lower_root,  # num_verts_orig + 15-29
            v_teeth_upper_edge,  # num_verts_orig + 30-44
            v_teeth_lower_edge,  # num_verts_orig + 45-59
            v_teeth_upper_root_back,  # num_verts_orig + 60-74
            v_teeth_upper_edge_back,  # num_verts_orig + 75-89
            v_teeth_lower_root_back,  # num_verts_orig + 90-104
            v_teeth_lower_edge_back,  # num_verts_orig + 105-119
        ], dim=0)
        num_verts_teeth = v_teeth.shape[0]
        self.v_template = torch.cat([self.v_template, v_teeth], dim=0)

        vid_teeth_upper_root = torch.arange(0, 15) + num_verts_orig
        vid_teeth_lower_root = torch.arange(15, 30) + num_verts_orig
        vid_teeth_upper_edge = torch.arange(30, 45) + num_verts_orig
        vid_teeth_lower_edge = torch.arange(45, 60) + num_verts_orig
        vid_teeth_upper_root_back = torch.arange(60, 75) + num_verts_orig
        vid_teeth_upper_edge_back = torch.arange(75, 90) + num_verts_orig
        vid_teeth_lower_root_back = torch.arange(90, 105) + num_verts_orig
        vid_teeth_lower_edge_back = torch.arange(105, 120) + num_verts_orig
        
        vid_teeth_upper = torch.cat([vid_teeth_upper_root, vid_teeth_upper_edge, vid_teeth_upper_root_back, vid_teeth_upper_edge_back], dim=0)
        vid_teeth_lower = torch.cat([vid_teeth_lower_root, vid_teeth_lower_edge, vid_teeth_lower_root_back, vid_teeth_lower_edge_back], dim=0)
        vid_teeth = torch.cat([vid_teeth_upper, vid_teeth_lower], dim=0)
      
        # update vertex masks
        # self.segment.v.register_buffer("teeth_upper", vid_teeth_upper)
        # self.segment.v.register_buffer("teeth_lower", vid_teeth_lower)
        # self.segment.v.register_buffer("teeth", vid_teeth)
        # self.segment.v.left_half = torch.cat([
        #     self.segment.v.left_half, 
        #     torch.tensor([
        #         5023, 5024, 5025, 5026, 5027, 5028, 5029, 5030, 5038, 5039, 5040, 5041, 5042, 5043, 5044, 5045, 5053, 5054, 5055, 5056, 5057, 5058, 5059, 5060, 5068, 5069, 5070, 5071, 5072, 5073, 5074, 5075, 5083, 5084, 5085, 5086, 5087, 5088, 5089, 5090, 5098, 5099, 5100, 5101, 5102, 5103, 5104, 5105, 5113, 5114, 5115, 5116, 5117, 5118, 5119, 5120, 5128, 5129, 5130, 5131, 5132, 5133, 5134, 5135, 
        #     ])], dim=0)

        # self.segment.v.right_half = torch.cat([
        #     self.segment.v.right_half, 
        #     torch.tensor([
        #         5030, 5031, 5032, 5033, 5034, 5035, 5036, 5037, 5045, 5046, 5047, 5048, 5049, 5050, 5051, 5052, 5060, 5061, 5062, 5063, 5064, 5065, 5066, 5067, 5075, 5076, 5077, 5078, 5079, 5080, 5081, 5082, 5090, 5091, 5092, 5093, 5094, 5095, 5097, 5105, 5106, 5107, 5108, 5109, 5110, 5111, 5112, 5120, 5121, 5122, 5123, 5124, 5125, 5126, 5127, 5135, 5136, 5137, 5138, 5139, 5140, 5141, 5142, 
        #     ])], dim=0)

        # # construct uv vertices for teeth
        # u = torch.linspace(0.62, 0.38, 15)
        # v = torch.linspace(1-0.0083, 1-0.0425, 7) 
        # v = v[[3, 2, 0, 1, 3, 4, 6, 5]]  # TODO: with this order, teeth_lower is not rendered correctly in the uv space
        # uv = torch.stack(torch.meshgrid(u, v, indexing='ij'), dim=-1).permute(1, 0, 2).reshape(num_verts_teeth, 2)  # (#num_teeth, 2)
        # num_verts_uv_orig = self.verts_uvs.shape[0]
        # num_verts_uv_teeth = uv.shape[0]
        # self.verts_uvs = torch.cat([self.verts_uvs, uv], dim=0)

        # shapedirs copy from lips
        self.shapedirs = torch.cat([self.shapedirs, torch.zeros_like(self.shapedirs[:num_verts_teeth])], dim=0)
        shape_dirs_mean = (self.shapedirs[vid_lip_outside_ring_upper, :, :self.SHAPE_SPACE_DIM] + self.shapedirs[vid_lip_outside_ring_lower, :, :self.SHAPE_SPACE_DIM]) / 2
        self.shapedirs[vid_teeth_upper_root, :, :self.SHAPE_SPACE_DIM] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_root, :, :self.SHAPE_SPACE_DIM] = shape_dirs_mean
        self.shapedirs[vid_teeth_upper_edge, :, :self.SHAPE_SPACE_DIM] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_edge, :, :self.SHAPE_SPACE_DIM] = shape_dirs_mean
        self.shapedirs[vid_teeth_upper_root_back, :, :self.SHAPE_SPACE_DIM] = shape_dirs_mean
        self.shapedirs[vid_teeth_upper_edge_back, :, :self.SHAPE_SPACE_DIM] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_root_back, :, :self.SHAPE_SPACE_DIM] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_edge_back, :, :self.SHAPE_SPACE_DIM] = shape_dirs_mean

        # posedirs set to zero
        posedirs = self.posedirs.reshape(len(self.parents)-1, 9, num_verts_orig, 3)  # (J*9, V*3) -> (J, 9, V, 3)
        posedirs = torch.cat([posedirs, torch.zeros_like(posedirs[:, :, :num_verts_teeth])], dim=2)  # (J, 9, V+num_verts_teeth, 3)
        self.posedirs = posedirs.reshape((len(self.parents)-1)*9, (num_verts_orig+num_verts_teeth)*3)  # (J*9, (V+num_verts_teeth)*3)

        # J_regressor set to zero
        self.J_regressor = torch.cat([self.J_regressor, torch.zeros_like(self.J_regressor[:, :num_verts_teeth])], dim=1)  # (5, J) -> (5, J+num_verts_teeth)

        # lbs_weights manually set
        self.lbs_weights = torch.cat([self.lbs_weights, torch.zeros_like(self.lbs_weights[:num_verts_teeth])], dim=0)  # (V, 5) -> (V+num_verts_teeth, 5)
        self.lbs_weights[vid_teeth_upper, 1] += 1  # move with neck
        self.lbs_weights[vid_teeth_lower, 2] += 1  # move with jaw

        # add faces for teeth
        f_teeth_upper = torch.tensor([
            [0, 31, 30],  #0
            [0, 1, 31],  #1
            [1, 32, 31],  #2
            [1, 2, 32],  #3
            [2, 33, 32],  #4
            [2, 3, 33],  #5
            [3, 34, 33],  #6
            [3, 4, 34],  #7
            [4, 35, 34],  #8
            [4, 5, 35],  #9
            [5, 36, 35],  #10
            [5, 6, 36],  #11
            [6, 37, 36],  #12
            [6, 7, 37],  #13
            [7, 8, 37],  #14
            [8, 38, 37],  #15
            [8, 9, 38],  #16
            [9, 39, 38],  #17
            [9, 10, 39],  #18
            [10, 40, 39],  #19
            [10, 11, 40],  #20
            [11, 41, 40],  #21
            [11, 12, 41],  #22
            [12, 42, 41],  #23
            [12, 13, 42],  #24
            [13, 43, 42],  #25
            [13, 14, 43],  #26
            [14, 44, 43],  #27
            [60, 75, 76],  # 56
            [60, 76, 61],  # 57
            [61, 76, 77],  # 58
            [61, 77, 62],  # 59
            [62, 77, 78],  # 60
            [62, 78, 63],  # 61
            [63, 78, 79],  # 62
            [63, 79, 64],  # 63
            [64, 79, 80],  # 64
            [64, 80, 65],  # 65
            [65, 80, 81],  # 66
            [65, 81, 66],  # 67
            [66, 81, 82],  # 68
            [66, 82, 67],  # 69
            [67, 82, 68],  # 70
            [68, 82, 83],  # 71
            [68, 83, 69],  # 72
            [69, 83, 84],  # 73
            [69, 84, 70],  # 74
            [70, 84, 85],  # 75
            [70, 85, 71],  # 76
            [71, 85, 86],  # 77
            [71, 86, 72],  # 78
            [72, 86, 87],  # 79
            [72, 87, 73],  # 80
            [73, 87, 88],  # 81
            [73, 88, 74],  # 82
            [74, 88, 89],  # 83
            [75, 30, 76],  # 84
            [76, 30, 31],  # 85
            [76, 31, 77],  # 86
            [77, 31, 32],  # 87
            [77, 32, 78],  # 88
            [78, 32, 33],  # 89
            [78, 33, 79],  # 90
            [79, 33, 34],  # 91
            [79, 34, 80],  # 92
            [80, 34, 35],  # 93
            [80, 35, 81],  # 94
            [81, 35, 36],  # 95
            [81, 36, 82],  # 96
            [82, 36, 37],  # 97
            [82, 37, 38],  # 98
            [82, 38, 83],  # 99
            [83, 38, 39],  # 100
            [83, 39, 84],  # 101
            [84, 39, 40],  # 102
            [84, 40, 85],  # 103
            [85, 40, 41],  # 104
            [85, 41, 86],  # 105
            [86, 41, 42],  # 106
            [86, 42, 87],  # 107
            [87, 42, 43],  # 108
            [87, 43, 88],  # 109
            [88, 43, 44],  # 110
            [88, 44, 89],  # 111
        ]).numpy()
        f_teeth_lower = torch.tensor([
            [45, 46, 15],  # 28           
            [46, 16, 15],  # 29
            [46, 47, 16],  # 30
            [47, 17, 16],  # 31
            [47, 48, 17],  # 32
            [48, 18, 17],  # 33
            [48, 49, 18],  # 34
            [49, 19, 18],  # 35
            [49, 50, 19],  # 36
            [50, 20, 19],  # 37
            [50, 51, 20],  # 38
            [51, 21, 20],  # 39
            [51, 52, 21],  # 40
            [52, 22, 21],  # 41
            [52, 23, 22],  # 42
            [52, 53, 23],  # 43
            [53, 24, 23],  # 44
            [53, 54, 24],  # 45
            [54, 25, 24],  # 46
            [54, 55, 25],  # 47
            [55, 26, 25],  # 48
            [55, 56, 26],  # 49
            [56, 27, 26],  # 50
            [56, 57, 27],  # 51
            [57, 28, 27],  # 52
            [57, 58, 28],  # 53
            [58, 29, 28],  # 54
            [58, 59, 29],  # 55
            [90, 106, 105],  # 112
            [90, 91, 106],  # 113
            [91, 107, 106],  # 114
            [91, 92, 107],  # 115
            [92, 108, 107],  # 116
            [92, 93, 108],  # 117
            [93, 109, 108],  # 118
            [93, 94, 109],  # 119
            [94, 110, 109],  # 120
            [94, 95, 110],  # 121
            [95, 111, 110],  # 122
            [95, 96, 111],  # 123
            [96, 112, 111],  # 124
            [96, 97, 112],  # 125
            [97, 98, 112],  # 126
            [98, 113, 112],  # 127
            [98, 99, 113],  # 128
            [99, 114, 113],  # 129
            [99, 100, 114],  # 130
            [100, 115, 114],  # 131
            [100, 101, 115],  # 132
            [101, 116, 115],  # 133
            [101, 102, 116],  # 134
            [102, 117, 116],  # 135
            [102, 103, 117],  # 136
            [103, 118, 117],  # 137
            [103, 104, 118],  # 138
            [104, 119, 118],  # 139
            [105, 106, 45],  # 140
            [106, 46, 45],  # 141
            [106, 107, 46],  # 142
            [107, 47, 46],  # 143
            [107, 108, 47],  # 144
            [108, 48, 47],  # 145
            [108, 109, 48],  # 146
            [109, 49, 48],  # 147
            [109, 110, 49],  # 148
            [110, 50, 49],  # 149
            [110, 111, 50],  # 150
            [111, 51, 50],  # 151
            [111, 112, 51],  # 152
            [112, 52, 51],  # 153
            [112, 53, 52],  # 154
            [112, 113, 53],  # 155
            [113, 54, 53],  # 156
            [113, 114, 54],  # 157
            [114, 55, 54],  # 158
            [114, 115, 55],  # 159
            [115, 56, 55],  # 160
            [115, 116, 56],  # 161
            [116, 57, 56],  # 162
            [116, 117, 57],  # 163
            [117, 58, 57],  # 164
            [117, 118, 58],  # 165
            [118, 59, 58],  # 166
            [118, 119, 59],  # 167
        ]).numpy()
        self.faces = np.concatenate([self.faces, f_teeth_upper+num_verts_orig, f_teeth_lower+num_verts_orig], axis=0)
        self.faces_tensor = torch.tensor(self.faces, dtype=torch.long)
        
        # self.faces = torch.cat([self.faces, f_teeth_upper+num_verts_orig, f_teeth_lower+num_verts_orig], dim=0)
        # self.textures_idx = torch.cat([self.textures_idx, f_teeth_upper+num_verts_uv_orig, f_teeth_lower+num_verts_uv_orig], dim=0)
 
    
    @property
    def num_expression_coeffs(self):
        return self.EXPRESSION_SPACE_DIM

    def name(self) -> str:
        return 'FLAME'
    
    def extra_repr(self):
        msg = [
            super(FLAME, self).extra_repr(),
            f'Number of Expression Coefficients: {self.EXPRESSION_SPACE_DIM}',
            f'Use face contour: {self.use_face_contour}',
        ]
        return '\n'.join(msg)
    
    def shape_blendshape(self, betas=None, expression=None):
        ''' 
        shape blend shape offsets
            Args:
            -----
                betas: torch.tensor, shape Bx100 
                    shape parameters
                expression: torch.tensor, shape Bx50
                    expression parameters
            Returns:
            --------
                offsets: torch.tensor, shape BxNx3 
                the shape blend shape offsets
        ''' 
        
        if betas is not None and expression is not None:
            shapes = torch.cat([betas, expression], dim=1) 
        elif betas is not None: 
            batch, dim = betas.shape
            shapes = torch.zeros(batch, self.SHAPE_SPACE_DIM+self.EXPRESSION_SPACE_DIM).float().cuda()
            shapes[:, :dim] = betas
        elif expression is not None:
            batch, dim = expression.shape
            shapes = torch.zeros(batch, self.SHAPE_SPACE_DIM+self.EXPRESSION_SPACE_DIM).float().cuda()
            shapes[:, self.SHAPE_SPACE_DIM:self.SHAPE_SPACE_DIM+dim] = expression
        
        offsets = blend_shapes(shapes, self.shapedirs) 
        return offsets

    def pose_blendshape(self, global_orient=None, neck_pose=None, jaw_pose=None, leye_pose=None, reye_pose=None):
        self.device, self.dtype = self.shapedirs.device, self.shapedirs.dtype
        if global_orient is None:
            global_orient = torch.zeros(1, 3, dtype=self.dtype).to(self.device)
        if neck_pose is None:
            neck_pose = torch.zeros(1, 3, dtype=self.dtype).to(self.device)
        if jaw_pose is None:
            jaw_pose = torch.zeros(1, 3, dtype=self.dtype).to(self.device)
        if leye_pose is None:
            leye_pose = torch.zeros(1, 3, dtype=self.dtype).to(self.device)
        if reye_pose is None:
            reye_pose = torch.zeros(1, 3, dtype=self.dtype).to(self.device)
        
        # batch = max(len(global_orient), len(neck_pose), len(jaw_pose), len(leye_pose), len(reye_pose))
        # if not batch == len(global_orient):
            
        full_pose = torch.stack([global_orient, neck_pose, jaw_pose, leye_pose, reye_pose], dim=1) 
        offsets = pose_blend_shapes(full_pose, self.posedirs, True)
        
        return offsets  

    def upsampling(self, mode='all'):
        '''
        subdivide the mesh to increase the number of vertices including v_template, lbs_weights, shapedir, posedir, and J_regressor 
            Args:
            -----
                mode: str, 
                    the mode of subdivide, 'all' or 'uniform'
        '''
        N = self.v_template.shape[0] 

        joints_cano = self.forward().joints_cano[0]

        if mode == 'all': 
            v_template, self.faces, unique = subdivide(self.v_template.cpu().numpy(), self.faces) 
            self.v_template = torch.tensor(v_template).to(self.v_template)
        else:   
            sparse_tri, sparse_tri_mask = self.segment.get_triangles(
                positive_parts=['neck', 'scalp', 'boundary', 'forehead'],
                negative_parts=['left_ear', 'right_ear', 'left_eyeball', 'right_eyeball'], 
                return_mask=True
            ) 
            v_template, faces, unique = subdivide(self.v_template.cpu().numpy(), sparse_tri)
            self.faces = np.concatenate([faces, self.faces[~sparse_tri_mask]]) 
        
        self.unique = torch.tensor(unique, dtype=torch.long) 
            
        self.lbs_weights = subdivide_inorder(self.lbs_weights, self.faces_tensor, unique)

        self.shapedirs = subdivide_inorder(
            self.shapedirs.reshape(N, -1), 
            self.faces_tensor, unique
            ).reshape(-1, 3, 400)
        
        dp = self.posedirs.shape[0]
        self.posedirs = subdivide_inorder(
            self.posedirs.reshape(dp, N, 3).permute(1, 0, 2).reshape(N, dp*3), 
            self.faces_tensor, unique
        ).reshape(-1, dp, 3).permute(1, 0, 2).reshape(dp, -1)
 
        # TODO: check if the J_regressor is correct
        # self.J_regressor = subdivide_inorder(
        #     self.J_regressor.transpose(0, 1),
        #     self.faces_tensor, unique
        # ).transpose(0, 1)
        self.update_J_regressor(joints_cano)
         
    def update_J_regressor(self, joints_cano, betas=None, expression=None):
        '''
        v @ J_regressor = joints 
        
        Args:
        -----
            joints_cano: torch.tensor, shape Jx3
            
        '''
        # remove 
        betas = betas if betas is not None else self.betas
        expression = expression if expression is not None else self.expression
        shape_offsets = self.shape_blendshape(betas, expression) 
        v_shaped = self.v_template + shape_offsets[0]
        
        J_regressor = np.linalg.lstsq(
            v_shaped.detach().cpu().numpy().T, 
            joints_cano.detach().cpu().numpy().T, 
            rcond=None
        )[0].T   # JxV 
        J = len(joints_cano)
        J_regressor = torch.tensor(J_regressor, dtype=self.dtype).to(joints_cano.device)
        # J_regressor = torch.cat([J_regressor, self.J_regressor[3:]], dim=0)
        self.J_regressor = J_regressor

    def set_params(self, params):
        ''' Set the parameters of the model '''
        for param_name, param in params.items():
            if hasattr(self, param_name):
                setattr(self, param_name, param)
            else:
                raise ValueError('Parameter {} does not exist'.format(param_name))

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        neck_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        expression: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        v_offsets: Optional[Tensor] = None,
        shapedirs: Optional[Tensor] = None, 
        posedirs: Optional[Tensor] = None,
        v_template: Optional[Tensor] = None, 
        **kwargs
    ) -> FLAMEOutput:
        '''
        Forward pass for the SMPLX model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            expression: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `expression` and use it
                instead. For example, it can used if expression parameters
                `expression` are predicted from some external model.
            jaw_pose: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `jaw_pose` and
                use this instead. It should either joint rotations in
                axis-angle format.
            jaw_pose: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `jaw_pose` and
                use this instead. It should either joint rotations in
                axis-angle format.
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
                output: ModelOutput
                A named tuple of type `ModelOutput`
        '''

        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = global_orient if global_orient is not None else self.global_orient
        jaw_pose = jaw_pose if jaw_pose is not None else self.jaw_pose
        neck_pose = neck_pose if neck_pose is not None else self.neck_pose
        leye_pose = leye_pose if leye_pose is not None else self.leye_pose
        reye_pose = reye_pose if reye_pose is not None else self.reye_pose
        betas = betas if betas is not None else self.betas
        expression = expression if expression is not None else self.expression

        batch_size = max(len(betas), len(global_orient), len(jaw_pose), len(leye_pose), len(reye_pose), len(neck_pose), len(expression))
        if not batch_size == self.batch_size:
            assert self.batch_size == 1, 'Batch size mismatch' 
            global_orient = global_orient.expand(batch_size, -1)
            jaw_pose = jaw_pose.expand(batch_size, -1)
            leye_pose = leye_pose.expand(batch_size, -1)
            reye_pose = reye_pose.expand(batch_size, -1)
            neck_pose = neck_pose.expand(batch_size, -1)
            betas = betas.expand(batch_size, -1)
            expression = expression.expand(batch_size, -1)
        
        if betas.shape[1] < self.SHAPE_SPACE_DIM:
            zero_beta = torch.zeros(betas.shape[0], self.SHAPE_SPACE_DIM - betas.shape[1], dtype=betas.dtype, device=betas.device)
            betas = torch.cat([betas, zero_beta], dim=1)
        if expression.shape[1] < self.EXPRESSION_SPACE_DIM:
            zero_expr = torch.zeros(expression.shape[0], self.EXPRESSION_SPACE_DIM - expression.shape[1], dtype=expression.dtype, device=expression.device)
            expression = torch.cat([expression, zero_expr], dim=1)

        if v_offsets is None and hasattr(self, 'v_offsets'):
                v_offsets = self.v_offsets 

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None and hasattr(self, 'transl'):
            transl = self.transl
        
        full_pose = torch.cat(
            [global_orient, neck_pose, jaw_pose, leye_pose, reye_pose], dim=1).reshape(-1, 5, 3)
        
        batch_size = max(betas.shape[0], global_orient.shape[0], jaw_pose.shape[0])
        # Concatenate the shape and expression coefficients
        scale = int(batch_size / betas.shape[0])
        if scale > 1:
            betas = betas.expand(scale, -1)

        shape_components = torch.cat([betas, expression], dim=-1)
        
        v_template = v_template if v_template is not None else self.v_template
        posedirs = posedirs if posedirs is not None else self.posedirs 
        if shapedirs is None:
            shapedirs = self.shapedirs 

        vertices, joints, vT, jT, v_cano, joints_cano = lbs(
            shape_components, 
            full_pose, 
            v_template,
            shapedirs, 
            posedirs,
            self.J_regressor, 
            self.parents,
            self.lbs_weights, 
            pose2rot=pose2rot,
            custom_out=True,  
            v_offsets=v_offsets   
        )

        ##### landmarks 
        if hasattr(self, 'full_lmk_faces_idx') and hasattr(self, 'full_lmk_bary_coords'): 
            lmk_faces_idx = self.full_lmk_faces_idx.repeat(batch_size, 1)   #[bach, J]
            lmk_bary_coords = self.full_lmk_bary_coords.repeat(batch_size, 1, 1)  #[bach, J, 3]
        else:
            lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1).contiguous()
            lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(batch_size, 1, 1)

            if self.use_face_contour:
                dyn_lmk_faces_idx, dyn_lmk_bary_coords = find_dynamic_lmk_idx_and_bcoords(
                    vertices, full_pose, self.dynamic_lmk_faces_idx,
                    self.dynamic_lmk_bary_coords,
                    self.neck_kin_chain,
                    pose2rot=True,
                )  
                lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
                lmk_bary_coords = torch.cat(
                    [dyn_lmk_bary_coords, lmk_bary_coords.expand(batch_size, -1, -1)], 1)
            
        landmarks = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords)
        
        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        joints_transform = self.vertex_joint_selector(vT, jT)
        
        # Add the landmarks to the joints
        joints = torch.cat([joints, landmarks], dim=1)

        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints=joints, vertices=vertices)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_transform[:, :, :3, 3] += transl.unsqueeze(dim=1)

        output = FLAMEOutput(vertices=vertices if return_verts else None,
                             joints=joints,
                             betas=betas,
                             expression=expression,
                             global_orient=global_orient,
                             neck_pose=neck_pose,
                             jaw_pose=jaw_pose,
                             joints_transform=joints_transform, 
                             v_cano=v_cano, 
                             joints_cano=joints_cano, 
                             full_pose=full_pose if return_full_pose else None)
        return output


class FLAMELayer(FLAME):
    def __init__(self, *args, **kwargs) -> None:
        ''' FLAME as a layer model constructor '''
        super(FLAMELayer, self).__init__(
            create_betas=False,
            create_expression=False,
            create_global_orient=False,
            create_neck_pose=False,
            create_jaw_pose=False,
            create_leye_pose=False,
            create_reye_pose=False,
            *args,
            **kwargs)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        neck_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        expression: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs
    ) -> FLAMEOutput:
        '''
        Forward pass for the SMPLX model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3x3
                Global rotation of the body. Useful if someone wishes to
                predicts this with an external model. It is expected to be in
                rotation matrix format. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                Shape parameters. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            expression: torch.tensor, optional, shape BxN_e
                If given, ignore the member variable `expression` and use it
                instead. For example, it can used if expression parameters
                `expression` are predicted from some external model.
            jaw_pose: torch.tensor, optional, shape Bx3x3
                Jaw pose. It should either joint rotations in
                rotation matrix format.
            transl: torch.tensor, optional, shape Bx3
                Translation vector of the body.
                For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
                output: ModelOutput
                A named tuple of type `ModelOutput`
        ''' 
        device, dtype = self.shapedirs.device, self.shapedirs.dtype
        if global_orient is None:
            batch_size = 1
            global_orient = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        else:
            batch_size = global_orient.shape[0]
        if neck_pose is None:
            neck_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 1, -1, -1).contiguous()
        if jaw_pose is None:
            jaw_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if leye_pose is None:
            leye_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if reye_pose is None:
            reye_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if betas is None:
            betas = torch.zeros([batch_size, self.SHAPE_SPACE_DIM],
                                dtype=dtype, device=device)
        if expression is None:
            expression = torch.zeros([batch_size, self.EXPRESSION_SPACE_DIM],
                                     dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        full_pose = torch.cat(
            [global_orient, neck_pose, jaw_pose, leye_pose, reye_pose], dim=1)

        shape_components = torch.cat([betas, expression], dim=-1)
        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)

 
        vertices, joints, vT, jT, v_cano, J_cano = lbs(
            shape_components, 
            full_pose, 
            self.v_template,
            shapedirs, 
            self.posedirs,
            self.J_regressor, 
            self.parents,
            self.lbs_weights, 
            pose2rot=False,
            custom_out=True,  
        )

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(
            dim=0).expand(batch_size, -1).contiguous()
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            batch_size, 1, 1)
        if self.use_face_contour:
            lmk_idx_and_bcoords = find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose, self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain,
                pose2rot=False,
            )
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = lmk_idx_and_bcoords
            lmk_faces_idx = torch.cat([lmk_faces_idx,
                                       dyn_lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat(
                [lmk_bary_coords.expand(batch_size, -1, -1),
                 dyn_lmk_bary_coords], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx,
                                       lmk_bary_coords)
 
        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        joints_transform = self.vertex_joint_selector(vT, jT)
         
        # Add the landmarks to the joints
        joints = torch.cat([joints, landmarks], dim=1)

        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints=joints, vertices=vertices)

        if apply_trans: 
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_transform[:, :, :3, 3] += transl.unsqueeze(dim=1)

        output = FLAMEOutput(
            vertices=vertices if return_verts else None,
            joints=joints,
            betas=betas,
            expression=expression,
            global_orient=global_orient,
            neck_pose=neck_pose,
            jaw_pose=jaw_pose,
            joints_transform=joints_transform,
            full_pose=full_pose if return_full_pose else None
        )
        return output

