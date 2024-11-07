import json
import pickle as pkl
import numpy as np
import torch
from matplotlib import cm as mpl_cm, colors as mpl_colors
from typing import Union 


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


class BaseSeg:
    def __init__(self):
        # self.device = device
        pass 

    def get_vertex_ids(self, part_name:Union[list, str])->np.ndarray:
        '''
        get the vertex ids of local part
            Args:
            -----
            part_name: str or list of str

            Returns:
            --------
            vertex_ids: np.ndarray, shape M 
        '''
        if isinstance(part_name, str):
            part_name = [part_name]
        
        vids_all = [] 
        for name in part_name:
            if name == 'face':
                v_mask = np.ones((self.N, 1))
                for key in ['right_ear', 'left_ear', 'neck', 'scalp', 'boundary']:
                    v_mask[self.segms[key]] = 0
                vids = np.where(v_mask)[0]
            else:
                vids = self.segms[name]
            
            vids_all.extend(vids)
            
        return np.array(vids_all) 


class FlameSeg:
    def __init__(self, flame_dir, faces, N=5023):  
        self.segms = pkl.load(open(f"{flame_dir}/FLAME_masks.pkl", "rb"), encoding='latin1') 
        self.N = N
        self._vc = None
        self.faces = faces 
        
        # additional parts 
        self.segms.update({
            'left_eyeball_cone': [
                837, 838, 840, 841, 842, 846, 847, 848, 1000, 1001, 1002, 1003, 1006, 1007, 1008, 1010, 1011, 1045, 1046, 1061, 1063, 1064, 1065, 1068, 1075, 1085, 1086, 1115, 1116, 1117, 1125, 1126, 1127, 1128, 1129, 1132, 1134, 1142, 1143, 1147, 1150, 1227, 1228, 1229, 1230, 1232, 1233, 1241, 1242, 1283, 1284, 1287, 1289, 1320, 1321, 1361, 3824, 3835, 3861, 3862, 3929
            ],
            'right_eyeball_cone': [
                2279, 2280, 2281, 2282, 2283, 2284, 2285, 2286, 2361, 2362, 2363, 2364, 2365, 2366, 2367, 2368, 2369, 2386, 2387, 2390, 2392, 2393, 2394, 2395, 2396, 2397, 2398, 2408, 2409, 2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2419, 2420, 2423, 2424, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2467, 2468, 2469, 2470, 2478, 2479, 2510, 3616, 3638, 3700, 3702, 3930
            ], 
            'left_eyelid': [
                807, 808, 809, 814, 815, 816, 821, 822, 823, 824, 825, 826, 827, 828, 829, 841, 842, 848, 864, 865, 877, 878, 879, 880, 881, 882, 883, 884, 885, 896, 897, 903, 904, 905, 922, 923, 924, 926, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 958, 959, 991, 992, 993, 994, 995, 999, 1000, 1003, 1006, 1008, 1011, 1023, 1033, 1034, 1045, 1046, 1059, 1060, 1061, 1062, 1093, 1096, 1101, 1108, 1113, 1114, 1115, 1125, 1126, 1132, 1134, 1135, 1142, 1143, 1144, 1146, 1147, 1150, 1151, 1152, 1153, 1154, 1170, 1175, 1182, 1183, 1194, 1195, 1200, 1201, 1202, 1216, 1217, 1218, 1224, 1227, 1230, 1232, 1233, 1243, 1244, 1283, 1289, 1292, 1293, 1294, 1320, 1329, 1331, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1361, 3827, 3832, 3833, 3835, 3853, 3855, 3856, 3861
            ], 
            'right_eyelid':[
                2264, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2282, 2283, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2297, 2298, 2299, 2303, 2304, 2305, 2312, 2313, 2314, 2315, 2323, 2324, 2325, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2364, 2365, 2367, 2369, 2381, 2382, 2383, 2386, 2387, 2388, 2389, 2390, 2391, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2411, 2412, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 2426, 2427, 2428, 2436, 2437, 2440, 2441, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2457, 2460, 2461, 2462, 2465, 2466, 2467, 2470, 2471, 2472, 2473, 2478, 2485, 2486, 2487, 2488, 2489, 2490, 2491, 2492, 2493, 2494, 2495, 2496, 2503, 2504, 2505, 2506, 2507, 2508, 2509, 2510, 3619, 3631, 3632, 3638, 3687, 3689, 3690, 3700, 
            ], 
            'left_iris':[
                3931, 3932, 3933, 3935, 3936, 3937, 3939, 3940, 3941, 3943, 3944, 3945, 3947, 3948, 3949, 3951, 3952, 3953, 3955, 3956, 3957, 3959, 3960, 3961, 3963, 3964, 3965, 3967, 3968, 3969, 3971, 3972, 3973, 3975, 3976, 3977, 3979, 3980, 3981, 3983, 3984, 3985, 3987, 3988, 3989, 3991, 3992, 3993, 3995, 3996, 3997, 3999, 4000, 4001, 4003, 4004, 4005, 4007, 4008, 4009, 4011, 4012, 4013, 4015, 4016, 4017, 4019, 4020, 4021, 4023, 4024, 4025, 4027, 4028, 4029, 4031, 4032, 4033, 4035, 4036, 4037, 4039, 4040, 4041, 4043, 4044, 4045, 4047, 4048, 4049, 4051, 4052, 4053, 4054, 4056, 4057, 4058, 
            ], 
            'right_iris': [
                4477, 4478, 4479, 4481, 4482, 4483, 4485, 4486, 4487, 4489, 4490, 4491, 4493, 4494, 4495, 4497, 4498, 4499, 4501, 4502, 4503, 4505, 4506, 4507, 4509, 4510, 4511, 4513, 4514, 4515, 4517, 4518, 4519, 4521, 4522, 4523, 4525, 4526, 4527, 4529, 4530, 4531, 4533, 4534, 4535, 4537, 4538, 4539, 4541, 4542, 4543, 4545, 4546, 4547, 4549, 4550, 4551, 4553, 4554, 4555, 4557, 4558, 4559, 4561, 4562, 4563, 4565, 4566, 4567, 4569, 4570, 4571, 4573, 4574, 4575, 4577, 4578, 4579, 4581, 4582, 4583, 4585, 4586, 4587, 4589, 4590, 4591, 4593, 4594, 4595, 4597, 4598, 4599, 4600, 4602, 4603, 4604, 
            ], 
            
            # lips 
            'lip_inside_ring_upper': [
                1595, 1746, 1747, 1742, 1739, 1665, 1666, 3514, 2783, 2782, 2854, 2857, 2862, 2861, 2731
            ], 
            'lip_inside_ring_lower': [
                1572, 1573, 1860, 1862, 1830, 1835, 1852, 3497, 2941, 2933, 2930, 2945, 2943, 2709, 2708
            ], 
            'lip_outside_ring_upper':[
                1713, 1715, 1716, 1735, 1696, 1694, 1657, 3543, 2774, 2811, 2813, 2850, 2833, 2832, 2830, 
            ],
            'lip_outside_ring_lower': [
                1576, 1577, 1773, 1774, 1795, 1802, 1865, 3503, 2948, 2905, 2898, 2881, 2880, 2713, 2712
            ], 
            'lip_inside_upper': [
                1588, 1589, 1590, 1591, 1594, 1595, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1724, 1725, 1739, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 2724, 2725, 2726, 2727, 2730, 2731, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2841, 2842, 2854, 2856, 2857, 2858, 2859, 2860, 2861, 2862, 3514, 3547, 3549
            ], 
            'lip_inside_lower': [
                1572, 1573, 1592, 1593, 1764, 1765, 1779, 1780, 1781, 1830, 1831, 1832, 1835, 1846, 1847, 1851, 1852, 1854, 1860, 1861, 1862, 2708, 2709, 2728, 2729, 2872, 2873, 2886, 2887, 2888, 2930, 2931, 2932, 2933, 2935, 2936, 2940, 2941, 2942, 2943, 2944, 2945, 3497, 3500, 3512, 
            ], 
            'lip_inside': [
                1572, 1573, 1580, 1581, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1718, 1719, 1722, 1724, 1725, 1728, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1764, 1765, 1777, 1778, 1779, 1780, 1781, 1782, 1827, 1830, 1831, 1832, 1835, 1836, 1846, 1847, 1851, 1852, 1854, 1860, 1861, 1862, 2708, 2709, 2716, 2717, 2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2835, 2836, 2839, 2841, 2842, 2843, 2854, 2855, 2856, 2857, 2858, 2859, 2860, 2861, 2862, 2863, 2872, 2873, 2884, 2885, 2886, 2887, 2888, 2889, 2929, 2930, 2931, 2932, 2933, 2934, 2935, 2936, 2940, 2941, 2942, 2943, 2944, 2945, 3497, 3500, 3512, 3513, 3514, 3533, 3547, 3549, 
            ]
            
        })
    
    @property
    def partnames(self):
        # 'eye_region', 'right_eye_region', 'left_eye_region' 
        #  'left_eyeball', 'right_eyeball', 
        # 'right_ear', 'left_ear', 'nose', 'forehead', 'lips', 
        #  'scalp', 'boundary', 'face', 'neck',
        return list(self.segms.keys())

    @property 
    def part2color(self):
        return {
            'face': [255, 85, 0],
            'left_eyeball':  [255, 85, 0],
            'right_eyeball': [255, 85, 0],
            'left_ear': [255, 85, 0],
            'right_ear': [255, 85, 0],
            
            'scalp': [0, 85, 255], 

            'neck': [255, 0, 255],

            'lips': [0, 0, 255],

            'left_eyeball': [255, 0, 170], 
            'right_eyeball': [255, 0, 170],

        }

    def get_vertex_ids(self, part_name:Union[list, str])->np.ndarray:
        '''
        get the vertex ids of local part
            Args:
            -----
            part_name: str or list of str

            Returns:
            --------
            vertex_ids: np.ndarray, shape M 
        '''
        if isinstance(part_name, str):
            part_name = [part_name]
        
        vids_all = [] 
        for name in part_name:
            if name == 'face':
                v_mask = np.ones((self.N, 1))
                for key in ['right_ear', 'left_ear', 'neck', 'scalp', 'boundary']:
                    v_mask[self.segms[key]] = 0
                vids = np.where(v_mask)[0]
            else:
                vids = self.segms[name]
            
            vids_all.extend(vids)
            
        return np.array(vids_all)

    def get_triangles(self, positive_parts, negative_parts=[], return_mask=False):
        '''
        get the triangles of local part
            Args:
            -----
            positive_parts: str or list of str
                the name of the part
            negative_parts: str or list of str
                the name of the part
            return_mask: bool
                whether return the mask of triangles
            
            Returns:
            --------
            triangles: np.ndarray, shape Mx3 
        ''' 
        if isinstance(positive_parts, str):
            if positive_parts == 'all':
                positive_parts = self.partnames
            else:
                positive_parts = [positive_parts]
        
        if isinstance(negative_parts, str):
            negative_parts = [negative_parts]

        v_mask = np.zeros((self.N)) 
        for name in positive_parts:
            v_mask[self.segms[name]] = 1
        
        for name in negative_parts:
            v_mask[self.segms[name]] = 0
            
        tri_mask = v_mask[self.faces].all(axis=1)

        if return_mask:
            return self.faces[tri_mask], tri_mask
        
        return self.faces[tri_mask]

    def get_vertices(self, vertices, part_name):
        '''
        get the vertices of local part
        
            Args
            ----------
            vertices: torch.tensor, shape BxNx3 or Nx3
            part_name: str, 
                the name of the part

            Returns
            ------- 
            vertices: torch.tensor, shape BxMx3 or Mx3
        '''
        assert len(vertices.shape) == 2 or len(vertices.shape) == 3
        if len(vertices.shape) == 2:
            return vertices[self.segms[part_name]]
        else:
            return vertices[:, self.segms[part_name]]

    # @property 
    # def vc(self):
    #     if self._vc is None:
    #         self._vc = segm_to_vertex_colors(self.segms, self.N)[:, :3]
    #         self._vc = torch.tensor(self._vc, dtype=torch.float, device=self.device)
    #     return self._vc 

    @property 
    def vc(self):
        if self._vc is None:
            vc = np.zeros((self.N, 3))
            for part_name, color in self.part2color.items():
                vc[self.segms[part_name]] = list(map(lambda x: x / 255, color))
            self._vc = torch.tensor(vc, dtype=torch.float, device=self.device)
        return self._vc 


class SmplxSeg(BaseSeg):
    def __init__(self, smplx_dir):
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
        self.segms = json.load(open(f"{smplx_dir}/smplx_vert_segementation.json", 'r'))
        self.flame_segs = pkl.load(open(f"{smplx_dir}/FLAME_masks.pkl", "rb"), encoding='latin1')
        self.flame_to_smplx_vid = np.load(f"{smplx_dir}/FLAME_SMPLX_vertex_ids.npy", allow_pickle=True)
        self.smplx_faces = np.load(f"{smplx_dir}/smplx_faces.npy")   
        self.N = 10475
        self._vc = None
    
    def mapping_smplx_to_flame(self, flame_vid):
        for key in ['right_ear', 'left_ear', 'nose', 'lips']:
            if key in flame_vid:
                return list(self.smplx_flame_vid[self.flame_segs["left_ear"]])

    def get_triangles(self, part_name):
        v_mask = np.zeros((self.N, 1))   
        v_mask[self.segms[part_name]] = 1 
        triangles = index_triangles_from_vertex_mask(v_mask, self.smplx_faces) 
        return triangles

    # def get_vertex_ids(self, part_name):
    #     return self.segms[part_name]

    def init_part_triangls(self):
        for part_name in self.smplx_segs.keys(): 
            setattr(self, part_name + '_tri', self.get_triangles(part_name))
    
    @property 
    def vc(self):
        if self._vc is None:
            self._vc = segm_to_vertex_colors(self.smplx_segs, self.N)[:, :3]
            self._vc = torch.tensor(self._vc, dtype=torch.float, device=self.device)
        return self._vc 
  
       