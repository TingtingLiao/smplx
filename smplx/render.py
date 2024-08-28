import os 
import random 
import torch
import torch.nn.functional as F
import numpy as np
import nvdiffrast.torch as dr
from kiui.op import scale_img_nhwc

from .lbs import batch_rodrigues 


class Renderer(torch.nn.Module):
    def __init__(self, gui=False, shading=False, hdr_path=None):
        super().__init__()    
        if not gui or os.name == 'nt': 
            self.glctx = dr.RasterizeCudaContext()
        else:
            self.glctx = dr.RasterizeGLContext()

        self.light_dir = np.array([0, 0])
        self.ambient_ratio = 0.5
        
        if shading and hdr_path is not None:
            import envlight
            if hdr_path is None:
                hdr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/lights/mud_road_puresky_1k.hdr') 
            self.light = envlight.EnvLight(hdr_path, scale=2, device='cuda')
            self.FG_LUT = torch.from_numpy(np.fromfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/lights/bsdf_256_256.bin"), dtype=np.float32).reshape(1, 256, 256, 2)).cuda()
            self.metallic_factor = 1
            self.roughness_factor = 1
 

    def render_bg(self, envmap_path):
        '''render with shading'''
        from PIL import Image 
        envmap = Image.open(envmap_path)
        h,w = self.res
        pos_int = torch.arange(w*h, dtype = torch.int32, device='cuda')
        pos = 0.5 - torch.stack((pos_int % w, pos_int // w), dim=1) / torch.tensor((w,h), device='cuda')
        a = np.deg2rad(self.fov_x)/2
        r = w/h
        f = torch.tensor((2*np.tan(a),  2*np.tan(a)/r), device='cuda', dtype=torch.float32)
        rays = torch.cat((pos*f, torch.ones((w*h,1), device='cuda'), torch.zeros((w*h,1), device='cuda')), dim=1)
        rays_norm = (rays.transpose(0,1) / torch.norm(rays, dim=1)).transpose(0,1)
        rays_view = torch.matmul(rays_norm, self.view_mats.inverse().transpose(1,2)).reshape((self.view_mats.shape[0],h,w,-1))
        theta = torch.acos(rays_view[..., 1])
        phi = torch.atan2(rays_view[..., 0], rays_view[..., 2])
        envmap_uvs = torch.stack([0.75-phi/(2*np.pi), theta / np.pi], dim=-1)
        self.bgs = dr.texture(envmap[None, ...], envmap_uvs, filter_mode='linear').flip(1)
        self.bgs[..., -1] = 0 # Set alpha to 0

    def shading(self, albedo, normal=None, mode='albedo'):  
        if mode == "albedo":
            return albedo 
        elif mode == "lambertian":
            assert normal is not None, "normal and light direction should be provided" 
            light_d = np.deg2rad(self.light_dir)
            light_d = np.array([
                np.cos(light_d[0]) * np.sin(light_d[1]),
                -np.sin(light_d[0]),
                np.cos(light_d[0]) * np.cos(light_d[1]),
            ], dtype=np.float32)
            light_d = torch.from_numpy(light_d).to(albedo.device)
            lambertian = self.ambient_ratio + (1 - self.ambient_ratio)  * (normal @ light_d).float().clamp(min=0)
            albedo = albedo * lambertian.unsqueeze(-1) 
            return albedo
              
        elif mode == "pbr": 
            xyzs, _ = dr.interpolate(self.mesh.v.unsqueeze(0), rast, self.mesh.f) # [1, H, W, 3]
            viewdir = safe_normalize(xyzs - pose[:3, 3])

            n_dot_v = (normal * viewdir).sum(-1, keepdim=True) # [1, H, W, 1]
            reflective = n_dot_v * normal * 2 - viewdir

            diffuse_albedo = (1 - metallic) * albedo

            fg_uv = torch.cat([n_dot_v, roughness], -1).clamp(0, 1) # [H, W, 2]
            fg = dr.texture(
                self.FG_LUT,
                fg_uv.reshape(1, -1, 1, 2).contiguous(),
                filter_mode="linear",
                boundary_mode="clamp",
            ).reshape(1, H, W, 2)
            F0 = (1 - metallic) * 0.04 + metallic * albedo
            specular_albedo = F0 * fg[..., 0:1] + fg[..., 1:2]

            diffuse_light = self.light(normal)
            specular_light = self.light(reflective, roughness)

            color = diffuse_albedo * diffuse_light + specular_albedo * specular_light # [H, W, 3]
            color = color * alpha + self.bg_color * (1 - alpha)

            buffer = color[0].detach().cpu().numpy()

        return color 

    def get_orthogonal_cameras(self, n=4, flipY=True):
        """
        Args:
            n: int number of cameras
            flipY: bool whether to flip Y axis
        Returns:
            mvp: torch.Tensor [n, 4, 4], batch of orthogonal cameras 
        """
        mvp = torch.eye(4)[None].expand(n, -1, -1).clone()  
        
        angles = np.linspace(0, np.pi*2, n, endpoint=False)
        R = batch_rodrigues(torch.tensor([[0, angle, 0] for angle in angles]).reshape(-1, 3)).float()  
        
        if flipY:
            flip_rot = batch_rodrigues(torch.tensor([np.pi, 0, 0]).reshape(1, 3)).reshape(3, 3).float()  
            R = torch.matmul(R, flip_rot) 

        mvp[:, :3, :3] = R 
        
        return mvp
    
    def forward(self, mesh, mvp,
                h=512,
                w=512,
                light_d=None,
                ambient_ratio=1.,
                shading_mode='albedo',
                spp=1,
                bg_color=None, 
                shading=False):
        """
        Args: 
            mesh: Mesh object
            mvp: [batch, 4, 4]
            h: int
            w: int
            light_d:
            spp: int
            ambient_ratio: float
            shading_mode: str rendering type albedo, lambertian or pbr
        Returns:
            color: [batch, h, w, 3]
            alpha: [batch, h, w, 1] 
        """
        assert shading_mode in ['albedo', 'lambertian', 'pbr'], "shading_mode should be albedo, lambertian or pbr"
        
        B = mvp.shape[0] 
        v_homo = F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).expand(B, -1, -1)
        v_clip = torch.bmm(v_homo, torch.transpose(mvp, 1, 2)).float()  # [B, N, 4]

        res = (int(h * spp), int(w * spp)) if spp > 1 else (h, w)
        rast, rast_db = dr.rasterize(self.glctx, v_clip, mesh.f, res)
        alpha = (rast[..., 3:] > 0).float()

        ### normal  
        if mesh.vn is None:
            mesh.auto_normal()
            
        normal, _ = dr.interpolate(mesh.vn[None, ...].float(), rast, mesh.f)
        normal = (normal + 1) / 2.
        
        if mesh.vt is not None and mesh.ft is not None:
            texc, texc_db = dr.interpolate(mesh.vt[None, ...], rast, mesh.ft, rast_db=rast_db, diff_attrs='all') 
        else:
            texc = None 
        
        ### albedo   
        if mesh.albedo is not None:
            albedo = dr.texture(mesh.albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear')  # [B, H, W, 3] 
        elif mesh.vc is not None:
            color, _ = dr.interpolate(mesh.vc[None, ..., :3].contiguous().float(), rast, mesh.f)
        else:
            color = None

        ### shading  
        if color is not None:
            color = torch.where(rast[..., 3:] > 0, color, torch.tensor(0).to(color.device))  # remove background
            color = self.shading(color, normal, mode=shading_mode)
            
        ### antialias
        normal = dr.antialias(normal, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
        alpha = dr.antialias(alpha, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
        if color is not None:
            color = dr.antialias(color, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]

        ### inverse super-sampling
        if spp > 1:
            if color is not None:
                color = scale_img_nhwc(color, (h, w))
            alpha = scale_img_nhwc(alpha, (h, w))
            normal = scale_img_nhwc(normal, (h, w))

        ### background
        if bg_color is not None:
            if color is not None:
                color = color * alpha + bg_color * (1 - alpha)
            normal = normal * alpha + bg_color * (1 - alpha)

        return {
            'image': color,
            'normal': normal,
            'alpha': alpha, 
            'uvs': texc
        }
