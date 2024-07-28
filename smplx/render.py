import os 
import random 
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
 


def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[
        1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]:  # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else:  # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC


class Renderer(torch.nn.Module):
    def __init__(self, gui=False):
        super().__init__()    
        if not gui or os.name == 'nt': 
            self.glctx = dr.RasterizeCudaContext()
        else:
            self.glctx = dr.RasterizeGLContext()

    def forward(self, mesh, mvp,
                h=512,
                w=512,
                light_d=None,
                ambient_ratio=1.,
                mode='rgb',
                spp=1,
                bg_color=None):
        """
        Args:
            spp: int
            mesh: Mesh object
            mvp: [batch, 4, 4]
            h: int
            w: int
            light_d:
            ambient_ratio: float
            mode: str rendering type rgb, normal, lambertian
        Returns:
            color: [batch, h, w, 3]
            alpha: [batch, h, w, 1]
            depth: [batch, h, w, 1]

        """
        B = mvp.shape[0] 
        v_clip = torch.bmm(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).expand(B, -1, -1),
                           torch.transpose(mvp, 1, 2)
                           ).float()  # [B, N, 4]
         

        res = (int(h * spp), int(w * spp)) if spp > 1 else (h, w)
        rast, rast_db = dr.rasterize(self.glctx, v_clip, mesh.f, res)

        ################################################################################
        # Interpolate attributes
        ################################################################################

        # Interpolate world space position
        alpha, _ = dr.interpolate(torch.ones_like(v_clip[..., :1]), rast, mesh.f)  # [B, H, W, 1]
        depth = rast[..., [2]]  # [B, H, W]
 
    
        if mesh.albedo is not None:
            texc, texc_db = dr.interpolate(mesh.vt[None, ...], rast, mesh.ft, rast_db=rast_db, diff_attrs='all') 
            albedo = dr.texture(
                mesh.albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')  # [B, H, W, 3]
            color = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device))  # remove background
        elif mesh.vc is not None:
            color, _ = dr.interpolate(mesh.vc[None, ..., :3].contiguous().float(), rast, mesh.f)
        else:
            color = None 
 
        if mode == "lambertian" and color is not None:
            lambertian = ambient_ratio + (1 - ambient_ratio) * (normal @ light_d.view(-1, 1)).float().clamp(min=0)
            color = color * lambertian.repeat(1, 1, 1, 3)

        # render vertex color  
        if mesh.vn is None:
            mesh.auto_normal()

        normal, _ = dr.interpolate(mesh.vn[None, ...].float(), rast, mesh.f)
        normal = (normal + 1) / 2.

        # antialias
        normal = dr.antialias(normal, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
        alpha = dr.antialias(alpha, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
        if color is not None:
            color = dr.antialias(color, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
          
        # inverse super-sampling
        if spp > 1:
            if color is not None:
                color = scale_img_nhwc(color, (h, w))
            alpha = scale_img_nhwc(alpha, (h, w))
            normal = scale_img_nhwc(normal, (h, w))

        if bg_color is not None:
            if color is not None:
                color = color * alpha + bg_color * (1 - alpha)
            normal = normal * alpha + bg_color * (1 - alpha)

        return {
            'image': color,
            'normal': normal,
            'alpha': alpha
        }
  