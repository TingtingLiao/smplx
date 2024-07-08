# SMPLX 
A smplx toolkit. 

## Install 
```bash 
pip install git+https://github.com/TingtingLiao/smplx.git 
```

## Usage 
- **SMPLX-D** support vertext offset on each vertex. 
```bash  
smplx_out = smpl_model( 
    global_orient=global_orient,
    body_pose=body_pose,
    jaw_pose=jaw_pose, 
    betas=betas,
    expression=expression,
    v_offsets=v_offsets,  #[V, 3]
) 
```

- **Upsampling smplx mesh**   
```bash 
smplx_out = smpl_model( 
    global_orient=global_orient,
    body_pose=body_pose,
    jaw_pose=jaw_pose, 
    betas=betas,
    expression=expression,
    v_offsets=v_offsets,  #[V', 3]
    upsample=True, 
)
``` 
- **SMPLX Segments**

![Screenshot from 2024-07-08 18-49-33](https://github.com/TingtingLiao/smplx/assets/45743512/504c4572-5039-4a77-946f-52ee14275376)

```bash 
from smplx.segment import SMPLXSeg
SMPLXSeg = SMPLXSeg('./data/smplx', device=device)
```

## Acknowledgments 
- [SMPLX](https://github.com/vchoutas/smplx). 
