# SMPLX 
A smplx toolkit. 

## Install 
```bash 
pip install git+https://github.com/TingtingLiao/smplx.git 
```

## Usage 
- **SMPLX-D** support vertext offset on each vertex. 
```bash  
smplx_out = self.smpl_model( 
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
smplx_out = self.smpl_model( 
    global_orient=global_orient,
    body_pose=body_pose,
    jaw_pose=jaw_pose, 
    betas=betas,
    expression=expression,
    v_offsets=v_offsets,  #[V', 3]
    upsample=True, 
)
``` 

## Acknowledgments 
- [SMPLX](https://github.com/vchoutas/smplx). 
