
import os.path as osp 
import os 
from typing import Optional, Dict, Union

from .smpl import SMPL, SMPLLayer
from .smplh import SMPLH, SMPLHLayer
from .smplx import SMPLX, SMPLXLayer
from .flame import FLAME, FLAMELayer
from .mano import MANO, MANOLayer

def build_layer(
    model_path: str,
    model_type: str = 'smpl',
    **kwargs
) -> Union[SMPLLayer, SMPLHLayer, SMPLXLayer, MANOLayer, FLAMELayer]:
    ''' Method for creating a model from a path and a model type

        Parameters
        ----------
        model_path: str
            Either the path to the model you wish to load or a folder,
            where each subfolder contains the differents types, i.e.:
            model_path:
            |
            |-- smpl
                |-- SMPL_FEMALE
                |-- SMPL_NEUTRAL
                |-- SMPL_MALE
            |-- smplh
                |-- SMPLH_FEMALE
                |-- SMPLH_MALE
            |-- smplx
                |-- SMPLX_FEMALE
                |-- SMPLX_NEUTRAL
                |-- SMPLX_MALE
            |-- mano
                |-- MANO RIGHT
                |-- MANO LEFT
            |-- flame
                |-- FLAME_FEMALE
                |-- FLAME_MALE
                |-- FLAME_NEUTRAL

        model_type: str, optional
            When model_path is a folder, then this parameter specifies  the
            type of model to be loaded
        **kwargs: dict
            Keyword arguments

        Returns
        -------
            body_model: nn.Module
                The PyTorch module that implements the corresponding body model
        Raises
        ------
            ValueError: In case the model type is not one of SMPL, SMPLH,
            SMPLX, MANO or FLAME
    '''

    if osp.isdir(model_path):
        model_path = os.path.join(model_path, model_type)
    else:
        model_type = osp.basename(model_path).split('_')[0].lower()

    if model_type.lower() == 'smpl':
        return SMPLLayer(model_path, **kwargs)
    elif model_type.lower() == 'smplh':
        return SMPLHLayer(model_path, **kwargs)
    elif model_type.lower() == 'smplx':
        return SMPLXLayer(model_path, **kwargs)
    elif 'mano' in model_type.lower():
        return MANOLayer(model_path, **kwargs)
    elif 'flame' in model_type.lower():
        return FLAMELayer(model_path, **kwargs)
    else:
        raise ValueError(f'Unknown model type {model_type}, exiting!')


def create(
    model_path: str,
    model_type: str = 'smpl',
    **kwargs
) -> Union[SMPL, SMPLH, SMPLX, MANO, FLAME]:
    ''' Method for creating a model from a path and a model type

        Parameters
        ----------
        model_path: str
            Either the path to the model you wish to load or a folder,
            where each subfolder contains the differents types, i.e.:
            model_path:
            |
            |-- smpl
                |-- SMPL_FEMALE
                |-- SMPL_NEUTRAL
                |-- SMPL_MALE
            |-- smplh
                |-- SMPLH_FEMALE
                |-- SMPLH_MALE
            |-- smplx
                |-- SMPLX_FEMALE
                |-- SMPLX_NEUTRAL
                |-- SMPLX_MALE
            |-- mano
                |-- MANO RIGHT
                |-- MANO LEFT

        model_type: str, optional
            When model_path is a folder, then this parameter specifies  the
            type of model to be loaded
        **kwargs: dict
            Keyword arguments

        Returns
        -------
            body_model: nn.Module
                The PyTorch module that implements the corresponding body model
        Raises
        ------
            ValueError: In case the model type is not one of SMPL, SMPLH,
            SMPLX, MANO or FLAME
    '''

    # If it's a folder, assume
    if osp.isdir(model_path):
        model_path = os.path.join(model_path, model_type)
    else:
        model_type = osp.basename(model_path).split('_')[0].lower()

    if model_type.lower() == 'smpl':
        return SMPL(model_path, **kwargs)
    elif model_type.lower() == 'smplh':
        return SMPLH(model_path, **kwargs)
    elif model_type.lower() == 'smplx':
        return SMPLX(model_path, **kwargs)
    elif 'mano' in model_type.lower():
        return MANO(model_path, **kwargs)
    elif 'flame' in model_type.lower():
        return FLAME(model_path, **kwargs)
    else:
        raise ValueError(f'Unknown model type {model_type}, exiting!')
