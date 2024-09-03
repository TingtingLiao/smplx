import numpy as np
import matplotlib.pyplot as plt 


def lbs_weights_to_colors(lbs_weights, cmap='Pastel1'):
    '''
    map lbs weights to rgb colors 

    Args: 
        lbs_weights: torch.tensor [N, J]
        cmap: str, matplotlib colormap name
            find more choice from https://matplotlib.org/stable/users/explain/colors/colormaps.html  
    Returns:
        colors: torch.tensor [N, 3]
    ''' 
    J = lbs_weights.shape[1] 
    
    mapper = getattr(plt.cm, cmap)
    J_colors = mapper(np.linspace(0, 1, J))
    J_ids = lbs_weights.argmax(dim=1)  
    colors = J_colors[J_ids]  
    return colors
