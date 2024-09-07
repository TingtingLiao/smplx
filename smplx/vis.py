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
    J_colors = mapper(np.array(range(J)))[:, :3]  # (J, 3)
    colors = lbs_weights @ J_colors # (N, 3)
    return colors



if __name__ == '__main__':
    lbs_weights = np.random.rand(100, 5)
    colors = lbs_weights_to_colors(lbs_weights)
    print(colors.shape) # (100, 3)
    
    