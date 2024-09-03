import numpy as np
import matplotlib.pyplot as plt 


def lbs_weights_to_color(lbs_weights):
    '''
    map lbs weights to rgb colors 
    
    Args: 
        lbs_weights: torch.tensor [N, J]
    
    Returns:
        colors: torch.tensor [N, 3]
    ''' 
    J = lbs_weights.shape[1] 
    J_color = plt.cm.rainbow(np.linspace(0, 1, num_colors))
    J_ids = lbs_weights.argmax(dim=1) 
    colors = J_color[max_idx]
    return colors
