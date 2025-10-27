import torch
import torch.nn as nn
import numpy as np

class UnderwaterPhysicsModel(nn.Module):
    """
    Simple model to simulate underwater image degradation.
    I = J * T + A * (1 - T)
    """
    def __init__(self):
        super(UnderwaterPhysicsModel, self).__init__()

    def forward(self, J, depth):
        """
        J: Clear image tensor (Batch, 3, H, W) range [0, 1]
        depth: A tensor of random depths (Batch, 1, 1, 1)
        """
        # 1. Set the Veiling Light (A)
        # This is the color of the "fog". We'll make it blue-green.
        # We add a little randomness to it.
        A_r = 0.1 + (torch.rand(1).item() * 0.1) # Red channel is weak
        A_g = 0.3 + (torch.rand(1).item() * 0.1) # Green channel
        A_b = 0.4 + (torch.rand(1).item() * 0.1) # Blue channel is strong
        A = torch.tensor([A_r, A_g, A_b]).view(1, 3, 1, 1).to(J.device)

        # 2. Create the Transmission Map (T)
        # This is how much light gets through. Deeper = less light.
        # beta is the "scattering coefficient"
        beta = 0.02 + (torch.rand(1).item() * 0.03) # Random fogginess
        T = torch.exp(-beta * depth)
        
        # 3. Apply the physics formula
        I = J * T + A * (1 - T)
        
        # Make sure values are still 0-1
        I = torch.clamp(I, 0.0, 1.0)
        
        return I, T, A