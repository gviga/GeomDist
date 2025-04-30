import numpy as np
import torch
import trimesh

def normalize_mesh(mesh):
    rescale = max(mesh.extents) / 2.
    tform = [
        -(mesh.bounds[1][i] + mesh.bounds[0][i]) / 2.
        for i in range(3)
    ]
    matrix = np.eye(4)
    matrix[:3, 3] = tform
    mesh.apply_transform(matrix)
    matrix = np.eye(4)
    matrix[:3, :3] /= rescale
    mesh.apply_transform(matrix)
    
    return mesh


def sample_sphere_volume(radius, center, num_points,device='cuda'):
    center = torch.tensor(center, dtype=torch.float32,device=device)
    r = radius * torch.rand(num_points,device=device)    ** (1/3)   # Sample radius with proper scaling for uniform distribution
    theta = torch.rand(num_points,device=device)    * 2 * np.pi           # Azimuthal angle
    phi = torch.acos(1 - 2 * torch.rand(num_points,device=device)   )     # Polar angle

    # Convert to Cartesian coordinates
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)

    points = torch.stack((x, y, z), dim=1) + center

    return points


