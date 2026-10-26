"""
Usage:
    python -m starter.camera_transforms --image_size 512
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer

import numpy as np 

def render_textured_cow(
    cow_path="data/cow.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative
    renderer = get_mesh_renderer(image_size=256)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="images/textured_cow4.jpg")
    args = parser.parse_args()
    # R_relative = Rcc1 = Rco @ Roc1 = Roc1
    # T_rel = tc1w - R_rel @ tc0w
    # Roc1 = np.array([
    #     [0, 1, 0],
    #     [-1, 0, 0],
    #     [0, 0, 1]
    # ] )

    Roc1 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ] )


    R_relative = Roc1 
    
    tc0w =np.array([0,0,3])
    tc1w = np.array([0.5,0,3])
    T_relative = tc1w - R_relative @ tc0w 

    image = render_textured_cow(cow_path=args.cow_path, image_size=args.image_size, R_relative=R_relative, T_relative=T_relative)
    plt.imsave(args.output_path, image)
