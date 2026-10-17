import argparse
import os
from datetime import datetime

import imageio
import pytorch3d
import torch

import math as m
import numpy as np 

from starter.utils import get_device, get_mesh_renderer

from pytorch3d.renderer import look_at_view_transform

def my_look_at_view_transform(dist, elev, azim, at=((0.25, 0.25, 0.25),)):
    """
    Reproduction of official look_at_view_transform

    So we need Rcw, but I think Rwc is easier to find, we just need to figure out how to rotate world axes to
    cam axes. Here are my thoughts

    - First rotate around y axis an angle azim 
    - Then, rotate around x axis an angle -elev
    - Then, rotate around y axis an angle pi

    Therefore, Rwc = Ry(azim) @ Rx(-elev) @ Ry(pi)
    
    So, that should be good for R
    Now T = tcw = - Rcw @ twc
    We will calculate twc by using spherical to cartesian coord
    Camera offset from the look-at point `at` (Y-up world, angles in degrees, r = dist):
    x = r*cos(elev)*sin(azim)
    y = r*sin(elev)
    z = r*cos(elev)*cos(azim)
    twc = at + [x, y, z]   (R is unaffected by `at`: only the viewing direction matters)

    so we have twc, and we can finish up
    """
    elev, azim = np.deg2rad(elev), np.deg2rad(azim)
    at = np.asarray(at, dtype=np.float64).reshape(3)

    Rwc = Ry(azim) @ Rx(-elev) @ Ry(m.pi)
    x = dist * np.sin(azim) * np.sin(m.pi/2 - elev)
    y = dist * np.sin(elev)
    z = dist * np.cos(azim) * np.sin(m.pi/2 - elev)

    twc = at + np.array([x, y, z])

    tcw = -Rwc.T @ np.array(twc)

    R = torch.tensor(Rwc, dtype=torch.float32)[None]
    T = torch.tensor(tcw, dtype=torch.float32)[None]

    return R, T

def Rx(angle_rad: float) -> np.ndarray:
    """Returns 3x3 rotation matrix for rotation around X-axis by angle_rad."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c]
    ])

def Ry(angle_rad: float) -> np.ndarray:
    """Returns 3x3 rotation matrix for rotation around Y-axis by angle_rad."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

def Rz(angle_rad: float) -> np.ndarray:
    """Returns 3x3 rotation matrix for rotation around Z-axis by angle_rad."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    
def create_R_T(elev, dist, num_frame, at=((0.25, 0.25, 0.25),)):
    """Returns R (N,3,3) and T (N,3) for N cameras orbiting `at` at fixed elev/dist (degrees)."""
    azims = torch.linspace(0, 360, num_frame + 1)[:-1]  # (N,) degrees, 360 == 0 so dropped

    Rs, Ts = look_at_view_transform(dist=dist, elev=elev, azim=azims, at=at)

    return Rs, Ts

def create_tetrahedron():
    verts = [torch.tensor([[0,0,0],[0,1,0],[1,0,0],[0,0,1]], dtype=torch.float32)]   # list of 1 × (4,3)
    faces = [torch.tensor([[0,3,1],[1,3,2],[0,1,2],[0,2,3]], dtype=torch.int64)]     # list of 1 × (4,3)
    colors = torch.tensor([[1,1,1],[0,1,0],[1,0,0],[0,0,1]], dtype=torch.float32)  # (4,3) RGB
    textures = pytorch3d.renderer.TexturesVertex(verts_features=colors[None])      # (1,4,3)

    mesh = pytorch3d.structures.Meshes(
        verts = verts,
        faces = faces,
        textures=textures
    )

    return mesh

def render_tetrahedron_360(elev=30, dist=3, num_frame=60, image_size=256, duration=50, output_path=None, device=None):
    """Renders a 360-degree orbit of the tetrahedron and saves it as a gif; returns the path."""
    if device is None:
        device = get_device()
    if output_path is None:
        output_path = f"images/tetrahedron_360_{datetime.now():%Y%m%d_%H%M%S}.gif"

    mesh = create_tetrahedron().to(device).extend(num_frame)          # batch of num_frame meshes
    Rs, Ts = create_R_T(elev, dist, num_frame)                        # (N,3,3), (N,3)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=Rs, T=Ts, fov=60, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    renderer = get_mesh_renderer(image_size=image_size, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)             # (N, H, W, 4)
    frames = (rend[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # (N, H, W, 3)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    imageio.mimsave(output_path, list(frames), duration=duration, loop=0)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_frame", type=int, default=60)
    parser.add_argument("--elev", type=float, default=30)
    parser.add_argument("--dist", type=float, default=3)
    parser.add_argument("--duration", type=int, default=50, help="ms per frame")
    args = parser.parse_args()

    path = render_tetrahedron_360(
        elev=args.elev, dist=args.dist, num_frame=args.num_frame,
        image_size=args.image_size, duration=args.duration, output_path=args.output_path,
    )
    print(f"saved {path}")
