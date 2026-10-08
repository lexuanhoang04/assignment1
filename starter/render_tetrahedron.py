import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer,
def my_look_at_view_transform(dist, elev, azim):
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
    z = x = rsin(azim)cos(pi/2-elev) (because normal x is now z in PyTorch3D/OpenGL convention)
    x = y = rsin(azim)sin(pi/2-elev)
    y = z = rcos(azim)
    r = dist

    so we have twc, and we can finish up
    """
def create_R_T():
    pass
def create_tetrahedron():
    verts = torch.tensor([[0,0,0],
                        [0,1,0],
                        [1,0,0],
                        [0,0,1]])
    
    faces = torch.tensor([[0,1,3],
                        [1,2,3],
                        [0,1,2],
                        [0,2,3]])
    
    mesh = pytorch3d.structures.Meshes(
        verts = verts,
        faces = faces
    )

    return mesh

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/cow_render.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    image = render_cow(cow_path=args.cow_path, image_size=args.image_size)
    plt.imsave(args.output_path, image)
