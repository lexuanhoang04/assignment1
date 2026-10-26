"""
5.1: Build point clouds from 2 RGB-D views (and their union) and render a 360 gif.

Usage:
    python problems/render_point_coud_rgbd.py
    python -m problems.render_point_coud_rgbd
"""
import sys
from pathlib import Path
import pytorch3d
from pytorch3d.structures import join_pointclouds_as_scene

# Make the repo root importable so `starter` resolves when run as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starter.render_generic import load_rgbd_views
from starter.utils import unproject_depth_image, get_points_renderer, get_device
from starter.render_tetrahedron import create_R_T

# views[i]: rgb (S,S,3), mask (S,S), depth (S,S), camera (PerspectiveCameras, CPU)
views = load_rgbd_views()
view1, view2 = views

rgb1, mask1, depth1, camera1 = view1["rgb"], view1["mask"], view1["depth"], view1["camera"]
rgb2, mask2, depth2, camera2 = view2["rgb"], view2["mask"], view2["depth"], view2["camera"]

print('shape of rgb1', rgb1.shape)
print('mask1', mask1.shape)
print('depth1', depth1.shape)
#
# unproject

points, rgba = unproject_depth_image(rgb1, mask1, depth1, camera1)

print('points shape', points.shape)
print('rgba', rgba.shape)

point_cloud = pytorch3d.structures.Pointclouds(
    points=[points], features=[rgba]
)

device = get_device()

# initialize camera
Rs, Ts = create_R_T(elev=30, dist=3, num_frame=60)                        # (N,3,3), (N,3)
cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=Rs, T=Ts, fov=60, device=device)
lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

renderer = get_points_renderer()
point_cloud = point_cloud.extend(60)               # 1 → num_frame clouds

rend = renderer(point_cloud, cameras=cameras, lights=lights)             # (N, H, W, 4)

output_path = "images/pc1.png"
os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
imageio.mimsave(output_path, list(frames), duration=duration, loop=0)