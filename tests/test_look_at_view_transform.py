"""Tests comparing my_look_at_view_transform against pytorch3d's look_at_view_transform.

Run from the assignment1 root:  python -m unittest tests.test_look_at_view_transform -v
"""
import itertools
import unittest

import torch
from pytorch3d.renderer import look_at_view_transform

from starter.render_tetrahedron import my_look_at_view_transform

DISTS = [0.5, 1.0, 3.0, 10.0]
ELEVS = [-80, -30, 0, 15, 30, 60, 89]  # degrees
AZIMS = [-170, -90, -45, 0, 30, 45, 90, 135, 180, 270]  # degrees
GRID = list(itertools.product(DISTS, ELEVS, AZIMS))
ATOL = 1e-5
ORIGIN = ((0.0, 0.0, 0.0),)


class TestMyLookAtViewTransform(unittest.TestCase):
    """Checks R (1,3,3) and T (1,3) against the official implementation and basic geometry."""

    def test_matches_reference_on_grid(self):
        """R and T match pytorch3d for a grid of (dist, elev, azim) in degrees."""
        for d, e, a in GRID:
            with self.subTest(dist=d, elev=e, azim=a):
                R_ref, T_ref = look_at_view_transform(dist=d, elev=e, azim=a)
                R, T = my_look_at_view_transform(d, e, a, at=ORIGIN)
                self.assertTrue(torch.allclose(R, R_ref, atol=ATOL), f"R\n{R}\nvs\n{R_ref}")
                self.assertTrue(torch.allclose(T, T_ref, atol=ATOL), f"T {T} vs {T_ref}")

    def test_shape_and_dtype(self):
        """Outputs are float32 tensors with shapes (1,3,3) and (1,3), like the reference."""
        R, T = my_look_at_view_transform(3.0, 30, 45, at=ORIGIN)
        R_ref, T_ref = look_at_view_transform(dist=3.0, elev=30, azim=45)
        self.assertEqual(R.shape, R_ref.shape)
        self.assertEqual(T.shape, T_ref.shape)
        self.assertEqual(R.dtype, R_ref.dtype)
        self.assertEqual(T.dtype, T_ref.dtype)

    def test_R_is_rotation(self):
        """R is orthonormal with det +1 (a proper rotation, not a reflection)."""
        for d, e, a in GRID:
            with self.subTest(dist=d, elev=e, azim=a):
                R, _ = my_look_at_view_transform(d, e, a, at=ORIGIN)
                eye = torch.eye(3)[None]
                self.assertTrue(torch.allclose(R @ R.transpose(1, 2), eye, atol=ATOL))
                self.assertTrue(torch.allclose(torch.linalg.det(R), torch.ones(1), atol=ATOL))

    def test_T_is_dist_along_camera_z(self):
        """Camera looks at the origin, so the world origin sits at (0,0,dist) in camera coords."""
        for d, e, a in GRID:
            with self.subTest(dist=d, elev=e, azim=a):
                _, T = my_look_at_view_transform(d, e, a, at=ORIGIN)
                expected = torch.tensor([[0.0, 0.0, d]])
                self.assertTrue(torch.allclose(T, expected, atol=ATOL))

    def test_camera_center_matches_spherical_formula(self):
        """C = -T @ R^T equals d*(cos e sin a, sin e, cos e cos a) (Y-up, azim from +Z toward +X)."""
        for d, e, a in GRID:
            with self.subTest(dist=d, elev=e, azim=a):
                R, T = my_look_at_view_transform(d, e, a, at=ORIGIN)
                C = (-T[:, None, :] @ R.transpose(1, 2))[:, 0]  # (1,3)
                er, ar = torch.deg2rad(torch.tensor(float(e))), torch.deg2rad(torch.tensor(float(a)))
                expected = d * torch.stack(
                    [torch.cos(er) * torch.sin(ar), torch.sin(er), torch.cos(er) * torch.cos(ar)]
                )[None]
                self.assertTrue(torch.allclose(C, expected, atol=ATOL))

    def test_known_poses(self):
        """Hand-checkable cases: azim=0/elev=0 is Ry(pi); azim=90 puts camera on +X looking along -X."""
        R, T = my_look_at_view_transform(2.0, 0, 0, at=ORIGIN)
        Ry_pi = torch.tensor([[-1.0, 0, 0], [0, 1, 0], [0, 0, -1]])[None]
        self.assertTrue(torch.allclose(R, Ry_pi, atol=ATOL))
        self.assertTrue(torch.allclose(T, torch.tensor([[0.0, 0.0, 2.0]]), atol=ATOL))

        R, _ = my_look_at_view_transform(2.0, 0, 90, at=ORIGIN)
        forward_world = R[0, :, 2]  # 3rd column of Rwc = camera +Z axis in world coords
        self.assertTrue(torch.allclose(forward_world, torch.tensor([-1.0, 0.0, 0.0]), atol=ATOL))

    def test_matches_reference_with_nonzero_at(self):
        """R and T match pytorch3d when looking at a point other than the origin."""
        at = ((0.25, 0.25, 0.25),)
        for d, e, a in GRID:
            with self.subTest(dist=d, elev=e, azim=a):
                R_ref, T_ref = look_at_view_transform(dist=d, elev=e, azim=a, at=at)
                R, T = my_look_at_view_transform(d, e, a, at=at)
                self.assertTrue(torch.allclose(R, R_ref, atol=ATOL))
                self.assertTrue(torch.allclose(T, T_ref, atol=ATOL))

    def test_degrees_not_radians(self):
        """Angles are degrees: azim=45 must differ from the result for 45 radians."""
        R45, _ = my_look_at_view_transform(3.0, 0, 45, at=ORIGIN)
        R45rad, _ = my_look_at_view_transform(3.0, 0, 45 * 180 / torch.pi, at=ORIGIN)
        self.assertFalse(torch.allclose(R45, R45rad, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
