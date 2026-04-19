import numpy as np 
from rfnetwork import const, conv
import pyvista as pv
import matplotlib.pyplot as plt

import rfnetwork as rfn
import unittest
from pathlib import Path

DATA_DIR = Path(__file__).parent / "../data"

class TestMeshUtils(unittest.TestCase):

    def test_get_edges(self):

        # make output folder
        (DATA_DIR / ".outputs").mkdir(exist_ok=True)

        len_patch = conv.in_mm(30)
        w_patch = conv.in_mm(30)
        w_slot = conv.in_mm(1.55)
        len_slot = conv.in_mm(7.5)
        len_leg = conv.in_mm(4.5)

        # upper slot offset (h-polarized patch)
        x_pos_h = conv.in_mm(-9.1)
        # lower (v-polarized patch)
        y_pos_v = conv.in_mm(-9.1)

        # conductor plane size
        sub_x0, sub_x1, sub_y0, sub_y1 = (-w_patch * 0.8, w_patch * 0.8, -len_patch * 0.8, len_patch * 0.8)

        # conductor layer with slot
        gnd_plane = pv.Rectangle([(sub_x0, sub_y0, 0), (sub_x0, sub_y1, 0), (sub_x1, sub_y1, 0)])
        # cutout slots
        slot_h = (x_pos_h-w_slot/2, x_pos_h+  w_slot/2, -len_slot/2, len_slot/2, 0, 0)
        leg1_h = (x_pos_h-len_leg/2, x_pos_h+len_leg/2, -len_slot/2 - w_slot/2, -len_slot/2 + w_slot/2, 0, 0)
        leg2_h = (x_pos_h-len_leg/2, x_pos_h+len_leg/2, len_slot/2 - w_slot/2, len_slot/2 + w_slot/2, 0, 0)

        # # cutout slot for lower (V)
        slot_v = (-len_slot/2, len_slot/2, y_pos_v - w_slot/2, y_pos_v + w_slot/2, 0, 0)
        leg1_v = (-len_slot/2 - w_slot/2, -len_slot/2 + w_slot/2, y_pos_v - len_leg/2, y_pos_v + len_leg/2, 0, 0)
        leg2_v = (len_slot/2 - w_slot/2, len_slot/2 + w_slot/2, y_pos_v - len_leg/2, y_pos_v + len_leg/2, 0, 0)

        for cutout in (slot_h, leg1_h, leg2_h, slot_v, leg1_v, leg2_v):
            gnd_plane = gnd_plane.clip_box(cutout).extract_surface(algorithm="dataset_surface")

        # get all edges in surface
        obj_edges = rfn.utils_mesh.get_object_edges(gnd_plane, group_faces=False)
        exterior_edges = rfn.utils_mesh.remove_interior_edges(obj_edges)

        grouped_edges = rfn.utils_mesh.get_object_edges(gnd_plane, group_faces=True)

        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(cmap.N)]

        # plot faces with cycling colors and exterior edges in bold black line
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_aspect("equal")

        for i, face in enumerate(grouped_edges):
            vertices = np.unique(np.array(face).reshape(-1, 3), axis=0)
            ax.fill(vertices[:, 0], vertices[:, 1], color=colors[i % cmap.N], alpha=0.5)

        for e in exterior_edges:
            # if np.all(np.abs(e[:, 0] - 0.0787) < 0.001):
            ax.plot(e[:, 0], e[:, 1], linewidth=2.5, color="k")

        # check number of line segments composing the exterior edges. These are segments joined end to end
        # from each of the faces that have an exterior edge.
        self.assertEqual(len(exterior_edges), 113)

        # write image
        fig.savefig(DATA_DIR / ".outputs/test_get_edges_fig1.png")
        test_im = plt.imread(DATA_DIR / ".outputs/test_get_edges_fig1.png")

        # compare with regression image
        ref_im = plt.imread(DATA_DIR / "regression/test_get_edges_fig1.png")

        np.testing.assert_array_almost_equal(ref_im, test_im, decimal=2)


if __name__ == "__main__":
    unittest.main()