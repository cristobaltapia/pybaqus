from pybaqus import open_fil
import pyvista as pv

res = open_fil("model_results.fil")

mesh = res.get_deformed_mesh(step=1, inc=1, scale=3)
U1 = res.get_nodal_result(var="U1", step=1, inc=1)
mesh.point_arrays["U1"] = U1


plot = pv.Plotter()
plot.add_mesh(mesh, show_edges=True, color="white",scalars="U1", show_scalar_bar=True)
plot.view_xy()
plot.show()