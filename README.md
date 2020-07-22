# Pybaqus

Pybaqus is a python library to imports the output files generated by [Abaqus][1] in the ASCII `*.fil` format, and creates a [VTK][2] object from the data.
The results can then be analyzed in pure python (i.e. no Abaqus licence needed) with the great tools provided by [PyVista][3].

# Features

Pybaqus is in a very early development stage.
Therefore, there are still many unimplemented functionalities.
However, basic operations like importing the mesh and the nodal and element results is implemeneted and can be used for some analysis.

The following feeatures are either already implemented or planned:

- [x] Import 2D meshes
- [ ] Import 3D meshes _(should work, but have not tested it)_
- [x] Import nodal results
- [x] Import element results
- [x] Element and node sets
- [x] Extrapolate element results to nodes _(implemented for some elements)_
- [ ] Interpolate element/nodal results to any point within the element considering the shape functions of the element
- [ ] Import history output
- [ ] Extrapolate element results from Gaussian points to element's nodes
- [x] Compute stresses along paths
- [ ] Compute section forces and moments
- [ ] Implement functions to easily create animations

# Installation

```
pip install git+https://github.com/cristobaltapia/pybaqus
```

# Quick-start

The first thing you need is to tell Abaqus that you want am ASCII `*.fil` result file.
To get that you need to write the following lines in your `*.inp` file, right after the step definition (`*End Step`):

```
...
*End Step
*FILE FORMAT, ASCII
*EL FILE
*S, E, COORD
*NODE FILE
*COORD, U
```

You can specify different output variables (as long as they are available for the elements you are using, of course).
After submitting your model you will get a `*.fil` file.
This is the file you need to import it with Pybaqus.

Import the `*.fil` file like this:

```python
from pybaqus import open_fil

res = open_fil("your_result.fil")

```

Great!
That was it. :)

Now you have your results as a VTK object, wrapped by PyVista, and there's nothing that can get in your way to analyze your results with pure python.

### Plot the mesh

```python
import pyvista as pv

mesh = res.get_mesh()


plot = pv.Plotter()
plot.add_mesh(mesh, show_edges=True, color="white")
plot.view_xy()
plot.show()
```

![Mesh](examples/mesh_hole.png)

Cool! But something's missing there. Colors!
We can plot some of our results like this:

```python
mesh = res.get_deformed_mesh(step=1, inc=1, scale=3)
s2 = res.get_nodal_result(var="S2", step=1, inc=1)
mesh.point_arrays["S2"] = s2

plot = pv.Plotter()
p.add_mesh(mesh, show_edges=True, color="white",
           scalars="S2", show_scalar_bar=True
plot.view_xy()
plot.show()
```

![Mesh](examples/mesh_results.png)

That's it!
Since the API is still under development, some of these functions might change.

And now you can de whatever you want with your results in python.
Have fun!

[1]: https://www.3ds.com/products-services/simulia/products/abaqus/
[2]: https://vtk.org/
[3]: https://www.pyvista.org/
