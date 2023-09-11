import numpy as np
import scipy as sp
from src.surface_laplacian import SurfaceLaplacian, operators
from half_edge import HalfEdgeModel 
from open3d.geometry import TriangleMesh
from geometric_tools import cartesian_to_spherical_coords
from geometric_plotter import Plotter
from isotropic_remesher import IsotropicRemesher
import pathlib 
filename = pathlib.Path(__file__).stem

path = pathlib.Path('data/mesh.npz')
if not path.exists():
    sphere = TriangleMesh().create_sphere(radius=1, resolution=15)
    model = HalfEdgeModel(sphere.vertices, sphere.triangles)
    remesher = IsotropicRemesher(model)
    remesher.isotropic_remeshing(
        .1, 
        iter=10, 
        explicit=False, 
        foldover=10,
        sliver=False
    )
    model.clean()
    np.savez(path, vertices=model.vertices, triangles=model.triangles)
else:
    npz = np.load(path)
    model = HalfEdgeModel(npz['vertices'], npz['triangles'])

spherical = cartesian_to_spherical_coords(model.vertices)
ρ, θ, φ = spherical[:,0], spherical[:,1], spherical[:,2]
f = np.cos(φ) # function with simple laplacian
lf = -2.*f # analytic laplacian of f
sl = SurfaceLaplacian(model)
M = sl.mass(operators.mixed).tocsc()
S = sl.stiffness(operators.cotan)
L = sp.sparse.linalg.inv(M) @ S

Plotter.set_export()

p = Plotter(False, figsize=(10,8))
p.add_trisurf(
    np.asarray(model.vertices),
    np.asarray(model.triangles),
    vertex_values=lf,
    vmin=lf.min(),
    vmax=lf.max(),
    colorbar=True
)
# estimation of laplacian with surface_laplacian
p.add_trisurf(
    np.asarray(model.vertices),
    np.asarray(model.triangles),
    vertex_values=L@f,
    vmin=lf.min(),
    vmax=lf.max(),
    translate=(3,0,0)
)
p.camera(view=(-90,0,0), zoom=1.),
p.save(folder='figs/', name=filename)

Plotter.show()