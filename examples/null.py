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

data = (np.asarray(model.vertices), np.asarray(model.triangles))
spherical = cartesian_to_spherical_coords(model.vertices)
ρ, θ, φ = spherical[:,0], spherical[:,1], spherical[:,2]
f = θ # function with simple laplacian
sl = SurfaceLaplacian(model)
M = sl.mass(operators.mixed).tocsc()
S = sl.stiffness(operators.cotan)
L = sp.sparse.linalg.inv(M) @ S
lf = L@f
vmin, vmax = lf.min(), lf.max()
# Plotter.set_export()

p = Plotter(False, figsize=(5,5))
p.add_trisurf(
    *data,
    vertex_values=lf,
    vmin=vmin,
    vmax=vmax,
    colorbar=True
)
p.camera(view=(-90,0,0), zoom=1.6),
p.save(folder='figs/', name=f"{filename}")

Plotter.show()