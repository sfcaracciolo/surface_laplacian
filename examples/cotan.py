import numpy as np
import scipy as sp
from src.surface_laplacian import SurfaceLaplacian, operators
from half_edge import HalfEdgeModel 
from open3d.geometry import TriangleMesh
from geometric_tools import cartesian_to_spherical_coords
import geometric_plotter
from isotropic_remesher import IsotropicRemesher
import pathlib 
import robust_laplacian 

sphere = TriangleMesh().create_sphere(radius=1, resolution=15)
model = HalfEdgeModel(sphere.vertices, sphere.triangles)

remesher = IsotropicRemesher(model)
remesher.isotropic_remeshing(
    .2, 
    iter=5, 
    explicit=False, 
    foldover=10,
    sliver=False
)
model.clean()

spherical = cartesian_to_spherical_coords(model.vertices)
ρ, θ, φ = spherical[:,0], spherical[:,1], spherical[:,2]
f = np.cos(φ) # function with simple laplacian
lf = -2.*f # analytic laplacian of f

sl = SurfaceLaplacian(model)
triangle_mesh = np.asarray(model.vertices), np.asarray(model.triangles)
vmin, vmax = lf.min(), lf.max()
ax_config = ((50,-150,0), 1.)

geometric_plotter.set_export()

# analytic laplacian plotting
ax = geometric_plotter.figure(figsize=(5,5))
geometric_plotter.plot_trisurf(ax, *triangle_mesh, vertex_colors=lf, vmin=vmin, vmax=vmax)
geometric_plotter.config_ax(ax, *ax_config),

geometric_plotter.execute(folder='E:\Repositorios\surface_laplacian\export\\', name=pathlib.Path(__file__).stem + '_1')


# estimation of laplacian with robust_laplacian

L, M = robust_laplacian.mesh_laplacian(*triangle_mesh)
ax = geometric_plotter.figure(figsize=(5,5))
estimation_rl = sp.sparse.linalg.inv(M) @ (-L) @ f
geometric_plotter.plot_trisurf(ax, *triangle_mesh, vertex_colors=estimation_rl, vmin=vmin, vmax=vmax)
geometric_plotter.config_ax(ax, *ax_config),

geometric_plotter.execute(folder='E:\Repositorios\surface_laplacian\export\\', name=pathlib.Path(__file__).stem + '_2')

# estimation of laplacian with surface_laplacian

M = sl.mass(operators.mixed).tocsc()
L = sl.stiffness(operators.cotan)
estimation_sl = sp.sparse.linalg.inv(M) @ L @ f
ax = geometric_plotter.figure(figsize=(5,5))
geometric_plotter.plot_trisurf(ax, *triangle_mesh, vertex_colors=estimation_sl, vmin=vmin, vmax=vmax)
geometric_plotter.config_ax(ax, *ax_config),

geometric_plotter.execute(folder='E:\Repositorios\surface_laplacian\export\\', name=pathlib.Path(__file__).stem+ '_3')


