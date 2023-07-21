import numpy as np 
from src.surface_laplacian import SurfaceLaplacian, operators
from open3d.geometry import TriangleMesh
from half_edge import HalfEdgeModel 
import igl

sphere = TriangleMesh().create_sphere(radius=1, resolution=15)
model = HalfEdgeModel(sphere.vertices, sphere.triangles)

# igl matrices
igl_data = (np.asarray(model.vertices), np.asarray(model.triangles))
mm_bary_igl = igl.massmatrix(*igl_data, igl.MASSMATRIX_TYPE_BARYCENTRIC)
mm_mixed_igl = igl.massmatrix(*igl_data, igl.MASSMATRIX_TYPE_VORONOI)
sm_igl = igl.cotmatrix(*igl_data)

sl = SurfaceLaplacian(model)
mm_bary =  sl.mass(operators.bary)
mm_mixed =  sl.mass(operators.mixed)
sm_cotan =  sl.stiffness(operators.cotan)

assert(np.allclose(mm_bary_igl.todense(), mm_bary.todense()))
assert(np.allclose(mm_mixed_igl.todense(), mm_mixed.todense()))
assert(np.allclose(sm_igl.todense(), sm_cotan.todense()))

print('OK')