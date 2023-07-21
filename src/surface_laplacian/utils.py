from half_edge import HalfEdgeModel
from vector_tools import TriPoint
import numpy as np 

def angle_factor(model: HalfEdgeModel, h_index:int) -> float:
    t1 = TriPoint(model.get_triangle_vertices_by_edge(h_index))
    # a1 = t1.get_angle(degree=False)
    a1 = t1.get_cosecant() - t1.get_cotangent()

    n_index = model.get_next_index_on_ring(h_index, True)
    t2 = TriPoint(model.get_triangle_vertices_by_edge(n_index))
    # a2 = t2.get_angle(degree=False)
    a2 = t2.get_cosecant() - t2.get_cotangent()
    # return (1.-np.cos(a1))/np.sin(a1) + (1.-np.cos(a2))/np.sin(a2)
    return a1 + a2