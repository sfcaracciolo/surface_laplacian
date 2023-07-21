from half_edge import HalfEdgeModel
from .utils import *

def rard(model: HalfEdgeModel, h_index: int) -> float:
    """Regular angles and regular distances weights"""
    ri = np.mean(list(map(lambda b: b.get_distance(), model.edge_ring(h_index))))
    M = model.valence(h_index)
    return 4./(M*ri**2)

def raid(model: HalfEdgeModel, h_index: int) -> float:
    """Regular angles and irregular distances weights"""
    ri = np.mean(list(map(lambda b: b.get_distance(), model.edge_ring(h_index))))
    rik = model.edge_len(h_index)
    M = model.valence(h_index)
    return 4./(M*ri*rik)

def iaid(model: HalfEdgeModel, h_index: int) -> float:
    """Irregular angles and irregular distances weights"""
    ri = np.mean(list(map(lambda b: b.get_distance(), model.edge_ring(h_index))))
    rik = model.edge_len(h_index)
    t = angle_factor(model, h_index)
    T = map(lambda h: angle_factor(model, h), model.one_ring(h_index))
    return 4.*t/(ri*rik*sum(T))

def cotan(model: HalfEdgeModel, h_index: int) -> float:
    p_index = model.get_next_index_on_ring(h_index, False)
    pt_index = model.get_twin_index(p_index)
    t1 = TriPoint(model.get_triangle_vertices_by_edge(pt_index))
    cot1 = t1.get_cotangent()

    n_index = model.get_next_index_on_ring(h_index, True)
    nn_index = model.get_next_index(n_index)
    t2 = TriPoint(model.get_triangle_vertices_by_edge(nn_index))
    cot2 = t2.get_cotangent()

    return (cot1 + cot2)/2.

def bary(model: HalfEdgeModel, h_index: int) -> float:
    return sum(map(lambda t: t.get_barycentric_region(), model.triangle_ring(h_index)))

def voro(model: HalfEdgeModel, h_index: int) -> float:
    return sum(map(lambda t: t.get_voronoi_region(), model.triangle_ring(h_index)))

def mixed(model: HalfEdgeModel, h_index: int) -> float:
    return sum(map(lambda t: t.get_mixed_voronoi_region(), model.triangle_ring(h_index)))
