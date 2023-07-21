from half_edge import HalfEdgeModel
import scipy as sp
import numpy as np

class SurfaceLaplacian:

    def __init__(self, model: HalfEdgeModel) -> None:
        self.model = model 

    def loop(func):

        def matrix_builder(self, w_fun):

            E = self.model.amount_of_half_edges()
            V = self.model.amount_of_vertices(clean=True)
            skip = []

            matrix = sp.sparse.dok_array((V, V), dtype=np.float32)

            for h_index in range(E):

                if h_index in self.model.unreferenced_half_edges:
                    continue
                
                v_index = self.model.get_start_vertex_index(h_index)

                if v_index in skip:
                    continue 

                if v_index in self.model.unreferenced_vertices:
                    continue
                
                func(self, w_fun, matrix, h_index, v_index)

                skip.append(v_index)

            return matrix
        
        return matrix_builder

    @loop
    def mass(self, w_fun, *args) -> sp.sparse.sparray:
        matrix, h_index, v_index = args
        matrix[v_index, v_index] = w_fun(self.model, h_index)

    @loop
    def stiffness(self, w_fun, *args) -> sp.sparse.sparray:
        matrix, h_index, v_index = args
        for h in self.model.one_ring(h_index):
            v = self.model.get_end_vertex_index(h)
            matrix[v_index, v] = w_fun(self.model, h)
            matrix[v_index, v_index] -= matrix[v_index, v]


