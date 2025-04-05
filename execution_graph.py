"""Testing module for experimenting with VSA computational graphs as
execution graphs.
"""

import time

import numpy as np

from vsa_graph.graph.graph import Bind, Connection, Graph, Vertex
from vsa_graph.vsa.hrr import HRR


def main() -> None:
    dim = 100
    input_a = Vertex("input_a", HRR.normal(dim).data)
    input_b = Vertex("input_b", HRR.normal(dim).data)
    output = Vertex("output", np.zeros(dim))

    cgraph = Graph([
        Connection(
            edge=Bind(),
            inputs=[input_a, input_b],
            outputs=[output]
        )
    ])
    cgraph.display()

    start = time.time()
    cgraph.forward()
    end = time.time()
    print(f"Time for execution of the computational graph: {end - start} s")


if __name__ == "__main__":
    main()
