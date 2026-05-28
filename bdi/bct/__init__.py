"""Bootstrap Consensus Tree (BCT) visualization.

Implementation of the Eder BCT algorithm for authorship verification, with
networkx/matplotlib visualization.

References:
    - Eder, M. (2017). Visualization in stylometry: cluster analysis using
      networks. Digital Scholarship in the Humanities, 32(1), 50-64.
    - https://computationalstylistics.github.io/projects/bootstrap-networks/
"""

from .algorithm import eder_bct
from .plot import plot_bct

__all__ = ["eder_bct", "plot_bct"]
