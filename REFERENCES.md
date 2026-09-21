# References

`netvoronoi` builds on established work on spatial analysis along networks and
network Voronoi diagrams. The additive-epsilon overlap rule implemented here is
presented as a practical tolerance extension of nearest-site network membership;
the project does not claim that ordinary network Voronoi diagrams are new.

## Network Voronoi background

- Okabe, A. & Sugihara, K. (2012). **Network Voronoi Diagrams.** In *Spatial
  Analysis Along Networks: Statistical and Computational Methods*, Chapter 4.
  Wiley. DOI: 10.1002/9781119967101.ch4.

- Okabe, A., Okunuki, K.-i. & Shiode, S. (2006). **SANET: A Toolbox for Spatial
  Analysis on a Network.** *Geographical Analysis*, 38(1), 57–66.
  DOI: 10.1111/j.0016-7363.2005.00674.x.

- Wang, M.-T. (2016). **Nearest neighbor query processing using the network
  Voronoi diagram.** *Data & Knowledge Engineering*, 103, 19–43.
  DOI: 10.1016/j.datak.2016.02.003.

## Python network backend

- Gaboardi, J., Rey, S. & Lumnitz, S. (2021). **spaghetti: spatial network
  analysis in PySAL.** *Journal of Open Source Software*, 6(62), 2826.
  DOI: 10.21105/joss.02826.

`netvoronoi` delegates spatial-network extraction and point-to-network snapping to
PySAL `spaghetti`, then uses SciPy sparse shortest-path routines and Shapely /
GeoPandas for Voronoi-specific calculations and geospatial output.
