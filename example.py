
import numpy as np
import hull2d as h2d
import hull3d as h3d

## 2D example
# specify the 'threshold value' N. "valid range of N falls in [0, 5]" (Park and Oh, 2012)
N = 3
# generate random 2d pointset
point_set = np.random.rand(100,2)
concave_hull = h2d.concave_hull_2d(point_set, tN=N)
# and plot the result
h2d.plot(point_set, concave_hull)



## 3D example
# specify the 'threshold value' N. "valid range of N falls in [0, 5]" (Park and Oh, 2012)
N = 3
# generate random 3d pointset
point_set = np.random.rand(50,3)
concave_hull = h3d.concave_hull_3d(point_set, tN=N)
# and plot the result
h3d.plot_3d(point_set, concave_hull)