import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from scipy.spatial import Delaunay

'''
2D Implementation of the following the paper:
A New Concave Hull Algorithm and Concaveness Measure for n-dimensional Datasets (2012)
by JIN-SEO PARK AND SE-JONG OH

Implementation created by Karoline Heiwolt, July 2023. 
Disclaimer: this is not optimised for efficiency. It is a direct implementation of the paper, using minimal dependencies, and is intended for research/testing.
'''

def plot(points, edges=None):
        '''
        Plot a set of points and (optionally) connecting edges in 2D
        inputs: 
            - a numpay array of coordinates, shape(n,2)
            - a collections.deque of edges, shape(m,2), in which each pair of indices represents an edge between two points
        '''
        x_coords, y_coords = zip(*points)

        # Create a new plot and set the axis limits
        fig, ax = plt.subplots()
        ax.set_xlim([min(x_coords)-1, max(x_coords)+1])
        ax.set_ylim([min(y_coords)-1, max(y_coords)+1])

        # Plot the points
        ax.scatter(x_coords, y_coords)

        # Plot the edges
        if edges is not None:
            for edge in edges:
                x1, y1 = points[edge[0]]
                x2, y2 = points[edge[1]]
                ax.plot([x1, x2], [y1, y2], 'k-')

        # Show the plot
        plt.show()
        return

def dist_points_to_segment(points,e1,e2):
    '''
    Computes the shortest distance between each point in a set of input points to a line segment defined by two end points
    If the point lies within the orthogonal region bounded by the two end points of the line segment that is simply the minimum distance between point and line (orth_dist)
    If the point lies outside of the line segment, the shortest distance is the distance to the closest line end point or corner point (corner_dist).

    Inputs:
        - points: a numpy array of 2D coordinates, shape(n,2)
        - e1 and e2: the two points describing the line segment, each a numpy array of shape (2,)
    Returns:
        - distances: a numpy array of shape (n,) containing the shortest distance between each point and the line segment
    '''
    orth_dist = dist_points_to_line(points, e1, e2)
    corner_dist = np.min(np.hstack((np.linalg.norm(points-e1, axis=1).reshape(-1,1), np.linalg.norm(points-e2, axis=1).reshape(-1,1))), axis=1)
    t = - (np.matmul((e1-points), (e2-e1))/(np.dot((e2-e1),(e2-e1)))) # projection of point onto line segment, if between 0 and 1 then the point lies within the space orthogonal to the line segment
    dists = np.where((t >= 0) & (t <= 1), orth_dist, corner_dist)
    return dists


def dist_points_to_line(points, e1, e2):
    '''
    Computes the shortest distance between each point in a set of input points to a line (with no endpoints).
    Inputs:
        - points: a numpy array of 2D coordinates, shape(n,2)
        - e1 and e2: the two points through which the line can be defined, each a numpy array of shape (2,)
    Returns:
        - d: a numpy array of shape (n,) containing the shortest (orthogonal) distance between each point and the line
    '''
    d=abs(np.cross(e2-e1,points-e1)/np.linalg.norm(e2-e1))
    return d

def concave_hull_2d(points, N=1):
    '''
    Computes the concave hull of a set of points in 2D
    Inputs:
        - points: a numpy array of 2D coordinates, shape(n,2)
        - N: the threshold value which determines how tightly the hull is fit to the points.
    Returns:
        - kept_edges: a collections.deque of edges, shape(m,2), in which each pair of indices represents an edge between two points
    '''
    tri = Delaunay(point_set) # Compute the Delaunay triangulation of the points
    edges = tri.convex_hull # Find the edges of the convex hull
    edge_deque = deque([sorted(edge) for edge in edges])
    kept_edges = deque([])

    while len(edge_deque) > 0:
        #print(len(edge_deque))
        edge = edge_deque.popleft() # pop from the left side to examine each edge of the convex hull in turn

        # Find the directly neighbouring edges of the current edge
        neighbour_edges = []
        to_do = [pair for pair in edge_deque if edge[0] in pair or edge[1] in pair]
        kept = [pair for pair in kept_edges if edge[0] in pair or edge[1] in pair]
        [neighbour_edges.append(el) for el in to_do if len(el) > 0]
        [neighbour_edges.append(el) for el in kept if len(el) > 0]
        neighbour_points = np.setdiff1d(np.unique(neighbour_edges), np.unique(edge)) # these are points with which the two current edge vertices share another edge

        # Find the closest 'inner' (i.e. currently not used for any edges) point to the current edge
        used_points = np.union1d(np.unique(np.asarray(edge_deque, dtype=np.int32).flatten()), np.unique(np.asarray(kept_edges, dtype=np.int32).flatten()))
        used_points = np.union1d(used_points, np.array(edge))
        inner_points = np.setdiff1d(np.arange(len(points)), used_points) # points which are not yet part of any hull edges
        if len(inner_points) > 0:
            # Find the shortest distance for each inner point to any point on the current edge, which is a line SEGMENT
            e1 = points[edge[0]]
            e2 = points[edge[1]]
            d = dist_points_to_segment(points[inner_points], e1, e2)

            # Find the shortest distance for each inner point to any point on the neighbouring edges as well
            neighbour_dist_1 = dist_points_to_segment(points[inner_points], points[neighbour_edges[0][0]], points[neighbour_edges[0][1]])
            neighbour_dist_2 = dist_points_to_segment(points[inner_points], points[neighbour_edges[1][0]], points[neighbour_edges[1][1]])

            # as per Algorithm 1 in the paper, at this stage we ONLY check the closest point as a potential candidate for digging
            # this might be undesirabe in some cases. Depending on the order in which edges are examined, the concave hull will come out differently
            # If the closest candidate is closer to a neighbour, we should potentially reexamine this edge later once neighbouring edges have been 'dug' sufficiently.
            closest_to_farthest = np.argsort(d)
            p = closest_to_farthest[0]
            if d[p] <= neighbour_dist_1[p] and d[p] <= neighbour_dist_2[p]: # "[candidate point] should not closer to neighbor edges" (p. 5) 
                candidate_index = inner_points[p]
                candidate_coordinate = points[candidate_index]

                # If there is a fitting candidate point, evaluate against the N parameter
                edge_length = np.linalg.norm(points[edge])
                decision_distance = min(np.linalg.norm(candidate_coordinate - points[edge[0]]), np.linalg.norm(candidate_coordinate - points[edge[1]]))
                if edge_length/decision_distance > N: # decision on digging process
                    # dig!
                    edge_deque.append(sorted([candidate_index, edge[0]]))
                    edge_deque.append(sorted([candidate_index, edge[1]]))
                else:
                    # keep!
                    kept_edges.append(edge)
            else:
                # keep!
                kept_edges.append(edge)
        else:
            # keep!
            kept_edges.append(edge)
        # Might want to consider reexamining some of these edges later, but for now we just keep them as per Algorithm 1
    return kept_edges


if __name__ == "__main__":
    # To test the 2d concave hull algorithm
    # specify the 'threshold value' N
    N = 0.1
    # generate random 2d pointset
    point_set = np.random.rand(100,2)
    concave_hull = concave_hull_2d(point_set, N=N)
    # and plot the result
    plot(point_set, concave_hull)
