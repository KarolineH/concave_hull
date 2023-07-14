from scipy.spatial import Delaunay
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib import cm

'''
3D Implementation of the following the paper:
A New Concave Hull Algorithm and Concaveness Measure for n-dimensional Datasets (2012)
by JIN-SEO PARK AND SE-JONG OH

Implementation created by Karoline Heiwolt, July 2023. 
Disclaimer: this is not optimised for efficiency. It is a direct implementation of the paper, using minimal dependencies, and is intended for research/testing.
'''

def plot_3d(points, triangles=None):
    '''
    Plot a set of points and (optionally) connecting faces (triangles) in 3D
    inputs: 
        - a numpay array of coordinates, shape(n,3)
        - a collections.deque of faces, shape(m,3), in which each triplet of indices represents a triangle surface between three points
    '''

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if triangles is None:
        ax.scatter(points[:,0], points[:,1], points[:,2]) # plot points only if no faces are given
    else:
        ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=triangles, cmap=cm.jet, linewidth=0.2) # plot points and faces as a triangle mesh surface
    plt.show()
    return

def dist_points_to_surface(points,t1,t2,t3):
    '''
    Computes the shortest distance between each point in a set of input points to a surface defined by three corner points
    If the point lies 'above' the surface, withint the orthogonal region bounded by the three corner points of the triangle, that is simply the minimum distance between point and plane (orth_dist)
    If the point lies outside of the triangle, the shortest distance is the distance to the closest corner (corner_dist).

    Inputs:
        - points: a numpy array of 3D coordinates, shape(n,3)
        - t1,t2,t3: the three corners describing the traingle surface, each a numpy array of shape (3,)
    Returns:
        - distances: a numpy array of shape (n,) containing the shortest distance between each point and the triangle surface
    '''
    # first get all we need for the surface equation
    # for coordinate format: ax + by + cz = d (or n1x + n2y + n3z = d)
    # vectors in the plane
    v1 = t3 - t1
    v2 = t2 - t1
    # normal vector to the plane
    normal = np.cross(v1,v2)
    normal = normal/np.linalg.norm(normal)
    # now insert one of the three corner coordinates into the plane equation to get d (doesn't matter which one)
    d = np.dot(normal, t1)

    # the line that goes through the point in question and through its orthogonal projection onto the plane is made up just of the point and the plane's normal vector
    # we can insert this line into the plane equation to get the orthogonal distance from the point to the plane (without bounds)
    scalars = d - np.dot(points, normal)

    # plug the scalar back into the line equation to get the orthogonal projection of the point onto the plane
    projections = points + scalars.reshape(-1,1) * normal.reshape(1,-1)

    # now check if those projections are WITHIN the bounds of the triangle
    within = check_point_in_triangle(projections, t1, t2, t3)

    # if within, assign the orthogonal distance, else assign the distance to the closest corner point
    orth_dist = abs(np.dot(points-t1, normal)) # orthogonal distance from point to surface
    corner_dist = np.min(np.hstack((np.hstack((np.linalg.norm(points-t1, axis=1).reshape(-1,1), np.linalg.norm(points-t2, axis=1).reshape(-1,1))), np.linalg.norm(points-t2, axis=1).reshape(-1,1))), axis=1)
    distances = np.where(within, orth_dist, corner_dist)
    return distances


def check_point_in_triangle(points, t1, t2, t3):
    '''
    Checks if each point in an array of candidate points lie within the region orthogonal to a triangle, bounded by the three corner points.
    Inputs:
        - points: a numpy array of 3D coordinates, shape(n,3) which lie in the same plane as the triangle
        - t1,t2,t3: the three corners describing the traingle surface, each a numpy array of shape (3,)
    Returns:
        - result: a numpy array of bools, shape (n,), where True indicates that the point lies within the triangle, and False indicates that it lies outside
    '''
        
    # find the vectors connecting the candidate points to all three corner points (unit vectors)
    v1 = (t1 - points)
    uv1 = v1 / np.linalg.norm(v1, axis=1)[:,None]
    v2 = (t2 - points)
    uv2 = v2 / np.linalg.norm(v2, axis=1)[:,None]
    v3 = (t3 - points)
    uv3 = v3 / np.linalg.norm(v3, axis=1)[:,None]

    # find the sum of angles between each triplet of vectors
    # if the sum is == 2pi, the point is within the triangle
    angle_sum = np.asarray([np.arccos(np.clip(np.dot(uv1[point,:], uv2[point,:]), -1.0, 1.0)) + np.arccos(np.clip(np.dot(uv2[point,:], uv3[point,:]), -1.0, 1.0)) + np.arccos(np.clip(np.dot(uv3[point,:], uv1[point,:]), -1.0, 1.0)) for point in range(len(points))])
    result = np.where(angle_sum == 2*np.pi, True, False)
    return result


def concave_hull_3d(points, tN=1):
    '''
    Computes the concave hull of a set of points in 3D
    Inputs:
        - points: a numpy array of 3D coordinates, shape(n,3)
        - N: the threshold value which determines how tightly the hull is fit to the points. "valid range of N falls in [0, 5]" (Park and Oh, 2012)
    Returns:
        - kept_triangles: a collections.deque of triangles, shape(m,3), in which each triplet of indices represents a triangle surface between three points
    '''
    tri = Delaunay(points) # Compute the Delaunay triangulation of the points in 3D
    triangles = tri.convex_hull # Find the triangles of the convex hull
    triangle_deque = deque([sorted(triangle) for triangle in triangles])
    kept_triangles = deque([])

    while len(triangle_deque) > 0:
        #print(len(triangle_deque))
        triangle = triangle_deque.popleft() # pop from the left side to iterate through the triangles one by one

        # Find the neighbouring surfaces of the current triangle
        neighbour_tri = []
        to_do = [triple for triple in triangle_deque if triangle[0] in triple or triangle[1] in triple or triangle[2] in triple]
        kept = [triple for triple in kept_triangles if triangle[0] in triple or triangle[1] in triple or triangle[2] in triple]
        [neighbour_tri.append(el) for el in to_do if len(el) > 0]
        [neighbour_tri.append(el) for el in kept if len(el) > 0]
        #neighbour_points = np.setdiff1d(np.unique(neighbour_tri), np.unique(triangle)) # points with which the three current triangle vertices share another edge/surface

        # Find the closest 'inner' point to the current triangle, these are points currently not used for any hull triangles
        used_points = np.union1d(np.unique(np.asarray(triangle_deque, dtype=np.int32).flatten()), np.unique(np.asarray(kept_triangles, dtype=np.int32).flatten()))
        used_points = np.union1d(used_points, np.array(triangle))
        inner_points = np.setdiff1d(np.arange(len(points)), used_points) # points which are not yet part of any hull edges

        if len(inner_points) > 0:
            # Find the shortest distance for each inner point to any point on the current triangle surface, which is a plane SEGMENT not just a plane extending in space
            t1 = points[triangle[0]]
            t2 = points[triangle[1]]
            t3 = points[triangle[2]]
            
            # find the shortest distance between points and surface
            d = dist_points_to_surface(points[inner_points], t1, t2, t3)
            neighbour_dists = np.asarray([dist_points_to_surface(points[inner_points], points[neighbour_tri[i][0]], points[neighbour_tri[i][1]], points[neighbour_tri[i][2]]) for i in range(len(neighbour_tri))])

            # then compare. If the closest point is closer to the current triangle than to any of the neighbouring triangles, consider it a candidate for digging.
            closest_to_farthest = np.argsort(d)
            p = closest_to_farthest[0]

            # as per Algorithm 2 in the paper, at this stage we ONLY check the closest point as a potential candidate for digging
            # this might be undesirabe in some cases. Depending on the order in which surfaces are examined, the concave hull will come out differently
            # If the closest candidate is closer to a neighbour, we should potentially reexamine this triangle later once neighbouring triangles have been 'dug' sufficiently.

            if d[p] < np.min(neighbour_dists[:,p]):
                candidate_index = inner_points[p]
                candidate_coordinate = points[candidate_index]

                # If there is a fitting candidate point, evaluate against the N parameter
                edge_length =  (np.linalg.norm(t1-t2) + np.linalg.norm(t2-t3) + np.linalg.norm(t3-t1)) / 3 # circumference of triangle divided by 3, or average edge length
                decision_distance = min(np.linalg.norm(candidate_coordinate - points[triangle], axis = 1))

                if edge_length/decision_distance > tN:
                    # dig!
                    triangle_deque.append(sorted([candidate_index, triangle[0], triangle[1]]))
                    triangle_deque.append(sorted([candidate_index, triangle[0], triangle[2]]))
                    triangle_deque.append(sorted([candidate_index, triangle[1], triangle[2]]))

                else:
                    # keep!
                    kept_triangles.append(triangle)
            else:
                # keep!
                kept_triangles.append(triangle)
        else:
            # keep!
            kept_triangles.append(triangle)
        # Might want to consider reexamining some of these edges later, but for now we just keep them as per Algorithm 2
    return kept_triangles

if __name__ == "__main__":
    # To test the 3d concave hull algorithm
    # specify the 'threshold value' N
    threshold_N = 3
    # generate random 3d pointset
    point_set = np.random.rand(50,3)
    plot_3d(point_set, None)
    # compute the concave hull
    concave_hull = concave_hull_3d(point_set, tN=threshold_N)
    # and plot the result
    plot_3d(point_set, concave_hull)