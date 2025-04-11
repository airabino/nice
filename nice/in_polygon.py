import numba
import numpy as np
from numba import jit, njit

'''
Ray-Tracing method for determinimg if a point is within a polygon.
x - longitudinal coordinate
y - lateral coordinate
polygon - the lon/lat coordinates of the exterior of the polygon

The basic method is to loop through the edges and to determine if the point
is directly to the left of the edge. If the point is directlt to the left
of exactly one edge then it is within the polygon, if it is directly to the
left of 2 edges then it is not within the polygon.

This function is compiled using Numba jit no-Python so it cannot be vectorized
'''

@jit(nopython = True, cache = True)
def _point_in_polygon(polygon, point):

    length = len(polygon) - 1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii < length:

        dy  = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy * dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):

            condition_0 = dy < 0 or dy2 < 0
            condition_1 = dy2 == 0
            condition_2 = (
                (point[0] == polygon[jj][0]) or
                (dy == 0 and (point[0] - polygon[ii][0]) * (point[0] - polygon[jj][0]) <= 0)
                )

            # non-horizontal line
            if condition_0:
                F = dy * (polygon[jj][0] - polygon[ii][0]) / (dy - dy2) + polygon[ii][0]

                if point[0] > F:
                # if line is left from the point - the ray moving towards left,
                #will intersect it

                    intersections += 1

                elif point[0] == F: # point on line

                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif condition_1 and condition_2:

                return 2

        ii = jj
        jj += 1

    #print 'intersections =', intersections
    return intersections & 1  

'''
Function for determining which points from an array are in a given polygon
'''
@njit(cache = True)
def _points_in_polygon(polygon, points):

    n = len(points)
    d = np.empty(n, dtype = numba.boolean) 

    for i in range(n):

        d[i] = _point_in_polygon(polygon, points[i])

    return d

def points_in_polygon(polygon, x, y):

    xy = np.asarray(tuple(zip(x, y)))

    xy_p = np.asarray(tuple(zip(*polygon.exterior.coords.xy)))

    d = _points_in_polygon(xy_p, xy)

    return d

# @njit(cache = True)
def _points_in_polygons(polygons, x, y):

    xy = np.asarray(tuple(zip(x, y)))

    d = []

    for polygon in polygons:

        xy_p = np.asarray(tuple(zip(*polygon.exterior.coords.xy)))

        d.append(_points_in_polygon(xy_p, xy))

    return np.array(d)











