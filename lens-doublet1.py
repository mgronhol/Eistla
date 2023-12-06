#!/usr/bin/env python3


from libEistla import *
import math
import matplotlib.pyplot as plt


# Computes two rays through a single convex lens


lens0 = ConicLens( Vector( 0, 0 ), 30, 7, 40, -40, 0, GLASS["N-BK7"], 0 )
lens1 = ConicLens( Vector( 6, 0 ), 30, 5, -40, 100000, 0, GLASS["N-F2"], 0 )

join_lens_surfaces( left = lens0, right = lens1 )

world = [lens0, lens1]
paths = []

ray0 = Ray( Vector( -10, -10), Vector(1, 0) )
ray1 = Ray( Vector( -10, 10), Vector(1, 0) )

ray2 = Ray( Vector( 20, -5), Vector(-1, 0) )
ray3 = Ray( Vector( 20, 5), Vector(-1, 0) )


rays = [ray0, ray1, ray2, ray3 ]

for ray in rays:
    path = raytrace( world, ray )
    #print( path )
    print( "path:")
    for p in path:
        print("\t", p )

    dl = 70 - path[-1].L 

    if path[-1].alive:
        last_point = path[-1].propagate(dl, inplace=False)
    else:
        last_point = path[-1].propagate(0, inplace=False)

    path.append( last_point )
    paths.append( path )


plt.figure()
plt.axis('equal')


N = 0
for elem in world:

    X = []
    Y = []

    for pt in elem.points:
        X.append( pt.x )
        Y.append( pt.y )

    X.append( elem.points[0].x )
    Y.append( elem.points[0].y )
    

    #plt.plot( X, Y, 'b-' )
    if isinstance( elem, ConicLens ):
        if N % 2 == 0:
            plt.fill( X, Y, 'b' )
        else:
            plt.fill( X, Y, 'y' )
    elif isinstance( elem, Baffle ):
        plt.fill( X, Y, 'k' )
    else:
        plt.fill( X, Y, 'g' )
    N += 1

for path in paths:

    pX = []
    pY = []
    for pt in path:
        pX.append( pt.origin.x )
        pY.append( pt.origin.y )

    plt.plot( pX, pY, 'r+-' )

plt.show()
