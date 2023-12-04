#!/usr/bin/env python3


from libEistla import *
import math
import matplotlib.pyplot as plt


# Computes two rays through a single convex lens


lens0 = SphericalLens( Vector( -10, 0), 30, 3, 40, 40, 1.33, 0 )




world = [lens0]
paths = []

ray0 = Ray( Vector( 10, 5), Vector(-1, 0) )
ray1 = Ray( Vector( 10, 5), Vector(-1, 0) )
ray1.towards( lens0.centre )

rays = [ray0, ray1 ]

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


for elem in world:

    X = []
    Y = []

    for pt in elem.points:
        X.append( pt.x )
        Y.append( pt.y )

    X.append( elem.points[0].x )
    Y.append( elem.points[0].y )
    

    #plt.plot( X, Y, 'b-' )
    if isinstance( elem, SphericalLens ):
        plt.fill( X, Y, 'b' )
    elif isinstance( elem, Baffle ):
        plt.fill( X, Y, 'k' )
    else:
        plt.fill( X, Y, 'g' )

for path in paths:

    pX = []
    pY = []
    for pt in path:
        pX.append( pt.origin.x )
        pY.append( pt.origin.y )

    plt.plot( pX, pY, 'r+-' )

plt.show()
