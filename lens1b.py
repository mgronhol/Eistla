#!/usr/bin/env python3


from libEistla import *
import math
import matplotlib.pyplot as plt


# Traces two sets of rays going through a simple two-element refractive system


lens0 = SphericalLens( Vector( -0, 0), 50, 8, 40, 40, 1.33, 0, Nsubdiv=1024 )
lens1 = SphericalLens( Vector( -65, 0), 30, 5, 50, 50, 1.33, 0, Nsubdiv=1024)

OPTICAL_AXIS = Ray( Vector( 0, 0), Vector( -1, 0 ) )

world = [lens0, lens1]
paths = []
rays = []

for i in range( -3, 4, 1 ):
    ray = Ray( Vector( 64.79, 10), Vector.from_angle( math.radians(i+ 180) ) )
    rays.append( ray )

for i in range( -3, 4, 1 ):
    ray = Ray( Vector( 64.79, 1), Vector.from_angle( math.radians(i+ 180) ) )
    rays.append( ray )


for ray in rays:
    path = raytrace( world, ray )
    #print( path )
    print( "path:")
    for p in path:
        print("\t", p )

    t = 80
    last_point = path[-1].propagate(t, inplace=False)

    path.append( last_point )
    paths.append( path )

divergence_analysis( paths )
fp_mean, fp_std = focal_point_analysis( paths )
print( "Focal point:", fp_mean, "+/-", fp_std)


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
