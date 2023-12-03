#!/usr/bin/env python3


from libEistla import *
import math
import matplotlib.pyplot as plt




mirror0 = ParabolicMirror( Vector(-800,0), 110, 10, 800, math.radians( 0 ), Nsubdiv=64 )
mirror1 = SphericalMirror( Vector( -50, 0), 26, 2, 10000, math.radians(45))
baffle0 = Baffle( Vector( -30, 0), 23, 1, 0)


world = [mirror0, mirror1, baffle0]
paths = []

for q in range( -50, 52, 2 ):
    print( ">>> q:", q)
    ray = Ray( Vector( 45, q ), Vector.from_angle(math.radians(180)))
    path = raytrace( world, ray )
    #print( path )
    print( "path:")
    for p in path:
        print("\t", p )

    dl = 1650 - path[-1].L 

    last_point = path[-1].propagate(dl, inplace=False)

    path.append( last_point )
    paths.append( path )

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
