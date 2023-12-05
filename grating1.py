#!/usr/bin/env python3


from libEistla import *
import math
import matplotlib.pyplot as plt



grating0 = TransmissionGrating( Vector( 0,0 ), 30, 1, 10000, math.radians(-5), lpm = 300, order = 1 )
grating1 = ReflectionGrating( Vector( -30,0 ), 30, 1, 10000, math.radians(35), lpm = 100, order = 1 )

world = [grating0, grating1]

ray0 = Ray( Vector( 10, 0), Vector(-1, 0) )
ray1 = Ray( Vector( 10, 1), Vector.from_angle( math.radians( 175 ) ) )
ray2 = Ray( Vector( 10, -1), Vector.from_angle( math.radians( 185 ) ) )

rays = [ray0, ray1, ray2]

paths = []
for ray in rays:
    path = raytrace( world, ray )
    #print( path )
    print( "path:")
    for p in path:
        print("\t", p )

    if path[-1].alive:
        last_point = path[-1].propagate(20, inplace=False)
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
    if isinstance( elem, ConicLens ):
        plt.fill( X, Y, 'b' )
    elif isinstance( elem, Baffle ):
        plt.fill( X, Y, 'k' )
    elif isinstance( elem, TransmissionGrating ):
        plt.fill( X, Y, 'm' )
    elif isinstance( elem, ReflectionGrating ):
        plt.fill( X, Y, 'm' )
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

