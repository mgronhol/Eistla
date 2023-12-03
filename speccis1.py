#!/usr/bin/env python3


from libEistla import *
import math
import matplotlib.pyplot as plt

# A Czerny-Turner spectrometer design

mirror0 = SphericalMirror( Vector( -50-3, 0), 15, 3, -55*2, math.radians(6) )
grating0 = ReflectionGrating( Vector( -10, 10 ), 15, 3, 10000, math.radians(45 - 8), lpm = 300, order = 1 )
mirror1 = SphericalMirror( Vector( -30, -35), 35, 3, -65*2, math.radians(80+0.85) )
detector = Baffle( Vector( -30, 30+2.5), 40, 1, math.radians(90) )


world = [mirror0, grating0, mirror1, detector]

ray0 = Ray( Vector( 5, 0), Vector.from_angle(math.radians(180+5)), wavelength = 650 )
ray1 = Ray( Vector( 5, 0), Vector.from_angle(math.radians(180-5)), wavelength = 650 )

ray2 = Ray( Vector( 5, 0), Vector.from_angle(math.radians(180+5)), wavelength = 535 )
ray3 = Ray( Vector( 5, 0), Vector.from_angle(math.radians(180-5)), wavelength = 535 )

ray4 = Ray( Vector( 5, 0), Vector.from_angle(math.radians(180+5)), wavelength = 420 )
ray5 = Ray( Vector( 5, 0), Vector.from_angle(math.radians(180-5)), wavelength = 420 )


rays = [ray0, ray1, ray2, ray3, ray4, ray5]

paths = []
for ray in rays:
    path = raytrace( world, ray )
    #print( path )
    print( "path:")
    for p in path:
        print("\t", p )

    
    last_point = path[-1].propagate(50, inplace=False)

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

    if path[0].wavelength > 600:
        plt.plot( pX, pY, 'r+-' )
    elif path[0].wavelength > 500:
        plt.plot( pX, pY, 'g+-' )
    else:
        plt.plot( pX, pY, 'b+-' )

plt.show()

