#!/usr/bin/env python3


from libEistla import *
import math
import matplotlib.pyplot as plt

import numpy as np

import scipy.optimize


# lens FFD = 1.45 + 5.5
# BFD = 2.46 + 4.3
# diameter = 4.8
n = 1.5
R1 = 7.1
R2 = 7.1

object_position = Vector( (-6.76-1)*1.2, 0.0)

lens0 = ConicLens( Vector(0,0), 4.8/2, 2, R1, -R2, 1.0, n, 0 )

astop = ApertureStop(Vector(2, 0), 3.0/2 )

grating = TransmissionGrating( Vector(5, 0), 5, 1, 1e4, math.radians(15), 625, 1 )

phi = -75
#lens1 = ConicLens( Vector(grating.centre.x,0) + Vector(5,0).rotate(math.radians(phi)), 4.8/2, 2, R1, -R2, 1.0, n, math.radians(phi) )


#world = [lens0, astop, grating, lens1 ]
world = [lens0, astop, grating ]

fp = find_focus_point(world, object_position )
print( "focal point:", fp)

baffle = Baffle( Vector( fp.x+.5, fp.y-.25), 15, 1, math.radians(phi))

world.append( baffle )


rays0 = []
rays0 = generate_ray_bundle( world, astop, object_position, wavelength = 630.0 )
rays1 = generate_ray_bundle( world, astop, object_position, wavelength = 550.0 )
rays1b = generate_ray_bundle( world, astop, object_position, wavelength = 540.0 )
rays2 = generate_ray_bundle( world, astop, object_position, wavelength = 415.0 )

paths = []

for ray in rays0:
    path = raytrace_sequential( world, ray )
    paths.append( path )

for ray in rays1:
    path = raytrace_sequential( world, ray )
    paths.append( path )

for ray in rays1b:
    path = raytrace_sequential( world, ray )
    paths.append( path )

for ray in rays2:
    path = raytrace_sequential( world, ray )
    paths.append( path )

rays0 = generate_ray_bundle( world, astop, object_position + Vector(0, 0.1), wavelength = 630.0 )
rays1 = generate_ray_bundle( world, astop, object_position + Vector(0, 0.1), wavelength = 550.0 )
rays1b = generate_ray_bundle( world, astop, object_position + Vector(0, 0.1), wavelength = 540.0 )
rays2 = generate_ray_bundle( world, astop, object_position + Vector(0, 0.1), wavelength = 415.0 )

for ray in rays0:
    path = raytrace_sequential( world, ray )
    paths.append( path )

for ray in rays1:
    path = raytrace_sequential( world, ray )
    paths.append( path )

for ray in rays1b:
    path = raytrace_sequential( world, ray )
    paths.append( path )

for ray in rays2:
    path = raytrace_sequential( world, ray )
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

    if path[0].wavelength > 600:
        plt.plot( pX, pY, 'r+-' )
    elif path[0].wavelength > 500:
        plt.plot( pX, pY, 'g+-' )
    else:
        plt.plot( pX, pY, 'b+-' )

astop.draw( plt )

plt.show()






