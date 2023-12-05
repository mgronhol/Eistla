#!/usr/bin/env python3


from libEistla import *
import math
import matplotlib.pyplot as plt

import numpy as np


# A Czerny-Turner spectrometer design

mirror0 = ConicMirror( Vector( -50-3, 0), 15, 3, -55*2,0,  math.radians(6 + 180) )
grating0 = ReflectionGrating( Vector( -10, 10 ), 15, 3, 10000, math.radians(45 - 8 ), lpm = 300, order = 1 )
mirror1 = ConicMirror( Vector( -30, -35), 35, 3, -65*2,0,  math.radians(80+0.85 + 180) )
detector = Baffle( Vector( -30, 30+2.5), 40, 1, math.radians(90) )



## Example how to do (simple) iterative design, here we want to put detector in focus

def quality_function( paths ):
    per_wavelength = {}
    for path in paths:
        last = path[-1]
        if last.wavelength not in per_wavelength:
            per_wavelength[last.wavelength] = []
        per_wavelength[last.wavelength].append( last )
    
    out = 0
    for wl in per_wavelength:
        xs = []
        ys = []
        for ray in per_wavelength[wl]:
            xs.append( ray.origin.x )
            ys.append( ray.origin.y )

        out += np.std( xs ) + np.std(ys)
    return out  





ray0 = Ray( Vector( 5, 0), Vector.from_angle(math.radians(180+5)), wavelength = 650 )
ray1 = Ray( Vector( 5, 0), Vector.from_angle(math.radians(180-5)), wavelength = 650 )

ray2 = Ray( Vector( 5, 0), Vector.from_angle(math.radians(180+5)), wavelength = 535 )
ray3 = Ray( Vector( 5, 0), Vector.from_angle(math.radians(180-5)), wavelength = 535 )

ray4 = Ray( Vector( 5, 0), Vector.from_angle(math.radians(180+5)), wavelength = 420 )
ray5 = Ray( Vector( 5, 0), Vector.from_angle(math.radians(180-5)), wavelength = 420 )


rays = [ray0, ray1, ray2, ray3, ray4, ray5]






world = [mirror0, grating0, mirror1, detector]

paths = []
for ray in rays:
    path = raytrace( world, ray )
    #print( path )
    print( "path:")
    for p in path:
        print("\t", p )

    if path[-1].alive:
        last_point = path[-1].propagate(50, inplace=False)
    else:
        last_point = path[-1].propagate(0, inplace=False)

    path.append( last_point )
    paths.append( path )

prev_quality = quality_function( paths )

print( "quality:", prev_quality )



detector_position = -0.5
delta = -0.5

# Iterate to find better detector position

for i in range( 20 ):
    detector = Baffle( Vector( -30, 30+2.5 + detector_position), 40, 1, math.radians(90) )
    world = [mirror0, grating0, mirror1, detector]

    paths = []
    for ray in rays:
        path = raytrace( world, ray )
        paths.append( path )

    quality = quality_function( paths )
    dquality = quality - prev_quality
    print( "quality:", quality, "detector position:", detector_position, "delta:", delta, "dq:", dquality )
    alpha = 0.92
    if quality < prev_quality:
        detector_position += delta
        delta = delta * alpha
    else:
        detector_position -= delta
        delta = delta * alpha

    prev_quality = quality





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

plt.show()

