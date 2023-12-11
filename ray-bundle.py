#!/usr/bin/env python3


from libEistla import *
import math
import matplotlib.pyplot as plt

import numpy as np
import scipy.optimize




lens0 = ConicLens( Vector( 0, 0), 30, 7, 40, -40, 0, 1.5, 0 )
lens1 = ConicLens( Vector( 50, 0), 20, 7, 60, -60, 0, 1.5, 0 )

astop = ApertureStop( Vector (25, 0), 10 )


world = [lens0, astop, lens1]

focus_pos = find_focus_point(world, Vector(-30, 0) )
print( "focus_pos", focus_pos )

baffle = Baffle( Vector( focus_pos.x, 0), 30, 1, 0)
world.append( baffle )

rays = generate_ray_bundle( world, astop, Vector( -30, 2.5) ) 

paths0 = []

for ray in rays:
    path = raytrace_sequential( world, ray )

    t = 10
    if path[-1].alive:
        last_point = path[-1].propagate(t, inplace=False)
    else:
        last_point = path[-1].propagate(0, inplace=False)

    path.append( last_point )
    paths0.append( path )

rays = generate_ray_bundle( world, astop, Vector( -30, -2.5) )

paths1 = []

for ray in rays:
    path = raytrace_sequential( world, ray )

    t = 10
    if path[-1].alive:
        last_point = path[-1].propagate(t, inplace=False)
    else:
        last_point = path[-1].propagate(0, inplace=False)

    path.append( last_point )
    paths1.append( path )

rays = generate_ray_bundle( world, astop, Vector( -30, 0) )

paths2 = []

for ray in rays:
    path = raytrace_sequential( world, ray )

    t = 10
    if path[-1].alive:
        last_point = path[-1].propagate(t, inplace=False)
    else:
        last_point = path[-1].propagate(0, inplace=False)

    path.append( last_point )
    paths2.append( path )




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
    else:
        plt.fill( X, Y, 'g' )

for path in paths0:

    pX = []
    pY = []
    for pt in path:
        pX.append( pt.origin.x )
        pY.append( pt.origin.y )

    plt.plot( pX, pY, 'r+-' )

for path in paths1:

    pX = []
    pY = []
    for pt in path:
        pX.append( pt.origin.x )
        pY.append( pt.origin.y )

    plt.plot( pX, pY, 'g+-' )

for path in paths2:

    pX = []
    pY = []
    for pt in path:
        pX.append( pt.origin.x )
        pY.append( pt.origin.y )

    plt.plot( pX, pY, 'c+-' )



astop.draw( plt )





plt.show()
