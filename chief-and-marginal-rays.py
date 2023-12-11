#!/usr/bin/env python3


from libEistla import *
import math
import matplotlib.pyplot as plt

import numpy as np
import scipy.optimize


# Computes two rays through a single convex lens

      

lens0 = ConicLens( Vector( 0, 0), 30, 7, 40, -40, 0, 1.5, 0 )
lens1 = ConicLens( Vector( 50, 0), 20, 7, 100, -100, 0, 1.5, 0 )


astop = ApertureStop( Vector (25, 0), 10 )


world = [lens0, astop, lens1]

marginal_ray = astop.solve_marginal_ray( world, Vector(-30,0 ) )
chief_ray = astop.solve_chief_ray( world, Vector(-30,0 ), 10 )

marginal_ray_path = raytrace_sequential( world, marginal_ray )

chief_ray_path = raytrace_sequential( world, chief_ray)





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


astop.draw( plt )

if True:
    if True:
        pX = []
        pY = []
        for pt in marginal_ray_path:
            pX.append( pt.origin.x )
            pY.append( pt.origin.y )

        plt.plot( pX, pY, 'g+-' )

    if True:
        pX = []
        pY = []
        for pt in chief_ray_path:
            pX.append( pt.origin.x )
            pY.append( pt.origin.y )

        plt.plot( pX, pY, 'y+-' )





plt.show()
