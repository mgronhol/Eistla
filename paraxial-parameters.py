#!/usr/bin/env python3


from libEistla import *
import math
import matplotlib.pyplot as plt

import pprint



lens0 = ConicLens( Vector( 0, 0), 30, 7, 40, -40, 0, 1.5, 0 )

astop = ApertureStop( Vector (-4, 0), 30 )

world = [astop, lens0]


params = ParaxialApproximation.get_system_parameters( world, Vector(-80, 0) )
pprint.pprint( params )





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





plt.show()