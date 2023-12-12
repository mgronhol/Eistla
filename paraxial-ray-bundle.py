#!/usr/bin/env python3


from libEistla import *
import math
import matplotlib.pyplot as plt




lens0 = ConicLens( Vector( 0, 0), 30, 7, 40, -40, 0, 1.5, 0 )
lens1 = ConicLens( Vector( 50, 0), 20, 7, 60, -60, 0, 1.5, 0 )

astop = ApertureStop( Vector (25, 0), 10 )


world = [lens0, astop, lens1]

paths0 = []
paths1 = []
paths2 = []

ray = Ray( Vector(-30, 0), Vector.from_angle( math.radians(2 ) ) )

paraxial_path = ParaxialApproximation.raytrace( world, ray )
print( paraxial_path )

mray = ParaxialApproximation.solve_marginal_ray( world, ray.origin )
print( "Paraxial marginal ray:", mray)

mray_trig = astop.solve_marginal_ray(world, ray.origin )
print( "Trigonometric marginal ray:", mray_trig)

cray = ParaxialApproximation.solve_chief_ray( world, ray.origin, 5 )
print( "Paraxial chief ray:", cray)

cray_trig = astop.solve_chief_ray(world, ray.origin, 5 )
print( "Trigonometric chief ray:", cray_trig)



path = []
for point in paraxial_path:
    path.append( ParaxialApproximation.to_ray( point ) )

paths0.append( path )

paths1.append( raytrace_sequential( world, ray ) )


print("")
focus_pos_parax = ParaxialApproximation.find_focus_point(world, Vector(-30, 0) )
print( "Paraxial focus point:", focus_pos_parax )

focus_pos_trig = find_focus_point(world, Vector(-30, 0) )
print( "Trigonometric focus point:", focus_pos_trig )

baffle = Baffle( Vector(focus_pos_parax.x, 0), 30, 1, 0 )

world.append( baffle )


ray_bundle = ParaxialApproximation.generate_ray_bundle( world, Vector(-30, 0) )

for ray in ray_bundle:
    paraxial_path = ParaxialApproximation.raytrace( world, ray )

    path = []
    for point in paraxial_path:
        path.append( ParaxialApproximation.to_ray( point ) )

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
