#!/usr/bin/env python3



import math
import cv2
import numpy as np

# Distance that the ray travels after interacting with a surface before new intersection round is performed
EPS = 100e-3


# General purpose 3D vector object, has overloaded arithmetic operators
class Vector( object ):

    @staticmethod
    def from_angle( theta ):
        return Vector( math.cos( theta ), math.sin( theta ) )

    def __init__(self, x, y, z = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z
    
    def __add__( self, rhs ):
        return Vector( self.x + rhs.x, self.y + rhs.y, self.z + rhs.z )

    def __sub__( self, rhs ):
        return Vector( self.x - rhs.x, self.y - rhs.y, self.z - rhs.z )

    def __neg__( self ):
        return Vector( -self.x, -self.y, -self.z )

    # if multiplied with a scalar -> scaled vector, if multiplied with another vector -> cross product
    def __mul__( self, rhs ):
        if isinstance( rhs, int ) or isinstance( rhs, float ):
            return Vector( self.x * rhs, self.y * rhs, self.z * rhs )
        else:
            return Vector(
                self.y * rhs.z - self.z * rhs.y,
                self.z * rhs.x - self.x * rhs.z,
                self.x * rhs.y - self.y * rhs.x
            )
    
    # XOR is used to represent dot product 
    def __xor__( self, rhs ):
        return self.x * rhs.x + self.y * rhs.y + self.z * rhs.z

    def length( self ):
        return math.sqrt( self.x**2 + self.y**2 + self.z**2 )

    # ~ operator is used to represent unit vector
    def __invert__( self ):
        l = self.length()
        return Vector( self.x / l, self.y / l , self.z / l )

    # Simple 2D rotation in xy-plane
    def rotate( self, theta ):
        nx = self.x * math.cos( theta ) - self.y * math.sin( theta )
        ny = self.x * math.sin( theta ) + self.y * math.cos( theta )
        return Vector( nx, ny, self.z )

    def __repr__( self ):
        return "Vector(x=%.2f, y=%.2f, z=%.2f)" % ( self.x, self.y, self.z )

    def __str__( self ):
        return repr( self )

    def angle( self ):
        return math.atan2( self.y, self.x )


# Single line segment
class Segment( object ):
    def __init__(self, A, B ) -> None:
        self.A = A
        self.B = B
        self.v2 = B - A
        self.N = ~Vector( -self.v2.y, self.v2.x )


# A ray of light with origin, direction and physical parameters about the light 
class Ray( object ):
    def __init__(self, origin, direction, n = 1.0, wavelength = 550, L = 0.0 ) -> None:
        self.origin = origin
        self.direction = direction
        self.invdir = Vector( -self.direction.y, self.direction.x )
        self.n = n
        self.wavelength = wavelength
        self.L = L
    
    # Set the ray to point towards a specific point
    def towards( self, point ):
        delta = point - self.origin

        self.direction = ~delta
        self.invdir = Vector( -self.direction.y, self.direction.x )

    # Compute intersection point between a ray and a line segment 
    def intersect( self, segment ):
        v1 = self.origin - segment.A
        v2 = segment.v2
        #v3 = Vector( -self.direction.y, self.direction.x )
        v3 = self.invdir

        if (v1 ^ self.direction) > 0:
            return False

        dv = v2 ^ v3
        if dv < 0:
            #v2 = -v2
            v3 = -v3
            dv = -dv

        if abs( dv ) < 1e-5:
            return False

        
        t1 = (v2 * v1).length() / dv
        t2 = (v1 ^ v3) / dv

        if t1 > 0:
            if 0 < t2 < 1:
                return t1
        return False
    
    # Propagate ray along its direction, accumulates propagated distance weighted with 1/n of current medium
    def propagate( self, dist, inplace = True ):
        if inplace:
            self.origin = self.origin + (self.direction * dist)
            self.L += dist / self.n
            return self
        else:
            return Ray( self.origin + (self.direction * dist), self.direction, n = self.n, wavelength = self.wavelength, L = self.L + dist / self.n )

    def rotate( self, theta, inplace = True ):
        if inplace:
            self.direction = self.direction.rotate( theta )
            self.invdir = Vector( -self.direction.y, self.direction.x )
            return self
        else:
            return Ray( self.origin, self.direction.rotate( theta ), n = self.n, wavelength = self.wavelength, L = self.L )

    def __repr__( self ):
        return "Ray( origin = %s, direction = %s, n = %.2f, λ = %.1fnm, L = %.1f )" % ( self.origin, self.direction, self.n, self.wavelength, self.L )

    def copy( self ):
        return Ray( self.origin*1, self.direction*1, n = self.n, wavelength = self.wavelength, L = self.L )


# A 2D bounding box, used to test if ray has a chance to hit an optical element
class BoundingBox( object ):

    @staticmethod
    def from_points( points ):
        rpoints = []
        for pt in points:
            rpoints.append( ( pt.x, pt.y ) )
        
        rect = cv2.minAreaRect( np.float32( rpoints ) )
        boxpoints = cv2.boxPoints( rect )

        corners = []
        for pt in boxpoints:
            corners.append( Vector( pt[0], pt[1] ) )

        return BoundingBox( *corners ) 

    def __init__(self, A, B, C, D ) -> None:
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.segments = []
        self.segments.append( Segment( A, B ) )
        self.segments.append( Segment( B, C ) )
        self.segments.append( Segment( C, D ) )
        self.segments.append( Segment( D, A ) )
    
    def intersect( self, ray ):
        results = []
        for segment in self.segments:
            dist = ray.intersect( segment )
            if dist:
                results.append( dist )
        
        if len( results ) > 0:
            results.sort()
            return results[0]
        else:
            return False
        
        
# Refraction accoring to Snell's law
# n1 and n2 are refractive indices of the corresponding media, can be either scalar values or functions of wavelength (in nm)
def snell( segment, ray, n1, n2 ):
    dotp = segment.N ^ ray.direction
    if dotp > 0:
        N = -segment.N
    else:
        N = segment.N
    
    S1 = ray.direction

    if isinstance( n1, float ):
        _n1 = n1
    else:
        _n1 = n1( ray.wavelength )
    
    if isinstance( n2, float ):
        _n2 = n2
    else:
        _n2 = n2( ray.wavelength )
    

    NxS1 = N*S1
    q = math.sqrt( 1 - (_n1 / _n2)**2 * (NxS1 ^ NxS1))
    
    S2 = (N * ((-N) * S1)) * (_n1 / _n2) - (N * q)

    return Ray( ray.origin, S2, n = _n2, wavelength = ray.wavelength, L = ray.L )

# Reflection from a sufrace
def reflect( segment, ray ):
    dotp = segment.N ^ ray.direction
    ndir = 1
    if dotp > 0:
        ndir = -1
    else:
        dotp = -dotp
    
    refN = segment.N * ndir
    reflected_dir = ray.direction + refN * (2 * dotp)

    return Ray( ray.origin, reflected_dir, n = ray.n, wavelength = ray.wavelength, L = ray.L )



def grating_diffraction( segment, ray, order, lpm, transmission = True ):
    d = 1.0/lpm
    d *= 1e6 # mm -> nm
    q = order * ray.wavelength / d

    dotp = segment.N ^ ray.direction
        
    if transmission:
        if dotp < 0:
            Ng = -segment.N
        else:
            Ng = segment.N
    else:
        if dotp < 0:
            Ng = segment.N
        else:
            Ng = -segment.N
        
    crossp = Ng * ray.direction
    sini = (crossp).length()

    try:
        theta_m = math.asin( sini + q )
    except ValueError:
        return False    
    
    if crossp.z < -1e-6:
        theta_m = -theta_m

    
    diff_dir = Ng.rotate( theta_m )
    return Ray( ray.origin, diff_dir, n = ray.n, wavelength = ray.wavelength, L = ray.L )

    



# A refractive lens with two spherical surfaces
class SphericalLens( object ):
    def __init__(self, centre, height, thickness, r1, r2, n, theta, Nsubdiv = 100 ) -> None:
        self.centre = centre
        self.r1 = r1
        self.r2 = r2
        self.n = n
        self.theta = theta
        self.height = height
        self.thickness = thickness
        h2 = height / 2
        t2 = thickness / 2

        phi0 = -math.atan2( h2, abs(r1) )
        phi1 = -phi0
        phi = phi0
        dphi = phi1 / Nsubdiv
        
        
        points = []

        while phi <= phi1:
            point = Vector( r1, 0).rotate( phi )
            points.append( point + Vector (-r1 + 2*t2, 0) )
            phi += dphi

        phi0 = -math.atan2( h2, abs(r2) )
        phi1 = -phi0
        phi = phi0
        dphi = phi1 / Nsubdiv
        
        while phi <= phi1:
            point = Vector( -r2, 0).rotate( phi )
            points.append( point + Vector( r2 - 2*t2, 0) )
            phi += dphi

        self.points = []

        for pt in points:
            self.points.append( pt.rotate( theta ) + centre )
        
        self.bbox = BoundingBox.from_points( self.points )

        self.segments = []
        for i in range( 1, len( self.points ) ):
            self.segments.append( Segment( self.points[i-1], self.points[i] ) )
        
        self.segments.append( Segment(self.points[-1], self.points[0] ))

    
    def _interact( self, ray, from_outside = True ):
        #print( "_interact:: ray=", ray)
        idx = 0
        mindist = 1e99
        for i in range( len( self.segments ) ):
            segment = self.segments[i]
            dist = ray.intersect( segment )
            if dist:
                if dist < mindist:
                    mindist = dist
                    idx = i
        #print( "_interact:: dbg, mindist=", mindist )
        new_ray = ray.propagate( mindist, inplace = False )
        #print( "_interact:: dbg, new_ray=", new_ray )
        #print( "_interact:: idx=", idx)
        if from_outside:
            return snell(self.segments[idx], new_ray, 1.0, self.n ).propagate( EPS )
        else:
            return snell(self.segments[idx], new_ray, self.n, 1.0 ).propagate( EPS )


    def interact( self, ray ):
        refracted_ray0 = self._interact( ray, from_outside=True)
        refracted_ray1 = self._interact( refracted_ray0, from_outside=False)
        return [refracted_ray0, refracted_ray1]


# A reflective mirror with one spherical surface
class SphericalMirror( object ):
    def __init__(self, centre, height, thickness, r1, theta, Nsubdiv = 100 ) -> None:
        self.centre = centre
        self.r1 = r1
        r2 = 10000
        if r1 < 0:
            r2 = -r2
        self.r2 = r2
        self.theta = theta
        self.height = height
        self.thickness = thickness
        h2 = height / 2
        t2 = thickness / 2

        phi0 = -math.atan2( h2, abs(r1) )
        phi1 = -phi0
        phi = phi0
        dphi = phi1 / Nsubdiv
        
        
        points = []

        while phi <= phi1:
            point = Vector( r1, 0).rotate( phi )
            points.append( point + Vector (-r1 + 2*t2, 0) )
            phi += dphi

        phi0 = -math.atan2( h2, abs(r2) )
        phi1 = -phi0
        phi = phi0
        dphi = phi1 / 10
        
        while phi <= phi1:
            point = Vector( -r2, 0).rotate( phi )
            points.append( point + Vector( r2 - 2*t2, 0) )
            phi += dphi

        self.points = []

        for pt in points:
            self.points.append( pt.rotate( theta ) + centre )
        
        self.bbox = BoundingBox.from_points( self.points )

        self.segments = []
        for i in range( 1, len( self.points ) ):
            self.segments.append( Segment( self.points[i-1], self.points[i] ) )
        
        self.segments.append( Segment(self.points[-1], self.points[0] ))

    
    def _interact( self, ray  ):
        #print( "_interact:: ray=", ray)
        idx = 0
        mindist = 1e99
        for i in range( len( self.segments ) ):
            segment = self.segments[i]
            dist = ray.intersect( segment )
            if dist:
                if dist < mindist:
                    mindist = dist
                    idx = i
        #print( "_interact:: dbg, mindist=", mindist )
        new_ray = ray.propagate( mindist, inplace = False )
        #print( "_interact:: dbg, new_ray=", new_ray )
        #print( "_interact:: idx=", idx)
        return reflect(self.segments[idx], new_ray ).propagate( EPS )
        

    def interact( self, ray ):
        reflected_ray = self._interact( ray )
        return [reflected_ray]


# Reflective parabolic mirror, one parabolic surface
class ParabolicMirror( object ):
    def __init__(self, centre, height, thickness, r1, theta, Nsubdiv = 32 ) -> None:
        self.centre = centre
        self.f1 = r1
        r2 = 10000
        if r1 < 0:
            r2 = -r2
        self.r2 = r2
        self.theta = theta
        self.height = height
        self.thickness = thickness
        h2 = height / 2
        t2 = thickness / 2

        phi0 = -math.atan2( h2, abs(r1) )
        phi1 = -phi0
        phi = phi0
        dphi = phi1/100
        
        dq = math.tan(dphi) * abs(r1) / Nsubdiv
        
        points = []

        a = 1/(4*r1)
        y = -h2
        while y < h2:
            x = a*y*y
            point = Vector( x + t2, y)
            points.append( point )
            y += dq



        phi0 = -math.atan2( h2, abs(r2) )
        phi1 = -phi0
        phi = phi0
        dphi = phi1/100
        
        while phi <= phi1:
            point = Vector( -r2, 0).rotate( phi )
            points.append( point + Vector( r2 - 2*t2, 0) )
            phi += dphi

        self.points = []

        for pt in points:
            self.points.append( pt.rotate( theta ) + centre )
        
        self.bbox = BoundingBox.from_points( self.points )

        self.segments = []
        for i in range( 1, len( self.points ) ):
            self.segments.append( Segment( self.points[i-1], self.points[i] ) )
        
        self.segments.append( Segment(self.points[-1], self.points[0] ))

    
    def _interact( self, ray  ):
        #print( "_interact:: ray=", ray)
        idx = 0
        mindist = 1e99
        for i in range( len( self.segments ) ):
            segment = self.segments[i]
            dist = ray.intersect( segment )
            if dist:
                if dist < mindist:
                    mindist = dist
                    idx = i
        #print( "_interact:: dbg, mindist=", mindist )
        new_ray = ray.propagate( mindist, inplace = False )
        #print( "_interact:: dbg, new_ray=", new_ray )
        #print( "_interact:: idx=", idx)
        return reflect(self.segments[idx], new_ray ).propagate( EPS )
        

    def interact( self, ray ):
        reflected_ray = self._interact( ray )
        return [reflected_ray]

# Rectangular baffle, stops light from passing through
class Baffle( object ):
    def __init__(self, centre, height, thickness, theta, Nsubdiv = 10 ) -> None:
        self.centre = centre
        r1 = 10000
        self.r1 = r1
        r2 = 10000
        if r1 < 0:
            r2 = -r2
        self.r2 = r2
        self.theta = theta
        self.height = height
        self.thickness = thickness
        h2 = height / 2
        t2 = thickness / 2

        phi0 = -math.atan2( h2, abs(r1) )
        phi1 = -phi0
        phi = phi0
        dphi = phi1 / Nsubdiv
        
        
        points = []

        while phi <= phi1:
            point = Vector( r1, 0).rotate( phi )
            points.append( point + Vector (-r1 + 2*t2, 0) )
            phi += dphi

        phi0 = -math.atan2( h2, abs(r2) )
        phi1 = -phi0
        phi = phi0
        dphi = phi1 / 10
        
        while phi <= phi1:
            point = Vector( -r2, 0).rotate( phi )
            points.append( point + Vector( r2 - 2*t2, 0) )
            phi += dphi

        self.points = []

        for pt in points:
            self.points.append( pt.rotate( theta ) + centre )
        
        self.bbox = BoundingBox.from_points( self.points )

        self.segments = []
        for i in range( 1, len( self.points ) ):
            self.segments.append( Segment( self.points[i-1], self.points[i] ) )
        
        self.segments.append( Segment(self.points[-1], self.points[0] ))
        

    def interact( self, ray ):
        return []


class TransmissionGrating( object ):
    def __init__(self, centre, height, thickness, r1, theta, lpm, order, Nsubdiv = 100 ) -> None:
        self.centre = centre
        self.r1 = r1
        r2 = 10000
        if r1 < 0:
            r2 = -r2
        self.r2 = r2
        self.theta = theta
        self.height = height
        self.thickness = thickness
        self.lpm = lpm
        self.order = order
        h2 = height / 2
        t2 = thickness / 2

        phi0 = -math.atan2( h2, abs(r1) )
        phi1 = -phi0
        phi = phi0
        dphi = phi1 / Nsubdiv
        
        
        points = []

        while phi <= phi1:
            point = Vector( r1, 0).rotate( phi )
            points.append( point + Vector (-r1 + 2*t2, 0) )
            phi += dphi

        phi0 = -math.atan2( h2, abs(r2) )
        phi1 = -phi0
        phi = phi0
        dphi = phi1 / 10
        
        while phi <= phi1:
            point = Vector( -r2, 0).rotate( phi )
            points.append( point + Vector( r2 - 2*t2, 0) )
            phi += dphi

        self.points = []

        for pt in points:
            self.points.append( pt.rotate( theta ) + centre )
        
        self.bbox = BoundingBox.from_points( self.points )

        self.segments = []
        for i in range( 1, len( self.points ) ):
            self.segments.append( Segment( self.points[i-1], self.points[i] ) )
        
        self.segments.append( Segment(self.points[-1], self.points[0] ))

    
    def _interact( self, ray, skip_grating = False  ):
        #print( "_interact:: ray=", ray)
        idx = 0
        mindist = 1e99
        for i in range( len( self.segments ) ):
            segment = self.segments[i]
            dist = ray.intersect( segment )
            if dist:
                if dist < mindist:
                    mindist = dist
                    idx = i
        #print( "_interact:: dbg, mindist=", mindist )
        new_ray = ray.propagate( mindist, inplace = False )
        #print( "_interact:: dbg, new_ray=", new_ray )
        #print( "_interact:: idx=", idx)
        #return reflect(self.segments[idx], new_ray ).propagate( EPS )
        if skip_grating:
            return new_ray.propagate( EPS )
        else:
            new_ray2 = grating_diffraction( self.segments[idx], new_ray, self.order, self.lpm, transmission=True )
            if not new_ray2:
                new_ray.direction = Vector( 0, 0 )
                return new_ray
            return new_ray2.propagate( EPS )


    def interact( self, ray ):
        ray0 = self._interact( ray, skip_grating=False )
        ray1 = self._interact( ray0, skip_grating=True )
        
        return [ray0, ray1]

class ReflectionGrating( object ):
    def __init__(self, centre, height, thickness, r1, theta, lpm, order, Nsubdiv = 100 ) -> None:
        self.centre = centre
        self.r1 = r1
        r2 = 10000
        if r1 < 0:
            r2 = -r2
        self.r2 = r2
        self.theta = theta
        self.height = height
        self.thickness = thickness
        self.lpm = lpm
        self.order = order
        h2 = height / 2
        t2 = thickness / 2

        phi0 = -math.atan2( h2, abs(r1) )
        phi1 = -phi0
        phi = phi0
        dphi = phi1 / Nsubdiv
        
        
        points = []

        while phi <= phi1:
            point = Vector( r1, 0).rotate( phi )
            points.append( point + Vector (-r1 + 2*t2, 0) )
            phi += dphi

        phi0 = -math.atan2( h2, abs(r2) )
        phi1 = -phi0
        phi = phi0
        dphi = phi1 / 10
        
        while phi <= phi1:
            point = Vector( -r2, 0).rotate( phi )
            points.append( point + Vector( r2 - 2*t2, 0) )
            phi += dphi

        self.points = []

        for pt in points:
            self.points.append( pt.rotate( theta ) + centre )
        
        self.bbox = BoundingBox.from_points( self.points )

        self.segments = []
        for i in range( 1, len( self.points ) ):
            self.segments.append( Segment( self.points[i-1], self.points[i] ) )
        
        self.segments.append( Segment(self.points[-1], self.points[0] ))

    
    def _interact( self, ray, skip_grating = False  ):
        #print( "_interact:: ray=", ray)
        idx = 0
        mindist = 1e99
        for i in range( len( self.segments ) ):
            segment = self.segments[i]
            dist = ray.intersect( segment )
            if dist:
                if dist < mindist:
                    mindist = dist
                    idx = i
        #print( "_interact:: dbg, mindist=", mindist )
        new_ray = ray.propagate( mindist, inplace = False )
        #print( "_interact:: dbg, new_ray=", new_ray )
        #print( "_interact:: idx=", idx)
        #return reflect(self.segments[idx], new_ray ).propagate( EPS )
        if skip_grating:
            return new_ray.propagate( EPS )
        else:
            new_ray2 = grating_diffraction( self.segments[idx], new_ray, self.order, self.lpm, transmission=False )
            if not new_ray2:
                new_ray.direction = Vector( 0, 0 )
                return new_ray

            return new_ray2.propagate( EPS )


    def interact( self, ray ):
        ray = self._interact( ray, skip_grating=False )
        return [ray]




# Traces a ray through the world (a list of optical elements)
def raytrace( world, ray ):
    current_idx = -1
    path = [ray.copy()]
    current_ray = ray.copy()
    done = False

    while not done:
        hits = []
        for idx in range( len( world ) ):
            if idx != current_idx:
                dist = world[idx].bbox.intersect( current_ray )
                #print( "raytrace:: idx=", idx, "dist=", dist)
                if dist:
                    hits.append( (idx, dist))
        #print( "hits:", hits)
        if len( hits ) > 0:
            sorted_hits = sorted( hits, key = lambda h: h[1])
            current_idx = sorted_hits[0][0]
            trace = world[ current_idx ].interact( current_ray )
            if len(trace) < 1:
                done = True
                current_ray.propagate( sorted_hits[0][1] )
                endray = Ray( current_ray.origin, Vector(0,0), n = current_ray.n, wavelength = current_ray.wavelength, L = current_ray.L )
                path.append( endray )
            else:
                path.extend( trace )
                current_ray = trace[-1].copy()
        else:
            done = True
    return path

# Analyses a list of traced paths and determines the spread of angles at the end (how collimated the beams are)
def divergence_analysis( paths ):
    dirs = []

    for path in paths:
        last = path[-1]
        d = last.direction
        if d.length() > 0.5:
            dirs.append( d )
    
    mean_dir = dirs[0]
    for i in range( 1, len(dirs) ):
        mean_dir = mean_dir + dirs[i]
    
    mean_dir = mean_dir * (1/len(dirs))

    angles = []
    for d in dirs:
        dotp = mean_dir ^ d
        angles.append( math.degrees( math.acos( dotp ) ) )
    
    #print( "Mean divergence:", np.mean( angles ) )
    return np.mean( angles )

# Function to compute where two rays intersect each other
def ray_ray_intersect( rayA, rayB ):

    # A + da * s = B + db * t 
    # A - B = da*s + db*t
    # da ^ (A-B) = s + (da^db) * t
    # da^(A-B) - (da^db) * t = s
    # 
    # A + da * ( da^(A-B) ) - da*(da^db)*t = B + db*t
    # (A - B) + da*(da^(A-B)) = (da*(da^db) + db) * t
    #  ( (A - B) + da*(da^(A-B)) ) ^ (da*(da^db) + db) / ((da*(da^db) + db)^(da*(da^db) + db)) = t
    
    # A - B = da*(da^(A-B) - (da^db) * t) + db*t
    # A - B = da*(da^(A-B)) - da*((da^db) * t) + db*t
    # db^(A-B) = (db^da)*(da^(A-B)) - da*((da^db) * t) + db^db*t
    # (db^(A-B))/(da^db) = da^(A-B)) - da*((da^db) * t + (1/da^db) * t
    # (db^(A-B))/(da^db) - da^(A-B)) = (da*((da^db) + (1/da^db)) * t

    # A - B = da*(da^(A-B)) - da*((da^db) * t) + db*t
    # (A - B)^db = (da^db)*(da^(A-B)) - (da^db)*((da^db) * t) + t
    # (A - B)^db - (da^db)*(da^(A-B)) = ( 1 - (da^db)*((da^db) ) * t


    AB = rayA.origin - rayB.origin

    dadb = rayA.direction ^ rayB.direction

    k1 = (AB^rayB.direction) - dadb * ( rayA.direction ^ AB  )
    k2 = 1 - dadb*dadb
    
    t = k1 / k2
    s = dadb*t - (rayA.direction ^ AB)     


    return s, t


# Analyses a list of traced paths and determines where the beams converge and how large the convergence area is
def focal_point_analysis( paths ):
    rays = []

    for path in paths:
        last = path[-1]
        d = last.direction
        if d.length() > 0.5:
            rays.append( last )


    fp_rays = []
    for i in range( len( rays ) - 1 ):
        for j in range( 1, len( rays ) ):
            if i != j:
                s, t = ray_ray_intersect( rays[i], rays[j] )
                #if s > 0 and t > 0:
                fpr = rays[i].propagate( s, inplace = False )
                fp_rays.append( fpr )

    Xs = []
    Ys = []
    for i in range( len( fp_rays ) ):
        Xs.append( fp_rays[i].origin.x )
        Ys.append( fp_rays[i].origin.y )
    
    return (np.mean(Xs), np.mean(Ys)), (np.std(Xs), np.std(Ys) )
        


# Generate a wavelength-dependent refractive index funtion using Sellmeier equation
def sellmeier_refractive_index( B1, B2, B3, C1, C2, C3 ):
    def refractive_index( wavelength ):
        l = wavelength / 1000 # nm -> µm
        l2 = l * l

        n2 = 1 + ( B1 * l2 ) / ( l2 - C1 ) + ( B2 * l2 ) / ( l2 - C2 ) + ( B3 * l2 ) / ( l2 - C3 )

        return math.sqrt( n2 )

    return refractive_index