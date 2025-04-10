#!/usr/bin/env python3



import math
import cv2
import numpy as np
import scipy.optimize

# General purpose 3D vector object, has overloaded arithmetic operators
class Vector( object ):

    @staticmethod
    def from_angle( theta ):
        return Vector( math.cos( theta ), math.sin( theta ) )

    @staticmethod
    def rodrigues_rotation( k, v, theta ):
        return (v * math.cos(theta)) + ((k*v)*math.sin(theta)) + k*((k^v)*(1-math.cos(theta)))

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

class NormalSegment( object ):
    def __init__(self, N) -> None:
        self.N = N

# A ray of light with origin, direction and physical parameters about the light 
class Ray( object ):
    def __init__(self, origin, direction, n = 1.0, wavelength = 550, L = 0.0, alive = True ) -> None:
        self.origin = origin
        self.direction = direction
        self.invdir = Vector( -self.direction.y, self.direction.x )
        self.n = n
        self.wavelength = wavelength
        self.L = L
        self.alive = alive
    
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
            self.L += dist * self.n
            return self
        else:
            return Ray( self.origin + (self.direction * dist), self.direction, n = self.n, wavelength = self.wavelength, L = self.L + dist * self.n, alive = self.alive )

    def rotate( self, theta, inplace = True ):
        if inplace:
            self.direction = self.direction.rotate( theta )
            self.invdir = Vector( -self.direction.y, self.direction.x )
            return self
        else:
            return Ray( self.origin, self.direction.rotate( theta ), n = self.n, wavelength = self.wavelength, L = self.L, alive = self.alive )

    def __repr__( self ):
        return "Ray( origin = %s, direction = %s, n = %.2f, λ = %.1fnm, L = %.1f )" % ( self.origin, self.direction, self.n, self.wavelength, self.L )

    def copy( self ):
        return Ray( self.origin*1, self.direction*1, n = self.n, wavelength = self.wavelength, L = self.L, alive = self.alive )


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



def grating_diffraction( segment, ray, order, lpm, blaze_angle = 0, transmission = True ):
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
    
    Ng = Ng.rotate( blaze_angle )

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





# Traces a ray through the world (a list of optical elements) in order
def raytrace_sequential( world, ray ):
    path = [ray.copy()]
    current_ray = ray.copy()
    
    for elem in world:
        dist = elem.bbox.intersect( current_ray )
        if dist:
            next_ray = current_ray.propagate( dist, inplace = False )
            trace = elem.interact( next_ray )
            if len(trace) < 1:
                current_ray.propagate( dist )
                endray = current_ray.copy()
                endray.alive = False
                path.append( endray )
                return path
            else:
                path.extend( trace )
                current_ray = trace[-1].copy()
        else:
            current_ray.propagate( dist )
            endray = current_ray.copy()
            endray.alive = False
            path.append( endray )
            return path

    return path

# Traces a ray through the world (a list of optical elements) without assuming the order
def raytrace_nonsequential( world, ray ):
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
            next_ray = current_ray.propagate( sorted_hits[0][1], inplace = False )
            trace = world[ current_idx ].interact( next_ray )
            if len(trace) < 1:
                done = True
                current_ray.propagate( sorted_hits[0][1] )
                endray = current_ray.copy()
                endray.alive = False
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
        if last.alive:
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
        if last.alive:
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



# Some glass parameters
# from SCHOTT AG
GLASS = {}
GLASS["N-BK7"] = sellmeier_refractive_index( 1.03961212, 0.231792344, 1.01046945, 0.00600069867, 0.0200179144, 103.560653 )
GLASS["N-SF2"] = sellmeier_refractive_index( 1.47343127, 0.163681849, 1.369208990, 0.01090190980, 0.0585683687, 127.4049330  )
GLASS["N-SF8"] = sellmeier_refractive_index( 1.55075812, 0.209816918, 1.462054910, 0.01143383440, 0.0582725652, 133.2416500 )
GLASS["N-PSK53A"] = sellmeier_refractive_index( 1.38121836, 0.196745645, 0.886089205, 0.00706416337, 0.0233251345, 97.4847345 )
GLASS["N-LASF9"] = sellmeier_refractive_index( 2.00029547, 0.298926886, 1.806918430, 0.0121426017, 0.0538736236, 156.5308290 )
GLASS["N-F2"] = sellmeier_refractive_index( 1.39757037, 0.159201403, 1.268654300, 0.00995906143, 0.0546931752, 119.2483460 )




class ConicFunctions( object ):
    @staticmethod
    def conic_z( x, y, k, c ):
        r = x**2 + y**2
        return c*r / ( 1 + math.sqrt( 1 - (1 + k)*c*c*r ) )

    @staticmethod
    def error_func( ray, q, k, c ):
        new_ray = ray.propagate(q, inplace = False)
        #print( "error_func, new_ray:", new_ray )
        rx = new_ray.origin.x
        ry = new_ray.origin.y
        rz = new_ray.origin.z

        cz = ConicFunctions.conic_z( rz, ry, k, c )
        return cz - rx

    @staticmethod
    def diff_error_func( ray, q, k, c ):
        error0 = ConicFunctions.error_func( ray, q - 1e-3, k, c )
        error1 = ConicFunctions.error_func( ray, q + 1e-3, k, c )

        
        #print( "error0:", error0, "error1:", error1)

        return (error1 - error0) / (2e-3)

    @staticmethod
    def find_intersection( ray, k, c, eps=1e-3 ):
        q = 0

        for i in range( 16 ):
            A = ConicFunctions.error_func( ray, q, k, c )
            B = ConicFunctions.diff_error_func( ray, q, k, c )
            #print( "debug, A:", A, "B:", B)
            q = q - A / B
            #print( "-->%i: A=%.3f B=%.3f q=%.3f"%(i,A, B, q))
            if abs(A) < eps:
                break
        return q
        

    @staticmethod
    def compute_normal( pos, k, c ):
        e = 1e-3
        h_z0 = ConicFunctions.conic_z( pos.z, pos.y-e, k, c )
        h_z1 = ConicFunctions.conic_z( pos.z, pos.y+e, k, c )

        v_z0 = ConicFunctions.conic_z( pos.z-e, pos.y, k, c )
        v_z1 = ConicFunctions.conic_z( pos.z+e, pos.y, k, c )

        vec_h = Vector( pos.x, pos.y+e, h_z1) - Vector( pos.x, pos.y-e, h_z0)
        vec_v = Vector( pos.x+e, pos.y, v_z1) - Vector( pos.x-e, pos.y, v_z0)

        n = ~(vec_h * vec_v)
        return Vector.rodrigues_rotation( Vector(0,1,0), n, math.radians(90) )

class AsphericFunctions( object ):
    @staticmethod
    def aspheric_z( x, y, k, c, params ):
        r = x**2 + y**2

        A = 0.0
        for i in range( len( params ) ):
            p = 4 + 2*i
            A += params[i]*r**p

        return c*r / ( 1 + math.sqrt( 1 - (1 + k)*c*c*r ) ) + A

    @staticmethod
    def error_func( ray, q, k, c, params ):
        new_ray = ray.propagate(q, inplace = False)
        #print( "error_func, new_ray:", new_ray )
        rx = new_ray.origin.x
        ry = new_ray.origin.y
        rz = new_ray.origin.z

        cz = AsphericFunctions.aspheric_z( rz, ry, k, c, params )
        return cz - rx

    @staticmethod
    def diff_error_func( ray, q, k, c, params ):
        error0 = AsphericFunctions.error_func( ray, q - 1e-3, k, c, params )
        error1 = AsphericFunctions.error_func( ray, q + 1e-3, k, c, params )

        
        #print( "error0:", error0, "error1:", error1)

        return (error1 - error0) / (2e-3)

    @staticmethod
    def find_intersection( ray, k, c, params, eps=1e-3 ):
        q = 0

        for i in range( 16 ):
            A = AsphericFunctions.error_func( ray, q, k, c, params )
            B = AsphericFunctions.diff_error_func( ray, q, k, c, params )
            #print( "debug, A:", A, "B:", B)
            q = q - A / B
            #print( "-->%i: A=%.3f B=%.3f q=%.3f"%(i,A, B, q))
            if abs(A) < eps:
                break
        return q
        

    @staticmethod
    def compute_normal( pos, k, c, params ):
        e = 1e-3
        h_z0 = AsphericFunctions.aspheric_z( pos.z, pos.y-e, k, c, params )
        h_z1 = AsphericFunctions.aspheric_z( pos.z, pos.y+e, k, c, params )

        v_z0 = AsphericFunctions.aspheric_z( pos.z-e, pos.y, k, c, params )
        v_z1 = AsphericFunctions.aspheric_z( pos.z+e, pos.y, k, c, params )

        vec_h = Vector( pos.x, pos.y+e, h_z1) - Vector( pos.x, pos.y-e, h_z0)
        vec_v = Vector( pos.x+e, pos.y, v_z1) - Vector( pos.x-e, pos.y, v_z0)

        n = ~(vec_h * vec_v)
        return Vector.rodrigues_rotation( Vector(0,1,0), n, math.radians(90) )



# A refractive lens with two conic surfaces
class ConicLens( object ):
    def __init__(self, centre, height, thickness, r1, r2, k, n, theta ) -> None:
        self.centre = centre
        self.r1 = r1
        self.r2 = r2
        self.n = n
        self.theta = theta
        self.height = height
        self.thickness = thickness
        self.k = k
        
        h2 = height / 2
        t2 = thickness / 2

        self.lhs_centre = Vector( -t2, 0 )
        self.rhs_centre = Vector( t2, 0)

        self.joined_lhs = None
        self.joined_rhs = None
        

        points = []

        dy = 1e-2

        y = -h2
        while y < h2:
            x = ConicFunctions.conic_z( 0, y, self.k, 1 / r1 ) 
            points.append( Vector( -r1*0 - t2 + x, y ) )
            y += dy
        
        y = h2
        while y > -h2:
            x = ConicFunctions.conic_z( 0, y, self.k, 1 / r2 ) 
            points.append( Vector(  r2*0 + t2 + x, y ) )
            y -= dy
        
        self.points = []
        for pt in points:
            self.points.append( pt.rotate( theta ) + centre )
        
        self.bbox = BoundingBox.from_points( self.points )


        self.segments = []
        for i in range( 1, len( self.points ) ):
            self.segments.append( Segment( self.points[i-1], self.points[i] ) )
        
        self.segments.append( Segment(self.points[-1], self.points[0] ))
        
    
    def _interact( self, ray, from_outside = True ):
        LHS = 1
        RHS = 2
        iray = ray.copy()
        iray.origin = iray.origin - self.centre

        iray.origin = iray.origin.rotate( -self.theta )
        iray.direction = iray.direction.rotate( -self.theta )

        dotp = iray.direction ^ Vector(-1,0,0)

        if not from_outside:
            dotp = -dotp

        side = 0
        if dotp < 0:
            side = LHS
            iray.origin = iray.origin - self.lhs_centre
            dz = ConicFunctions.find_intersection( iray, self.k, 1/self.r1 )
            iray.propagate( dz, inplace = True )
            normal = ConicFunctions.compute_normal( iray.origin, self.k, 1/self.r1 )
        else:
            side = RHS
            iray.origin = iray.origin - self.rhs_centre
            dz = ConicFunctions.find_intersection( iray, self.k, 1/self.r2 )
            iray.propagate( dz, inplace = True )
            normal = ConicFunctions.compute_normal( iray.origin, self.k, 1/self.r2 )

        contact_height = abs(iray.origin ^ Vector(0,1,0) )        
        if contact_height > self.height/2:
            return False


        if from_outside:
            if (side == LHS) and (self.joined_lhs is not None):
                new_ray = iray.copy()
            elif (side == RHS) and (self.joined_rhs is not None):
                new_ray = iray.copy()
            else:
                new_ray = snell( NormalSegment(normal), iray, 1.0, self.n )
        else:
            if (side == LHS) and (self.joined_lhs is not None):
                new_ray = snell( NormalSegment(normal), iray, self.n, self.joined_lhs.n )
            elif (side == RHS) and (self.joined_rhs is not None):
                new_ray = snell( NormalSegment(normal), iray, self.n, self.joined_rhs.n )
            else:
                new_ray = snell( NormalSegment(normal), iray, self.n, 1.0 )

        if dotp < 0:
            new_ray.origin = new_ray.origin + self.lhs_centre
        else:
            new_ray.origin = new_ray.origin + self.rhs_centre

        new_ray.origin = new_ray.origin.rotate( self.theta )
        new_ray.direction = new_ray.direction.rotate( self.theta )
        new_ray.origin = new_ray.origin + self.centre

        return new_ray
        
    def interact( self, ray ):
        out = []
        try:
            refracted_ray0 = self._interact( ray, from_outside=True)
            if not refracted_ray0:
                return []
            out.append( refracted_ray0 )
            try:
                refracted_ray1 = self._interact( refracted_ray0, from_outside=False)
                if not refracted_ray1:
                    refracted_ray0.alive = False
                    return [refracted_ray0]
                out.append( refracted_ray1 )
            except ValueError:
                out[0].alive = False
        except ValueError:
            pass
        return out


class ConicMirror( object ):
    def __init__(self, centre, height, thickness, r1, k, theta ) -> None:
        self.centre = centre
        self.r1 = r1
        self.r2 = 10e4
        r2 = self.r2
        self.theta = theta
        self.height = height
        self.thickness = thickness
        self.k = k
        h2 = height / 2
        t2 = thickness / 2
        self.n = -1

        self.lhs_centre = Vector( -t2, 0 )
        self.rhs_centre = Vector( t2, 0)

        points = []

        dy = 1e-2

        y = -h2
        while y < h2:
            x = ConicFunctions.conic_z( 0, y, 0, 1 / r1 ) 
            points.append( Vector( -r1*0 - t2 + x, y ) )
            y += dy
        
        y = h2
        while y > -h2:
            x = ConicFunctions.conic_z( 0, y, 0, 1 / r2 ) 
            points.append( Vector(  r2*0 + t2 + x, y ) )
            y -= dy
        
        self.points = []
        for pt in points:
            self.points.append( pt.rotate( theta ) + centre )
        
        self.bbox = BoundingBox.from_points( self.points )


        self.segments = []
        for i in range( 1, len( self.points ) ):
            self.segments.append( Segment( self.points[i-1], self.points[i] ) )
        
        self.segments.append( Segment(self.points[-1], self.points[0] ))
        
    
    def _interact( self, ray ):
        iray = ray.copy()
        iray.origin = iray.origin - self.centre

        iray.origin = iray.origin.rotate( -self.theta )
        iray.direction = iray.direction.rotate( -self.theta )

        dotp = iray.direction ^ Vector(-1,0,0)

        if dotp < 0:
            iray.origin = iray.origin - self.lhs_centre
            dz = ConicFunctions.find_intersection( iray, 0, 1/self.r1 )
            iray.propagate( dz, inplace = True )
            normal = ConicFunctions.compute_normal( iray.origin, 0, 1/self.r1 )
        else:
            iray.origin = iray.origin - self.rhs_centre
            dz = ConicFunctions.find_intersection( iray, 0, 1/self.r2 )
            iray.propagate( dz, inplace = True )
            normal = ConicFunctions.compute_normal( iray.origin, 0, 1/self.r2 )

        contact_height = abs(iray.origin ^ Vector(0,1,0) )
        if contact_height > self.height/2:
            return False

        #new_ray = snell( NormalSegment(normal), iray, self.n, 1.0 )
        new_ray = reflect( NormalSegment(normal), iray )

        if dotp < 0:
            new_ray.origin = new_ray.origin + self.lhs_centre
        else:
            new_ray.origin = new_ray.origin + self.rhs_centre

        new_ray.origin = new_ray.origin.rotate( self.theta )
        new_ray.direction = new_ray.direction.rotate( self.theta )
        new_ray.origin = new_ray.origin + self.centre

        return new_ray

        
    def interact( self, ray ):
        reflected_ray = self._interact( ray )
        if not reflected_ray:
            return []
        else:
            return [reflected_ray]

        
class ReflectionGrating( object ):
    def __init__(self, centre, height, thickness, r1, theta, lpm, order, blaze_angle = 0 ) -> None:
        self.centre = centre
        self.r1 = r1
        self.r2 = 10e4
        r2 = self.r2
        self.theta = theta
        self.height = height
        self.thickness = thickness
        self.k = 0
        self.lpm = lpm
        self.order = order
        self.blaze_angle = blaze_angle
        h2 = height / 2
        t2 = thickness / 2

        self.lhs_centre = Vector( -t2, 0 )
        self.rhs_centre = Vector( t2, 0)

        points = []

        dy = 1e-2

        y = -h2
        while y < h2:
            x = ConicFunctions.conic_z( 0, y, 0, 1 / r1 ) 
            points.append( Vector( -r1*0 - t2 + x, y ) )
            y += dy
        
        y = h2
        while y > -h2:
            x = ConicFunctions.conic_z( 0, y, 0, 1 / r2 ) 
            points.append( Vector(  r2*0 + t2 + x, y ) )
            y -= dy
        
        self.points = []
        for pt in points:
            self.points.append( pt.rotate( theta ) + centre )
        
        self.bbox = BoundingBox.from_points( self.points )


        self.segments = []
        for i in range( 1, len( self.points ) ):
            self.segments.append( Segment( self.points[i-1], self.points[i] ) )
        
        self.segments.append( Segment(self.points[-1], self.points[0] ))
        
    
    def _interact( self, ray ):
        iray = ray.copy()
        iray.origin = iray.origin - self.centre

        iray.origin = iray.origin.rotate( -self.theta )
        iray.direction = iray.direction.rotate( -self.theta )

        dotp = iray.direction ^ Vector(-1,0,0)

        if dotp < 0:
            iray.origin = iray.origin - self.lhs_centre
            dz = ConicFunctions.find_intersection( iray, 0, 1/self.r1 )
            iray.propagate( dz, inplace = True )
            normal = ConicFunctions.compute_normal( iray.origin, 0, 1/self.r1 )
        else:
            iray.origin = iray.origin - self.rhs_centre
            dz = ConicFunctions.find_intersection( iray, 0, 1/self.r2 )
            iray.propagate( dz, inplace = True )
            normal = ConicFunctions.compute_normal( iray.origin, 0, 1/self.r2 )

        contact_height = abs(iray.origin ^ Vector(0,1,0) )
        if contact_height > self.height/2:
            return False

        #new_ray = snell( NormalSegment(normal), iray, self.n, 1.0 )
        #new_ray = reflect( NormalSegment(normal), iray )
        new_ray = grating_diffraction( NormalSegment(normal), iray, self.order, self.lpm, self.blaze_angle, transmission=False )


        if dotp < 0:
            new_ray.origin = new_ray.origin + self.lhs_centre
        else:
            new_ray.origin = new_ray.origin + self.rhs_centre

        new_ray.origin = new_ray.origin.rotate( self.theta )
        new_ray.direction = new_ray.direction.rotate( self.theta )
        new_ray.origin = new_ray.origin + self.centre

        return new_ray

        
    def interact( self, ray ):
        reflected_ray = self._interact( ray )
        if not reflected_ray:
            return []
        else:
            return [reflected_ray]


class Baffle( object ):
    def __init__(self, centre, height, thickness,  theta ) -> None:
        self.centre = centre
        self.r1 = 10e4
        r1 = self.r1
        self.r2 = 10e4
        r2 = self.r2
        self.theta = theta
        self.height = height
        self.thickness = thickness
        self.k = 0
        self.n = 1.0
        h2 = height / 2
        t2 = thickness / 2

        self.lhs_centre = Vector( -t2, 0 )
        self.rhs_centre = Vector( t2, 0)

        points = []

        dy = 1e-1

        y = -h2
        while y < h2:
            x = ConicFunctions.conic_z( 0, y, 0, 1 / r1 ) 
            points.append( Vector( -r1*0 - t2 + x, y ) )
            y += dy
        
        y = h2
        while y > -h2:
            x = ConicFunctions.conic_z( 0, y, 0, 1 / r2 ) 
            points.append( Vector(  r2*0 + t2 + x, y ) )
            y -= dy
        
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
    def __init__(self, centre, height, thickness, r1, theta, lpm, order, blaze_angle = 0 ) -> None:
        self.centre = centre
        self.r1 = r1
        self.r2 = 10e4
        r2 = self.r2
        self.theta = theta
        self.height = height
        self.thickness = thickness
        self.k = 0
        self.lpm = lpm
        self.order = order
        self.blaze_angle = blaze_angle
        h2 = height / 2
        t2 = thickness / 2

        self.lhs_centre = Vector( -t2, 0 )
        self.rhs_centre = Vector( t2, 0)

        points = []

        dy = 1e-2

        y = -h2
        while y < h2:
            x = ConicFunctions.conic_z( 0, y, 0, 1 / r1 ) 
            points.append( Vector( -r1*0 - t2 + x, y ) )
            y += dy
        
        y = h2
        while y > -h2:
            x = ConicFunctions.conic_z( 0, y, 0, 1 / r2 ) 
            points.append( Vector(  r2*0 + t2 + x, y ) )
            y -= dy
        
        self.points = []
        for pt in points:
            self.points.append( pt.rotate( theta ) + centre )
        
        self.bbox = BoundingBox.from_points( self.points )


        self.segments = []
        for i in range( 1, len( self.points ) ):
            self.segments.append( Segment( self.points[i-1], self.points[i] ) )
        
        self.segments.append( Segment(self.points[-1], self.points[0] ))
        
    
    def _interact( self, ray, skip_grating = False ):
        iray = ray.copy()
        iray.origin = iray.origin - self.centre

        iray.origin = iray.origin.rotate( -self.theta )
        iray.direction = iray.direction.rotate( -self.theta )

        dotp = iray.direction ^ Vector(-1,0,0)

        if dotp < 0:
            iray.origin = iray.origin - self.lhs_centre
            dz = ConicFunctions.find_intersection( iray, 0, 1/self.r1 )
            iray.propagate( dz, inplace = True )
            normal = ConicFunctions.compute_normal( iray.origin, 0, 1/self.r1 )
        else:
            iray.origin = iray.origin - self.rhs_centre
            dz = ConicFunctions.find_intersection( iray, 0, 1/self.r2 )
            iray.propagate( dz, inplace = True )
            normal = ConicFunctions.compute_normal( iray.origin, 0, 1/self.r2 )

        contact_height = abs(iray.origin ^ Vector(0,1,0) )
        if contact_height > self.height/2:
            return False

        #new_ray = snell( NormalSegment(normal), iray, self.n, 1.0 )
        #new_ray = reflect( NormalSegment(normal), iray )
        if skip_grating:
            new_ray = iray.copy()
        else:
            new_ray = grating_diffraction( NormalSegment(normal), iray, self.order, self.lpm, self.blaze_angle, transmission=True )


        if dotp < 0:
            new_ray.origin = new_ray.origin + self.lhs_centre
        else:
            new_ray.origin = new_ray.origin + self.rhs_centre

        new_ray.origin = new_ray.origin.rotate( self.theta )
        new_ray.direction = new_ray.direction.rotate( self.theta )
        new_ray.origin = new_ray.origin + self.centre

        return new_ray

        
    def interact( self, ray ):

        ray0 = self._interact( ray, skip_grating=False )
        if not ray0:
            return []
    
        ray1 = self._interact( ray0, skip_grating=True )
        if not ray1:
            ray0.alive = False
            return [ray0] 

        return [ray0, ray1]

class ApertureStop( object ):
    def __init__(self, centre, height ) -> None:
        self.centre = centre
        self.height = height/2
        self.thickness = 0

        self.top = self.centre + Vector( 0, self.height)
        self.bottom = self.centre - Vector( 0, self.height)
        
        self._internal = Ray( self.bottom, ~(self.top - self.bottom))

        self.points = [self.top + Vector(0, 1e-3), self.top + Vector(1e-3, 0) + Vector(0, 1e-3), self.bottom + Vector(0, -1e-3), self.bottom + Vector(1e-3, 0) + Vector(0, -1e-3)] # slight expansion to reduce numerical instability
        

        self.bbox = BoundingBox.from_points( self.points )

    def draw( self, ax ):

        ax.plot( [self.centre.x - 1, self.centre.x + 1], [self.centre.y + self.height, self.centre.y + self.height], '-k')
        ax.plot( [self.centre.x, self.centre.x], [self.centre.y + self.height, self.centre.y + self.height+1], '-k')
        
        ax.plot( [self.centre.x - 1, self.centre.x + 1], [self.centre.y - self.height, self.centre.y - self.height], '-k')
        ax.plot( [self.centre.x, self.centre.x], [self.centre.y - self.height, self.centre.y - self.height-1], '-k')
        


    def interact( self, ray ):
        out = [ray.copy()] # if the ray hits the aperture stop, it is ok to go through
        return out

    def crossing_point( self, ray ):
        s, t = ray_ray_intersect( ray, self._internal)
        return t


    # Marginal ray goes through the edge of the stop
    def marginal_ray_error_func( self, world, object_position, top = True, wavelength=550.0 ):
        def error_func( angle ):
            ray = Ray( object_position, Vector.from_angle(angle), wavelength=wavelength )
            path = raytrace_sequential( world, ray )
            t = self.crossing_point( path[-1] )
            if top:
                return abs(t - self.height*2)
            else:
                return abs(t)
        return error_func
    
    # Chief ray goes through the centre of the stop
    def chief_ray_error_func( self, world, object_position, object_height, wavelength=550.0 ):
        def error_func( angle ):
            ray = Ray( object_position + Vector(0, object_height), Vector.from_angle(angle), wavelength=wavelength )
            path = raytrace_sequential( world, ray )
            t = self.crossing_point( path[-1] )
            return abs(t - self.height)
        return error_func


    # Find marginal ray hitting this aperture stop, angle_bounds = range of angles (in radian) to search
    def solve_marginal_ray( self, world, object_position, angle_bounds = (-0.8, 0.8), top = True, wavelength=550.0 ):
        
        # Select elements on the left hand side of the system relative to stop
        world_lhs = []
        for elem in world:
            if elem is not self:
                world_lhs.append( elem )
            else:
                break
        
        # Find marginal ray by minimising the corresponding error function
        error_func = self.marginal_ray_error_func( world_lhs, object_position, top, wavelength=wavelength )
        result = scipy.optimize.minimize_scalar( error_func, bounds=angle_bounds, method='bounded' )
        
        marginal_ray = Ray( object_position, Vector.from_angle( result.x ), wavelength = wavelength )
        return marginal_ray

    # Find chief ray hitting centre of this aperture stop, angle_bounds = range of angles (in radian) to search
    def solve_chief_ray( self, world, object_position, object_height, angle_bounds = (-0.8, 0.8), wavelength=550.0 ):
        
        # Select elements on the left hand side of the system relative to stop
        world_lhs = []
        for elem in world:
            if elem is not self:
                world_lhs.append( elem )
            else:
                break
        
        # Find chief ray by minimising the corresponding error function
        error_func = self.chief_ray_error_func( world_lhs, object_position, object_height, wavelength=wavelength )
        result = scipy.optimize.minimize_scalar( error_func, bounds=angle_bounds, method='bounded' )
        
        chief_ray = Ray( object_position + Vector(0, object_height), Vector.from_angle( result.x ), wavelength=wavelength )
        return chief_ray


def generate_ray_bundle( world, aperture_stop, object_position, wavelength=550.0 ):

    marginal_ray0 = aperture_stop.solve_marginal_ray( world, object_position, top = True, wavelength=wavelength )
    marginal_ray1 = aperture_stop.solve_marginal_ray( world, object_position, top = False, wavelength=wavelength )
    chief_ray = aperture_stop.solve_chief_ray( world, object_position, 0, wavelength=wavelength )

    return [marginal_ray0, chief_ray, marginal_ray1]
    
def find_focus_point( world, object_position, wavelength=550.0, method = None ):
    stop = None
    for elem in world:
        if isinstance( elem, ApertureStop ):
            stop = elem
            break
    
    if stop is None:
        return False
    
    rays = generate_ray_bundle( world, stop, object_position, wavelength = wavelength )
    paths = []
    for ray in rays:
        path = raytrace_sequential( world, ray )
        paths.append( path )
    
    points = []

    for i in range( len( paths ) ):
        for j in range( i, len( paths ) ):
            if i != j:
                s, t = ray_ray_intersect( paths[i][-1], paths[j][-1] )
                pos = paths[i][-1].propagate( s, inplace = False )
                points.append( pos.origin )
    
    Xs = [ p.x for p in points ]
    Ys = [ p.y for p in points ]
    if method is not None:
        return Vector( method( Xs ), method( Ys ) )
    else:
        return Vector( np.mean( Xs ), np.mean( Ys ) )



def join_lens_surfaces( left, right ):
    left.joined_rhs = right
    right.joined_lhs = left


def read_from_zmx( fn, position, theta ):
    output = []
    surfaces = []
    next_distance = 0
    current_surface = {}
    with open( fn, 'r' ) as handle:
        for line in handle:
            line = line.strip().replace("\0", "")
            if len( line ) < 1:
                continue
            
            parts = line.split()
            op = parts[0]
            args = parts[1:]

            if op == "SURF":
                if len( current_surface ) > 0:
                    if current_surface["diameter"] > 1e-3:
                        surfaces.append( current_surface )
                current_surface = {"Z": next_distance, "last": False, "diameter": -1, "conic": 0 }

            elif op == "CURV":
                current_surface["C"] = float( args[0] )

            elif op ==  "DISZ":
                if args[0] == "INFINITY":
                    next_distance = 0
                else:
                    next_distance = float( args[0] )

            elif op == "GLAS":
                current_surface["glass"] = args[0]
            
            elif op == "DIAM":
                current_surface["diameter"] = 2*float(args[0])

            elif op == "MEMA":
                current_surface["mechanical_diameter"] = 2*float(args[0])
            
            elif op == "CONI":
                current_surface["conic"] = float(args[0])

            elif op == "MAZH":
                current_surface["last"] = True
    
    if len( surfaces ) == 2:
        if surfaces[0]["glass"] == "MIRROR":
            try:
                R1 = 1 / surfaces[0]["C"]
            except ZeroDivisionError:
                R1 = 10e3
            
            try:
                R2 = 1 / surfaces[1]["C"]
            except ZeroDivisionError:
                R2 = 10e3
            
            height = surfaces[0]["mechanical_diameter"]
            thickness = surfaces[1]["Z"]

            #if abs(surfaces[0]["conic"] + 1) < 1e-6:
            #    output.append( ParabolicMirror(Vector(0, surfaces[0]["Z"]) + position, height, thickness, R1, theta))
            #else:
            #    output.append( SphericalMirror( Vector(0, surfaces[0]["Z"]) + position, height, thickness, R1, theta ))
            output.append( ConicMirror( Vector(0, surfaces[0]["Z"]) + position, height, thickness, R1, surfaces[0]["conic"], theta ))

        else:
            try:
                R1 = 1 / surfaces[0]["C"]
            except ZeroDivisionError:
                R1 = 10e3
            
            try:
                R2 = 1 / surfaces[1]["C"]
            except ZeroDivisionError:
                R2 = 10e3
            
            height = surfaces[0]["mechanical_diameter"]
            
            nidx = GLASS[ surfaces[0]["glass"] ]
            thickness = surfaces[1]["Z"]

            #output.append( SphericalLens(Vector( surfaces[0]["Z"], 0) + position, height, thickness, R1, R2, nidx, theta) )
            output.append( ConicLens(Vector( surfaces[0]["Z"], 0) + position, height, thickness, R1, R2, surfaces[0]["conic"], nidx, theta) )

    return output


class ParaxialApproximation( object ):

    @staticmethod
    def to_ray( paraxial_ray ):
        return Ray( Vector(paraxial_ray[0], paraxial_ray[1][0][0]), Vector.from_angle( paraxial_ray[1][1][0] ) )

    @staticmethod
    def element_to_matrix( element, wavelength = 550.0 ):
        if isinstance( element, ApertureStop ):
            R1 = np.eye(2)
            T1 = np.eye(2)
            R2 = np.eye(2)
            return R1, T1, R2
            
        
        
        if isinstance( element.n, float ):
            n = element.n
        else:
            n = element.n( wavelength )
        
        power1 = (n - 1.0) / element.r1
        power2 = (1.0 - n) / element.r2
        
        R1 = np.array([[1,0], [-power1, 1]])
        T1 = np.array([[1, element.thickness / n], [0, 1]])
        R2 = np.array([[1,0], [-power2, 1]])
        
        return R1, T1, R2
    
    @staticmethod
    def raytrace( world, ray ):
        path = []
        
        w0 = ray.direction.angle() * 1.0
        y0 = ray.origin.y
        L = np.array([[y0], [w0]])
        
        current_x = ray.origin.x
        path.append((current_x, L))

        
        for elem in world:
            R1, T1, R2 = ParaxialApproximation.element_to_matrix( elem, ray.wavelength )
            
            t = elem.centre.x - current_x - elem.thickness / 2
            T = np.array([[1, t / 1.0], [0, 1]])

            L = np.matmul( T, L )
            current_x += t
            L = np.matmul( R1,  L)
            path.append( (current_x, L))

            L = np.matmul( T1, L )

            current_x += elem.thickness
            L = np.matmul( R2,  L)
            path.append( (current_x, L))
        
        return path
            
    @staticmethod
    def compute_system_matrix( world, object_position, wavelength = 550.0, skip_virtual = False ):
        M = np.eye(2)
        
        current_x = object_position.x
        
        for elem in world:
            if skip_virtual:
                if isinstance(elem, ApertureStop):
                    continue
            R1, T1, R2 = ParaxialApproximation.element_to_matrix( elem, wavelength=wavelength )
            t = elem.centre.x - current_x - elem.thickness / 2
            T = np.array([[1, t / 1.0], [0, 1]])
            
            M = np.matmul( T, M )
            M = np.matmul( R1, M )
            M = np.matmul( T1, M )
            M = np.matmul( R2, M )
            
            current_x += t
            current_x += elem.thickness
        
        return M, current_x
    
    @staticmethod
    def compute_inverse_system_matrix( world, wavelength = 550.0, skip_virtual = False ):
        M = np.eye(2)
        
        current_x = 0
        first = True

        for elem in world:
            if skip_virtual:
                if isinstance(elem, ApertureStop):
                    continue

            R1, T1, R2 = ParaxialApproximation.element_to_matrix( elem, wavelength=wavelength )
            if first:
                first = False
                t = 0
            else:
                t = elem.centre.x - current_x - elem.thickness / 2
            T = np.array([[1, t / 1.0], [0, 1]])
            
            M = np.matmul( T, M )
            M = np.matmul( R1, M )
            M = np.matmul( T1, M )
            M = np.matmul( R2, M )
            
            current_x += t
            current_x += elem.thickness
        
        return np.linalg.inv(M)
    
    @staticmethod
    def solve_marginal_ray( world, object_position, angle_bounds = (-0.2, 0.2), top = True, wavelength = 550.0 ):
        world_lhs = []
        stop = None
        for elem in world:
            world_lhs.append( elem )
            if isinstance( elem, ApertureStop ):
                stop = elem
                break
        M, _ = ParaxialApproximation.compute_system_matrix( world_lhs, object_position, wavelength=wavelength )

        def error_func( ray_angle ):
            L0 = np.array([[object_position.y], [ray_angle]])
            L1 = np.matmul( M, L0 )
            if top:
                return abs( +stop.height - L1[0][0] )
            else:
                return abs( -stop.height - L1[0][0] )

        result = scipy.optimize.minimize_scalar( error_func, bounds=angle_bounds, method='bounded'  )
        
        return Ray( object_position, Vector.from_angle( result.x ), wavelength=wavelength )


    @staticmethod
    def solve_chief_ray( world, object_position, height, angle_bounds = (-0.2, 0.2), wavelength = 550.0 ):
        world_lhs = []
        for elem in world:
            world_lhs.append( elem )
            if isinstance( elem, ApertureStop ):
                break
        M, _ = ParaxialApproximation.compute_system_matrix( world_lhs, object_position, wavelength=wavelength )

        def error_func( ray_angle ):
            L0 = np.array([[object_position.y + height], [ray_angle]])
            L1 = np.matmul( M, L0 )
            return abs( L1[0][0] )

        result = scipy.optimize.minimize_scalar( error_func, bounds=angle_bounds, method='bounded'  )
        
        return Ray( object_position + Vector(0, height), Vector.from_angle( result.x ), wavelength=wavelength)


    @staticmethod
    def generate_ray_bundle( world, object_position, angle_bounds = (-0.2, 0.2), wavelength = 550.0 ):
        
        marginal_ray0 = ParaxialApproximation.solve_marginal_ray( world, object_position, angle_bounds=angle_bounds, wavelength=wavelength, top = True )
        marginal_ray1 = ParaxialApproximation.solve_marginal_ray( world, object_position, angle_bounds=angle_bounds, wavelength=wavelength, top = False )
        chief_ray = ParaxialApproximation.solve_chief_ray( world, object_position, 0, angle_bounds=angle_bounds, wavelength=wavelength)

        return marginal_ray0, chief_ray, marginal_ray1
    
    @staticmethod
    def find_focus_point( world, object_position, wavelength = 550.0 ):
        rays = ParaxialApproximation.generate_ray_bundle( world, object_position, wavelength = wavelength )

        results = []
        for ray in rays:
            path = ParaxialApproximation.raytrace( world, ray )

            results.append( ParaxialApproximation.to_ray( path[-1] ) )
     
        points = []

        for i in range( len( results ) ):
            for j in range( i, len( results ) ):
                if i != j:
                    s, t = ray_ray_intersect( results[i], results[j] )
                    pos = results[i].propagate( s, inplace = False )
                    points.append( pos.origin )
        
        Xs = [ p.x for p in points ]
        Ys = [ p.y for p in points ]
        
        return Vector( np.mean( Xs ), np.mean( Ys ) )     
                
        
        
        
    @staticmethod
    def get_system_parameters( world, object_position, wavelength = 550.0 ):
        OpticalAxis = Ray( object_position, Vector.from_angle(0) )
        M, length = ParaxialApproximation.compute_system_matrix( world, object_position, wavelength=wavelength, skip_virtual=True )

        marginal_ray = ParaxialApproximation.solve_marginal_ray( world, object_position, wavelength=wavelength )
        chief_ray = ParaxialApproximation.solve_chief_ray( world, object_position, height = 1, wavelength=wavelength )

        Lcollimated0 = np.array([[1], [0]])
        Lcollimated1 = np.matmul( M, Lcollimated0 )
        
        # Back focal distance
        BFD = -Lcollimated1[0][0] / Lcollimated1[1][0]

        invM = ParaxialApproximation.compute_inverse_system_matrix( world, wavelength=wavelength, skip_virtual=True )

        # Tracing back to front
        Lrev0 = np.array([[1], [0]])
        Lrev1 = np.matmul( invM, Lrev0 )

        # Front focal distance
        FFD = -Lrev1[0][0] / Lrev1[1][0]

        Lmarginal0 = np.array([[0], [marginal_ray.direction.angle()]])
        Lmarginal1 = np.matmul( M, Lmarginal0 )

        mag_angular = Lmarginal1[1][0] / Lmarginal0[1][0]


        marginal_focus = -Lmarginal1[0][0] / Lmarginal1[1][0]

        NA = abs(Lmarginal1[1][0])


        Lchief0 = np.array([[chief_ray.origin.y], [chief_ray.direction.angle()]])
        Lchief1 = np.matmul( M, Lchief0 )

        Tfocus = np.array([[1, marginal_focus], [0, 1]])

        Lchief2 = np.matmul( Tfocus, Lchief1 )
        Lmarginal2 = np.matmul( Tfocus, Lmarginal1 )

        mag_transverse = Lchief2[0][0] / Lchief0[0][0]
        
        power = -Lcollimated1[1][0] / Lcollimated0[0][0]
        EFL = 1.0 / power


        # front focal length
        power_f = Lrev1[1][0] / Lrev0[0][0]
        ffl = -1/power_f

        # back focal length
        power_b = Lcollimated1[1][0] / Lcollimated0[0][0]
        bfl = -1/power_b
        

        # Entrance pupil
        ms, cs = ray_ray_intersect( OpticalAxis, chief_ray )
        
        ep_centre = chief_ray.propagate( cs, inplace = False )
        ep_top = math.tan( marginal_ray.direction.angle() ) * ms
        
        ep_x = ep_centre.origin.x
        ep_d = ep_top * 2
        

        # F-number
        Fn = EFL / ep_d 


        # Exit pupil
        chief_ray2 = ParaxialApproximation.to_ray((length + BFD, Lchief2))
        marginal_ray2 = ParaxialApproximation.to_ray((length + BFD, Lmarginal2))
        
        chief_ray2.direction = -chief_ray2.direction
        marginal_ray2.direction = -marginal_ray2.direction

        ms2, cs2 = ray_ray_intersect( OpticalAxis, chief_ray2 )
        xp_centre = chief_ray2.propagate( cs2, inplace = False )
        xp_x = xp_centre.origin.x
        
        dst = (xp_centre.origin - chief_ray2.origin) ^ (-OpticalAxis.direction)
        
        xp_top = abs(math.tan( marginal_ray2.direction.angle() )) * dst
        
        xp_d = 2*xp_top


        
        # Still working out which is the best way to compute these
        ep_x = Lchief0[0][0] / Lchief0[1][0] - object_position.x
        xp_x = Lchief2[0][0] / Lchief2[1][0] - marginal_focus


        return { "efl": EFL, "bfd": BFD,"bfl": bfl, "ffd": FFD, 'ffl':ffl, 'm(angular)': mag_angular, 'm(transverse)': mag_transverse, "m(pupil)": xp_d/ep_d, "NA": NA, "f/#": Fn, "power": power, "EPx": ep_x, "EPd": ep_d, "XPx": xp_x, "XPd": xp_d }