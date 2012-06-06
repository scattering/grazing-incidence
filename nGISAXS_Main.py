#!/usr/bin/python
###################################################################
# IntegralGISAXS-parallel.py
# version 0.8.0
# December 17, 2009
###################################################################
# Author: Kevin G. Yager
# Affiliation: NIST Polymers
###################################################################
# Description:
#  This script computes a detector-image for a GISAXS experiment,
# where a beam is reflected off a surface and the off-specular
# scattering (qx-qz) is recorded on a detector.
#  The code uses a real-space 'box' that is finite in size and
# subdivided into finite slices. The real-space potential in this
# box is computed using a user-modifiable function. The scattering
# is then computed by explicitly integrating the appropriate equation,
# taking account of the incident and exit wavefunctions.
#  The wavefunctions are approximated by using a Parratt-like
# reflectivity formalism, based on the box potential.
#
#  This version of the code uses parallel-processing to compute
# different parts of the detector image simultaneously. It will thus
# automatically take advantage of all CPUs and cores available.
# This parallelization offers no particular advantage on single-core
# machines (and will in fact make the script run slightly slower).
###################################################################
# INPUT:
#  The user should update the function "Potential" to point to the
# potential function of their choosing.
#  A series of initial definitions ("simulation values", below) should
# also be updated to reflect the particular situation being simulated.
###################################################################
# OUTPUT:
#  The main output is a series of images that show the reciprocal
# space intensity (detector image). A slice through real-space is
# also generated. The reflectivity curve can be optionally output.
# Of course, all the data structures could also be saved to disk for
# other analysis.
###################################################################
# KNOWN BUGS AND LIMITATIONS:
#  The finite partitioning of real-space can produce artifacts at
# q-values corresponding to the box-size and the partitioning size.
# for disordered systems, these artifacts are not significant. For
# highly ordered systems, they can produce false scattering on par
# with the actual data of interest. This can be avoided by averaging
# multiple simulations that have different size and/or partitioning.
###################################################################
# CHANGELOG:
# 2009-Dec-17: Cleaned-up code comments.
###################################################################






# Import Python libraries
###################################################################
#from nGISAXS_InstrumentDefs import loadexpparms
from math import radians, degrees, pi, sin, cos, asin, sqrt, exp, tan, log, atan, floor  # basic math
from matplotlib import ticker
import pylab as p
import numpy        # numerical python
import os           # for interacting with the host system

import pp           # parallel-python to enable parallel processing



print( ">>> Off Specular Calculation..." )



###################################################################
# Defined functions
###################################################################

# Define an iterator that yields the values along each axis
def axis( dimension):
    for index in range(limits[dimension]['num']):
        value = index*limits[dimension]['step']
        yield index, value

def axis_pp( dimension, limits ):
    for index in range(limits[dimension]['num']):
        value = index*limits[dimension]['step']
        yield index, value


def axes():
    for ix, x in axis('x'):
        for iy, y in axis('y'):
            for iz, z in axis('z'):
                yield ix, x, iy, y, iz, z

def yzaxes():
    for iz, z in axis('z'):
        for iy, y in axis('y'):
            yield iy, y, iz, z

# Define an iterator that enumerates backwards
def reverse_enumerate( sequence ):
    for iz in range(len(sequence)-1, -1, -1):
        yield iz, sequence[iz]

def print_step( string ):
    elapsed = (time.clock()-start_time)
    elapsed_string = '%3d'%elapsed    
    print( '['+elapsed_string+'s] : ' + string )



# Plotting
###################################################################


# Plotting
def auto_contourlevels( matrix, min_val=None, max_val=None ):
    lvl_start = numpy.min(matrix)
    lvl_end = numpy.max(matrix)
    lvl_spread = lvl_end-lvl_start
    if max_val:
        lvl_end = lvl_start + max_val*lvl_spread
    if min_val:
        lvl_start += min_val*lvl_spread
    lvl_n = 64
    lvl_step = (lvl_end-lvl_start)/lvl_n
    contourlevels = [ lvl_start+lvl_step*n for n in range(lvl_n) ]
    return contourlevels

def film_min_max( V_box ):

    vmin=numpy.min( V_box[:,:,1:limits['z']['num']-1] )
    vmax=numpy.max( V_box[:,:,limits['z']['num']-1] )

    vmin *= 0.99
    vmax *= 1.01

    return vmin, vmax
    



# Plot Z-averaged potential
def plot_Z(V_z, Z_z):
    print( "Plotting Vz..." )

    fig = p.figure()
    p.plot( Z_z, abs(V_z) )
    p.xlabel('z (nm)')
    p.ylabel('V')
    p.savefig( 'potential_Z_SLDavg.png' )

def plot_Y(V_box, limits):
    print( "Plotting Vy..." )

    Y_y = numpy.empty( (limits['y']['num']), numpy.float )
    V_y = numpy.empty( (limits['y']['num']), numpy.complex )
    for iy, y in axis('y'): # V_z, Z_z used to plot average V(y)
        Y_y[iy] = iy*limits['y']['step']
        V_y[iy] = numpy.average(V_box[:,iy,:])

    fig = p.figure()
    p.plot( Y_y, abs(V_y) )
    p.xlabel('y (nm)')
    p.ylabel('V')
    p.savefig( 'potential_Y_SLDavg.png' )

def plot_X(V_box, limits):
    print( "Plotting Vx..." )

    X_x = numpy.empty( (limits['x']['num']), numpy.float )
    V_x = numpy.empty( (limits['x']['num']), numpy.complex )
    for ix, x in axis('x'): # V_z, Z_z used to plot average V(x)
        X_x[ix] = ix*limits['x']['step']
        V_x[ix] = numpy.average(V_box[ix,:,:])

    fig = p.figure()
    p.plot( X_x, abs(V_x) )
    p.xlabel('x (nm)')
    p.ylabel('V')
    p.savefig( 'potential_X_SLDavg.png' )

def plot_YZContour(V_box, limits):
    print("Plotting Y-Z contour...")
    
    Y_values = numpy.empty( (limits['y']['num'], limits['z']['num']), numpy.float )
    Z_values = numpy.empty( (limits['y']['num'], limits['z']['num']), numpy.float )
    YZ_slice = numpy.empty( (limits['y']['num'], limits['z']['num']), numpy.float )
    
    # Generate 2-D arrays with y and z values for axes of contour plot
    # Flip Y array over so that air is on top (y=max) and substrate on bottom (y=0)
    for iy, y in axis('y'):
        for iz, z in axis('z'):
            Y_values[iy,iz] = y
            Z_values[iy,iz] = z

    YZ_slice = abs(V_box[0,:,:])
    fig = p.figure()
    fig = p.figure( figsize=(7.5,7.5) )
    p.subplot(111, aspect='equal' )
    p.contourf(Y_values,Z_values,numpy.fliplr(YZ_slice), antialiased=False)    
    p.xlabel('x (nm)')
    p.ylabel('z (nm)')
    p.savefig( 'potential_Y_contour.png' )
    
# Plot detector image
def plot_detector( detector_smeared, qy_grid, qz_grid, min_v, max_v, filename):
    print( "Plotting detector image..." )

    contourlevels = auto_contourlevels( detector_smeared, min_val=min_v, max_val=max_v )

    fig = p.figure( figsize=(7.5,7.5) )
    p.subplot(111, aspect='equal' )

    p.contourf(qy_grid, qz_grid, detector_smeared, contourlevels, antialiased=False, extend='both')
    
    p.xlabel(r'$Q_x \hspace{0.5} (\rm{nm}^{-1})$')
    p.ylabel(r'$Q_z \hspace{0.5} (\rm{nm}^{-1})$')
    filenamepng = filename + '.png'
    p.savefig( filenamepng )

    directory = defs_i['directory']

    filenamecsv = filename + '.csv'
    f = open(filenamecsv, 'w')
    for line in range(len(detector_smeared)):
        for column in range(len(detector_smeared[line,:])):
            f.write(str(detector_smeared[line,column])+', ')
        f.write('\n')
    f.close()


# Output the "qz" axis (the central vertical line on the detector)
def plot_qz_axis( detector_smeared, qz_grid, defs_i, axis_scaling=None ):
    print( "Plotting I(qz)..." )

    iqy = defs_i['qy_slice_1']
    fig = p.figure()
    p.plot( qz_grid[0,:], detector_smeared[iqy,:])
    if axis_scaling:
        p.axis( axis_scaling )
    p.xlabel(r'$Q_z \hspace{0.5} (\rm{nm}^{-1})$')
    p.ylabel('Log Intensity (a.u.)')
    p.semilogy()
    p.savefig( 'GISAXS_Iqz19.png' )

    iqy = defs_i['qy_slice_2']
    fig = p.figure()
    p.plot( qz_grid[0,:], detector_smeared[iqy,:] )
    if axis_scaling:
        p.axis( axis_scaling )
    p.xlabel(r'$Q_z \hspace{0.5} (\rm{nm}^{-1})$')
    p.ylabel('Log Intensity (a.u.)')
    p.semilogy()
    p.savefig( 'GISAXS_Iqz57.png' )

def plot_qz_axis2( detector_smeared, qz_grid, expIqz, defs_i, axis_scaling=None ):
    print( "Plotting I(qz)..." )

    iqy = defs_i['qy_slice_1']
    fig = p.figure()
    p.plot( qz_grid[0,:], detector_smeared[iqy,:], qz_grid[0,:], expIqz[0,:], 'r+' )
    if axis_scaling:
        p.axis( axis_scaling )
    p.xlabel(r'$Q_z \hspace{0.5} (\rm{nm}^{-1})$')
    p.ylabel('Log Intensity (a.u.)')
    p.semilogy()
    p.savefig( 'GISAXS_Iqz19.png' )

    iqy = defs_i['qy_slice_2']
    fig = p.figure()
    p.plot( qz_grid[0,:], detector_smeared[iqy,:], qz_grid[0,:], expIqz[1,:], 'r+' )
    if axis_scaling:
        p.axis( axis_scaling )
    p.xlabel(r'$Q_z \hspace{0.5} (\rm{nm}^{-1})$')
    p.ylabel('Log Intensity (a.u.)')
    p.semilogy()
    p.savefig( 'GISAXS_Iqz57.png' )
    

# Output the "qy" axis (the bottom horizontal line on the detector)
def plot_qy_axis( detector_smeared, qy_grid2, defs_i, axis_scaling=None ):

    iqz = defs_i['qz_slice']

    print( "Plotting I(qy)..." )
    fig = p.figure()
    p.plot( qy_grid2[:,iqz], detector_smeared[:,iqz] )
    if axis_scaling:
        p.axis( axis_scaling )
    p.xlabel(r'$Q_y \hspace{0.5} (\rm{nm}^{-1})$')
    p.ylabel('Log Intensity (a.u.)')
    p.semilogy()
    p.savefig( 'GISAXS_Iqy.png' )



# Potential definitions
###################################################################

def update_potential(axis_rotate, defs_i, V_box):

    eta = axis_rotate[0]
    phi = axis_rotate[1]
    theta = axis_rotate[2]

    rotation_elements = [[  cos(eta)*cos(phi)*cos(theta)-sin(eta)*sin(theta) ,
                                -cos(eta)*cos(phi)*sin(theta)-sin(eta)*cos(theta) ,
                                -cos(eta)*sin(phi)                                   ],
                         [  sin(eta)*cos(phi)*cos(theta)+cos(eta)*sin(theta) ,
                                -sin(eta)*cos(phi)*sin(theta)+cos(eta)*cos(theta) ,
                                sin(eta)*sin(phi)                                    ],
                         [ -sin(phi)*cos(theta) ,
                               sin(phi)*sin(theta) ,
                               cos(phi)                                              ]]

#    V_box = Potential_OverlapGrating(rotation_elements, defs_i, V_box)
#    V_box = Potential_Trapezoid(rotation_elements, defs_i, V_box)
    V_box = Potential_SixTrapezoid(rotation_elements, defs_i, V_box)
#    V_box = Potential_Cylinders(rotation_elements, defs_i, V_box)
    
    Z_z = numpy.empty( (limits['z']['num']), numpy.float )
    V_z = numpy.empty( (limits['z']['num']), numpy.complex )
    for iz, z in axis('z'): # V_z, Z_z used to plot average V(z)
        Z_z[iz] = iz*limits['z']['step']
        V_z[iz] = numpy.average(V_box[:,:,iz])

    return V_z, Z_z


def Potential_SimpleGrating(rotation_elements, defs_i, V_box):

    #Enter sample parameters
    grating_period = 800    # nm
    grating_thickness = 50.0    #nm         
    linewidth = 0.5*grating_period  #nm       
    total_film_thickness = 50.0  #nm

    # Enter scattering length density (SLD) as either delta, mu or qc2, mu
    # If you prefer to enter scattering length densities as delta, mu then set do_delta=True)
    # If you prefer to enter scattering length densities as qc2, mu then set do_delta=False)
    # 2*delta = thetac^2, beta = mu*wavelength/(4*pi)
    # Assume theta << 1, sin(theta) = theta
    # 2*delta - 2i*beta = (4*pi/wavelength)^2 * 2*delta - 2*i*(4*pi/wavelength)*mu

    qc2_ambient_real = 0.0 #nm^-2
    qc2_film_real = 6e-2 #nm^-2
    qc2_substrate_real = 1e-1 #nm^-2

    delta_ambient = 0.0               # unitless
    delta_film = 3.5e-6     
    delta_substrate = 8e-6            # unitless

    mu_ambient = 0.0
    mu_film = 3.5e-7
    mu_substrate = 1e-6

    do_delta = False
    if do_delta:
        pref1 = 16*pi*pi/defs_i['wavelength']/defs_i['wavelength']
        pref2 = 4*pi/defs_i['wavelength']
        qc2_ambient = 2*pref1*delta_ambient - 2j*pref2*mu_ambient
        qc2_film = 2*pref1*delta_film - 2j*pref2*mu_film
        qc2_substrate = 2*pref1*delta_substrate - 2j*pref2*mu_substrate
    else:
        pref2 = 4*pi/defs_i['wavelength']   
        qc2_ambient = qc2_ambient_real - 2j*pref2*mu_ambient
        qc2_film = qc2_film_real - 2j*pref2*mu_film
        qc2_substrate = qc2_substrate_real - 2j*pref2*mu_substrate

        
    print('---------Sample Parameters------------')
    print('Qc2(Superstrate): '+str(qc2_ambient))
    print('Qc2(Film): '+str(qc2_film))
    print('Qc2(Substrate): '+str(qc2_substrate))
    print(' ')
    print('Total Film Thickness: '+str(total_film_thickness))
    print('Grating Thickness: '+str(grating_thickness))
    print('Grating Period, Linewidth: '+str(grating_period)+' '+str(linewidth))
    print(' ')

    for ix, x, iy, y, iz, z in axes():
        if iz<1:
            V_box[:,:,0] = qc2_ambient
        elif z < grating_thickness:
            V_box[ix,iy,iz] = qc2_ambient
            if y%grating_period > (grating_period-linewidth):
                V_box[ix,iy,iz] = qc2_substrate
        else:
            V_box[ix,iy,iz] = qc2_substrate

    return V_box

            


def Potential_Trapezoid(rotation_elements, defs_i, V_box):

    # Model utilizes the squared complex critical q vector (qc^2) for each pixel in the simulation cell
    # Useful Formulas include:
    #       (1) Scattering Length Density 
    #               SLD = sum(scattering lengths)/molar volume  (this is typically a complex number SLD' + i*SLD'')
    #
    #       (2) refractive index n = 1 - delta - i*beta (delta and beta are dimensionless)
    #                            n = 1 - Lam^2*(SLD'/2pi) - i*Lam*(SLD''/4pi)
    #                                   Therefore:  delta = Lam^2*SLD'/2pi
    #                                               beta = Lam*(SLD''/4pi)
    #       (3) qc^2:   thetac (real part only) = sqrt(2*delta)
    #                   beta = mu*wavelength/(4*pi)  (Neutron community often uses mu for adsorbtion term)
    #                   qc2 = 16pi^2/Lam^2*2delta - 2i*4pi*mu/Lam (typical neutron version)
    #                   qc2 = 16pi^2/Lam^2*2delta - 2i*16pi^2/Lam^2*beta
    #                   qc2 = 16pi*(SLD') - 2i*4pi/Lam*(SLD'')
    #
    # SLD at approx 14 keV (real part not a strong function of energy)
    # PS - 9.6e-6 + 7e-9i A^-2
    # PMMA - 10.9e-6 + 8e-9 A^-2
    # delta + mu (74%PS-b-26%PMMA) = 9.938e-6 + 7.3e-9i A^-2
    #

    
    sld_grating_r = 20.0e-4 
    sld_matrix_r = 0.0e-4
    sld_substrate_r = 20.0e-4            # nm^-2

    sld_grating_c = 1.0e-5 
    sld_matrix_c = 0.0e-7
    sld_substrate_c = 1.0e-5            # nm^-2

    qc2_ambient = 0.0j
    qc2_grating = 16*pi*(sld_grating_r) - 2j*4*pi/defs_i['wavelength']*(sld_grating_c)
    qc2_matrix = 16*pi*(sld_matrix_r) - 2j*4*pi/defs_i['wavelength']*(sld_matrix_c)
    qc2_substrate = 16*pi*(sld_substrate_r) - 2j*4*pi/defs_i['wavelength']*(sld_substrate_c)

    #Enter sample parameters
    # Define an array of trapezoids by four parameters: periodicity, avg height, avg thickness, avg linewidth,
    # and avg sidewall angle (SWA). The SWA is the angle of the sidewall to the substrate, and is defined as
    # positive when the air side of the line is smaller than the foot of the trapezoid. The SWA on both sides
    # of the line is assumed to be identical.
    #
    # The routine calculates the width of the header, and builds the trapezoid in layers to the substrate,
    # calculating the width at each layer. Since the sidewall angle is typically close to 90 deg or exactly
    # 90 deg, we use the complementary angle beta = (90 - SWA) in the calculations.
    
    avg_grating_period = 89.0    # nm
    avg_grating_thickness = 439.0    #nm         
    avg_linewidth = 62  #nm       
    avg_sidewall_ang = 90.0 #deg
    
    beta_deg = 90.0 - avg_sidewall_ang
    beta_rad = radians(beta_deg)
    head_linewidth = avg_linewidth - 2.0*(avg_grating_thickness/2.0)*tan(beta_rad)
    foot_linewidth = avg_linewidth + 2.0*(avg_grating_thickness/2.0)*tan(beta_rad)
    lwperiodratio = avg_linewidth/avg_grating_period
    avg_qc2grating = lwperiodratio*qc2_grating + (1-lwperiodratio)*qc2_matrix
        
    print('---------Sample Parameters------------')
    print('Qc2(Superstrate): '+str(qc2_ambient))
    print('Qc2(Silicon Grating): '+str(avg_qc2grating))
    print('Grating Period, LW/Period: '+str(avg_grating_period)+' '+str(lwperiodratio))
    print('Avg Trapezoid Height, LW, SWA: '+str(avg_grating_thickness)+' '+ \
          str(avg_linewidth)+' '+str(avg_sidewall_ang))
    print('Head LW, Foot LW: '+str(head_linewidth)+' '+str(foot_linewidth))
    print(' ')

    for ix, x, iy, y, iz, z in axes():
        if iz<1:
            V_box[:,:,0] = qc2_ambient
        elif z < avg_grating_thickness:
            linewidth_z = head_linewidth + 2.0*z*tan(beta_rad)
            V_box[ix,iy,iz] = qc2_matrix
            if y % avg_grating_period > (avg_grating_period-linewidth_z)/2.0 and \
               y % avg_grating_period < (avg_grating_period+linewidth_z)/2.0:
                V_box[ix,iy,iz] = qc2_grating
        else:
            V_box[ix,iy,iz] = qc2_substrate
            


    return V_box  


def Potential_SixTrapezoid(rotation_elements, defs_i, V_box):

    z_incr = limits['z']['step']
    # Substrate, Superstrate, and Matrix parameters
    sld_matrix_r = 0.0e-4
    sld_substrate_r = 19.5e-4            # nm^-2

    sld_matrix_c = 0.0e-7
    sld_substrate_c = 1.0e-5            # nm^-2

    qc2_ambient = 0.0j
    qc2_matrix = 16*pi*(sld_matrix_r) - 2j*4*pi/defs_i['wavelength']*(sld_matrix_c)
    qc2_substrate = 16*pi*(sld_substrate_r) - 2j*4*pi/defs_i['wavelength']*(sld_substrate_c)

    V_box[:,:,0] = qc2_substrate    

    # Film parameters
    lper = 86.2    # nm

    #Specify sld_r, sld_c, layer_thick, foot_width, and sidewall_ang for each layer
    #### Layer 1
    sld_grating_r = 19.5e-4 
    sld_grating_c = 1.0e-5 
    qc2_grating = 16*pi*(sld_grating_r) - 2j*4*pi/defs_i['wavelength']*(sld_grating_c)

    layer_height = 69.3  #nm  layer heights are additive, at last layer, layer height = film thickness        
    foot_width = 73.3  #nm  --- build trapezoid from foot to head      
    sidewall_ang = 83.9 #deg
    beta_rad = radians(90.0 - sidewall_ang)
    
    iz = 1
    z = z_incr
    prev_layer_height = 0.0 #nm
    while z < layer_height:
        z_rel = z-prev_layer_height
        w_z = foot_width - 2.0*z_rel*tan(beta_rad)
        iz += 1
        z += z_incr
        for ix, x in axis('x'):
            for iy, y in axis('y'):
                V_box[ix,iy,iz] = qc2_matrix
                if y % lper > (lper-w_z)/2.0 and y % lper < (lper+w_z)/2.0:
                    V_box[ix,iy,iz] = qc2_grating

    # Layer 2
    sld_grating_r = 19.5e-4 
    sld_grating_c = 1.0e-5 
    qc2_grating = 16*pi*(sld_grating_r) - 2j*4*pi/defs_i['wavelength']*(sld_grating_c)
    prev_layer_height = layer_height #nm
    layer_height = 69.3 + 125.1  #nm  layer heights are additive, at last layer, layer height = film thickness        
    foot_width = w_z  #nm  --- build trapezoid from foot to head      
    sidewall_ang = 87.5 #deg
    beta_rad = radians(90.0 - sidewall_ang)
    
    while z < layer_height:
        z_rel = z-prev_layer_height
        w_z = foot_width - 2.0*z_rel*tan(beta_rad)
        iz += 1
        z += z_incr
        for ix, x in axis('x'):
            for iy, y in axis('y'):
                V_box[ix,iy,iz] = qc2_matrix
                if y % lper > (lper-w_z)/2.0 and y % lper < (lper+w_z)/2.0:
                    V_box[ix,iy,iz] = qc2_grating

    # Layer 3
    sld_grating_r = 19.5e-4 
    sld_grating_c = 1.0e-5 
    qc2_grating = 16*pi*(sld_grating_r) - 2j*4*pi/defs_i['wavelength']*(sld_grating_c)
    prev_layer_height = layer_height #nm
    layer_height = 69.3 + 125.1 + 105.0  #nm  layer heights are additive        
    foot_width = w_z  #nm  --- build trapezoid from foot to head      
    sidewall_ang = 87.0 #deg
    beta_rad = radians(90.0 - sidewall_ang)
    
    while z < layer_height:
        z_rel = z-prev_layer_height
        w_z = foot_width - 2.0*z_rel*tan(beta_rad)
        iz += 1
        z += z_incr
        for ix, x in axis('x'):
            for iy, y in axis('y'):
                V_box[ix,iy,iz] = qc2_matrix
                if y % lper > (lper-w_z)/2.0 and y % lper < (lper+w_z)/2.0:
                    V_box[ix,iy,iz] = qc2_grating

    # Layer 4 - SiN layer
    sld_grating_r = 22.8e-4 
    sld_grating_c = 1.0e-5 
    qc2_grating = 16*pi*(sld_grating_r) - 2j*4*pi/defs_i['wavelength']*(sld_grating_c)
    prev_layer_height = layer_height #nm
    layer_height = 69.3 + 125.1 + 105.0 + 62.4  #nm  layer heights are additive        
    foot_width = w_z  #nm  --- build trapezoid from foot to head      
    sidewall_ang = 88.8 #deg
    beta_rad = radians(90.0 - sidewall_ang)
    
    while z < layer_height:
        z_rel = z-prev_layer_height
        w_z = foot_width - 2.0*z_rel*tan(beta_rad)
        iz += 1
        z += z_incr
        for ix, x in axis('x'):
            for iy, y in axis('y'):
                V_box[ix,iy,iz] = qc2_matrix
                if y % lper > (lper-w_z)/2.0 and y % lper < (lper+w_z)/2.0:
                    V_box[ix,iy,iz] = qc2_grating

    # Layer 5
    sld_grating_r = 19.5e-4 
    sld_grating_c = 1.0e-5 
    qc2_grating = 16*pi*(sld_grating_r) - 2j*4*pi/defs_i['wavelength']*(sld_grating_c)
    prev_layer_height = layer_height #nm
    layer_height = 69.3 + 125.1 + 105.0 + 62.4 + 70.8  #nm  layer heights are additive        
    foot_width = w_z  #nm  --- build trapezoid from foot to head      
    sidewall_ang = 85.0 #deg
    beta_rad = radians(90.0 - sidewall_ang)
    
    while z < layer_height:
        z_rel = z-prev_layer_height
        w_z = foot_width - 2.0*z_rel*tan(beta_rad)
        iz += 1
        z += z_incr
        for ix, x in axis('x'):
            for iy, y in axis('y'):
                V_box[ix,iy,iz] = qc2_matrix
                if y % lper > (lper-w_z)/2.0 and y % lper < (lper+w_z)/2.0:
                    V_box[ix,iy,iz] = qc2_grating

    # Layer 6
    sld_grating_r = 19.5e-4 
    sld_grating_c = 1.0e-5 
    qc2_grating = 16*pi*(sld_grating_r) - 2j*4*pi/defs_i['wavelength']*(sld_grating_c)
    prev_layer_height = layer_height #nm
    layer_height = 69.3 + 125.1 + 105.0 + 62.4 + 70.8 + 6.7  #nm  layer heights are additive        
    foot_width = w_z  #nm  --- build trapezoid from foot to head      
    sidewall_ang = 32.4 #deg
    beta_rad = radians(90.0 - sidewall_ang)
    
    while z < layer_height:
        z_rel = z-prev_layer_height
        w_z = foot_width - 2.0*z_rel*tan(beta_rad)
        iz += 1
        z += z_incr
        for ix, x in axis('x'):
            for iy, y in axis('y'):
                V_box[ix,iy,iz] = qc2_matrix
                if y % lper > (lper-w_z)/2.0 and y % lper < (lper+w_z)/2.0:
                    V_box[ix,iy,iz] = qc2_grating

    while (z > layer_height and z < limits['z']['size']):
        V_box[ix,iy,iz] = qc2_ambient
        iz += 1
        z += z_incr

    V_temp = numpy.copy(V_box)
    for iz, z in axis('z'):
        for ix, x in axis('x'):
            for iy, y in axis('y'):
                V_box[ix,iy,iz] = V_temp[ix,iy,-iz]

    return V_box




def Potential_Trapezoid_FixedSLD(rotation_elements, defs_i, V_box):

    # Model utilizes the squared complex critical q vector (qc^2) for each pixel in the simulation cell
    # Useful Formulas include:
    #       (1) Scattering Length Density 
    #               SLD = sum(scattering lengths)/molar volume  (this is typically a complex number SLD' + i*SLD'')
    #
    #       (2) refractive index n = 1 - delta - i*beta (delta and beta are dimensionless)
    #                            n = 1 - Lam^2*(SLD'/2pi) - i*Lam*(SLD''/4pi)
    #                                   Therefore:  delta = Lam^2*SLD'/2pi
    #                                               beta = Lam*(SLD''/4pi)
    #       (3) qc^2:   thetac (real part only) = sqrt(2*delta)
    #                   beta = mu*wavelength/(4*pi)  (Neutron community often uses mu for adsorbtion term)
    #                   qc2 = 16pi^2/Lam^2*2delta - 2i*4pi*mu/Lam (typical neutron version)
    #                   qc2 = 16pi^2/Lam^2*2delta - 2i*16pi^2/Lam^2*beta
    #                   qc2 = 16pi*(SLD') - 2i*4pi/Lam*(SLD'')
    #
    # SLD at approx 14 keV (real part not a strong function of energy)
    # PS - 9.6e-6 + 7e-9i A^-2
    # PMMA - 10.9e-6 + 8e-9 A^-2
    # delta + mu (74%PS-b-26%PMMA) = 9.938e-6 + 7.3e-9i A^-2
    #

    #Enter sample parameters
    avg_grating_period = 140.0    # nm
    avg_grating_thickness = 90.0    #nm         
    avg_linewidth = 0.5*avg_grating_period  #nm       
    avg_sidewall_ang = 10.0 #deg, 0 deg is vertical, + deg has smaller top than bottom

    sld_r = numpy.zeros((9),numpy.float)
    sld_r[0] = 4.128110e-004
    sld_r[1] = 4.484910e-004
    sld_r[2] = 5.081110e-004
    sld_r[3] = 5.081110e-004
    sld_r[4] = 5.816800e-004
    sld_r[5] = 6.947640e-004
    sld_r[6] = 6.947640e-004

    sld_c[0] = 2.476510e-007
    sld_c[1] = 8.936010e-007
    sld_c[2] = 7.455490e-007
    sld_c[3] = 7.455490e-007
    sld_c[4] = 8.432400e-007
    sld_c[5] = 9.000000e-007
    sld_c[6] = 9.000000e-007

    thick[0] = 1.141610e+001
    thick[1] = 9.823440e+000
    thick[2] = 1.182120e+001
    thick[3] = 0.000000e+000
    thick[4] = 1.165360E+001
    thick[5] = 4.863660E+000
    thick[6] = 0.000000E+000
           
    sld_grating_r = 20.0e-4 
    sld_matrix_r = 0.0e-4
    sld_substrate_r = 20.0e-4            # nm^-2

    sld_grating_c = 1.0e-5 
    sld_matrix_c = 0.0e-7
    sld_substrate_c = 1.0e-5            # nm^-2

    qc2_ambient = 0.0j
    qc2_grating = 16*pi*(sld_grating_r) - 2j*4*pi/defs_i['wavelength']*(sld_grating_c)
    qc2_matrix = 16*pi*(sld_matrix_r) - 2j*4*pi/defs_i['wavelength']*(sld_matrix_c)
    qc2_substrate = 16*pi*(sld_substrate_r) - 2j*4*pi/defs_i['wavelength']*(sld_substrate_c)

    

    lsratio = avg_linewidth/avg_grating_period
    qc2grating = lsratio*qc2_grating + (1-lsratio)*qc2_matrix
        
    print('---------Sample Parameters------------')
    print('Qc2(Superstrate): '+str(qc2_ambient))
    print('Qc2(Silicon Grating): '+str(qc2grating))
    print('Grating Thick, Period, LW: '+str(avg_grating_thickness)+' '+ \
          str(avg_grating_period)+' '+str(avg_linewidth))
    print('Qc2(Sidewall Angle): '+str(avg_sidewall_ang))
    print('Qc2(Substrate): '+str(qc2_substrate))
    print(' ')

    beta = radians(avg_sidewall_ang)
    top_linewidth = avg_linewidth - 2.0 * avg_grating_thickness/2.0 * tan(beta)
    for ix, x, iy, y, iz, z in axes():
        if iz<1:
            V_box[:,:,0] = qc2_ambient
        elif z < avg_grating_thickness:
            linewidth_z = top_linewidth + 2.0*z*tan(beta)
#            print iz, z, linewidth_z
            V_box[ix,iy,iz] = qc2_matrix
            if y % avg_grating_period > (avg_grating_period-linewidth_z)/2.0 and \
               y % avg_grating_period < (avg_grating_period+linewidth_z)/2.0:
                V_box[ix,iy,iz] = qc2_grating
        else:
            V_box[ix,iy,iz] = qc2_substrate

    return V_box

           
def Potential_Cylinders(rotation_elements, defs_i, V_box ):

    lattice_spacing = 30.0 #nm
    cylinder_diameter = (lattice_spacing/2.0) #nm
    film_thickness = 100

    sld_cylinder_r = 10.0e-4 
    sld_matrix_r = 0.0e-4
    sld_substrate_r = 10.0e-4            # nm^-2

    sld_cylinder_c = 7e-7 
    sld_matrix_c = 8.0e-7
    sld_substrate_c = 1.0e-5            # nm^-2

    qc2_ambient = 0.0j
    qc2_cylinder = 16*pi*(sld_cylinder_r) \
                   - 2j*4*pi/defs_i['wavelength']*(sld_cylinder_c)
    qc2_matrix = 16*pi*(sld_matrix_r) \
                 - 2j*4*pi/defs_i['wavelength']*(sld_matrix_c)
    qc2_substrate = 16*pi*(sld_substrate_r) \
                    - 2j*4*pi/defs_i['wavelength']*(sld_substrate_c)


    # iterate over V_box pixels (ix,iy,iz) and retain the real space position of each
    # pixel (x_cur, y_cur, z_cur)
    # layer 1: ambient atmosphere
    # layer 2-n: film with cylinders oriented along x and packed hexagonally in y-z plane
    # layer n+1: substrate
    #
    # within film layer, calculate the nearest lattice point (lat_ycur, lat_cur) and
    # calculate whether we are within 1 radius of that point. If so, fill with cylinder
    # sld, if not, leave as the matrix sld.
    
    for iy, y_cur, iz, z_cur in yzaxes():    
        if iz<1:
            V_box[:,:,0] = qc2_ambient
        elif z_cur < film_thickness:
            V_box[:, iy, iz] = qc2_matrix
            latnum_zcur = round(z_cur / lattice_spacing) #number of lattice layer in z
            lat_zcur = latnum_zcur * lattice_spacing
            if latnum_zcur%2 > 0: #Shift every other layer by have unit cell
                latnum_ycur = round(y_cur / lattice_spacing + 0.5)
                lat_ycur = latnum_ycur*lattice_spacing - 0.5*lattice_spacing
            else:
                lat_ycur = round(y_cur / lattice_spacing)*lattice_spacing

            # test to see if we are within 1 radius of lat_ycur, lat_zcur
            yz_dist = numpy.sqrt(numpy.square(z_cur - lat_zcur)+numpy.square(y_cur - lat_ycur))
            if yz_dist < cylinder_diameter/2.0:
                    V_box[:, iy, iz] = qc2_cylinder
        else:
            V_box[ix,iy,iz] = qc2_substrate
                
    return V_box


# Calculate Scattering Functions
###################################################################
    
def Calculate_Efield(angle, defs_i, limits, V_z):


    iq_stop = 0
    q_list = numpy.zeros( (limits['z']['num']), numpy.complex  )
    q_list[0] = 2*(2*3.14159/defs_i['wavelength'])*sin( angle )
    E_0 = q_list[0]**2
    for iq, V in zip(range(len(q_list)), V_z):
        if iq>0:
            q_list[iq] = numpy.sqrt( E_0 - V )
            if V_z[iq]<E_0 and iq < len(q_list)-1:
                iq_stop = iq
            
    Macc = numpy.identity( 2, numpy.complex )
    M = numpy.zeros( (limits['z']['num'],2,2), numpy.complex )

    for iz, Z in reverse_enumerate( Z_z ):
        if iz > 0 and iz < iq_stop+1:
            qdelz = q_list[iz]*limits['z']['step']
            M[iz] = numpy.array( [[ numpy.cos(qdelz), 1.0/q_list[iz]*numpy.sin(qdelz) ],
                                  [ -1.0*q_list[iz]*numpy.sin(qdelz), numpy.cos(qdelz) ]] )

            Macc = numpy.dot(Macc,M[iz])

    detMacc = Macc[0,0]*Macc[1,1] - Macc[0,1]*Macc[1,0]

    Efield = numpy.zeros( (limits['z']['num'],2,1), numpy.complex )
    alpha0 = q_list[0]*Z_z[0]
    rnumer = q_list[0]*q_list[iq_stop+1]*Macc[0,1] + Macc[1,0] - 1.0j*q_list[iq_stop+1]*Macc[0,0] + 1.0j*q_list[0]*Macc[1,1]
    rdenom = q_list[0]*q_list[iq_stop+1]*Macc[0,1] - Macc[1,0] + 1.0j*q_list[iq_stop+1]*Macc[0,0] + 1.0j*q_list[0]*Macc[1,1]
    rs = numpy.exp(2.0j*alpha0)*rnumer/rdenom
    
    tdenom = q_list[0]*q_list[iq_stop+1]*Macc[0,1] - Macc[1,0] + 1.0j*q_list[iq_stop+1]*Macc[0,0] + 1.0j*q_list[0]*Macc[1,1]
    ts_prime = 2.0j*q_list[0]/tdenom

    Efield[0,0] = numpy.exp(1.0j*alpha0) + rs*numpy.exp(-1.0j*alpha0)
    Efield[0,1] = 1.0j*q_list[0]*(numpy.exp(1.0j*alpha0) - rs*numpy.exp(-1.0j*alpha0))
    Etrans_prime = Macc[0,0]*Efield[0,0] + Macc[0,1]*Efield[0,1]
    for iz in range( 1, len(Z_z) ):
        if(iz < iq_stop+1):       
            Efield[iz] = numpy.dot(M[iz],Efield[iz-1])
        else:
            beta_prime = q_list[iz]*(Z_z[iz]-Z_z[iq_stop+1]+limits['z']['step'])
            Efield[iz,0] = Etrans_prime*numpy.exp(1.0j*beta_prime)
   
    return q_list, Efield, rs, Etrans_prime[0]


def pixela_dq_pixelb(dq, theta_scan, phi_scan, defs_i):

    #Find the number of the pixel (pixel b) that is dq offset from the current pixel (pixel a)
    k_wavevector = 2.0*3.14159/defs_i['wavelength']
    qy_a = k_wavevector*cos(theta_scan)*sin(phi_scan)
    qy_b = qy_q + dq
    pixel_b

    
def pixel_to_angle( pix_y, pix_z, defs_i ):
    # Conversion from detector pixel coordinates to angles

    theta_scan = atan( pix_z*defs_i['pixel_size']/defs_i['detector_distance']) \
                 + defs_i['theta_scan_floor']
    phi_scan = atan( pix_y*defs_i['pixel_size']/defs_i['detector_distance'] )
    return theta_scan, phi_scan


def compute_detector_row( iqz, z_integral, defs_i, limits, theta_scan ):

#    print( "Starting iqz: " + str(iqz) )


    xv = numpy.zeros( (limits['x']['num'],1), numpy.complex )
    yv = numpy.zeros( (1,limits['y']['num']), numpy.complex )

    detector_row = numpy.zeros( (defs_i['qy_num']), numpy.float )
    Scatt_vol = numpy.square(1.0*limits['x']['num']*limits['y']['num'])
    prefactor = defs_i['beam intensity']
    k_wavevector = 2.0*3.14159/defs_i['wavelength']
    for iqy in range(len(detector_row)):    
        working_xy = numpy.copy(z_integral)

        phi_scan = math.atan( iqy*defs_i['pixel_size']/defs_i['detector_distance'] )
        theta_scantop = math.atan( (iqz*defs_i['pixel_size']+defs_i['pixel_size']/2.0)\
                                   /defs_i['detector_distance']) + defs_i['theta_scan_floor']
        theta_scanbot = math.atan( (iqz*defs_i['pixel_size']-defs_i['pixel_size']/2.0)\
                                   /defs_i['detector_distance']) + defs_i['theta_scan_floor']

        qx_pix = k_wavevector*\
                 (math.cos(theta_scan)*math.cos(phi_scan) - math.cos(defs_i['theta_incident']) )
        qx_pixtop = k_wavevector*\
                 (math.cos(theta_scantop)*math.cos(phi_scan) - math.cos(defs_i['theta_incident']) )
        qx_pixbot = k_wavevector*\
                 (math.cos(theta_scanbot)*math.cos(phi_scan) - math.cos(defs_i['theta_incident']) )

##        for ix, r_x in axis_pp('x', limits):
##            xv[ix] = cmath.exp( 1j*qx_pix*r_x ) * limits['x']['step']

##  Smear from bottom angle to top angle of a given pixel.
##  Remove for Toshiba calculation
        num_qxpix = 4
        dqx_pix = (qx_pixtop - qx_pixbot)/num_qxpix
        qx_pix = qx_pixbot
        xv[0] = 0.0j
        for iqx in range(num_qxpix):
            if (qx_pix == 0.0):
                xv[0] = 1.0
            else:
                xv[0] += numpy.sin(qx_pix*defs_i['xcohlen'])\
                         /(qx_pix*defs_i['xcohlen'])                       
            qx_pix += dqx_pix


        qy_pix = k_wavevector*math.cos(theta_scan)*math.sin(phi_scan)
        for iy, r_y in axis_pp('y', limits):
             yv[0,iy] = cmath.exp( 1j*qy_pix*r_y ) * limits['y']['step']
        
        # Multiply the integrand components through working_xy
        # Thus working_xy contains a 2D array of all the integrand
        # components we need to sum for the integral.
        working_xy *= yv              # for ix, x in axis('x'): working_xy[ix,:] *= yv
        working_xy *= xv              # for iy, y in axis('y'): working_xy[:,iy] *= xv
        
        ReflectionAmplitude = numpy.sum(working_xy)
        ReflectionIntensity = abs(ReflectionAmplitude*ReflectionAmplitude.conjugate())

        detector_row[iqy] = prefactor*ReflectionIntensity/Scatt_vol

    return detector_row


def Generate_GISAXSParmScreen(defs_i):

    #Calculate these parameters only for the purpose of displaying on the screen
    #Note that qy is a function of theta, so the min, max, dqy are calculated at the specular point only
    #Might consider a better graphical representation in the future to demonstrate the magnitude of
    #warping with the range of theta on qy.
    k_wavevector = 2*3.14159/defs_i['wavelength']
    k_incident = k_wavevector * sin(defs_i['theta_incident'])
    k_floor = k_wavevector * sin(defs_i['theta_scan_floor'])
    phi_scan_max = numpy.arctan(defs_i['qy_num']/defs_i['detector_distance'])
    theta_scan_max = numpy.arctan(defs_i['qz_num']/defs_i['detector_distance'])+defs_i['theta_scan_floor']
    det_qz_min = k_wavevector*(sin(defs_i['theta_incident']) + sin(defs_i['theta_scan_floor']))
    det_qz_max = k_wavevector*(sin(defs_i['theta_incident']) + sin(theta_scan_max))
    det_qy_max = k_wavevector*cos(defs_i['theta_incident'])*sin(phi_scan_max)
    det_qy_min = -1.0*det_qy_max
    det_qx_min = k_wavevector*(cos(defs_i['theta_scan_floor']) - cos(defs_i['theta_incident']) )
    det_qx_max = k_wavevector*(cos(theta_scan_max) - cos(defs_i['theta_incident']) )
    det_dqx = abs((det_qx_max - det_qx_min )/defs_i['qz_num'])
    det_dqy = abs((det_qy_max - det_qy_min )/defs_i['qy_num'])
    det_dqz = abs((det_qz_max - det_qz_min )/defs_i['qz_num'])

    # For GI-SAXS data, define number of pixels to integrate on the detector
    k_out_max = k_wavevector*sin(atan(defs_i['qz_num']*defs_i['pixel_size']/(defs_i['detector_distance']))+ defs_i['theta_scan_floor'])

    print('--------Instrument parameters-------------')
    print('Incident Theta (rad): ' + str(defs_i['theta_incident']))
    print('Detector Distance (mm) : ' + str(defs_i['detector_distance']))
    print('Wavelength (nm) : ' + str(defs_i['wavelength']))
    print('Pixel size (mm) : ' + str(defs_i['pixel_size']))
    print('Detector Size (ZxY pixels) : '+str(defs_i['qz_num'])+' x '+str(2*defs_i['qy_num']))
    print(' ')
    print('--------Scattering Vectors---------------')
    print('k,incident (nm^-1) : ' + str(k_incident))
    print('Detector kz,min: ' + str(k_floor)+ ' kz,max: '+str(k_out_max))
    print(' ')
    print('--------Detector q-ranges: ')
    print('Qx min,max: '+str(det_qx_min)+', '+str(det_qx_max)+', '+str(det_dqx))
    print('Qy min,max: '+str(det_qy_min)+', '+str(det_qy_max)+', '+str(det_dqy))
    print('Qz min,max: '+str(det_qz_min)+', '+str(det_qz_max)+', '+str(det_dqz))


def Calculate_Detector_Image( defs_i, limits, V_z, V_box):

    qy_num = defs_i['qy_num']
    qz_num = defs_i['qz_num']
    k_wavevector = 2*3.14159/defs_i['wavelength']
    
    detector_grid = numpy.zeros( (qy_num,qz_num), numpy.float )
    z_integral = numpy.zeros( (limits['x']['num'], limits['y']['num']), numpy.complex )
    z_integral_base = numpy.zeros( (limits['x']['num'], limits['y']['num'], limits['z']['num']), numpy.complex )

    Generate_GISAXSParmScreen(defs_i)

    Psi1 = numpy.zeros(len(V_z), numpy.complex)

    # Calculate wavefunction for incident wave, only need to do once for GISAXS
    theta_incident = defs_i['theta_incident']
    q_i, Efield, rs, Etrans = Calculate_Efield( theta_incident, defs_i, limits, V_z )       
    for iz in range(len(V_z)):
        Psi1[iz] = Efield[iz,0,0]

    # Calculate the z-integral except for the exit angle wavefunction, which changes for each detector row        
    z_integral_base = (V_box)*Psi1*limits['z']['step']

    for iqz in range(len(detector_grid[0])):

        theta_scan, phi_scan = pixel_to_angle( 0, iqz, defs_i )
        
        q_s, Efield, rs, Etrans = Calculate_Efield( theta_scan, defs_i, limits, V_z )       


        Psi2 = numpy.zeros(len(V_z), numpy.complex)
        for iz in range(len(V_z)):
            Psi2[iz] = Efield[iz,0,0]

        z_integral_base_temp = numpy.copy(z_integral_base) 
        z_integral_base_temp *= Psi2
        z_integral = numpy.sum( z_integral_base_temp, axis=2 )
            
        # Launch each row as a parallel job

        #detector_row = compute_detector_row( iqz, z_integral )
        # job_server.submit( function_name, (list of arguments), (list of functions it depends on), (list of imports) )
        jobs.append( (iqz, job_server.submit( compute_detector_row, (iqz, z_integral, defs_i, limits, theta_scan), (axis_pp, ), ("numpy", "math", "cmath") ) ) )


    for iqz, job in jobs:
        detector_row = job()
        theta_scan, phi_scan = pixel_to_angle( 0, iqz, defs_i )        
        for iqy in range(len(detector_grid)):
            if theta_scan > 0:
                detector_grid[iqy,iqz] = detector_row[iqy]
            else:
                detector_grid[iqy,iqz] = 1.0e-18*detector_row[iqy]
                

    # Perform a mirror operation to put the specular (qy=0) in the middle of an array that is qy_num*2 in width
    detector_grid2 = numpy.concatenate( (numpy.flipud(detector_grid), detector_grid), axis=0 )

    # Smear the data using a simple Gaussian along qy
    qx_grid = numpy.zeros( (qy_num*2,qz_num), numpy.float )
    qy_grid = numpy.zeros( (qy_num*2,qz_num), numpy.float )
    qz_grid = numpy.zeros( (qy_num*2,qz_num), numpy.float )
    for iqz in range(qz_num):
        for iqy in range(qy_num*2):
            iqy_shift = iqy - qy_num
            theta_scan, phi_scan = pixel_to_angle( iqy_shift, iqz, defs_i )
            qx = k_wavevector*(cos(theta_scan)* cos(phi_scan) - cos(defs_i['theta_incident']) )
            qy = k_wavevector*cos(theta_scan)*sin(phi_scan)
            qz = k_wavevector*(sin(defs_i['theta_incident']) + sin(theta_scan))
            qx_grid[iqy,iqz] = -1.0*qx
            qy_grid[iqy,iqz] = qy
            qz_grid[iqy,iqz] = qz

    detector_smeared = gauss_qsmear( defs_i, detector_grid2, qx_grid, qy_grid, qz_grid)

    detector_smeared += defs_i['background']    
    return qx_grid, qy_grid, qz_grid, detector_smeared           

def gauss_qsmear( defs_i, detector_grid, qx_grid, qy_grid, qz_grid):

    qy_num = defs_i['qy_num']
    qz_num = defs_i['qz_num']
    rad_smear = defs_i['radial_smear_pix']
    sigqy2 = defs_i['sigqy2']
    detector_smeared = numpy.zeros((qy_num*2,qz_num), numpy.float)
    # Loop over every pixel on the unsmeared image with y_pix, z_pix
    for y_pix in range(len(detector_grid)):
        for z_pix in range(len(detector_grid[0])):
            #define the q values for the current pixel
            qx_pix = qx_grid[y_pix, z_pix]
            qy_pix = qy_grid[y_pix, z_pix]
            qz_pix = qz_grid[y_pix, z_pix]
            for dy_pix in range(-1*int(rad_smear), int(rad_smear)+1):
                for dz_pix in range(-1*int(rad_smear), int(rad_smear)+1):
                    d_pix = numpy.sqrt(dy_pix*dy_pix + dz_pix*dz_pix)
                    if d_pix < rad_smear:
                        yp_pix = y_pix+dy_pix
                        zp_pix = z_pix+dz_pix
                        if(yp_pix > -1 and yp_pix < qy_num*2-1):
                            if(zp_pix > -1 and zp_pix < qz_num-1):
                                qxp_pix = qx_grid[yp_pix, zp_pix]
                                qyp_pix = qy_grid[yp_pix, zp_pix]
                                qzp_pix = qz_grid[yp_pix, zp_pix]
                                dq2 = numpy.square(qxp_pix-qx_pix)\
                                    +numpy.square(qyp_pix-qy_pix)+numpy.square(qzp_pix-qz_pix)
#                                print y_pix, z_pix, yp_pix, zp_pix, dq2
                                detector_smeared[y_pix,z_pix] += \
                                    numpy.exp(-0.5*dq2/sigqy2)*detector_grid[yp_pix, zp_pix]

    return detector_smeared

def lorentz_qsmear( defs_i, detector_grid, qx_grid, qy_grid, qz_grid):

    qy_num = defs_i['qy_num']
    qz_num = defs_i['qz_num']
    rad_smear = defs_i['radial_smear_pix']
    sigqy2 = defs_i['sigqy2']
    detector_smeared = numpy.zeros((qy_num*2,qz_num), numpy.float)
    # Loop over every pixel on the unsmeared image with y_pix, z_pix
    for y_pix in range(len(detector_grid)):
        for z_pix in range(len(detector_grid[0])):
            #define the q values for the current pixel
            qx_pix = qx_grid[y_pix, z_pix]
            qy_pix = qy_grid[y_pix, z_pix]
            qz_pix = qz_grid[y_pix, z_pix]
            for dy_pix in range(-1*int(rad_smear), int(rad_smear)+1):
                for dz_pix in range(-1*int(rad_smear), int(rad_smear)+1):
                    d_pix = numpy.sqrt(dy_pix*dy_pix + dz_pix*dz_pix)
                    if d_pix < rad_smear:
                        yp_pix = y_pix+dy_pix
                        zp_pix = z_pix+dz_pix
                        if(yp_pix > -1 and yp_pix < qy_num*2-1):
                            if(zp_pix > -1 and zp_pix < qz_num-1):
                                qxp_pix = qx_grid[yp_pix, zp_pix]
                                qyp_pix = qy_grid[yp_pix, zp_pix]
                                qzp_pix = qz_grid[yp_pix, zp_pix]
                                dq2 = numpy.square(qxp_pix-qx_pix)\
                                    +numpy.square(qyp_pix-qy_pix)+numpy.square(qzp_pix-qz_pix)
#                                print y_pix, z_pix, yp_pix, zp_pix, dq2
                                detector_smeared[y_pix,z_pix] += \
                                    sigqy2/(dq2 + sigqy2)*detector_grid[yp_pix, zp_pix]

    return detector_smeared


def compute_detector_pixel(theta_incident, theta_scan, phi_scan, limits, defs_i, V_z, V_box):

    k_wavevector = 2*3.14159/defs_i['wavelength']
    xv = numpy.zeros( (limits['x']['num'],1), numpy.complex )
    yv = numpy.zeros( (1,limits['y']['num']), numpy.complex )
    z_integral_base_temp = numpy.zeros( (limits['x']['num'], limits['y']['num'], limits['z']['num']), numpy.complex )
    z_integral = numpy.zeros( (limits['x']['num'], limits['y']['num']), numpy.complex )
    Psi1 = numpy.zeros(len(V_z), numpy.complex)
    Psi2 = numpy.zeros(len(V_z), numpy.complex)
    prefactor = 1.0
    Scatt_vol = numpy.square(1.0*limits['x']['num']*limits['y']['num'])
    
    q_i, Efield, rs_i, Etrans_i = Calculate_Efield( theta_incident, defs_i, limits, V_z  )
    for iz in range(len(Z_z)):
        Psi1[iz] = Efield[iz,0,0]
        
    q_s, Efield, rs_s, Etrans_s = Calculate_Efield( theta_scan, defs_i, limits, V_z  )       
    for iz in range(len(Z_z)):
        Psi2[iz] = Efield[iz,0,0]

    Psi_tot = numpy.sum(Psi1*Psi2)
    Esum_pix = abs(Psi_tot*Psi_tot.conjugate())
    z_integral_base_temp = Psi1*(V_box)*Psi2*limits['z']['step']
    z_integral = numpy.sum( z_integral_base_temp, axis=2 )
    
    x_component = k_wavevector*(cos(theta_scan)* cos(phi_scan) - cos(theta_incident) )        
    for ix, x in axis_pp('x', limits):
        xv[ix] = numpy.exp( 1j*x_component*x ) * limits['x']['step']
    
    y_component = k_wavevector * cos(theta_scan) * sin(phi_scan)
    for iy, y in axis_pp('y', limits):
        yv[0,iy] = numpy.exp( 1j*y_component*y ) * limits['y']['step']
        
    working_xy = numpy.copy(z_integral)
    working_xy *= yv
    working_xy *= xv

    RAmplitude = prefactor*( numpy.sum(working_xy) )
    RIntensity = prefactor * abs(RAmplitude*RAmplitude.conjugate())/Scatt_vol + defs_i['background']
    
    return RIntensity, Esum_pix


   

def Reflect_1D(angles_th_i, angles_th_s, angles_ph_s, defs_i, limits, V_z, V_box):
    #   Calculates full model reflected intensity at a given pixel defined by three angles:
    #   theta incident, theta scan, phi scan
    #   Routine iterates over a series of three angle triplets

    numangs = defs_i['1D_scan_numangles']
    k_wavevector = 2*3.14159/defs_i['wavelength']
    qx_start = k_wavevector * (cos(defs_i['th_s_start'])*cos(defs_i['ph_s_start']) - cos(defs_i['th_i_start']))
    qx_end = k_wavevector * (cos(defs_i['th_s_end'])*cos(defs_i['ph_s_end']) - cos(defs_i['th_i_end']))
    qy_start = k_wavevector * cos(defs_i['th_s_start'])*sin(defs_i['ph_s_start'])
    qy_end = k_wavevector * cos(defs_i['th_s_end'])*sin(defs_i['ph_s_end'])
    qz_start = k_wavevector * (sin(defs_i['th_s_start']) + sin(defs_i['th_i_start']))
    qz_end = k_wavevector * (sin(defs_i['th_s_end']) + sin(defs_i['th_i_end']))
    qx_incr = (qx_end - qx_start)/numangs
    qy_incr = (qy_end - qy_start)/numangs
    qz_incr = (qz_end - qz_start)/numangs
    print('--------Instrument parameters-------------')
    print('Wavelength (nm) : ' + str(defs_i['wavelength']))
    print(' ')
    print('--------Detector q-ranges: ')
    print('Qx min,max,incr: '+str(qx_start)+', '+str(qx_end)+', '+str(qx_incr))
    print('Qy min,max,incr: '+str(qy_start)+', '+str(qy_end)+', '+str(qy_incr))
    print('Qz min,max,incr: '+str(qz_start)+', '+str(qz_end)+', '+str(qz_incr))

    refl = numpy.zeros((numangs,2), numpy.float )
    qx_scan = numpy.empty(numangs)
    qy_scan = numpy.empty(numangs)
    qz_scan = numpy.empty(numangs)
    for ix in range(numangs):
        theta_i = angles_th_i[ix]
        theta_s = angles_th_s[ix]
        phi_s = angles_ph_s[ix]
        qx_scan[ix] = k_wavevector * (cos(theta_s)*cos(phi_s) - cos(theta_i))
        qy_scan[ix] = k_wavevector * cos(theta_s) * sin(phi_s)
        qz_scan[ix] = k_wavevector*(sin(theta_i)+sin(theta_s))
        Reflect, Esum_pix = compute_detector_pixel(theta_i, theta_s, phi_s, limits, defs_i, V_z, V_box)
        refl[ix,0] = Reflect

    return qx_scan, qy_scan, qz_scan, refl[:,0]




def calculate_reflectivity(angles, defs_i, limits, V_z  ):
    #   Calculates the rigorous reflectivity using the transfer matrix method on the
    #   x-y averaged sld profile along z.
    #   Note that this method is only using a single wavefunction. Eventually, the result
    #   here should be approximated closely by Reflect1D(qx=0,qy=0,qz).

    refl = numpy.zeros( (len(angles),4), numpy.float )
    coeff = numpy.zeros( (len(angles),2), numpy.complex)
    for ia, angle in enumerate(angles):
        qi, Efield, rs, ts = Calculate_Efield(angle, defs_i, limits, V_z  )
        Psi_tot = numpy.sum(Efield[:,0,0])
        Esum_pix = abs(Psi_tot*Psi_tot.conjugate())
        coeff[ia,0] = rs
        coeff[ia,1] = ts
        refl[ia,0] = Esum_pix
        refl[ia,1] = abs(qi[0])
        refl[ia,2] = abs(rs*rs.conjugate())
        refl[ia,3] = abs(ts*ts.conjugate())
        
    return refl[:,1], refl[:,2], refl[:,3], coeff[:,0], coeff[:,1], refl[:,0]



def calculate_wavefunction(angle_in, angle_out, defs_i, limits, V_z ):
        

    Psi1 = numpy.zeros(limits['z']['num'], numpy.complex)
    Psi2 = numpy.zeros(limits['z']['num'], numpy.complex)

    q_i, Efield, r1, t1 = Calculate_Efield(angle_in, defs_i, limits, V_z )
    for iz in range(len(Z_z)):
        Psi1[iz] = Efield[iz,0,0]
        
    q_i, Efield, r2, t2 = Calculate_Efield(angle_out, defs_i, limits, V_z )
    for iz in range(len(Z_z)):
        Psi2[iz] = Efield[iz,0,0]
    
    PsiTotal = abs(Psi1 * Psi1.conjugate() * Psi2 * Psi2.conjugate())
    
    return PsiTotal


def loadexpparms(scan_type):
    # Dictionary containing all definitions for the instrumental configuration
    defs_i = {}

    # 14.2 keV = 0.087311 nm
    defs_i['wavelength'] = 0.087311 #nm

    ####### Specular Reflectivity Parms ##############        
    if scan_type == 'Specular':
        defs_i['wavelength'] = 0.087311 #nm
        start_angle = 0.01 #deg
        end_angle = 0.2 #deg
        incr_angle = 0.0001 #deg
        defs_i['spec_start_angle'] = radians(start_angle)
        defs_i['spec_end_angle'] = radians(end_angle)
        defs_i['spec_incr_angle'] = radians(incr_angle)

    ####### Off-Specular Reflectivity Parms ##############    
    if scan_type == '1DScan':
        #theta incident
        defs_i['wavelength'] = 0.087311 #nm
        defs_i['background'] = 1. #arbitrary
        defs_i['th_i_vary'] = False
        theta_i_start = 0.2 #deg
        theta_i_end = 0.2 #deg
        theta_i_incr = 0.001 #deg
        if defs_i['th_i_vary']:
            defs_i['1D_scan_numangles'] = (theta_i_start-theta_i_end)/theta_i_incr
            
        #theta scan
        defs_i['th_s_vary'] = True
        theta_s_start = 0.1 #deg
        theta_s_end = 0.4 #deg
        theta_s_incr = 0.001 #deg
        if defs_i['th_s_vary']:
            defs_i['1D_scan_numangles'] = abs((theta_s_end-theta_s_start)/theta_s_incr)

        #phi scan
        defs_i['ph_s_vary'] = False
        phi_s_start = 0.0#deg
        phi_s_end = 0.0#deg
        phi_s_incr = 0.0#deg
        if defs_i['ph_s_vary']:
            defs_i['1D_scan_numangles'] = abs((phi_s_end-phi_s_start)/phi_s_incr)

        # Set these values to True if they are to be varied in scan
        # Set to False if they are to be constant, the constant value is the "start" value above
        defs_i['th_i_start']=radians(theta_i_start)
        defs_i['th_s_start']=radians(theta_s_start)
        defs_i['ph_s_start']=radians(phi_s_start)
        defs_i['th_i_end']=radians(theta_i_end)
        defs_i['th_s_end']=radians(theta_s_end)
        defs_i['ph_s_end']=radians(phi_s_end)
        defs_i['th_i_incr']=radians(theta_i_incr)
        defs_i['th_s_incr']=radians(theta_s_incr)
        defs_i['ph_s_incr']=radians(phi_s_incr)
        

    ####### GI-SAXS Parms ##############        
    if scan_type == 'GISAXS':
        defs_i['beam intensity'] = 1.5e9 #arbitrary units
        defs_i['wavelength'] = 0.08731 #nm

        defs_i['wavelength_num'] = 100
        defs_i['wavelength_incr']= .001 #radians

        theta_incident = 0.01 #degrees
        theta_incr = 0.005 #degrees
        theta_scan_floor = 0.001 #degrees
        
        defs_i['detector_distance']= 4140.0     #mm
        defs_i['pixel_size']= 0.210             # mm
        defs_i['theta_incident']= radians(theta_incident)       #radians
        defs_i['theta_incident_num'] = 100
        defs_i['theta_incident_incr']= radians(theta_incr) #radians
        
        defs_i['theta_scan_floor']= radians(theta_scan_floor)   #radians
        defs_i['qy_num'] = 200                  #pixels (note: final image is 2*qynum X qznum in size)
        defs_i['qz_num'] = 80                  #pixels
        defs_i['background'] = 1. #arbitrary
        defs_i['radial_smear_pix']= 5 #nm^-1
        defs_i['sigqy2'] = 0.00001 #nm^-2
        defs_i['xcohlen'] = 317000 #nm
        defs_i['qxeqzero'] = 1.0e-5 #nm^-1

        #define constant values of pixels (iqz and iqy) for which we will generate
        #I(qy at constant qz_slice) and I(qz at constant qy_slice)
        # Note that this slice is taken AFTER the mirror transform of I(qy).
        # As a result, I(qz) at qy_slice=qz_num/2 will be the line through a specular point (qy = 0)
        defs_i['directory'] = 'Z:\\My Documents\\Programming\\nGISAXS_v5'
        defs_i['slice_file_1'] = 'ExpIQz19.txt'
        defs_i['slice_file_2'] = 'ExpIQz38.txt'

        qz_slice_pix = round(defs_i['detector_distance']/defs_i['pixel_size'] \
                                       * (tan(defs_i['theta_incident']) - tan(defs_i['theta_scan_floor'])))
        defs_i['qz_slice'] = int(qz_slice_pix)+10
        defs_i['qy_slice_1'] = int(round(defs_i['qy_num'])+19)
        defs_i['qy_slice_2'] = int(round(defs_i['qy_num'])+57)
        defs_i['outcsv_file'] = 'GISAXS_LogI_0225.csv'

        defs_i['leftqy_abs'] = -0.8 # nm^-1
        defs_i['bottqz_abs'] = 0.0 # nm^-1
        defs_i['numqy_abs'] = defs_i['qy_num'] * 2
        defs_i['numqz_abs'] = defs_i['qz_num']
        defs_i['rightqy_abs'] = 0.8 # nm^-1
        defs_i['topqz_abs'] = 0.6 # nm^1

    return defs_i

def buildcell():

    # Define the integration limits
    # The number of z layers and the 'num' in the limits dictionary are crucial.
    # They define the partitioning of real-space, which influences the accuracy
    # of the result (and may introduce artifacts if one is not careful).

##  Trapezoid cell for Toshiba samples
    x_max = 400000 #nm
    num_layers_x = 1  #Should be an integer
    dx_incr = 1.0*x_max/num_layers_x
    qx_nyquist = 2.0*pi/dx_incr
    qx_minfreq = 2.0*pi/x_max

    y_max = 8620    #nm
    num_layers_y = 8620 #Should be an integer
    dy_incr = 1.0*y_max/num_layers_y
    qy_nyquist = 2.0*pi/dy_incr
    qy_minfreq = 2.0*pi/y_max

    z_max = 444 #nm
    num_layers_z = 444 #Should be an integer
    dz_incr = 1.0*z_max/num_layers_z
    qz_nyquist = 2.0*pi/dz_incr
    qz_minfreq = 2.0*pi/z_max

    # Cell for cylinders for Epps group
##    x_max = 500 #nm
##    num_layers_x = 100  #Should be an integer
##    dx_incr = 1.0*x_max/num_layers_x
##    qx_nyquist = 2.0*pi/dx_incr
##    qx_minfreq = 2.0*pi/x_max
##
##    y_max = 500 #nm
##    num_layers_y = 500 #Should be an integer
##    dy_incr = 1.0*y_max/num_layers_y
##    qy_nyquist = 2.0*pi/dy_incr
##    qy_minfreq = 2.0*pi/y_max
##
##    z_max = 100 #nm
##    num_layers_z = 100 #Should be an integer
##    dz_incr = 1.0*z_max/num_layers_z
##    qz_nyquist = 2.0*pi/dz_incr
##    qz_minfreq = 2.0*pi/z_max

    limits = { \
        'x': { 'size':x_max, 'num':num_layers_x, 'step':dx_incr }, \
        'y': { 'size':y_max, 'num':num_layers_y, 'step':dy_incr}, \
        'z': { 'size':z_max, 'num':num_layers_z, 'step':dz_incr }, \
        }

    V_box = numpy.empty( (limits['x']['num'], limits['y']['num'], limits['z']['num']), numpy.complex )

    print ("-------------Cell Dimensions----------------")
    print ("X: max:"+str(limits['x']['size'])+" num:"+str(limits['x']['num'])+" dx:"+str(limits['x']['step']))
    print ("Y: max:"+str(limits['y']['size'])+" num:"+str(limits['y']['num'])+" dy:"+str(limits['y']['step']))
    print ("Z: max:"+str(limits['z']['size'])+" num:"+str(limits['z']['num'])+" dz:"+str(limits['z']['step']))
    print (' ')

    print ("-------------Integration Limits-------------")
    print ("dQx,min: "+str(qx_minfreq)+" dQy,min: "+str(qy_minfreq)+" dQz,min: "+str(qz_minfreq))
    print ("Qx,max: "+str(qx_nyquist)+" Qy,max: "+str(qy_nyquist)+" Qz,max: "+str(qz_nyquist))
    print (" ")

    return limits, V_box


def ReadExpData(file_len, filename_str):

    print ("------------Reading Experimental Data Files ------------------")
    print ("File: "+filename_str)
    print (' ')
    ExpIq = numpy.zeros(file_len, numpy.float)
    f = open(filename_str, 'r')
    for line in range(file_len):
        ExpIq[line]=float(f.readline()[0:-1])
    f.close()
    return ExpIq

def ExtractQyQzPlane(detector_smeared, detector_qyqz, qx_grid, qy_grid, qz_grid, defs_i):

    leftqy_abs = defs_i['leftqy_abs']
    bottqz_abs = defs_i['bottqz_abs']
    numqy_abs = defs_i['numqy_abs']
    numqz_abs = defs_i['numqz_abs']
    rightqy_abs = defs_i['rightqy_abs']
    topqz_abs = defs_i['topqz_abs']

    dqy_abs = (rightqy_abs - leftqy_abs)/numqy_abs
    dqz_abs = (topqz_abs - bottqz_abs)/numqz_abs
    qy_num = defs_i['qy_num']
    qz_num = defs_i['qz_num']

    detector_qyqz -= defs_i['background']
    for iqz in range(len(detector_smeared[0])):
        for iqy in range(len(detector_smeared)):
            if abs(qx_grid[iqy, iqz])<defs_i['qxeqzero']:
                qy_theta = qy_grid[iqy,iqz]
                qz_theta = qz_grid[iqy,iqz]
                iqy_abs = int((qy_theta-leftqy_abs)/dqy_abs)
                iqz_abs = int((qz_theta-bottqz_abs)/dqz_abs)
                if (iqy_abs < numqy_abs and iqy_abs > 0 and iqz_abs < numqz_abs and iqz > 0):
                    detector_qyqz[iqy_abs,iqz_abs] = detector_smeared[iqy,iqz]
            
    detector_qyqz += defs_i['background']

######################################################################

############## START MAIN PROGRAM HERE ###############################

######################################################################

p.rcParams['axes.labelsize'] = 'x-large'
p.rcParams['xtick.labelsize'] = 'large'
p.rcParams['ytick.labelsize'] = 'large'



# Prepare for parallel computation
ppservers = ()
ncpus = 7
job_server = pp.Server(ncpus, ppservers=ppservers)
jobs = []




if True:
#for i_calculate in range(5):
    do_build_cell = True
    do_build_film = True
    do_GISAXS = True
    do_1DScan = False
    do_Specular = False
    do_wavefunction = False
            
        
    if do_build_cell:
        # define space for empty simulation cell, this size can be varied in the main program loop.
        defs_i = loadexpparms('buildcell') #parameters describing instrument configuration
        limits, V_box = buildcell()

    if do_build_film:    
        axis_rotate = numpy.zeros(3, numpy.float)
        V_z, Z_z = update_potential(axis_rotate, defs_i, V_box)
        plot_Z(V_z, Z_z)
        plot_Y(V_box, limits)
        plot_X(V_box, limits)
        plot_YZContour(V_box, limits)
        
    if do_Specular:
        print("Calculating Specular Reflectivity...")
        defs_i = loadexpparms('Specular') #parameters describing instrument configuration
        reflect_angles = numpy.arange(defs_i['spec_start_angle'], defs_i['spec_end_angle'], \
                                      defs_i['spec_incr_angle'])
        qs, Reflect, Trans, R_coeff, T_coeff, Esum = \
            calculate_reflectivity(reflect_angles, defs_i, limits, V_z)

        fig = p.figure()
        fig.subplots_adjust( bottom=0.2 )
        p.plot( qs, Reflect )
        p.xlabel(r'$Q_z \hspace{0.5} (\rm{nm}^{-1})$')      
        p.ylabel('R')
        p.semilogy()
        p.savefig( 'reflectivity.png' )

        fig = p.figure()
        fig.subplots_adjust( bottom=0.2 )
        p.plot( qs, Trans )
        p.xlabel(r'$Q_z \hspace{0.5} (\rm{nm}^{-1})$')      
        p.ylabel('|E(substrate)|')
        p.savefig( 'transmittance.png' )

        R_real = numpy.real(R_coeff)
        R_imag = numpy.imag(R_coeff)
        fig = p.figure()
        fig.subplots_adjust( bottom=0.2 )
        p.plot( R_real, R_imag )
        p.xlabel('Re r')
          
        p.ylabel('Im r')
        p.savefig( 'rcoeff.png' )

        fig = p.figure()
        fig.subplots_adjust( bottom=0.2 )
        p.plot( qs, Esum )
        p.xlabel(r'$Q_z \hspace{0.5} (\rm{nm}^{-1})$')
        p.ylabel('Sum(E(z))')
        p.semilogy()
        p.savefig( 'Esum_Spec_qz.png' )

    if do_GISAXS:
        print "Calculating GISAXS..."
        defs_i = loadexpparms('GISAXS') #parameters describing instrument configuration
        do_read_exp_data = False
        if do_read_exp_data:
            total_files = 2
            file_len = 200
            expIqz = numpy.empty((total_files, file_len), numpy.float)

            numfile = 1
            filename_str = defs_i['slice_file_1']
            expIqz[numfile-1][:] = ReadExpData(file_len, filename_str)

            numfile = 2
            filename_str = defs_i['slice_file_2']
            expIqz[numfile-1][:] = ReadExpData(file_len, filename_str)

        # Manually calculate a range of incident angles and/or wavelengths for smearing along qz-qx
        qy_num = defs_i['qy_num']
        qz_num = defs_i['qz_num']
        theta_incr = defs_i['theta_incident_incr']
        theta_incident = defs_i['theta_incident'] #radians
        wavelength_incr = defs_i['wavelength_incr']
        wavelength = defs_i['wavelength']
        qy_abs = numpy.zeros((qy_num*2,qz_num), numpy.float)
        qz_abs = numpy.zeros((qy_num*2,qz_num), numpy.float)
        leftqy_abs = defs_i['leftqy_abs']
        bottqz_abs = defs_i['bottqz_abs']
        numqy_abs = defs_i['numqy_abs']
        numqz_abs = defs_i['numqz_abs']
        rightqy_abs = defs_i['rightqy_abs']
        topqz_abs = defs_i['topqz_abs']
        dqy_abs = (rightqy_abs - leftqy_abs)/numqy_abs
        dqz_abs = (topqz_abs - bottqz_abs)/numqz_abs
        for iqz in range(qz_num):
            for iqy in range(qy_num*2):
                qz_abs[iqy][iqz] = bottqz_abs + iqz*dqz_abs
                qy_abs[iqy][iqz] = leftqy_abs + iqy*dqy_abs
        
        detector_qyqz = numpy.zeros((qy_num*2,qz_num), numpy.float) + defs_i['background']

        #for i_theta in range(defs_i['theta_incident_num']-1):
        for i_theta in range(defs_i['theta_incident_num']-1):
            qx_grid, qy_grid, qz_grid, detector_smeared = Calculate_Detector_Image(defs_i, limits, V_z, V_box)
            print degrees(theta_incident)*1000
            filename1_str = ('DetectorImage_theta'+str(int(degrees(theta_incident)*1000)))
            filename2_str = ('QyQzImage_wave')

            
            for iqz in range(len(detector_smeared[0])):
                for iqy in range(len(detector_smeared)):
                    detector_smeared[iqy,iqz] = log( detector_smeared[iqy,iqz] )
            plot_detector( detector_smeared, qy_grid, qz_grid, min_v=0.0, max_v=0.8, filename=filename1_str )

            ExtractQyQzPlane(detector_smeared, detector_qyqz, qx_grid, qy_grid, qz_grid, defs_i)
            plot_detector( detector_qyqz, qy_abs, qz_abs, min_v=0.0, max_v=0.8, filename=filename2_str )

            theta_incident += theta_incr
            defs_i['theta_incident'] = theta_incident
#            wavelength -= wavelength_incr
#            defs_i['wavelength'] = wavelength
            
        plot_qz_axis(detector_smeared, qz_abs, defs_i)
        #plot_qy_axis(detector_smeared, qy_grid, defs_i)



    if do_1DScan:
        print("Calculating Off-Specular Reflectivity...")
        defs_i = loadexpparms('1DScan') #parameters describing instrument configuration
        if(defs_i['th_i_vary']):
            angles_th_i = numpy.arange(defs_i['th_i_start'],defs_i['th_i_end'],defs_i['th_i_incr'])
        else:
            angles_th_i = numpy.ones(defs_i['1D_scan_numangles'], numpy.float) * defs_i['th_i_start']
        if(defs_i['th_s_vary']):
            angles_th_s = numpy.arange(defs_i['th_s_start'],defs_i['th_s_end'],defs_i['th_s_incr'])
        else:
            angles_th_s = numpy.ones(defs_i['1D_scan_numangles'], numpy.float) * defs_i['th_s_start']
        if(defs_i['ph_s_vary']):
            angles_ph_s = numpy.arange(defs_i['ph_s_start'],defs_i['ph_s_end'],defs_i['ph_s_incr'])
        else:
            angles_ph_s = numpy.ones(defs_i['1D_scan_numangles'], numpy.float) * defs_i['ph_s_start']

        
        Reflect = numpy.zeros( defs_i['1D_scan_numangles'] )
        
        qx_scan, qy_scan, qz_scan, Reflect \
                 = Reflect_1D(angles_th_i, angles_th_s, angles_ph_s, defs_i, limits, V_z, V_box)
        
        fig = p.figure()
        fig.subplots_adjust( bottom=0.2 )
        p.plot( qz_scan, Reflect )
        p.xlabel(r'$Qz \hspace{0.5} (\rm{nm}^{-1})$')    
        p.ylabel('I(qx,qy,qz)')
        p.semilogy()
        p.savefig( 'reflect_1d_qx.png' )

        
        


    if do_wavefunction:
        angle_in = 0.05
        angle_out = 0.18

        print("Calculating Wavefunction...")
        PsiTotal = calculate_wavefunction(angle_in, angle_out, defs_i, limits, V_z)
        
        fig = p.figure()
        fig.subplots_adjust( bottom=0.2 )
        p.plot( Z_z, PsiTotal )
        p.xlabel('z (nm)')      
        p.ylabel('$|Psi|^2$')
        p.savefig( 'wavefunction.png' )
            
        










