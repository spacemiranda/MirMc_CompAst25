"""
Script for animating photon scatter, made for Computational Methods, finished at last October 29th 2025 (happy near-Halloween!!)
--Miranda McCarthy
"""
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches
from matplotlib import colormaps
from astropy import constants as c
from astropy import units as u
import time
import glob
import pdb
import argparse
from time import sleep
base_rng = np.random.default_rng(4242) # initializing an rng here, but the seed will be set later.

# basic functions, that I use across homeworks.
# i should possibly consider setting this up as a separate module that can be imported in my github... hm.
# printif
def printif(thingtoprint, printit = True):
    """simple space saving fn - conditional print that only prints the first argument given if the second argument == True.
       I use this for debug messages and as an easy way to set verbosity in long functions!
       Args:
           thingtoprint - the object to actually print
           printit = True - if True, print thingtoprint, if False, don't print it. Default True so it doesn't throw an error when nothing is given as the second argument."""
    if printit:
        print(thingtoprint)
# update and return dict
def updateandreturndict(ogdict, updatewith):
    """A really, really simple function that basically runs dict.update() without returning None, but rather the updated dictionary. made this to make custom kwarg variables dictionaries for nested functions less annoying, ages ago, and now i use it a lot. might be a better alternative out there?
    Runs:
        ogdict.update(updatewith)
        return(ogdict)
    """
    ogdict.update(updatewith)
    return(ogdict)
# if any in 
def ifanyofthesein(anyofthese, areinthis):
    """ Another simple space saving function I use a lot. Rather than checking if foo in [(list of things)], it checks if any item in a list is in a given object.
        Args:
            anyofthese - a list of things to check if they are in areinthis.
            areinthis - whatever you want to search for the items in anyofthese.
        Returns:
            True, if any item in anyofthese is in areinthis. Else False."""
    for itm in anyofthese:
        if itm in areinthis:
            return(True)
# a couple new functions i'm adding to my basic functions list; for dealing with astropy units
def stripunit(val):
    """check if val is an astropy Quantity and if so return it without the unit (as a float)"""
    if type(val) == u.quantity.Quantity:
        val = val.value
    return(val)

def addunit(val, unit):
    """convert value to given unit; if no unit is already on the value then that unit is added."""
    if type(val) != u.quantity.Quantity:
        val = val*unit
    else:
        val = val.to(unit)
    return(val)
# handle the 'bools'
def boolify(argval):
    """
    take value and if it's not zero, it's true. otherwise it's false.
    I bet this is something that can be done in one line with just bool(argval) to get the same effect.
    but also, I don't want to take chances with how this is handled being different across different versions of any package, so I'm hardcoding this. Born of some ancient issue I had with argparse and how it was interpreting bools that I honestly forget the full context of"""
    if argval == 0:
        argval = False
    else:
        argval = True
    return(argval)

# Functions specific to this hw / usecase:
def ne_sun(r, Rsun = c.R_sun, useunits =True):
    """
    Equation for solar electron density. 
    Args:
        r - radius
        Rsun = c.R_sun - solar radius
        useunits = True - whether or not to use units. I have this set to true by default, and then pretty much nowhere else actually use the units in this particular script.
    Will run (putting full code here bc it's short and helpful for debugging later):
        r = addunit(r, u.m)
        outval = (2.5e26*(u.cm**-3)) * (np.e**(-r/(0.096*Rsun)))
        if useunits:
            return(outval)
        else:
            return(stripunit(outval))
    """
    r = addunit(r, u.m)
    outval = (2.5e26*(u.cm**-3)) * (np.exp(-r/(0.096*Rsun)))
    if useunits:
        return(outval)
    else:
        return(stripunit(outval))

def mfp(n = 1e20*(u.cm**-3), sigma = 6.652e-25*(u.cm**2), outputunit = u.m, useunits = True):
    """
    Equation to calculate mean free path.
    Args:
        n = 1e20*u.cm**-3 - n_e for the slab
        sigma = 6.652e-25*(u.cm**2) - sigma_t, thompson cross scattering
        outputunit = u.m - units to output in
        useunits = True - whether or not to use units. I have this set to true by default, and then pretty much nowhere else actually use the units in this particular script.
    Will run:
        addunit(n, u.cm**-3)
        addunit(sigma, u.cm**2)
        outval = (1/(n*sigma))
        if useunits:
            return(outval.to(outputunit))
        else:
            outval = outval.to(outputunit)
            return(stripunit(outval))
    """
    addunit(n, u.cm**-3)
    addunit(sigma, u.cm**2)
    outval = (1/(n*sigma))
    if useunits:
        return(outval.to(outputunit))
    else:
        outval = outval.to(outputunit)
        return(stripunit(outval))

# Core functions for calculating the simulation of the scatter. 
# Side note, I almost deleted pretty much all of this on accident. Shoutout to Jupyter's %%history function for letting me retrieve the code I'd inputted after accidentally hitting the extremely wrong button and undoing 24 hours worth of work <3 
# FOR THE SCATTERING:
def scatter_slab(numscatters = 100000, n = 1e20*(u.cm**-3), sigma = 6.652e-25*(u.cm**2), 
                 init_x = 500, init_y = 500, init_angle = 0, timeit = True, 
                bonusprints = True, rng = base_rng, lenslab = 1000):
    """ Function for calculating the scattering of a photon through a uniform slab.
    Args:
        numscatters = 100000 - number of scattering events to calculate before stopping. Often in the slab with default arguments, it'll escape before you reach this point.
        n = 1e20 cm**-3 - n_e electron density
        sigma = 6.652e-25 cm**2 - thompson cross section
        init_x, init_y, init_angle = 500, 500, 0 - initial x, y, and angle of motion for the photon.
        timeit = True - print time taken to run (with start time.time() - end time.time()) at the end of the function. This will print regardless of bonusprints.
        bonusprints = True - print extra messages for errors, debugging, etc. a mini verbosity controller basically
        rng = base_rng - the rng that is used for the random number generation.
        lenslab = 1000 - the extent of the slab in meters in x and y (note this function ALWAYS assumes the slab is a rectangle with its lower left vortex at 0, 0). Used to judge if the photon escaped or not.
        """
    # get time
    starttime = time.time()
    # define initial internal variables
    init_r = np.sqrt((init_x**2 + init_y**2))
    init_mfp = mfp(n, useunits = False)
    init_s = 0
    # track if it escaped
    escaped = False
    # define dict
    scatter_dict = {'x':[init_x], 'y':[init_y], 'r':[0], 't':[0], 's':[0], 'angle':[init_angle], 'mfp':[init_mfp], 'escaped':[escaped]}
    
    # run scattering loop
    for stepthru in np.arange(0, numscatters):
        # see how far it went
        disttraveled = rng.exponential(init_mfp)
        # so now it scattered, get its x and y 
        init_angle = rng.uniform(0, 2*np.pi)
        init_x = (disttraveled*np.cos(init_angle)) + init_x
        init_y = (disttraveled*np.sin(init_angle)) + init_y
        # add to total path length (remember to do this before making init_r the new r!!) the new distance
        init_s = init_s + np.sqrt((init_x**2 + init_y**2))
        # set new r
        init_r = np.sqrt((init_x**2 + init_y**2))
        # get current time
        init_t = init_s/c.c.value
        # check if it escaped
        # if it went out the "front"/top:
        if ifanyofthesein([True], np.array([init_x, init_y]) > lenslab):
            escaped = True
        if ifanyofthesein([True], np.array([init_x, init_y]) < 0):
            escaped = True
        # add these values to the table
        scatter_dict['x'].append(init_x)
        scatter_dict['y'].append(init_y)
        scatter_dict['r'].append(init_r)
        scatter_dict['angle'].append(init_angle)
        scatter_dict['mfp'].append(init_mfp)
        scatter_dict['t'].append(init_t)
        scatter_dict['s'].append(init_s)
        scatter_dict['escaped'].append(escaped)
        if escaped == True:
            printif("ESCAPED!!!", bonusprints)
            break
    # make pandas table of full result
    fulltable = pd.DataFrame(scatter_dict)
    # time
    if timeit:
        endtime = time.time()
        print("Time elapsed:", endtime-starttime)
    # return
    return(fulltable)

# scatter slab (vectorized)
def scatter_slab_VECTORIZED(numscatters = 100000, n = 1e20*(u.cm**-3), sigma = 6.652e-25*(u.cm**2), 
                 init_x = 500, init_y = 500, init_angle = 0, timeit = True, 
                bonusprints = True, rng = base_rng, lenslab = 1000, mfp_fn = mfp):
    """ Function for calculating the scattering of a photon through a uniform slab.
    NOTE HOWEVER, because the photon takes such a short amount of time to escape the slab if starting in the center, practically speaking this takes longer to run than the non-vectorized one, because it'll escape in a very short number of scattering events!!!
    Args:
        numscatters = 100000 - number of scattering events to calculate before stopping. Often in the slab with default arguments, it'll escape before you reach this point.
        n = 1e20 cm**-3 - n_e electron density
        sigma = 6.652e-25 cm**2 - thompson cross section
        init_x, init_y, init_angle = 500, 500, 0 - initial x, y, and angle of motion for the photon.
        timeit = True - print time taken to run (with start time.time() - end time.time()) at the end of the function. This will print regardless of bonusprints.
        bonusprints = True - print extra messages for errors, debugging, etc. a mini verbosity controller basically
        rng = base_rng - the rng that is used for the random number generation.
        lenslab = 1000 - the extent of the slab in meters in x and y (note this function ALWAYS assumes the slab is a rectangle with its lower left vortex at 0, 0). Used to judge if the photon escaped or not.
        """
    # get time
    starttime = time.time()
    # handle numscatters:
    if type(numscatters) != int:
        printif("Hey, your numscatters isn't an int! Making it an int", bonusprints)
        numscatters = int(numscatters)
    
    # define initial internal variables
    init_r = np.sqrt((init_x**2 + init_y**2))
    init_mfp = mfp_fn(n, useunits = False)
    init_s = 0
    escaped = False
    # bc none of these guys are actually getting out we are gonna calc the mfp ahead of time.
    disttraveled = rng.exponential(init_mfp, size = numscatters)
    # get the list of angles
    anglevals = rng.uniform(0, 2*np.pi, size = numscatters)
    
    # get the list of xvals
    xvals = (disttraveled*np.cos(anglevals)) + init_x
    yvals = (disttraveled*np.sin(anglevals)) + init_y
    # get the r vals
    rvals = np.sqrt((xvals**2 + yvals**2))
    # get the mfp vals
    mfpvals = np.full(len(disttraveled), mfp_fn(n, useunits = False))
    # get the current total distance traveled (final value should be sum of disttraveled!)
    svals = np.array([itm+np.sum(disttraveled[:ind]) for ind, itm in enumerate(disttraveled)])
    # get time vals
    tvals = svals/c.c.value
    # did it escape
    escapedx = np.array([(lambda x: False if 0 < x < lenslab else True)(r) for r in xvals])
    escapedy = np.array([(lambda y: False if 0 < y < lenslab else True)(r) for r in yvals])
    escaped = np.logical_or(escapedx, escapedy)

    # insert initial zero values (so the first row is the initial step)
    xvals = np.insert(xvals, 0, init_x)
    yvals = np.insert(yvals, 0, init_y)
    rvals = np.insert(rvals, 0, init_r)
    tvals = np.insert(tvals, 0, 0)
    anglevals = np.insert(anglevals, 0, init_angle)
    svals = np.insert(svals, 0, init_s)
    mfpvals = np.insert(mfpvals, 0, init_mfp)
    disttraveled = np.insert(disttraveled, 0, 0)
    escaped = np.insert(escaped, 0, False)
    
    # make the dict
    scatter_dict = {'x':xvals, 'y':yvals, 'r':rvals, 't':tvals, 's':svals, 'angle':anglevals, 'mfp':mfpvals, 'disttraveled':disttraveled, 'escaped':escaped}
        
    # make pandas table of full result
    fulltable = pd.DataFrame(scatter_dict)
    # time
    if timeit:
        endtime = time.time()
        print("Time elapsed:", endtime-starttime)
    # return
    return(fulltable)


# for the SUN
def scatter_sun(numscatters = 100000, ne_fn = ne_sun, sigma = 6.652e-25*(u.cm**2), mfp_fn = mfp,
                 init_x = 0, init_y = 0, init_angle = 0, timeit = True, 
                bonusprints = True, rng = base_rng, escaperadius = c.R_sun.value):
    """ Function for calculating the scattering of a photon through The Sun. The code itself is very similar to the slab, but the n_e and mfp are allowed to change with radius, and it tracks actual radius / time / full distance traveled. This is the "slow", for-loop version, but it won't throw an error if the mfp varies too much - because it does the simulation step by step it calculates the scattering based on the actual mfp at every radius.
    Args:
        numscatters = 100000 - number of scattering events to calculate before stopping.
        ne_fn = ne_sun - function to get the electron density in the sun as a function of radius
        sigma = 6.652e-25 cm**2 - thompson cross section. (note no need for n here because it is calculated based on the function for the sun!)
        mfp_fn = mfp - function for the mean free path
        init_x, init_y, init_angle = 0, 0, 0 - initial x, y, and angle of motion for the photon.
        timeit = True - print time taken to run (with start time.time() - end time.time()) at the end of the function. This will print regardless of bonusprints.
        bonusprints = True - print extra messages for errors, debugging, etc. a mini verbosity controller basically
        rng = base_rng - np.random.default_rng() to use
        escaperadius = c.R_sun.value - the radius past which the photon has escaped!!"""
    # get time
    starttime = time.time()
    
    # define initial internal variables
    init_r = np.sqrt((init_x**2 + init_y**2))
    init_mfp = mfp_fn(ne_fn(init_r), useunits = False)
    init_s = 0
    scatter_dict = {'x':[init_x], 'y':[init_y], 'r':[0], 't':[0], 's':[0], 'angle':[init_angle], 'mfp':[init_mfp], 'escaped':[False]}
    # track if it escaped (hint: for the sun it probably won't)
    escaped = False
    # run scattering loop
    for stepthru in np.arange(0, numscatters):
        # see how far it went
        disttraveled = rng.exponential(init_mfp)
        # so now it scattered, get its x and y 
        init_angle = rng.uniform(0, 2*np.pi)
        init_x = (disttraveled*np.cos(init_angle)) + init_x
        init_y = (disttraveled*np.sin(init_angle)) + init_y
        # add to total path length (remember to do this before making init_r the new r!!) the new distance
        init_s = init_s + np.sqrt((init_x**2 + init_y**2))
        # set new r
        init_r = np.sqrt((init_x**2 + init_y**2))
        # get new mfp
        init_mfp = mfp_fn(ne_fn(init_r), useunits = False)
        # get current time
        init_t = init_s*c.c
        # check if it escaped:
        if init_r > escaperadius:
            escaped = True
        # add these values to the table
        scatter_dict['x'].append(init_x)
        scatter_dict['y'].append(init_y)
        scatter_dict['r'].append(init_r)
        scatter_dict['angle'].append(init_angle)
        scatter_dict['mfp'].append(init_mfp)
        scatter_dict['t'].append(init_t)
        scatter_dict['s'].append(init_s)
        scatter_dict['escaped'].append(escaped)
        if escaped:
            printif("ESCAPED!!!", bonusprints)
            break
        
    # make pandas table of full result
    fulltable = pd.DataFrame(scatter_dict)
    # time
    if timeit:
        endtime = time.time()
        print("Time elapsed:", endtime-starttime)
    # return
    return(fulltable)

# scatter sun (fast)
def scatter_sun_FAST(numscatters = 100000, ne_fn = ne_sun, sigma = 6.652e-25*(u.cm**2), mfp_fn = mfp,
                 init_x = 0, init_y = 0, init_angle = 0, timeit = True, mfp_tolerance = 1e-8,
                bonusprints = True, rng = base_rng, escaperadius = c.R_sun.value):
    """ Function for calculating the scattering of a photon through The Sun. The code itself is very similar to the slab, but the n_e and mfp are allowed to change with radius, and it tracks actual radius / time / full distance traveled. This is the "fast" version where because the photon will never get anywhere significantly far from the center as far as the difference in the mean free path is concerned, we can calculate the scattering without any for loops. 
    Args:
        numscatters = 100000 - number of scattering events to calculate before stopping.
        ne_fn = ne_sun - function to get the electron density in the sun as a function of radius
        sigma = 6.652e-25 cm**2 - thompson cross section. (note no need for n here because it is calculated based on the function for the sun!)
        mfp_fn = mfp - function for the mean free path
        init_x, init_y, init_angle - initial x, y, and angle of motion for the photon.
        timeit = True - print time taken to run (with start time.time() - end time.time()) at the end of the function. This will print regardless of bonusprints.
        mfp_tolerance = 1e-8 - maximum difference between min and max mfp past which our assumption that the change in radius isn't significant enough to matter breaks.
        rng = base_rng - rng to use
        escaperadius = c.R_sun.value - the r past which we have escaped
        bonusprints = True - print extra messages for errors, debugging, etc. a mini verbosity controller basically"""
    # get time
    starttime = time.time()
    # handle numscatters:
    if type(numscatters) != int:
        printif("Hey, your numscatters isn't an int! Making it an int", bonusprints)
        numscatters = int(numscatters)
    
    # define initial internal variables
    init_r = np.sqrt((init_x**2 + init_y**2))
    init_mfp = mfp_fn(ne_fn(init_r), useunits = False)
    init_s = 0
    escaped = False
    # bc none of these guys are actually getting out we are gonna calc the mfp ahead of time.
    disttraveled = rng.exponential(init_mfp, size = numscatters)
    # get the list of angles
    anglevals = rng.uniform(0, 2*np.pi, size = numscatters)
    
    # get the list of xvals
    xtraveled = (disttraveled*np.cos(anglevals))
    ytraveled = (disttraveled*np.sin(anglevals))
    xvals = np.array([itm+np.sum(xtraveled[:ind]+init_x) for ind, itm in enumerate(xtraveled)])
    yvals = np.array([itm+np.sum(ytraveled[:ind]+init_y) for ind, itm in enumerate(ytraveled)])
    # get the r vals
    rvals = np.sqrt((xvals**2 + yvals**2))
    # get the mfp vals
    mfpvals = mfp_fn(ne_fn(rvals), useunits = False)
    # check that our assumption stands:
    if abs(np.max(mfpvals)-np.min(mfpvals)) > mfp_tolerance:
        #pdb.set_trace()
        raise ValueError(f"Whatever initial conditions you have fed this function, it has broken your conditino that the mfp not vary by more than {mfp_tolerance} across all its scatters!")
    # get the current total distance traveled (final value should be sum of disttraveled!)
    svals = np.array([itm+np.sum(disttraveled[:ind]) for ind, itm in enumerate(disttraveled)])
    # get time vals
    tvals = svals*c.c.value
    # did it escape
    escaped = np.array([(lambda x: False if abs(x) < escaperadius else True)(r) for r in rvals])

    # insert initial zero values (so the first row is the initial point)
    xvals = np.insert(xvals, 0, init_x)
    yvals = np.insert(yvals, 0, init_y)
    rvals = np.insert(rvals, 0, init_r)
    tvals = np.insert(tvals, 0, 0)
    anglevals = np.insert(anglevals, 0, init_angle)
    svals = np.insert(svals, 0, init_s)
    mfpvals = np.insert(mfpvals, 0, init_mfp)
    escaped = np.insert(escaped, 0, False)
    
    # make the dict
    scatter_dict = {'x':xvals, 'y':yvals, 'r':rvals, 't':tvals, 's':svals, 'angle':anglevals, 'mfp':mfpvals, 'escaped':escaped}
    
        
    # make pandas table of full result
    fulltable = pd.DataFrame(scatter_dict)
    # time
    if timeit:
        endtime = time.time()
        print("Time elapsed:", endtime-starttime)
    # return
    return(fulltable)


# animation functions
def anim_scatter(scattersim, savename = 'slabanim.gif', framestorun = 'default', overwrite = False,
                 cmap = 'cool', ax = None, fig = None, return_updatefunc = False,
                 fig_kwargs = {}, xlims = 'auto', ylims = 'auto', zoomcushion = 10, ax_kwargs = {}, 
                 slab_kwargs = {'alpha':0.2, 'color':'gray'},slab_out_kwargs = {'alpha':0.3, 'color':'gray'},
                 add_slabpatch_kwargs = {}, add_slabpatch_out_kwargs = {}, 
                 arrow_kwargs = {'width':0.042}, photon_kwargs = {'c':'cyan', 'label':'photon'}, 
                 cushion_frames = 10, funcanim_kwargs = {'blit':True}, pilwriter_kwargs = {'fps':10}, 
                 pltclose = True, pltshow = False, plttight = False, anisave_kwargs = {}, timeit = True,
                 bonusprints = False, photonlabel = True):
    """Function for animating the scattering of a photon through a slab.

    Args:
        scattersim - table of x, y, and angles for the points at which the photon scattered.
        savename = 'slabanim.gif' - filename to saveto.
        framestorun = 'default' - if default will be len(scattersim)+cushion_frames. Alternatively, give a single number here
        overwrite = False - allow overwriting of gif file
        cmap = 'cool' - cmap for the arrows delineating the path of the scatter.
        ax, fig = None - if you want to plot this to a specific figure/axis then pass them here.
        return_updatefunc = False - if you want to return the update_internal function instead of running and saving the animation, set this to True.
        fig_kwargs = {} - kwargs for plt.figure()
        xlims = 'auto', ylims = 'auto' - sets the x and y limits. if left as 'auto' it'll automatically set them either based on the assumption that the slab is a 1km slab with its bottom corner at 0, 0 . if set to 'scale' it'll scale based on the max and min of the scatter simulation.
        zoomcushion = 10 - how much to increase the x and y past their min/max is setting xlims/ylims to scale
        ax_kwargs = {'xlabel':'x (m)', 'ylabel':'y (m)', 'title':'Slab'} - kwargs for ax.set(). x and y will be autoset by lims_fudgefactor and the min/max x and y of the scattered photon unless x and y limits are explicitly passed here.
        slab_kwargs = {'alpha':0.2, 'color':'gray'} - kwargs for the patch that is the "slab"
        slab_out_kwargs = {'alpha':0.3, 'color':'gray'} - kwargs for the patch that is the "slab"
        add_slabpatch_kwargs = {}
        add_slabpatch_out_kwargs = {}
        arrow_kwargs = {'width':0.042} - initial kwargs for the arrow that makes the path (the width will stay the same, the color will update according to the cmap)
        photon_kwargs = {'c':'cyan', 'label':'photon'} - kwargs for the photon
        cushion_frames = 10 - number of extra frames to add at the end
        funcanim_kwargs = {'blit':True} - kwargs for FuncAnimation()
        pilwriter_kwargs = {'fps':10} - kwargs for PillowWriter
        anisave_kwargs = {} - kwargs for anim.save()
        pltclose, pltshow, plttight = True, False, False - whether to run plt.close() / plt.show() / plt.tight() after saving the function.
        timeit = True - whether to print time elapsed (indep of bonusprints)
        bonusprints = False - whether to print extra debuggin messages
        
    """
    starttime = time.time()
    # set up figure
    if fig == None:
        fig = plt.figure(**fig_kwargs)
    if ax == None:
        ax = fig.add_subplot(111)
    else:
        ax = ax
    # set up colors
    colors = colormaps[cmap](np.linspace(0, 1, len(scattersim)))
    # set up general plotting kwargs 
    # set up axis limits
    if type(xlims) == str:
        if xlims.lower() in ['auto', 'default']:
            xlims = (-10, 1010)
        elif xlims.lower() in ['zoom', 'scale']:
            xlims = (np.min(scattersim['x'])-zoomcushion, np.max(scattersim['x'])+zoomcushion)   
        if ylims.lower() in ['auto', 'default']:
            ylims = (-10, 1010)
        elif ylims.lower() in ['zoom', 'scale']:
            ylims = (np.min(scattersim['y'])-zoomcushion, np.max(scattersim['y'])+zoomcushion)
    #print(xlims)
    ax_kwargs = updateandreturndict({'xlim':xlims,
                                     'ylim':ylims, 
                                    'xlabel':'x (m)', 'ylabel':'y (m)', 'title':'Slab'}, 
                                   ax_kwargs)
    ax.set(**ax_kwargs)
    # add patch over the slab
    slab_kwargs = updateandreturndict({'alpha':0.2, 'color':'gray'}, slab_kwargs)
    ax.add_patch(patches.Rectangle((0, 0), 
                                    1000, 1000,
                                    label = 'slab',
                                    **slab_kwargs),
                  **add_slabpatch_kwargs)
    # set up arrow and photon
    arrow_kwargs = updateandreturndict({'width':0.042}, arrow_kwargs)
    photon_kwargs = updateandreturndict({'c':'cyan', 'label':'photon'}, photon_kwargs)
    og_x = scattersim['x'].iloc[0]
    og_y = scattersim['y'].iloc[0]
    arrow = ax.add_patch(patches.Arrow(og_x, og_y, 0, 0, color = colors[0], **arrow_kwargs))
    photon = ax.scatter(og_x, og_y, edgecolors = colors[-1], **photon_kwargs)
    # update_internal function
    def update_internal(framenum, photon = photon, arrow = arrow, path_info = scattersim, photonlabel = photonlabel):
        if framenum >= len(path_info):
            frame = len(path_info)-1
        else:
            frame = framenum
        photon_x = path_info['x'].iloc[frame]
        photon_y = path_info['y'].iloc[frame]
        oldxy = photon.get_offsets()[0]
        scatterangle = path_info['angle'].iloc[frame]
        photon.set_offsets([photon_x, photon_y])
        photon.set_edgecolor(colors[-frame])
        arrow = ax.add_patch(patches.Arrow(oldxy[0], oldxy[1], 
                                               (photon_x-oldxy[0]), (photon_y-oldxy[1]), 
                                                color = colors[frame], **arrow_kwargs))
        #ax.set_title(f"x={path_info['x'].iloc[frame]:.2f}, y={path_info['y'].iloc[frame]:.2f}, scatter #{frame}")
        if photonlabel == True:
            photon.set_label(f'time = {path_info['t'].iloc[frame]:.2e} s')
            ax.legend()
        return(photon, arrow)
    if framestorun == 'default':
        framestorun = len(scattersim)+cushion_frames
    ani = animation.FuncAnimation(fig = fig, func = update_internal, frames = framestorun, blit = True)
    if return_updatefunc:
        endtime = time.time()
        if timeit:
            print(f"Time elapsed: {endtime-starttime}")
        return(update_internal)
    else:
        pilwriter = animation.PillowWriter(**pilwriter_kwargs)
        if savename[-4:] != '.gif':
            printif("Hey, you didn't add .gif to your save name! Adding it for you...", bonusprints)
            savename = savename + '.gif'
        if overwrite == False:
            if glob.glob(savename) != []:
                raise ValueError(f"You are trying to overwrite a file at {savename} and overwrite = False.")
            else:
                ani.save(savename, writer = pilwriter, **anisave_kwargs)
        else:
            ani.save(savename, writer = pilwriter, **anisave_kwargs)
        if plttight:
            plt.tight_layout()
        if pltshow:
            plt.show()
        if pltclose:
            plt.close()
        endtime = time.time()
        if timeit:
            print(f"Time elapsed: {endtime-starttime}")

def anim_scatter_sun(scattersim, savename = 'fancysun.gif', frameindices = 'default', framestepsize= 500, numframes = 'default', overwrite = False,
                 cmap = 'cool', ax = None, fig = None, return_updatefunc = False,
                 fig_kwargs = {}, cushion_x = 0.0005, cushion_y = 0.0005, ax_kwargs = {}, 
                 slab_kwargs = {'alpha':0.2, 'color':'gray'},slab_out_kwargs = {'alpha':0.3, 'color':'orange'},
                 add_slabpatch_kwargs = {}, add_slabpatch_out_kwargs = {}, 
                 arrow_kwargs = {'width':0.042}, photon_kwargs = {'c':'cyan', 'label':'photon'}, 
                 cushion_frames = 10, funcanim_kwargs = {'blit':True}, pilwriter_kwargs = {'fps':10}, 
                 pltclose = True, pltshow = False, plttight = False, anisave_kwargs = {}, timeit = True,
                 bonusprints = False, photonlabel = True, colorful_lines = True):
    """Function for animating the scattering of a photon through the sun.

    Args:
        scattertab - table of x, y, and angles for the points at which the photon scattered.
        savename = 'fancysun.gif' - filename to saveto.
        frameindices = 'default' - if left default will generate indices to reference as we step through the frames using np.arange(0, len(justscattered) + framestepsize, framestepsize)
        framestepsize = 500 - frame step size thru the indices of the sun scatter simulation (for every frame, go this many indices further in the table) so we don't make 100000 frame gifs
        numframes = 'default' - if not left default, rather than using frameindices and framestepsize to determine the indices, it'll use frameindices = np.arange(0, int(len(justscattered)), len(justscattered)/numframes, dtype = int) to determine the indices
        overwrite = False - allow overwriting of gif file
        cmap = 'cool' - cmap for the arrows delineating the path of the scatter.
        ax, fig = None - if you want to plot this to a specific figure/axis then pass them here.
        return_updatefunc = False - if you want to return the update_internal function instead of running and saving the animation, set this to True.
        fig_kwargs = {} - kwargs for plt.figure()
        cushion_x = 0.0005, cushion_y = 0.0005 - amount to cushion the x and y limits
        ax_kwargs = {'xlabel':'x (m)', 'ylabel':'y (m)', 'title':'Sun'} - kwargs for ax.set(). x and y will be autoset by lims_fudgefactor and the min/max x and y of the scattered photon unless x and y limits are explicitly passed here.
        slab_kwargs = {'alpha':0.2, 'color':'gray'} - kwargs for the patch that is the "slab"
        slab_out_kwargs = {'alpha':0.3, 'color':'gray'} - kwargs for the patch that is the "slab"
        add_slabpatch_kwargs = {}
        add_slabpatch_out_kwargs = {}
        arrow_kwargs = {'width':0.042} - initial kwargs for the arrow that makes the path (the width will stay the same, the color will update according to the cmap)
        photon_kwargs = {'c':'cyan', 'label':'photon'} - kwargs for the photon
        cushion_frames = 10 - number of extra frames to add at the end
        funcanim_kwargs = {'blit':True} - kwargs for FuncAnimation()
        pilwriter_kwargs = {'fps':10} - kwargs for PillowWriter
        anisave_kwargs = {} - kwargs for anim.save()
        pltclose, pltshow, plttight = True, False, False - whether to run plt.close() / plt.show() / plt.tight() after saving the function.
        timeit = True
        bonusprints = False
        
    """
    starttime = time.time()
    # MAKE THE TABLE LIMITED TO WHAT ACTUALLY SCATTERED
    justscattered = scattersim
    # set up figure
    if fig == None:
        fig = plt.figure(**fig_kwargs)
    if ax == None:
        ax = fig.add_subplot(111)
    # set up colors
    colors = colormaps[cmap](np.linspace(0, 1, len(justscattered)))

    # set up arrow and photon kwargs
    arrow_kwargs = updateandreturndict({'width':0.042}, arrow_kwargs)
    photon_kwargs = updateandreturndict({'c':'cyan', 'label':'photon'}, photon_kwargs)
    og_x = scattersim['x'].iloc[0]
    og_y = scattersim['y'].iloc[0]
    
    # set up general plotting kwargs 
    # set up axis limits
    ax_kwargs = updateandreturndict({'xlim':(-cushion_x+ np.min(justscattered['x']), 
                                            cushion_x+ np.max(justscattered['x'])),
                                     'ylim':(-cushion_y+ np.min(justscattered['y']), 
                                            cushion_y+ np.max(justscattered['y'])), 
                                    'xlabel':'x (m)', 'ylabel':'y (m)'}, 
                                   ax_kwargs)
    ax.set(**ax_kwargs)
    # add patch over the slab
    axxlim = ax.get_xlim()
    axylim = ax.get_ylim()
    slab_kwargs = updateandreturndict({'alpha':0.2, 'color':'gray', 
                                      'xy':(-cushion_x+ np.min(justscattered['x']),
                                            -cushion_y+ np.min(justscattered['y'])),
                                            'height':2*(abs(np.min(justscattered['y']))+abs(np.max(justscattered['y']))), 
                                            'width':2*(abs(np.min(justscattered['x']))+abs(np.max(justscattered['x'])))}, slab_kwargs)
    slab_out_kwargs = updateandreturndict({'alpha':0.3, 'color':'orange', 
                                          'xy':(og_x,-cushion_y+ np.min(justscattered['y'])),
                                            'height':2*(abs(np.min(justscattered['y']))+abs(np.max(justscattered['y']))),  
                                            'width':2*(abs(np.min(justscattered['x']))+abs(np.max(justscattered['x'])))}, slab_out_kwargs)
    ax.add_patch(patches.Rectangle(**slab_kwargs), 
                 **add_slabpatch_kwargs)
    ax.add_patch(patches.Rectangle(**slab_out_kwargs),
                  **add_slabpatch_out_kwargs)

    #arrow = ax.add_patch(patches.Arrow(og_x, og_y, 0, 0, color = colors[0], **arrow_kwargs))
    linebehind, = ax.plot(og_x, og_y, c = 'k', zorder = 3, alpha = 0.5)
    photon = ax.scatter(og_x, og_y, zorder = 10, edgecolors = colors[-1], **photon_kwargs)

    # handle the frames - 
    if numframes == 'default': # you let the number of frames be determined by the indices and stepsize.
        if type(frameindices) == str:
            if frameindices.lower() != 'default':
                raise ValueError("Hey, the only string you can pass frameindices is 'default' to tell it to use the default value - anything else, and you need to pass it as an array!!")
            frameindices= np.arange(0, len(justscattered) + framestepsize, framestepsize)
        else:
            frameindices = frameindices
    else:
        frameindices = np.arange(0, int(len(justscattered)), len(justscattered)/numframes, dtype = int)
    
    # update_internal function
    #def update_internal(framenum, photon = photon, arrow = arrow, path_info = justscattered):
    def update_internal(framenum, photon = photon, linebehind = linebehind, path_info = justscattered):
        if framenum < len(frameindices):
            frame = frameindices[framenum]
            if frame >= len(path_info):
                frame = len(path_info) - 1 
            photon_x = path_info['x'].iloc[frame]
            photon_y = path_info['y'].iloc[frame]
            oldxy = photon.get_offsets()[0]
            scatterangle = path_info['angle'].iloc[frame]
            photon.set_offsets([photon_x, photon_y])
            photon.set_edgecolor(colors[-frame])
            photon.set(zorder = 10)
            linebehind.set_data([path_info['x'][:frame], path_info['y'][:frame]])
            if colorful_lines:
                linebehind.set_color(colors[frame])
            ax.set_title(f"frame {framenum}, index {frame}")
            if photonlabel:
                photon.set_label(f't = {path_info['t'].iloc[frame]:.2e}s')
                ax.legend()
            #return(photon, arrow)
        return(photon, linebehind,)
        
    ani = animation.FuncAnimation(fig = fig, func = update_internal, frames = len(frameindices)+cushion_frames, blit = True)
    if return_updatefunc:
        endtime = time.time()
        if timeit:
            print(f"Time elapsed: {endtime-starttime}")
        return(update_internal)
    else:
        pilwriter = animation.PillowWriter(**pilwriter_kwargs)
        if savename[-4:] != '.gif':
            printif("Hey, you didn't add .gif to your save name! Adding it for you...", bonusprints)
            savename = savename + '.gif'
        if overwrite == False:
            if glob.glob(savename) != []:
                raise ValueError(f"You are trying to overwrite a file at {savename} and overwrite = False.")
            else:
                ani.save(savename, writer = pilwriter, **anisave_kwargs)
        else:
            ani.save(savename, writer = pilwriter, **anisave_kwargs)
        if plttight:
            plt.tight_layout()
        if pltshow:
            plt.show()
        if pltclose:
            plt.close()
        endtime = time.time()
        if timeit:
            print(f"Time elapsed: {endtime-starttime}")


def anim_fancy_sun(scattersim, savename = 'fancysun.gif', frameindices = 'default', framestepsize= 500, numframes = 'default', overwrite = False,
                 cmap = 'cool', ax = None, fig = None, return_updatefunc = False,
                 fig_kwargs = {}, cushion_x = 0.0005, cushion_y = 0.0005, ax_kwargs = {}, 
                 slab_kwargs = {'alpha':0.2, 'color':'gray'},slab_out_kwargs = {'alpha':0.3, 'color':'orange'},
                 add_slabpatch_kwargs = {}, add_slabpatch_out_kwargs = {}, 
                 arrow_kwargs = {'width':0.042}, photon_kwargs = {'c':'cyan', 'label':'photon'}, 
                 cushion_frames = 10, funcanim_kwargs = {'blit':True}, pilwriter_kwargs = {'fps':10}, 
                 pltclose = True, pltshow = False, plttight = False, anisave_kwargs = {}, timeit = True,
                 bonusprints = False, photonlabel = True, colorful_lines = True):
    """Function for animating the scattering of a photon through the sun, WITH adjusting x/y axes to the scale of its path.

    Args:
        scattertab - table of x, y, and angles for the points at which the photon scattered.
        savename = 'fancysun.gif' - filename to saveto.
        frameindices = 'default' - if left default will generate indices to reference as we step through the frames using np.arange(0, len(justscattered) + framestepsize, framestepsize)
        framestepsize = 500 - frame step size thru the indices of the sun scatter simulation (for every frame, go this many indices further in the table) so we don't make 100000 frame gifs
        numframes = 'default' - if not left default, rather than using frameindices and framestepsize to determine the indices, it'll use frameindices = np.arange(0, int(len(justscattered)), len(justscattered)/numframes, dtype = int) to determine the indices
        overwrite = False - allow overwriting of gif file
        cmap = 'cool' - cmap for the arrows delineating the path of the scatter.
        ax, fig = None - if you want to plot this to a specific figure/axis then pass them here.
        return_updatefunc = False - if you want to return the update_internal function instead of running and saving the animation, set this to True.
        fig_kwargs = {} - kwargs for plt.figure()
        cushion_x = 0.0005, cushion_y = 0.0005 - amount to cushion the x and y limits
        ax_kwargs = {'xlabel':'x (m)', 'ylabel':'y (m)', 'title':'Sun'} - kwargs for ax.set(). x and y will be autoset by lims_fudgefactor and the min/max x and y of the scattered photon unless x and y limits are explicitly passed here.
        slab_kwargs = {'alpha':0.2, 'color':'gray'} - kwargs for the patch that is the "slab"
        slab_out_kwargs = {'alpha':0.3, 'color':'gray'} - kwargs for the patch that is the "slab"
        add_slabpatch_kwargs = {}
        add_slabpatch_out_kwargs = {}
        arrow_kwargs = {'width':0.042} - initial kwargs for the arrow that makes the path (the width will stay the same, the color will update according to the cmap)
        photon_kwargs = {'c':'cyan', 'label':'photon'} - kwargs for the photon
        cushion_frames = 10 - number of extra frames to add at the end
        funcanim_kwargs = {'blit':True} - kwargs for FuncAnimation()
        pilwriter_kwargs = {'fps':10} - kwargs for PillowWriter
        anisave_kwargs = {} - kwargs for anim.save()
        pltclose, pltshow, plttight = True, False, False - whether to run plt.close() / plt.show() / plt.tight() after saving the function.
        timeit = True
        bonusprints = False
        
    """
    starttime = time.time()
    # MAKE THE TABLE LIMITED TO WHAT ACTUALLY SCATTERED
    justscattered = scattersim
    # set up figure
    if fig == None:
        fig = plt.figure(**fig_kwargs)
    if ax == None:
        ax = fig.add_subplot(111)
    # set up colors
    colors = colormaps[cmap](np.linspace(0, 1, len(justscattered)))

    # set up arrow and photon kwargs
    arrow_kwargs = updateandreturndict({'width':0.042}, arrow_kwargs)
    photon_kwargs = updateandreturndict({'c':'cyan', 'label':'photon'}, photon_kwargs)
    og_x = scattersim['x'].iloc[0]
    og_y = scattersim['y'].iloc[0]
    
    # set up general plotting kwargs 
    # set up axis limits
    ax_kwargs = updateandreturndict({'xlim':(-cushion_x+ og_x, cushion_x+ og_x),
                                     'ylim':(-cushion_y+ og_y, cushion_y+ og_y), 
                                    'xlabel':'x (m)', 'ylabel':'y (m)'}, 
                                    ax_kwargs)
    ax.set(**ax_kwargs)
    # add patch over the slab
    axxlim = ax.get_xlim()
    axylim = ax.get_ylim()
    slab_kwargs = updateandreturndict({'alpha':0.2, 'color':'gray', 
                                      'xy':(-cushion_x+ np.min(justscattered['x']),
                                            -cushion_y+ np.min(justscattered['y'])),
                                            'height':2*(abs(np.min(justscattered['y']))+abs(np.max(justscattered['y']))), 
                                            'width':2*(abs(np.min(justscattered['x']))+abs(np.max(justscattered['x'])))}, slab_kwargs)
    slab_out_kwargs = updateandreturndict({'alpha':0.3, 'color':'gray', 
                                          'xy':(og_x, -cushion_y+ np.min(justscattered['y'])),
                                                'height':2*(abs(np.min(justscattered['y']))+abs(np.max(justscattered['y']))),  
                                            'width':2*(abs(np.min(justscattered['x']))+abs(np.max(justscattered['x'])))}, slab_out_kwargs)
    ax.add_patch(patches.Rectangle(**slab_kwargs), 
                 **add_slabpatch_kwargs)
    ax.add_patch(patches.Rectangle(**slab_out_kwargs),
                  **add_slabpatch_out_kwargs)

    #arrow = ax.add_patch(patches.Arrow(og_x, og_y, 0, 0, color = colors[0], **arrow_kwargs))
    linebehind, = ax.plot(og_x, og_y, c = 'k', zorder = 3, alpha = 0.5)
    photon = ax.scatter(og_x, og_y, zorder = 10, edgecolors = colors[-1], **photon_kwargs)

    # handle the frames - 
    if numframes == 'default': # you let the number of frames be determined by the indices and stepsize.
        if type(frameindices) == str:
            if frameindices.lower() != 'default':
                raise ValueError("Hey, the only string you can pass frameindices is 'default' to tell it to use the default value - anything else, and you need to pass it as an array!!")
            frameindices= np.arange(0, len(justscattered) + framestepsize, framestepsize)
        else:
            frameindices = frameindices
    else:
        frameindices = np.arange(0, int(len(justscattered)), len(justscattered)/numframes, dtype = int)
    
    # update_internal function
    #def update_internal(framenum, photon = photon, arrow = arrow, path_info = justscattered):
    def update_internal(framenum, photon = photon, linebehind = linebehind, path_info = justscattered):
        if framenum < len(frameindices):
            frame = frameindices[framenum]
            if frame >= len(path_info):
                frame = len(path_info) - 1 
            photon_x = path_info['x'].iloc[frame]
            photon_y = path_info['y'].iloc[frame]
            oldxy = photon.get_offsets()[0]
            scatterangle = path_info['angle'].iloc[frame]
            photon.set_offsets([photon_x, photon_y])
            photon.set_edgecolor(colors[-frame])
            photon.set(zorder = 10)
            linebehind.set_data([path_info['x'][:frame], path_info['y'][:frame]])
            if colorful_lines:
                linebehind.set_color(colors[frame])
            ax.set_title(f"frame {framenum}, index {frame}")
            if photonlabel:
                photon.set_label(f'scatter = {scatterangle:.2f}')
                ax.legend()
            #pdb.set_trace()
            if frame != 0:
                ax.set_xlim((-cushion_x+ np.min(path_info['x'][:frame]), cushion_x+ np.max(path_info['x'][:frame])))
                ax.set_ylim((-cushion_y+ np.min(path_info['y'][:frame]), cushion_y+ np.max(path_info['y'][:frame])))
            #return(photon, arrow)
        return(photon, linebehind,)
        
    ani = animation.FuncAnimation(fig = fig, func = update_internal, frames = len(frameindices)+cushion_frames, blit = True)
    if return_updatefunc:
        endtime = time.time()
        if timeit:
            print(f"Time elapsed: {endtime-starttime}")
        return(update_internal)
    else:
        pilwriter = animation.PillowWriter(**pilwriter_kwargs)
        if savename[-4:] != '.gif':
            printif("Hey, you didn't add .gif to your save name! Adding it for you...", bonusprints)
            savename = savename + '.gif'
        if overwrite == False:
            if glob.glob(savename) != []:
                raise ValueError(f"You are trying to overwrite a file at {savename} and overwrite = False.")
            else:
                ani.save(savename, writer = pilwriter, **anisave_kwargs)
        else:
            ani.save(savename, writer = pilwriter, **anisave_kwargs)
        if plttight:
            plt.tight_layout()
        if pltshow:
            plt.show()
        if pltclose:
            plt.close()
        endtime = time.time()
        if timeit:
            print(f"Time elapsed: {endtime-starttime}")

# handle the 'bools'
def boolify(argval):
    # I bet this is something that can be done in one line with just bool(argval) to get the same effect.
    # but also, I don't want to take chances with how this is handled being different across different versions of any package, so I'm hardcoding this.
    if argval == 0:
        argval = False
    else:
        argval = True
    return(argval)


# now begin the actual script part:
if __name__ == '__main__':
    print('woo')
    # hoo boy. time for a lot of arguments
    parser = argparse.ArgumentParser(description = "Script for simulating the path of photons through a uniform slab and through the Sun. This function has five runmodes: 'explore', which will run every run mode in sequence; 'p1', which will run a simulatii")
    parser.add_argument('--runmode', default = 'explore', type = str, help = f"The mode in which to run this script. Default is 'explore', which will run all parts relating to the homework and the bonus parts.")
    parser.add_argument('--slowdown', default = 0, type = float, help = f"This script can use sleep(slowdown) to pause the script at certain points and give you more time to read printouts in terminal. Increase slowdown to pause longer.")
    parser.add_argument("--seed", default = 4242, type = int, help = f"The random seed number to be used for rng purposes.")
    parser.add_argument("--cmap", default = 'cool', type = str, help = f"The default cmap to be used when making animations.")
    parser.add_argument("-n", default = 1e20*(u.cm**-3), help = f"The default electron density for the slab.")
    parser.add_argument("--sigma", default = 6.652e-25*(u.cm**2), help = f"The default sigma_t = Thompson cross section for the slab.")
    parser.add_argument("--slab_xyang", default = (500, 500, 0), nargs = 3, type = float, help = f"initial X, Y, and angle for the slab simulation (ie, midway through the slab, with no initial angle).")
    parser.add_argument("--sun_xyang", default = (0, 0, 0), nargs = 3, type = float, help = f"initial X, Y, and angle for the sun simulation (ie, center of the sun, with no initial angle).")
    parser.add_argument("--ns_slab", default = 100000, help = f"Number of scattering events to simulate (until escape at least) for the slab.")
    parser.add_argument("--ns_sun", default = 100000, help = f"Number of scattering events to simulate (until escape at least) for the sun.")
    parser.add_argument("--slabgif_p1", default = 'slabanim.gif', type = str, help = f"Filename to save the animation of the single photon in the slab (from part 1 of the homework).")
    parser.add_argument("--sungif_p1", default = 'sunanim.gif', type = str, help = f"Filename to save the animation of the single photon in the sun (from part 2 of the homework).")
    parser.add_argument("--sun_numframes", default = 100, type = int, help = f"Number of frames to use when animating the scatter through the sun. NOTE THAT THIS WILL ANIMATE THE ENTIRETY OF THE SCATTER FOR A SINGLE PHOTON; it just adjusts the size of the steps that are taken through the simulated path of the photon to create a gif with an according number of frames.")
    parser.add_argument('--slab_fps', default = 10, type = int, help = f"FPS for slab gif")
    parser.add_argument('--sun_fps', default = 10, type = int, help = f"FPS for sun gif")
    parser.add_argument('--slab_examples', default = 50, type = int, help = f"Number of example photons to generate for an animation of multiple photons going thorugh the slab.")
    parser.add_argument('--sun_examples', default = 10, type = int, help = f"Number of example photons to generate for an animation of multiple photons going thorugh the sun.")
    parser.add_argument('--slabmultgif', default = 'multi_SLAB.gif', type = str, help = f"Name of file to which the simulation of the slab with many photons is saved")
    parser.add_argument('--sunmultgif', default = 'multi_SUN.gif', type = str, help = f"Name of file to which the simulation of the sun with many photons is saved")
    parser.add_argument('--mfptol', default = 1e-8, type = float, help = f"The tolerance for the mean free path- ie, if you're running the sun simulation with the fast (vectorized) method, how much variance in the mfp from where the photon starts to where it ends before you decide your initial conditions don't work with the assumption that the mfp doesn't vary enough to matter (which is the assumptions that we need to make for the fast method).")
    parser.add_argument('--fastsun', default = 1, type = int, help = f"Whether or not to use the fast method for the sun scattering. 1= True, 0  = False. Defaults to True, which means that we assume it won't scatter far enough for the change in mfp with radius to matter; adjust how much variation in mfp you will accept with mfptol.")
    parser.add_argument('--fastslab', default = 0, type = int, help = f"Whether or not to use the fast method for the slab scattering. 1= True, 0  = False. Defaults to 0, False, because it actually takes *less* time for the default conditions for the photon to escape than it would for the non for-loop function to finish running.")
    parser.add_argument('--sunzoomgif', default = 'sun_zoom.gif', type = str, help = f"Name of file to which the simulation of the sun with a single photon, but adjusting the axes to 'zoom out' as the photon travels, is saved.")
    parser.add_argument('--bet', default = 7, type = int, help = f"Index (from 0 to 9) of the photon you want to bet will escape the slab first.")
    parser.add_argument('--betgif', default = 'slab_bet.gif', type = str, help = f"Name of file to which the simulation of the slab where you are betting on which photon escapes first goes.")
    # place to put arg for the slab frames, if i have time
    #parse.

    args = parser.parse_args()
    print("Args used: \n", args, '\n\n')

    runmode = args.runmode
    slowdown = args.slowdown
    seed = args.seed
    cmap = args.cmap
    n = args.n
    sigma = args.sigma
    slab_x, slab_y, slab_angle = [float(x) for x in args.slab_xyang]
    sun_x, sun_y, sun_angle = [float(x) for x in args.sun_xyang]
    ns_slab = args.ns_slab
    ns_sun = args.ns_sun
    slabgif_p1 = args.slabgif_p1
    sungif_p1 = args.sungif_p1
    sun_numframes = args.sun_numframes
    slab_fps = args.slab_fps
    sun_fps = args.sun_fps
    slab_examples = args.slab_examples
    sun_examples = args.sun_examples
    slabmultgif = args.slabmultgif
    sunmultgif = args.sunmultgif
    mfptol = args.mfptol
    fastsun = boolify(args.fastsun)
    fastslab = boolify(args.fastslab)
    sunzoomgif = args.sunzoomgif
    bet = args.bet
    betgif = args.betgif
    print(f"Non-for-loop version of slab simulation: {fastslab}")
    print(f"Non-for-loop version of sun simulation: {fastsun}")

    base_rng = np.random.default_rng(seed)

    runopts_dict = {'explore':ifanyofthesein(['all', 'explore', 'explor', 'ex', 'run all'], runmode.lower()), 
                    'run_slab':ifanyofthesein(['run_slab', 'run slab', 'p1', 'runp1', 'run p1', 'run_p1', 'p1_run'], runmode.lower()), 
                    'run_sun':ifanyofthesein(['run_sun', 'run sun', 'p2', 'runp2', 'run p2', 'run_p2', 'p2_run'], runmode.lower()),
                    'slab_many':ifanyofthesein(['p3', 'run_p3', 'run p3', 'slab many', 'slab_many', 'run_slab_many'], runmode.lower()), 
                    'sun_many':ifanyofthesein(['p4', 'run_p4', 'run p4', 'sun many', 'sun_many', 'run_sun_many'], runmode.lower()), 
                    'sun_rescale':ifanyofthesein(['p5', 'run_p5', 'run p5', 'sun rescale', 'sun_rescale', 'run_sun_rescale', 'zoom'], runmode.lower()), 
                    'bet_slab':ifanyofthesein(['p6', 'run_p6', 'run p6', 'bet', 'bet_slab', 'bet slab'], runmode.lower()), 
                    
                   }
    if runopts_dict['explore']:
        for ky in runopts_dict:
            runopts_dict[ky] = True

    # print intro
    print('\n\n')
    print('***'*10)
    print("Welcome to my script for exploring photon scattering! Fair warning - this script makes a number of animations, and the resulting .gifs can get large. To avoid giving the user the chance to accidentally freeze up the script by trying to make a gif with 10000 frames, I've hardcoded more things than I usually do in the script itself, so there are a number of options to customize the gifs in the functions that I made than I actually use. If you want to explore the functions used in this script with more flexibility, I suggest using this more like a module than a script and importing some of this into a Jupyter notebook to play around with.")
    print('***'*10)
    print('\n\n')
    sleep(slowdown)

    if runopts_dict['run_slab']:
        print(f"Part 1: Perform a Monte Carlo simulation of a photon going throuh a slab with ne = 1e20 cm**-3 and a width of 1 km. For the sake of this simulation by default I set it to start in the center of the slab, you are currently running with init_x = {slab_x}, init_y = {slab_y}, init_angle = {slab_angle}.")
        print(f"Running the simulation for a single photon...")
        if fastslab:
            slabtab = scatter_slab_VECTORIZED(numscatters=ns_slab, init_x = slab_x, init_y = slab_y, init_angle = slab_angle, 
                              n = n, sigma = sigma, rng = base_rng)
            print(f"Finding point of first escape (since all after are irrelevant) and cutting down table")
            pointescape = np.where(slabtab['escaped'] == True)[0][0]+1
            slabtab = slabtab.iloc[0:pointescape]
            pdb.set_trace()
        else:
            slabtab = scatter_slab(numscatters=ns_slab, init_x = slab_x, init_y = slab_y, init_angle = slab_angle, 
                      n = n, sigma = sigma, rng = base_rng)
        #pdb.set_trace()
        print(f"OK, done! Now running the animation:")
        anim_scatter(slabtab, savename = slabgif_p1, overwrite = True, 
                    cmap = cmap, pilwriter_kwargs = {'fps':slab_fps})
        print(f"Review the animation, and you should see that the single photon scatters out and escaped pretty quickly if you used the default arguments!")
        print(f"Final values for the photon:")
        print(slabtab.iloc[-1])
        sleep(slowdown)
    #part 2
    if runopts_dict['run_sun']:
        print(f"Part 2: perform a simulation of a photon going through the Sun, following the fit for the Sun's electron density as a function of radius taken from equation 4.2 in Neutrino Astrophysics (John Bachall). I set it to start in the center of the sun by default; you are currently running with init_x = {sun_x}, init_y = {sun_y}, init_angle = {sun_angle}")
        print(f"Running the simulation for a single photon...")
        if fastsun:
            suntab = scatter_sun_FAST(numscatters=ns_sun, init_x = sun_x, init_y = sun_y, init_angle = sun_angle,
                            sigma = sigma, rng = base_rng)
        else:
            suntab = scatter_sun(numscatters=ns_sun, init_x = sun_x, init_y = sun_y, init_angle = sun_angle,
                            sigma = sigma, rng = base_rng)
            
        print(f"OK, done! Now running the animation:")
        anim_scatter_sun(suntab, savename = sungif_p1, overwrite = True, cmap = cmap, numframes = sun_numframes,  pilwriter_kwargs = {'fps':sun_fps})
        print(f"Review the animation! You should see that the photon didn't come close to escaping at all (which makes sense given the density of the sun!!) if you used the default arguments.")
        print(f"Final values for the photon:")
        print(suntab.iloc[-1])
        sleep(slowdown)
    #part 3
    if runopts_dict['slab_many']:
        print(f"Now, let's look at what happens with the slab if you do it for many photons and not just 1. By default this will run 50 example photons; you currently have {slab_examples} photons set to run.")
        print(f"Running the simulation for {slab_examples} photon(s)...")
        slab_runs = []
        starttime = time.time()
        for num in np.arange(0, slab_examples):
            if fastslab:
                temptab = scatter_slab_VECTORIZED(bonusprints = False, timeit =False, init_x =slab_x, init_y = slab_y, init_angle = slab_angle, n = n, sigma= sigma, rng = base_rng)
                print(f"Finding point of first escape (since all after are irrelevant) and cutting down table")
                pointescape = np.where(temptab['escaped'] == True)[0][0]+1
                temptab = temptab.iloc[0:pointescape]
            else:
                temptab = scatter_slab(bonusprints = False, timeit =False, init_x =slab_x, init_y = slab_y, init_angle = slab_angle, n = n, sigma= sigma, rng = base_rng)
            slab_runs.append(temptab)
        print(f"Time elapsed: {time.time() - starttime}")
        print(f"OK, done! Now running the animation:")
        # because it's multiple in one we are going to have it return the update functions and then actually do the writer and saving the animation external to the functions.
        starttime = time.time() 
        fig = plt.figure()
        ax1 = fig.add_subplot()
        ufnlist = []
        # get the first update function; this one is external so i can have this set the slab_kwargs
        uf1 = anim_scatter(slab_runs[0],overwrite=True, savename = slabmultgif, return_updatefunc=True, 
                    pltclose=False, pltshow=False, ax = ax1, 
                               photonlabel = False, fig = fig, timeit=False)
        ufnlist.append(uf1)
        # get the others, but blank out the slab, so it doesn't get way higher opacity than the one-photon version
        for phtn in slab_runs[1:10]:
            ufnlist.append(anim_scatter(phtn,overwrite=True, savename = slabmultgif, return_updatefunc=True, 
                    pltclose=False, pltshow=False, ax = ax1, fig = fig, 
                                        photonlabel = False, slab_kwargs={'alpha':0}, timeit=False))
        def tempufn(framen, ufnlist = ufnlist):
            for ufn in ufnlist:
                ufn(framen)
        pilwriter = animation.PillowWriter(fps = slab_fps)
        ani = animation.FuncAnimation(fig = fig,func = tempufn, frames = np.max([len(tab) for tab in slab_runs])+5)
        ani.save(slabmultgif, writer = pilwriter)
        print(f"Time elapsed: {time.time() - starttime}")
        escapedtheslab = 0
        for tab in slab_runs:
            if True in tab['escaped'].values:
                escapedtheslab +=1
        print(f"Number escaped: {escapedtheslab}")
        print(f"Done! You should see some range in how long it takes for them to escape - some zoom out after just one or two scatters, others bounce around a lot more.")
    
    #part 4
    if runopts_dict['sun_many']:
        print(f"Now, let's look at what happens with the Sun if you do it for many photons and not just 1. By default this will run 10 example photons (because it's more intensive for the sun!); you currently have {sun_examples} photons set to run.")
        print(f"Running the simulation for {sun_examples} photon(s)...")
        sun_runs = []
        starttime = time.time()
        for num in np.arange(0, sun_examples):
            if fastsun:
                try:
                    temptab = scatter_sun_FAST(bonusprints = False, timeit =False, init_x =sun_x, init_y = sun_y, init_angle = sun_angle, sigma= sigma, rng = base_rng)
                except ValueError: # if get a value error, raise a different value error that's more specific
                    raise ValueError(f"Hey, whatever initial conditions you gave this, it's broken the assumption that the MFP won't vary by enough to matter (variation in mfp < {mfptol}) that allows us to use the fast function for simulating the sun. If you REALLY want to use whatever initial conditions you gave this script, you need to switch to the slower function by setting slowsun = 1.")
            else:
                temptab = scatter_sun(bonusprints = False, timeit = False, init_x = sun_x, init_y = sun_y, init_angle = sun_angle, sigma = sigma, rng = base_rng)
            sun_runs.append(temptab)
        print(f"Time elapsed: {time.time() - starttime}")
        print(f"OK, done! Now running the animation, with a gray/orange box splitting the background into in front of / behind the point the photon starts at (in x):")
        # because it's multiple in one we are going to have it return the update functions and then actually do the writer and saving the animation external to the functions.
        starttime = time.time() 
        fig = plt.figure()
        ax1 = fig.add_subplot()
        ufnlist = []
        # get the first update function; this one is external so i can have this set the sun_kwargs
        uf1 = anim_scatter_sun(sun_runs[0],overwrite=True, savename = sunmultgif, return_updatefunc=True, pltclose=False, pltshow=False, ax = ax1, ax_kwargs = {'xlim':(-0.05, 0.05), 'ylim':(-0.05, 0.05)}, slab_kwargs = {'xy':(-0.05, -0.05), 'height':1, 'width':1}, slab_out_kwargs = {'xy':(0, -0.05), 'height':1, 'width':1}, photonlabel = False, fig = fig, timeit=False, numframes = sun_numframes)
        ufnlist.append(uf1)
        # get the others, but blank out the sun, so it doesn't get way higher opacity than the one-photon version
        for phtn in sun_runs[1:10]:
            ufnlist.append(anim_scatter_sun(phtn,overwrite=True, savename = sunmultgif, return_updatefunc=True, 
                    pltclose=False, pltshow=False, ax = ax1, ax_kwargs = {'xlim':(-0.05, 0.05), 'ylim':(-0.05, 0.05)}, slab_kwargs = {'alpha':0}, slab_out_kwargs = {'alpha':0}, photonlabel = False, fig = fig, timeit=False, numframes = sun_numframes))
        def tempufn(framen, ufnlist = ufnlist):
            for ufn in ufnlist:
                ufn(framen)
        pilwriter = animation.PillowWriter(fps = sun_fps)
        ani = animation.FuncAnimation(fig = fig,func = tempufn, frames = sun_numframes)
        ani.save(sunmultgif, writer = pilwriter)
        print(f"Time elapsed: {time.time() - starttime}")
        print(f"Done! You should see some range in how long it takes for them to escape - some zoom out after just one or two scatters, others bounce around a lot more.")
        sleep(slowdown)
    # run 5
    if runopts_dict['sun_rescale']:
        print(f"For fun, let's make a version of the sun animation that rescales the animation based on where the photon is (it 'zooms out' to follow the photon)!")
        print(f"Running the initial simulation...")
        starttime = time.time()
        if fastsun:
            suntab = scatter_sun_FAST(bonusprints = False, timeit = False, init_x = sun_x, init_y = sun_y, init_angle = sun_angle, sigma = sigma, rng = base_rng)
        else:
            suntab = scatter_sun(bonusprints = False, timeit = False, init_x = sun_x, init_y = sun_y, init_angle = sun_angle, sigma = sigma, rng = base_rng)
        print(f"Time elapsed: {time.time() - starttime}")
        print(f"OK, done! Now running the animation, with a gray/orange box splitting the background into in front of / behind the point the photon starts at (in x):")
        starttime = time.time()
        anim_fancy_sun(suntab, savename = sunzoomgif, overwrite = True, cmap = cmap, numframes = sun_numframes,  pilwriter_kwargs = {'fps':sun_fps})
        print(f"Time elapsed: {time.time() - starttime}")
        print(f"Done!!")
    # run 6
    if runopts_dict['bet_slab']:
        print(f"For fun; use the --bet argument to bet on which photon (from 0 to 9) is going to escape the slab in the fewest number of steps!!! You have bet (or have left the default bet as): {bet}")
        print(f"Running the simulation for {slab_examples} photon(s)...")
        slab_runs = []
        starttime = time.time()
        for num in np.arange(0, slab_examples):
            if fastslab:
                temptab = scatter_slab_VECTORIZED(bonusprints = False, timeit =False, init_x =slab_x, init_y = slab_y, init_angle = slab_angle, n = n, sigma= sigma, rng = base_rng)
            else:
                temptab = scatter_slab(bonusprints = False, timeit =False, init_x =slab_x, init_y = slab_y, init_angle = slab_angle, n = n, sigma= sigma, rng = base_rng)
            slab_runs.append(temptab)
        print(f"Time elapsed: {time.time() - starttime}")
        print(f"Checking which was shortest...")
        shortesttab = np.min([len(tab) for tab in slab_runs])
        for ind, tab in enumerate(slab_runs):
            if len(tab) == shortesttab:
                if ind == bet:
                    print(ind, 'was your bet, you won!!!')
                else:
                    print(ind, 'was not your bet :(')
        
        print(f"Now running the animation so you can see them go!!")
        # because it's multiple in one we are going to have it return the update functions and then actually do the writer and saving the animation external to the functions.
        starttime = time.time() 
        fig = plt.figure()
        ax1 = fig.add_subplot()
        ufnlist = []
        # get the first update function; this one is external so i can have this set the slab_kwargs
        uf1 = anim_scatter(slab_runs[0],overwrite=True, savename = betgif, return_updatefunc=True, 
                    pltclose=False, pltshow=False, ax = ax1, 
                               photonlabel = False, fig = fig, timeit=False)
        ufnlist.append(uf1)
        # get the others, but blank out the slab, so it doesn't get way higher opacity than the one-photon version
        for phtn in slab_runs[1:10]:
            ufnlist.append(anim_scatter(phtn,overwrite=True, savename = betgif, return_updatefunc=True, 
                    pltclose=False, pltshow=False, ax = ax1, fig = fig, 
                                        photonlabel = False, slab_kwargs={'alpha':0}, timeit=False))
        def tempufn(framen, ufnlist = ufnlist):
            for ufn in ufnlist:
                ufn(framen)
        pilwriter = animation.PillowWriter(fps = slab_fps)
        ani = animation.FuncAnimation(fig = fig,func = tempufn, frames = np.max([len(tab) for tab in slab_runs])+5)
        ani.save(betgif, writer = pilwriter)
        print(f"Time elapsed: {time.time() - starttime}")