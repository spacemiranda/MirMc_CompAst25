"""
Script for calculating and animating the orbit of space junk: namely, a ball bearing around a cylinder in space!
For CUNY Computational Methods in Physics, Fall 2025
--Miranda McCarthy
"""

# imports
import numpy as np
import pandas as pd
import astropy.constants as c
import astropy.units as u

import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot

import copy
import time
from time import sleep
import pdb
import glob

from matplotlib import colormaps
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import argparse
#init_notebook_mode(connected=True)  

# basic functions I reuse across notebooks
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
    """Another simple space saving function I use a lot. Rather than checking if foo in [(list of things)], it checks if any item in a list is in a given object.
        Args:
            anyofthese - a list of things to check if they are in areinthis.
            areinthis - whatever you want to search for the items in anyofthese.
        Returns:
            True, if any item in anyofthese is in areinthis. Else False."""
    for itm in anyofthese:
        if itm in areinthis:
            return(True)
def boolify(argval):
    """I bet this is something that can be done in one line with just bool(argval) to get the same effect... but also, I don't want to take chances with how this is handled being different across different versions of any package, so I'm hardcoding this."""
    if argval == 0:
        argval = False
    else:
        argval = True
    return(argval)
    
# orbits.
def f_junk(rvec, L = 2, M = 10, G = 1, t = None):
    """Get the derivatives for the gravitational force of a cylinder of space junk with mass M and length L. This is the broken down four-first-order version of the two second order equations for x'' and y'' ; where we've broken it into our four equations for x, for y, for dx/dt, for dy/dt.
    Args:
        rvec - the initial vector [x, y, dx/dt, dy/dt]
        L = 2 - cylinder length
        M = 10 - cylinder mass
        G = 1 - our "G" (very very rounded up to make it larger)
        t = None
    """
    # this is pretty much just like the example in class with the slight difference in the denominator.
    x = rvec[0] # equation for x- hint, that it'll become basically the dx/dt value in the next step
    y = rvec[1] # equation for y- hint, that it'll become basically the dy/dt value in the next step
    dxdt = rvec[2]
    dydt = rvec[3]
    # calc r
    r = np.sqrt((x**2) + (y**2))
    # get the secondary derivatives, or acceleration
    dvxdt = -G*M*x / (r**2)*((r**2)+(0.25*(L**2)))**0.5
    dvydt = -G*M*y / (r**2)*((r**2)+(0.25*(L**2)))**0.5
    return(np.array([dxdt, dydt, dvxdt, dvydt]))

# rk4 solver
def runge_kutta_vectorized(h, rval, outto, inpfn,
                           inpfn_kwargs = {'L':2, 'M':10, 'G':1, 't':None}):
    """
    A fourth-order runge kutta solver.
    Args:
        h - step
        rval - initial vector [x, y, dx/dt, dy/dt]
        outto - how far out in timesteps we want to go (ie, we're going through the array np.arange(0, outto, h))
        inpfn - input function that we are solving
        inpfn_kwargs = {'L':2, 'M':10, 'G':1, 't':None} - input function kwargs
    """
    inpfn_kwargs = updateandreturndict({'L':2, 'M':10, 'G':1, 't':None}, inpfn_kwargs)
    arange_t = np.arange(0, outto, h)
    rvals = [rval]
    for ind, tval in enumerate(arange_t[1:]):
        k1 = h*inpfn(rvals[ind], **inpfn_kwargs)
        k2 = h*inpfn(rvals[ind]+0.5*k1, **inpfn_kwargs)
        k3 = h*inpfn(rvals[ind]+0.5*k2, **inpfn_kwargs)
        k4 = h*inpfn(rvals[ind]+k3, **inpfn_kwargs)
        new_r = rvals[ind] + (k1+2*k2+2*k3+k4)/6
        rvals.append(new_r)
        #pdb.set_trace()
    return(rvals)

# plotting
def basic_plotly_orbit(inpdict, xcol, ycol, colorcol = None, plotline = True, plotscatter = True):
    """Basic function to plot the orbit of the space junk around the cylinder.
    Args:
        inpdict - input dictionary with data for plotly to plot.
        xcol - name of x column in dictionary
        ycol - name of y column in dictionary
        colorcol - name of column to use to set the colormap of the scatterplot points."""
    if [plotline, plotscatter] == [False, False]:
        raise ValueError("You must plot either or both of the line or scatter plots!")
    if plotline:
        f1 = px.line(inpdict, x = xcol, y = ycol)
    else:
        f1 = px.line(None)
    if plotscatter:
        f2 = px.scatter(inpdict, x = xcol, y = ycol, color = colorcol)
        figtitle = f"Orbit of space junk (scatter color = {colorcol})"
    else:
        f2 = px.scatter(None)
        figtitle = 'Orbit of space junk'
    fig = go.Figure(data = f1.data + f2.data)
    fig.update_layout(title = {'text':figtitle}, 
                 xaxis = {'title':{'text':xcol}}, 
                 yaxis = {'title':{'text':ycol}}, )
    fig.show()
    
def basic_plotly_anim(inpdf, xcol, ycol, plotline = True):
    """Basic function to plot the orbit of the space junk around the cylinder.
    Args:
        inpdf - input dataframe with data for plotly to plot.
        xcol - name of x column in dictionary
        ycol - name of y column in dictionary
        plotline = True - whether or not to also plot the orbit line."""

    if type(inpdf) == dict:
        print("Hey, you passed a dict instead of a DataFrame!! Attempting to convert it to a pandas DataFrame.")
        inpdf = pd.DataFrame(inpdf)
    fig = px.scatter(inpdf, x = xcol, y = ycol, animation_frame= 't')
    if plotline:
        fig.add_trace(go.Scatter(x = inpdf[xcol], y = inpdf[ycol],
                         mode = 'lines', legendgroup='orbitline', zorder = 0))
        fig.update_traces(patch={'line':{'dash':'dot'}}, selector={'legendgroup':'orbitline'})
    
    fig.update_layout( title = {'text':'Orbit of the space junk'} )
    fig.show()
# space junk animation
def space_junk_anim(inpdf, savename = 'defaultanim.gif', overwrite = False,
                  xcol = 'x', ycol = 'y', colorcol = 't',
                   imagebg = None, decimalpts = 0.2,
                   orbitline_kwargs = {}, cylinder_kwargs = {}, orbscatter_kwargs = {},
                  frameindices = 'default', framestepsize = 'default',
                  numframes = 'default', cushion_frames = 0,
                  fig = None, ax = None,  
                  fig_kwargs = {}, ax_kwargs = {}, 
                  pltclose = True, pltshow = False, plttight = True ,
                  pilwriter_kwargs = {'fps':20}, timeit = True, 
                  funcanim_kwargs = {'blit':True}, anisave_kwargs = {}, 
                  return_update_func = False, bonusprints =True):

    """Animate the orbit of space junk!
    
    Args:
        inpdf - input dataframe.
        savename = 'defaultanim.gif' - where to save the animation to.
        overwrite = False - whether to allow overwriting of the resulting gif.
        
        xcol = 'x', ycol = 'y' - names of the x and y columns from the dataframe
        framecol = 't' - name of the column that is changing with each timestep, so determines the label that appears.
        imagebg = 'jsc_auroraaustralis.png' - if not None, name of the file to use as the bg of the gif
        decimalpts = 0.2 - the number of decimal places to show according to number:decimalptsf formattin
        orbitline_kwargs = {} - kwargs for the line of the orbit
        cylinder_kwargs = {} - kwargs for the marker for the cylinder.
        orbitscatter_kwargs = {} - kwargs for the orbit's scatter plot.
        
        frameindices = 'default' - indices for the frames you want plotted. if left 'default' will generate indices as np.arange(0, len(inpdf), len(inpdf)/numframes)
        framestepsize = 'default' - if you don't want to automatically determine the number of frames, you can tell it the number of steps across the index numbers you want to go per frame, and it'll calculate the frame indices as frameindices = np.arange(0, len(inpdf) + framestepsize, framestepsize)
        numframes = 'default' - number of frames to have in the resulting animation. Set like this to avoid making giant gifs on accident.
            By default, if this is left default and frameindices/framestepsize are too, then if the inpdf is shorter than 100 rows it'll set numframes = len(inpdf), and if it's longer than 100 it'll cap it at 100 frames and scale the frame indices accordingly.
        cushion_frame = 0 - number of frames to add on the end as a "cushion" to the loop.
        fig = None - figure, if left None, will generate one with fig = plt.figure(**fig_kwargs)
        ax = None - ax, if left None, will generate one as ax = fig.add_subplot(111)
        fig_kwargs = {} - kwargs passed to the setting up of the plt.figure()
        ax_kwargs = {} - kwargs passed to ax.set()
        pltclose = True, pltshow = False, plttight = False - bools controlling if plt.close(), plt.show() and plt.tight_layout() respectivel are run at the end of this function (not in that order).
        pilwriter_kwargs = {'fps':20} - kwargs for animation.PillowWriter()
        timeit = True - whether or not to print the time elapsed.
        funcanim_kwargs = {'blit':True} - kwargs for the animation.FuncAnimation() in addition to figure, fun, and blit (which are determined within this function).
        anisave_kwargs = {} - kwargs for saving the animation when running ani.save() (where ani is the FuncAnimation).
        return_update_func = False - if True, will just run everything through defining the update function and then return the update function (for use when combining animation functions).
        bonusprints = True - print optional messages
    """
    starttime = time.time()
    # set up figure
    if fig == None:
        fig = plt.figure(**updateandreturndict({'figsize':(5,5)}, fig_kwargs))
    if ax == None:
        ax = fig.add_subplot(111)
    # set up frames and various conditions of numframes, framestepsize, frameindices - these are much less important for this HW compared to the sun animation one, since we don't have 10000-row long tables to attempt to animate
    if numframes == 'default': # let number of frames go to 100 
        if len(inpdf) < 100:
            numframes = len(inpdf)
        else:
            numframes = 100
    if framestepsize != 'default':
        if frameindices == 'default':
            frameindices = np.arange(0, len(inpdf)+framestepsize, framestepsize, dtype = int)
        else:
            frameindices = frameindices
    else:
        if frameindices == 'default':
            frameindices = np.arange(0, int(len(inpdf)), int(len(inpdf)/numframes), dtype = int)
        else:
            frameindices = frameindices
    # set up lines
    orbitline_kwargs = updateandreturndict({'c':'white', 'ls':'--', 'zorder':2, 'alpha':0.72}, orbitline_kwargs)
    cylinder_kwargs = updateandreturndict({'c':'cyan', 'zorder':3}, cylinder_kwargs)
    orbscatter_kwargs = updateandreturndict({'cmap':'plasma', 'zorder':3}, orbscatter_kwargs)
    orbitline, = ax.plot(inpdf[xcol], inpdf[ycol], **orbitline_kwargs)
    cylinderscatter = ax.scatter(0, 0, **cylinder_kwargs)
    orbitscatter = ax.scatter([], [], c= [], **orbscatter_kwargs)
    blankscatter = ax.scatter([], [], alpha = 0, s = 0.1)
    # set up ax
    ax_kwargs = updateandreturndict({'title':'Orbit of space junk', 'xlabel':xcol, 'ylabel':ycol, 'facecolor':'lightgray'}, ax_kwargs)
    # add image bg if needed
    if imagebg != None:
        ax.imshow(plt.imread(imagebg), extent= [ax.get_xlim()[0], ax.get_xlim()[1], ax.get_ylim()[0], ax.get_ylim()[1]])
    ax.set(**ax_kwargs)
    if plttight:
        plt.tight_layout()
    
    def update_internal(framenum, orbitline = orbitline, orbitscatter = orbitscatter, path_info = inpdf, blankscatter = blankscatter):
        if framenum < len(frameindices):
            frame = frameindices[framenum]
            if frame >= len(path_info):
                frame = len(path_info) - 1
            orbitscatter = ax.scatter(inpdf[xcol].iloc[:frame], inpdf[ycol].iloc[:frame],
                                       c = inpdf[colorcol].iloc[:frame], **orbscatter_kwargs) 
            blankscatter.set_label(f"t = {inpdf[colorcol].iloc[frame]:{decimalpts}f}")
            orbitline = orbitline
            ax.legend()
        return(orbitline, orbitscatter)

    ani = animation.FuncAnimation(fig = fig, func = update_internal, frames = len(frameindices)+cushion_frames, **funcanim_kwargs)
    if return_update_func:
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

# for the magic mass!
def magicmass(M, theta, minmass = 1, trigfn = np.sin):
    """What if the mass magically varied as a function of time (or whatever else you put in for theta?) This function returns a value for mass given by:
        abs(M*trigfn(theta) + minmass)
        Where trigfn can be put in as a string that this will translate to a numpy trig function. Warning that some may lead to overflow or other errors, depending on how you've set the initial M arguments. In the spirit of Python and its lack of guardrails, and since this isn't meant to represent anything physical anyway, I'll let you put in things like arcsin that may give RuntimeWarnings for some values of theta. Play around and find out - sin gives a pretty result for M = 10 and minmass = 1, so the default trigfn is 'sin' and the default minmass is 1.
    """
    trigdict = {'sin':np.sin, 'cos':np.cos, 'tan':np.tan, 'arcsin':np.arcsin, 'arccos':np.arccos, 'arctan':np.arctan}
    trigfn = trigdict[trigfn]
    return(abs(M*trigfn(theta) + minmass))

def rk4_magicmass(h, rval, outto, inpfn, magicmassfn = magicmass, minmass =1, trigfn = 'sin',
                       inpfn_kwargs = {'L':2, 'M':10, 'G':1, 't':None}, varyval = 'M'):
    """A slightly modified version of the RK4 solver that allows you to vary the mass as a function of timestep. Technically, I could have also made a version of the f_grav function that actually uses the timestep argument to vary mass. However, I figured I'd rather keep the function that reflects the actual physics entirely physical and let the space-wizard magically varying mass happen outside of that entirely.
    
    Args:
        h - step
        rval - initial vector [x, y, dx/dt, dy/dt]
        outto - how far out in timesteps we want to go (ie, we're going through the array np.arange(0, outto, h))
        inpfn - input function that we are solving
        inpfn_kwargs = {'L':2, 'M':10, 'G':1, 't':None} - input function kwargs
    Special for this version of the RK4:
        magicmassfn = magicmass - function that varies the mass
        minmass = 1 - minimum value for mass
        trigfn = sin - trig function to use in varying the mass
    
    """
    inpfn_kwargs = updateandreturndict({'L':2, 'M':10, 'G':1, 't':None}, inpfn_kwargs)
    arange_t = np.arange(0, outto, h)
    rvals = [rval]
    # save the first value of M from which the next ones will be calculated
    new_M = inpfn_kwargs[varyval]
    for ind, tval in enumerate(arange_t[1:]):
        # get the new mass based on the mass function, and update the inputs to the function being solved accordingly
        new_M = magicmassfn(new_M, tval, minmass, trigfn = trigfn)
        inpfn_kwargs = updateandreturndict(inpfn_kwargs, {varyval:new_M})
        # back to your regularly scheduled rk4
        k1 = h*inpfn(rvals[ind], **inpfn_kwargs)
        k2 = h*inpfn(rvals[ind]+0.5*k1, **inpfn_kwargs)
        k3 = h*inpfn(rvals[ind]+0.5*k2, **inpfn_kwargs)
        k4 = h*inpfn(rvals[ind]+k3, **inpfn_kwargs)
        new_r = rvals[ind] + (k1+2*k2+2*k3+k4)/6
        rvals.append(new_r)
        #pdb.set_trace()
    return(rvals)

if __name__  == '__main__':

    parser = argparse.ArgumentParser(description = "Script for simulating the orbit of pieces of space junk around each other, using numerical solvers for the orbit. Runmode options include: 'plotlyjunk', which will just calculate the orbit of the space junk and make a plotly animation of it; 'mplanim', which will do the same thing but make a matplotlib gif instead; 'magicmass', which will allow you to pick a variable out of cylinder mass, cylinder length, or G and vary it as a function of time (as the hypothetical space wizard magically varies these parameters and those parameters alone) using a given trig function and make a gif of the resulting orbit; or 'explore', which will run all three in sequence.")
    # general script args
    parser.add_argument('--runmode', default = 'explore', type = str, help = f"The mode in which to run this script. Default is 'explore', which will run all parts relating to the homework AND the bonus parts.")
    parser.add_argument('--slowdown', default = 0, type = float, help = f"This script can use sleep(slowdown) to pause the script at certain points and give you more time to read printouts in terminal. Increase slowdown to pause longer.")
    parser.add_argument("--seed", default = 4242, type = int, help = f"The random seed number to be used for rng purposes.")
    # specific hw args - setting up analysis
    parser.add_argument('--solver', default = 'rk4', type = str, help = f"The solver to use. Default rk4 for Runge-Kutta to the 4th order.")
    # set up args for the solver
    parser.add_argument('--hstep', default = 0.01, type = float, help = f"The timestep to use. Default 0.01 (seconds).")
    parser.add_argument('-r', '--rval', default = (1, 0, 0, 1), nargs = 4, type = float, help = f"The vector for the radius and velocity, in x, y, dx/dt, and dy/dt. Default [1, 0, 0, 1]. Enter as four separate values in a row!")
    parser.add_argument('--outto', default = 10, type = float, help = f"How many seconds to run out to, default 10 (seconds).")
    parser.add_argument('--length', default = 2, type = float, help = f"Length of the cylinder, default 2.")
    parser.add_argument('--mass', default = 10, type = float, help = f"Mass of the cylinder, default 10.")
    parser.add_argument('--gravity', default = 1, type = float, help = f"Value we're using for G. NOTE THAT WE ARE DEFAULTING TO G = 1!!!")
    # args for plotting related things
    parser.add_argument('--imagebg', default = 'jsc_auroraaustralis.png', type = str, help = f"Image to use as background for the matplotlib animation. from https://images.nasa.gov/details/S39-23-036")
    parser.add_argument('--numframes', default = 50, type = int, help = f"Number of frames to have in the matplotlib gif.")
    parser.add_argument('--fps', default = 20, type = int, help = f"FPS for the gifs when generated.")
    parser.add_argument('--decimalpts', default = 2, type = int, help = f"Formatting for the number of places to show in the gif labels of time following number:decimalptsf formatting. Default =2, so it'll divide by 10 (2 -> 0.2) in accordance with how that kind of formatting string works.")
    parser.add_argument('--mplgif', default = 'spacejunkorbit.gif', type = str, help = f"Name of file to which to save the matplotlib gif of the space junk orbit.")
    parser.add_argument('--magicgif', default = 'magicmassorbit.gif', type = str, help = f"Name of file to which to save the matplotlib gif of the space junk orbit in the case that the mass of the cylinder is varied with time.")
    parser.add_argument('--overwrite', default = 1, type = int, help = f"Whether or not to allow overwriting of gifs (default is 1 = True).")
    # args for the magic mass!
    parser.add_argument('--masstrig', default = 'sin', type = str, help = f"For the bonus part where you magically vary the mass according to some function M(t) = abs(M*masstrig(t)+minmass), this is the trig function that is used in place of masstrig. Fair warning: some inputs for the trig function here may give overflow errors depending on how you've set the base mass and minimum mass!")
    parser.add_argument('--minmass', default = 1, type = float, help = f"For the bonus part where you magically vary the mass according to some function M(t) = abs(M*masstrig(t)+minmass), this is the minimum mass value so you can avoid going to zero mass and vanishing the cylinder entirely. Fair warning: some inputs for the trig function here may give overflow errors depending on how you've set the base mass and minimum mass!")
    parser.add_argument('--magicanim', default = 'mpl', type = str, help = f"Which method to use to animate the orbit of the ball bearing if the cylinder mass is magically being varied - either mpl for matplotlib or plotly for plotly. I recommend matplotlib, but to each their own!")
    parser.add_argument('--varyval', default = 'M', type = str, help = f"Bonus trick: this controls what variable out of L/G/M you change with the magicmass function, in the case where a space wizard varies properties of the cylinder as a function of time. If you want to vary L (or G, even!) rather than the mass of the cylinder, you can set this to L or G accordingly.")

    args = parser.parse_args()
    print(f"Args used: {args}")

    # general script args
    runmode = args.runmode
    slowdown = args.slowdown
    seed = args.seed
    # specific hw args - define the solver, and 
    solver = args.solver
    if ifanyofthesein(['rk4', 'runge', 'kutta', 'rk'], solver.lower()):
        solver = runge_kutta_vectorized
    else:
        print("I don't have any other solver options here, so we're using RK4 for now. Sometime I might add more!")
        solver = runge_kutta_vectorized
    # define the timestep arguments, rvector, et cetera
    h = args.hstep
    rval = [float(val) for val in args.rval]
    outto = args.outto
    L = args.length
    M = args.mass
    G = args.gravity
    # define tihngs relevant to the gifs - background, gif names, etc
    imagebg = args.imagebg
    if imagebg == 'None':
        imagebg = None
    numframes = args.numframes
    fps = args.fps
    decimalpts = args.decimalpts / 10
    mplgif = args.mplgif
    magicgif = args.magicgif
    overwrite = boolify(args.overwrite)
    #bonus part args
    masstrig = args.masstrig
    minmass = args.minmass
    magicanim = args.magicanim
    varyval = args.varyval
    if varyval not in ['L', 'l', 'M', 'm', 'G', 'g', 'len', 'length', 'mass']:
        raise ValueError(f"Invalid argument for varyval {varyval}; it should be one of L, M, or G.")
    
    # run options dictionary
    runopts_dict = {'explore':ifanyofthesein(['all', 'explore', 'explor', 'ex', 'run all'], runmode.lower()), 
                    'plotlyjunk':ifanyofthesein(['plotly', 'pj', 'plotly junk', 'plotlyjunk'], runmode.lower()), 
                    'mplanim':ifanyofthesein(['mplanim', 'mpl orbit','mpl junk', 'mpl anim'], runmode.lower()), 
                    'magicmass':ifanyofthesein(['magicmass', 'magic', 'mass', 'varymass', 'magic mass'], runmode.lower())
                   }
    if runopts_dict['explore']:
        for ky in runopts_dict:
            runopts_dict[ky] = True
    # putting this here since the basic run will be used by pretty much all subsequent parts.
    trans_solver_result = np.transpose(solver(h, rval, outto, inpfn = f_junk, inpfn_kwargs = {'L':L, 'M':M, 'G':G}))
    orbit_dict = {'x':trans_solver_result[0], 'y':trans_solver_result[1], 
                     'dxdt':trans_solver_result[2], 'dydt':trans_solver_result[3], 
                     't':np.arange(0, outto, h), 'frame':np.arange(0, len(trans_solver_result[0]))}
    orbit_df = pd.DataFrame(orbit_dict)

    # run it!
    if runopts_dict['plotlyjunk']:
        print(f"With plotly, plot and then animate the orbit of a piece of space junk (ball bearing) orbiting another piece of space junk (cylinder) from t = 0 to t = {outto}, with initial x, y, dx/dt, dy/dt of {rval}.")
        print("Plot:")
        basic_plotly_orbit(orbit_df, 'x', 'y', 't')
        print("Animate:")
        basic_plotly_anim(orbit_df, 'x', 'y')
        print(f"Note that it does indeed have this funky precessing orbit!")
        sleep(slowdown)
    if runopts_dict['mplanim']:
        print(f"Animate, with matplotlib, the orbit of a piece of space junk (ball bearing) orbiting another piece of space junk (cylinder) from t = 0 to t = {outto}, with initial x, y, dx/dt, dy/dt of {rval}.")
        print(f"Note that, if you are using imagebg = jsc_auroraaustralis , the background image isn't itself a gif - something about how it gets rendered in MPL makes it shimmer though!")
        print(f"(this particular picture was taken by astronauts on Discovery on film, which is very cool!! It comes from here: https://images.nasa.gov/details/S39-23-036)")
        space_junk_anim(orbit_df, savename = mplgif, numframes = numframes, imagebg = imagebg, overwrite = overwrite, pilwriter_kwargs={'fps':fps}, decimalpts = decimalpts)
        sleep(slowdown)
    if runopts_dict['magicmass']:
        print(f"A space wizard has appeared, and is magically making the mass of the cylinder vary as a function of time! (Or, depending on your input for varyval, {varyval} is varying!) Using the masstrig argument {masstrig} to determine the trig function, and the minmass argument {minmass} to set its minimum mass, the {varyval} of the cylinder will now vary according to:")
        print(f"{varyval}(t) = abs({varyval}*{masstrig}(t)) + {minmass}")
        print(f"First, we'll have to rerun the solver.")
        starttime = time.time()
        mmass_trans_solver_result = np.transpose(rk4_magicmass(h, rval, outto, inpfn = f_junk, inpfn_kwargs = {'L':L, 'M':M, 'G':G}, 
                                                              magicmassfn = magicmass, trigfn = masstrig, minmass = minmass, varyval = varyval))
        mmass_orbit_dict = {'x':mmass_trans_solver_result[0], 'y':mmass_trans_solver_result[1], 
                         'dxdt':mmass_trans_solver_result[2], 'dydt':mmass_trans_solver_result[3], 
                         't':np.arange(0, outto, h), 'frame':np.arange(0, len(mmass_trans_solver_result[0]))}
        mmass_orbit_df = pd.DataFrame(mmass_orbit_dict)
        print(f"Took {time.time() - starttime} seconds to run the solver!!")
        print(f"Let's see how strange the results of this space wizard's experiment look. Animate, with {magicanim}, the orbit of the space junk:")
        if ifanyofthesein(['mpl', 'matplotlib'], magicanim.lower()):
            space_junk_anim(mmass_orbit_df, savename = magicgif, numframes = numframes, imagebg = imagebg, overwrite = overwrite,
                           ax_kwargs = {'title':f"Orbit of space junk ( {varyval}(t) = |{varyval}*{masstrig}(t)| + {minmass} )"}, 
                           pilwriter_kwargs={'fps':fps}, decimalpts = decimalpts)
        elif ifanyofthesein(['plotly', 'plt'], magicanim.lower()):
            basic_plotly_anim(mmass_orbit_df, 'x', 'y')
        else:
            print("WARNING: I don't understand your input for magicanim, so i'm defaulting to the faster (plotly) method.")
            basic_plotly_anim(mmass_orbit_df, 'x', 'y')
        sleep(slowdown)
    print("Hint: if the gifs are too big, turn off the image background!")