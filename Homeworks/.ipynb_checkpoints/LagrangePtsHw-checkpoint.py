"""Computational Astronomy Homework, week of September 25th

 Write a program that uses either Newton’s method or the secant method to solve for the distance r from the Earth to the L1 point. Compute a solution accurate to at least four significant figures. 
 (Note, Newton's method is so accurate and fast at getting to the right answer that you have to be pretty far off with the initial guess for r, or at a very high tolerance, to get it to take more than a couple iterations to find the solution when given the default constants!)
- Miranda McCarthy, September 25th 2025
"""
# import libraries
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from time import sleep
import argparse
import sys
#import pdb # debugging package

# define the functions we'll be using
# basic ones I use in most scripts
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

# functions for this HW
# get the r of the L1
def L1_function_r(r, G = const.G.value, M_earth = const.M_earth.value, m_moon = 7.348e22, 
                 omega = 2.662e-6, R = 3.844e8):
    """The function that we'll plug into Newton's method to solve for r, the radius of L1!
        Essentially: G*M_earth/r**2 - G*m_moon/(R-r)**2 - r*(omega**2) (=0 if we find the solution of r!)

        Args:
            r - radius of L1
            G = const.G.value - gravitational constant
            M_earth = const.M_earth.value - mass of Earth
            m_moon = 7.348e22 - the mass of the moon
            omega = 2.662e-6 - the angular velocity of the Moon and the satellite.
            R = 3.844e8 - the distance between Earth and the Moon.
        returns:
            G*M_earth/r**2 - G*m_moon/(R-r)**2 - r*(omega**2)
    """
    p1 = (G*M_earth)/(r**2)
    p2 = -(G*m_moon)/((R-r)**2)
    p3 = -(omega**2)*r
    return(p1 + p2 + p3)
# derivative of above function
def deriv_L1_function(r, G = const.G.value, M_earth = const.M_earth.value, m_moon = 7.348e22, 
                 omega = 2.662e-6, R = 3.844e8):
    """The derivative of the L1 function. 
    
        Args:
            r - radius of L1
            G = const.G.value - gravitational constant
            M_earth = const.M_earth.value - mass of Earth
            m_moon = 7.348e22 - the mass of the moon
            omega = 2.662e-6 - the angular velocity of the Moon and the satellite.
            R = 3.844e8 - the distance between Earth and the Moon.
        returns:
            G*M_earth/r**2 - G*m_moon/(R-r)**2 - r*(omega**2)
    """
    p1 = (-2*G*M_earth)/(r**3)
    p2 = -(2*G*m_moon)/(R-r)**3
    p3 = -(omega**2)
    return(p1 + p2 + p3)
# newtonian method
def newt_method(inpfn, deriv_inpfn, initial_guess, nitersmax = 100, tolerance= 1e-8,
                bonusprints = True, return_if_fails = None, printprecision = 0.3, printmode = 'e',
                inpfn_kwargs = {}, deriv_kwargs={}):
    """Newton's method for numerically solving a function!
        Args:
            inpfn - the function to solve
            deriv_inpfn - the derivative of inpfn
            initial_guess - the initial guess for the solution
            nitersmax = 100 - the maximum number of iterations to run.
            tolerance = 1e-8 - the precision - ie how close the current value and current guess of the solution are, and if they're below the tolerance, we consider the solution found.
            bonusprints = True - controls optional print statements.
            return_if_fails = None - value to return if a solution is not found.
            printprecision = 0.3 - how far out to round numbers when printed (following f'{val:{printprecision}{printmode}}').
            printmode = 'e' - whether to print numbers in the format 'e' (scientific notation) or 'f' (decimal).
            inpfn_kwargs = {} - kwargs for the input fn.
            deriv_kwargs = {} - kwargs for the derivative of input fn. (In most cases, should == inpfn_kwargs).
    
    """
    niters = 0
    current_guess = initial_guess
    if tolerance < sys.float_info.epsilon:
        print(f"Hey, your tolerance of {tolerance} is below machine precision {sys.float_info.epsilon}! Be aware that it's going to be effectively machine precision!")
    while niters < nitersmax:
        currentval = current_guess - inpfn(current_guess, **inpfn_kwargs)/deriv_inpfn(current_guess, **deriv_kwargs)
        if abs(currentval - current_guess) <= tolerance:
            printif(f'Solution found: x = {currentval:{printprecision}{printmode}} after {niters:{printprecision}{printmode}} iterations', bonusprints)
            printif(f"Tolerance: {tolerance:{printprecision}{printmode}}; x - fn(x) = {(currentval-current_guess):{printprecision}{printmode}}", bonusprints)
            return(currentval)
        else:
            current_guess = currentval
            niters+=1
    printif(f'Failed: no solution where abs(x-fn(x)) <= {tolerance:{printprecision}{printmode}} after {niters:{printprecision}{printmode}} iterations.', bonusprints)
    printif(f"x-fn(x) = {currentval:{printprecision}{printmode}} - {current_guess:{printprecision}{printmode}} = {(currentval - current_guess):{printprecision}{printmode}}", bonusprints)
    return(return_if_fails)

def polar_plot_L1( initial_guess, G = const.G.value, M_earth = const.M_earth.value, m_moon = 7.348e22, R = 3.844e8, omega = 2.663e-6,nitersmax = 1e6, tolerance = 1e-8, printprecision = 0.4, printmode = 'e', L1_fn = L1_function_r, deriv_L1_fn = deriv_L1_function,L1_fn_kwargs = {}, deriv_L1_fn_kwargs = {},r_line_kwargs = {'alpha':0.5, 'ls':'--', 'c':'magenta'},R_line_kwargs = {'alpha':0.5, 'ls':'--', 'c':'blue'},newt_method_kwargs = {'bonusprints':False, 'return_if_fails':None}, ax_set_kwargs = {'rorigin':0},passfig = None, passax = None, pltshow = True):
    """Makes a polar plot modified to basically be a diagram with Earth in the center, the moon near the edge, and L1 between them.
    Args:
        initial_guess - initial guess for the solution of r_L1.
        G = const.G.value - gravitational constant
        M_earth = const.M_earth.value - mass of Earth
        m_moon = 7.348e22 - the mass of the moon
        omega = 2.662e-6 - the angular velocity of the Moon and the satellite.
        R = 3.844e8 - the distance between Earth and the Moon.
        nitersmax = 100 - the maximum number of iterations to run.
        tolerance = 1e-8 - the precision - ie how close the current value and current guess of the solution are, and if they're below the tolerance, we consider the solution found.
        printprecision = 0.3 - how far out to round numbers when printed (following f'{val:{printprecision}{printmode}}').
        printmode = 'e' - whether to print numbers in the format 'e' (scientific notation) or 'f' (decimal).
        L1_fn = L1_function_r - function for L1 (should be 0 if the solution is found).
        deriv_L1_fn = deriv_L1_function - derivative of L1_fn.
        L1_fn_kwargs = {} - kwargs for L1_fn.
        deriv_L1_fn_kwargs = {} - kwargs for deriv_L1_fn.
        r_line_kwargs = {'alpha':0.5, 'ls':'--', 'c':'magenta'} - kwargs for the line marking r.
        R_line_kwargs = {'alpha':0.5, 'ls':'--', 'c':'blue'} - kwargs for the line marking R.
        newt_method_kwargs = {'bonusprints':False, 'return_if_fails':None} - kwargs for newt_method
        ax_set_kwargs = {'rorigin':0} - kwargs to pass ax.set()
        passfig = None - figure to plot on
        passax = None - ax to plot on
        pltshow = True - whether to run plt.show() at the end.
    """

    # handle figure and ax
    if passfig == None:
        fig = plt.figure(layout = 'constrained')
    else:
        fig = passfig
    if passax == None:
        ax = fig.add_subplot(projection = 'polar')
    else:
        ax = passax

    # handle dictionaries
    L1_fn_kwargs = updateandreturndict({'G':G, 'M_earth':M_earth, 'm_moon':m_moon, 'R':R, 'omega':omega}, L1_fn_kwargs)
    deriv_L1_fn_kwargs = updateandreturndict({'G':G, 'M_earth':M_earth, 'm_moon':m_moon, 'R':R, 'omega':omega}, deriv_L1_fn_kwargs)
    newt_method_kwargs = updateandreturndict({'nitersmax':nitersmax, 'tolerance':tolerance, 'bonusprints':True, 'return_if_fails':None,
                                              'printprecision':printprecision, 'printmode':printmode,
                                              'inpfn_kwargs':L1_fn_kwargs, 'deriv_kwargs':deriv_L1_fn_kwargs}, 
                                            newt_method_kwargs)

    # calculate the solution for the r value
    rsoln = newt_method(inpfn=L1_fn, deriv_inpfn=deriv_L1_fn, initial_guess= initial_guess, **newt_method_kwargs)
    # handle dictionaries for the plotting:
    r_line_kwargs = updateandreturndict({'alpha':0.5, 'ls':'--', 'c':'magenta', 'label':f"r = {rsoln:{printprecision}{printmode}}"}, r_line_kwargs)
    R_line_kwargs = updateandreturndict({'alpha':0.5, 'ls':'--', 'c':'blue', 'label':f"R = {R:{printprecision}{printmode}}"}, R_line_kwargs)
   
    # set r_max to the rounded-up version of the distance to the far body, plus 2
    ax_set_kwargs = updateandreturndict({'rmax':np.round(R*1.5), 'rorigin':0, 'rticks':[], 
                                        'xticklabels':[]}, ax_set_kwargs)
    ax.set(**ax_set_kwargs)
       
    # draw a line through the r value and R value
    ax.axhline(rsoln, **r_line_kwargs)
    ax.axhline(R, **R_line_kwargs)
    # draw the bodies
    ax.scatter(0, 0, zorder =2)
    ax.scatter(0, rsoln, zorder =2, label = 'L1') 
    ax.scatter(0, R, zorder =2)

    ax.grid(False)
    plt.legend(loc= 'upper left')

def r_fnof_val(init_r_guess, partovary_name, partovary_vals, 
               og_r_fn = L1_function_r, deriv_r_fn = deriv_L1_function, 
               nitersmax = 1e6, bonusprints = False, return_if_fails = np.nan, tolerance = 1e-8,
               newt_method_kwargs = {}):
    """Calculates the solution of r with a variety of values for a given constant.
    Args:
        init_r_guess - initial guess for r.
        partovary_name - which parameter (constant) to vary.
        partovary_vals - the values of that parameter to calculate a solution for r with.
        og_r_fn = L1_function_r - the function that we want to use to get the solution for r.
        deriv_r_fn = deriv_L1_function - derivative of og_r_fn
        nitersmax = 1e6 - maximum iterations for newt_method
        bonusprints = False - verbosity for newt_method
        return_if_fails = np.nan - what to return if no solution found for r
        tolerance = 1e-8 - the precision - ie how close the current value and current guess of the solution are, and if they're below the tolerance, we consider the solution found.
        newt_method_kwargs = {} - the kwargs for the newt_method function.
    
    """

    newt_method_kwargs = updateandreturndict({'nitersmax':nitersmax, 'bonusprints':bonusprints, 'return_if_fails':return_if_fails, 'tolerance':tolerance}, 
                                            newt_method_kwargs)
    
    outpt = []
    for parval in partovary_vals:
        rsoln = newt_method(L1_function_r, deriv_L1_function, init_r_guess, 
                    inpfn_kwargs={partovary_name:parval}, deriv_kwargs={partovary_name:parval},
                            **newt_method_kwargs)
        outpt.append(rsoln)
    return(np.array(outpt))

def plot_r_fnof_val(init_r_guess, partovary_name, partovary_init_val, partovary_percent_andsteprat = (0.5, 1.5, 1/100),
                    passfig = None, passax = None, pltshow = False, plttight = True, uselabel = 'DEFAULT', printprecision = 0.2,
                    printmode = 'e',
                    pltfigure_kwargs = {}, figaddsubplot_kwargs= {}, r_fnof_val_kwargs = {}, plot_kwargs = {}):
    """plot_r_fnof_val plots r_fnof_val (see documentation for that function).
    Args:
        init_r_guess - initial guess for r
        partovary_name - which constant to vary
        partovary_percent_andsteprat  = (0.5, 1.5, 0.01)- Percentages, and the step ratio, by which to vary the constants in pars_to_vary when exploring how the solution for r changes with the constants. Default is (0.5, 1.5, 0.01), which will therefore make an array of constant values to plug into r_fnof_val (the function that calculates r as it changes with the constant values) like np.arange(0.5*constant, 1*constant, (0.01)*constant) . SO, makes an array from 50 percent of the constant to 150 percent of the constant with 100 points in between them. Note that it can take a few seconds to run with this default, so consider increasing the step size (ie, to 0.1).
        passfig = None, passax = None - fig and ax to plot to, if not generating one inside the function.
        pltshow = False, plttight = True - whether to run plt.show() and plt.tight_layout() at the end, respectively.
        uselabel = 'DEFAULT'  - label for the line plotted; defaults to showing you the parameter varied and its original value.
        printprecision = 0.2 - how far out to round numbers when printed (following f'{val:{printprecision}{printmode}}').
        printmode = 'e' - whether to print numbers in the format 'e' (scientific notation) or 'f' (decimal).
        pltfigure_kwargs = {} - kwargs to pass plt.figure
        figaddsubplot_kwargs = {} - kwargs for fig.add_subplot
        r_fnof_val_kwargs = {} - kwargs for r_fnof_val
        plot_kwargs = {} - kwargs for ax.plot
        
    
    """
    
    fig = passfig
    if passfig == None:
        fig = plt.figure(**pltfigure_kwargs)
    ax = passax
    if passax == None:
        ax = fig.add_subplot(**figaddsubplot_kwargs)

    percentrange = np.arange(partovary_percent_andsteprat[0], partovary_percent_andsteprat[1], partovary_percent_andsteprat[2])
    # now, translate that into the relevant values
    partovary_vals_arange = np.arange(partovary_percent_andsteprat[0], partovary_percent_andsteprat[1], partovary_percent_andsteprat[2])*partovary_init_val
    r_fnof_result = r_fnof_val(init_r_guess, partovary_name=partovary_name, partovary_vals=partovary_vals_arange, **r_fnof_val_kwargs)
    if uselabel == 'DEFAULT':
        uselabel = f"{partovary_name} (og value: {partovary_init_val:{printprecision}{printmode}})"
    else:
        uselabel = uselabel 

    plot_kwargs = updateandreturndict({'label':uselabel}, plot_kwargs)
    ax.plot(percentrange, r_fnof_result, **plot_kwargs)
    if plttight:
        plt.tight_layout()
    if pltshow:
        plt.show()
        
            
# the script part
if __name__ == '__main__':

    # set up arguments with argparse
    parser = argparse.ArgumentParser(description = "Script for calculating the solution for the radius of the first Lagrangian point between the Earth and Moon. Also has extra options for other interesting things that can be done with the function for r_L1, as controlled by the runmode argument!")
    # args:
    # general run settings
    parser.add_argument('--runmode', default = 'explore', type = str, help = f"Controls what parts of this script are run. Enter your desired commands here all in one string, separated by spaces. Options include 'test_r' (plots the equation for L1 versus its derivative, so you have a sense of where to place the initial guess), 'calc_r' (does the basic calculation of r using Newton's method), 'polar_plot' (will make a polar plot that has been modified to be essentially a diagram of the Earth and Moon (or whatever central body and orbiting body) versus where L1 is between them), 'vary_constants' (will make a plot comparing solutions for r as a function of one or more of the constants (R, G, M_earth, m_moon, or omega) included in the equation for L1, varying that constant from some minimum to maximum percentage) and 'explore' (will run all of the above options in sequence - this is the default option). Consider also increasing the value of --slowdown from 0, which will tell Python to wait for a few seconds between making plots and switching from part to part of the script so you have time to read the printed values.")
    parser.add_argument('--slowdown', default = 0, type = float, help=f"This script can use sleep(slowdown) to pause the script at certain points if desired to give you more time to read the printouts in terminal. Increase slowdown to pause longer. (Since it usually takes a couple seconds for vary_constants to run, no slowdown is included between the explanation of vary_constants and its plot actually being made, if it is run).")
    parser.add_argument('--explainbonus', default = 'True', type = str, help = f"Controls the printing of the explanation of different run modes that (by default) will print at the end of the script if not all the optional modes are run.")
    # args for the default run (calc_r)
    parser.add_argument('-r', default = 1e5, type = float, help=f"Initial guess for the value of r.")
    parser.add_argument('-G', default = const.G.value, type=float, help = f"Gravitational constant, defaults to {const.G.value}.")
    parser.add_argument('-R', default = 3.844e8, type = float, help = f"Distance between Earth and the Moon. Defaults to 3.844e8")
    parser.add_argument('--M_earth', default = const.M_earth.value, type=float, help = f"Mass of Earth, defaults to {const.M_earth.value}.")
    parser.add_argument('--m_moon', default = 7.348e22, type=float, help = f"Mass of the moon. Default value 7.348e22")
    parser.add_argument('--omega', default = 2.662e-6, type=float, help = f"Angular velocity of the moon (and the satellite we are placing at r). Default value 2.662e-6")
    parser.add_argument('--nitersmax', default = 1e6, type=float, help = f"Maximum number of iterations to go through when running Newton's method.")
    parser.add_argument('--tolerance', default = 1e-8, type=float, help = f"The precision for Newton's method.")
    parser.add_argument('--printprecision', default = 0.5, type=float, help = f"Precision for print outputs of the Newton's method function.")
    parser.add_argument('--printmode', default = 'e', type=str, help = f"Mode to print outputs of the Newton's method function ('e' for scientific notation, 'f' for decimal).")
    # plot r as a fn of a given constant
    parser.add_argument('--pars_to_vary', nargs = '+',  default = ('R','G','M_earth','m_moon','omega'), type = str, help = f"Which parameters to vary when plotting how the solution for r changes when varying one of the constants involved in the equation for L1. Options (and the default argument) are: ('R','G','M_earth','m_moon','omega')")
    parser.add_argument('--percent_step_rat', nargs = 3, default = (0.5, 1.5, 0.01), type = float, help = f"Percentages, and the step ratio, by which to vary the constants in pars_to_vary when exploring how the solution for r changes with the constants. Default is (0.5, 1.5, 0.01), which will therefore make an array of constant values to plug into r_fnof_val (the function that calculates r as it changes with the constant values) like np.arange(0.5*constant, 1*constant, (0.01)*constant) . SO, makes an array from 50 percent of the constant to 150 percent of the constant with 100 points in between them. Note that it can take a few seconds to run with this default, so consider increasing the step size (ie, to 0.1).")
    #debug
    parser.add_argument('--printargs', default = False, type = bool, help = "If True, will print args before running the script. Default False.")

    args = parser.parse_args()
    runmode = args.runmode
    slowdown = args.slowdown
    explainbonus = args.explainbonus # i don't know why this one made me make it a string, but the printargs was fine as is???
    r = args.r
    G = args.G
    R = args.R
    M_earth = args.M_earth
    m_moon = args.m_moon
    omega = args.omega
    nitersmax = args.nitersmax
    tolerance = args.tolerance
    printprecision = args.printprecision
    printmode = args.printmode
    # parttovary
    pars_to_vary = [x for x in args.pars_to_vary]
    percent_step_rat = tuple([float(x) for x in args.percent_step_rat])
    #debug
    if args.printargs:
        print(args)
    # warn about r
    if r > 1e13:
        print(f"HEADS UP: your input guess for r {r:0.4e} is a: getting big enough that you might have trouble finding a solution, and may get divide-by-zero warnings if running with 'vary_constants' in the runmode, and b: about two orders of magnitude greater than the distance between Earth and the Sun.")
    if r > R:
        print(f"HEADS UP: your guess for the r {r:0.4e} is greater than the value you've given for the distance between Earth and the Moon (R = {R:0.4e}).")

    # combine args that are used together as kwargs a lot into one dictionary
    l1_kwargs = {'G':G, 'R':R, 'M_earth':M_earth, 'm_moon':m_moon, 'omega':omega}
        
    show_r_plot = False    
    if ifanyofthesein(['demo', 'test_r', 'show_r_plot'], runmode.lower()):
        show_r_plot=True
    calc_r_default = False
    if ifanyofthesein(['calc_r', 'get r', 'calc r', 'calc_r_default'], runmode.lower()):
        calc_r_default = True
    polar_plot = False
    if ifanyofthesein(['polar', 'plot polar', 'polar_plot', 'polarplot'], runmode.lower()):
        polar_plot = True
    vary_constants = False
    if ifanyofthesein(['vary_constants', 'vary constants', 'plot vary', 'cvary', 'plot_vary'], runmode.lower()):
        vary_constants = True
    everything = False
    if ifanyofthesein(['explore', 'everything', 'all'], runmode.lower()):
        everything = True
        show_r_plot = True
        calc_r_default = True
        polar_plot = True
        vary_constants = True

    if show_r_plot:
        print("Before we go through the rest of this, let's look at a plot of the actual function for L1 that we want to get the root of and its derivative to get a sense of what r value to use as the initial guess.")
        sleep(slowdown)
        fig = plt.figure()
        ax = fig.add_subplot()
        xrange = np.arange(1e2, 1e8, 50)
        ax.plot(xrange, L1_function_r(xrange), label = 'fn')
        ax.plot(xrange, deriv_L1_function(xrange), label = 'deriv fn')#, ls = '--')plt.legend()
        ax.set(xscale = 'log')#, yscale = 'log')
        ax.axvline(3.844e8, c = 'red', ls = '--', label='Earth-Moon distance')
        plt.legend()
        plt.show()


    if calc_r_default:
        print(f"Calculating r_L1, the radius of the first Lagrangian point between the Earth and the Moon with an initial guess of {r}.\n")
        sleep(slowdown)
        rsoln = newt_method(L1_function_r, deriv_L1_function, initial_guess= r, nitersmax = nitersmax, tolerance = tolerance, printprecision = printprecision, printmode = printmode, inpfn_kwargs = l1_kwargs, deriv_kwargs = l1_kwargs)
        if rsoln != None:
            print("\nInterestingly, though this is accurate (as far as Python is concerned) to within {tolerance}, the Wikipedia page for Lagrange points gives the location as 3.2639e8 m. So, let's compare what Python thinks of that versus our answer by comparing how close we get to zero when plugging them into the function L1_function_r:")
            print(f"Our calculated L1 of {rsoln:{printprecision}{printmode}}: {L1_function_r(rsoln)}")
            print(f"Wikipedia (3.2639e8): {L1_function_r(3.2639e8)}")
            print(f"You should see why this script prefers our solution - our calculated L1 radius gives an answer closer to zero when plugged into the equation we're trying to get to zero, 0 = GM/ r^2 - Gm/(R-r)^2 - r*omega^2 .\n\n")

        sleep(slowdown)
        
    if polar_plot:
        print(f"Make a polar plot that plots the Earth, Moon and L1 relative to each other.\n\n")
        sleep(slowdown)
        polar_plot_L1(initial_guess = r, nitersmax = nitersmax, tolerance = tolerance, printprecision = printprecision, printmode = printmode, **l1_kwargs)
        plt.show()

        sleep(slowdown)

    if vary_constants:
        print(f"Plot r as a function of various constants - IE, pass a list of constants and a range of percentages by which you wish to vary them, and this will plot the solution the Newton's method gets for r as a function of the percentage a constant is being varied by. (For an example, just don't change any arguments and let this run as default.) Where the line breaks, that's where no solution was found within the given tolerance / max iterations.\n This may take a few seconds!! If you want it to go faster, increase the ratio that is used to determine the step size of the arange of percentages this function steps through (the last argument in percent_step_rat, currently {percent_step_rat[-1]}).\n\n")

        if r > 5e13:
            print(f"HEADS UP: Your input value for r is getting large enough that you may start to encounter RuntimeWarnings and not get valid solutions, especially when varying omega. Also, remember that 1 AU is ~1.5e11 m. ")
        fig = plt.figure()
        ax = fig.add_subplot()

        for parval in pars_to_vary:
            plot_r_fnof_val(init_r_guess = r, partovary_name = parval, partovary_init_val = l1_kwargs[parval], partovary_percent_andsteprat = percent_step_rat, passfig = fig, passax = ax)
        
        plt.legend()
        plt.title(r'Solutions for r$_{L1}$ as a function of various constants')
        plt.xlabel("Percentage of constant's original value")
        plt.ylabel('r')
        plt.tight_layout()
        plt.show()
    
    if not everything:
        if explainbonus == 'True':
            print("If you haven't tried, explore the other modes of this script by entering runmode = explore! Options include:")
            print(f"'calc_r' - will just do the core part of this homework, calculating the radius r of the Lagrange point and comparing our answer to the 'known' one.\n")
            print(f"'polar_plot' - will make a polar plot that has been modified to be essentially a diagram of the Earth and Moon (or whatever central body and orbiting body) versus where L1 is between them.")
            print(f"'vary_constants' - will make a plot comparing solutions for r as a function of one or more of the constants (R, G, M_earth, m_moon, or omega) included in the equation for L1, varying that constant from some minimum to maximum percentage.")
        else:
            print(f'Done! Thank you for running!')