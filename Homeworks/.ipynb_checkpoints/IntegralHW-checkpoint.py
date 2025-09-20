"""Computational Astronomy Homework, week of September 18th

Write a program to calculate E(x) = integral 0->x e**(-t**2)dt for x from 0 to 3 in steps of 0.1. Choose your method of integration and number of slices.
When you're convinced it's working, make a graph of E(x) as a function of x.

...I went a little bit a lot overboard doing the extra bits on this particular one, so there's a number of extra parts. Run this with -runmode "explore" to see all of them. Thanks!

- Miranda McCarthy, September 19th 2024
"""

# imports
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# define the functions we'll be using

# the actual equation to test for the HW
def e_to_minus_tsq(t):
    """ Simple function returning e**(-t**2)."""
    return(np.e**(-t**2))
# a basic polynomial from prior exercises
def basic_polynomial(x, xpow = 4, a =1, b = -2, c = 1):
    """ A basic polynomical function of the form:
            a*( x**(xpower) ) + b*x + c
        Defaults to a simple polynomial:
            x**4 -2*x + 1
        Args:
           x - input value to the function
        Kwargs:
           xpow - power to which the a*(x**xpow) term is raised (default 4)
           a - coefficient of the a*(x**xpow) term (default 1)
           b - coefficient of the b*x term (default -2)
           c - constant at the end (default 1)
    """
    return(a*(x**xpow) + b*x + c)
# a variety of integration functions
def trap_integral(inpfn, ll, ul, nslices):
    """Trapezoidal integration function. 
        Args: 
            inpfn - the function to integrate.
            ll - the lower limit of the integration.
            ul - the upper limit of the integration.
            nslices - the number of slices to use in the in the integration."""
    delx = (ul-ll)/nslices
    f_ll_term = 0.5 * inpfn(ll)
    f_ul_term = 0.5 * inpfn(ul)
    f_sum_term = np.sum(inpfn( ll + delx*(np.arange(0, nslices)) ))
    return(delx*(f_ll_term+f_ul_term+f_sum_term))

def simpsons_rule(inpfn, ll, ul, nslices):
    """Simpson's rule integration function. 
        Args: 
            inpfn - the function to integrate.
            ll - the lower limit of the integration.
            ul - the upper limit of the integration.
            nslices - the number of slices to use in the in the integration."""
    delx = (ul-ll)/nslices
    central_term = (inpfn(ll)+inpfn(ul))
    a_and_b_term = inpfn(ll)+inpfn(ul)
    odds_term = 4*np.sum(  inpfn(  (ll + ((2*np.arange(1, nslices//2)) - 1)*delx) )  )
    evens_term = 2*np.sum(  inpfn(  (ll + (2*np.arange(1, ( (nslices//2)-1 ))*delx ))  ) )
    return((delx/3)*(central_term+a_and_b_term+odds_term+evens_term))


def integral_over_x(x, lowerlim = 0, stepsize = 0.1, nslices = 1000, inpfn = e_to_minus_tsq, integ_fn = simpsons_rule, special_hwpartab_case = False):
    """Function that returns the integral of another function, as in:
            E(x) = integral(f(x))
        for values in np.arange(0, x, stepsize) with some stepsize stepsize. IF special_hwparta_case = True, it'll actually integrate from 0-3 (see the note in this documentation for special_hwparta_case).
       Parameters:
           x - input value x.
       Optional parameters:
           lowerlim = 0 - lower limit from which to integrate up to x.
           stepsize = 0.1 - stepsize to generate the range of x values to get the integral of the input function for.
           nslices = 1000 - number of slices to break up the integration into. (CHECK)
           inpfn = e_to_minus_tsq - the input function that is integrated over; defaults to e**(-t**2)
           integ_fn = simpsons_rule - the function for integration that is used. defaults to simpsons_rule. Any function passed to this must take arguments in the pattern (input fn, lower limit, upper limit, number of steps)!
           special_hwpartab_case = False - for the homework, it says to integrate with x from 0 to 3 and steps of 0.1. But np.arange(0, 3, 0.1) will max out at 2.9, and not end with the integral at x=3. I've spent way longer than expected debugging the rest of this and don't want to make a workaround for np.arange being open at the end that works in all cases but I know np.arange(0, x+stepsize, stepsize) works in this particular case. So, I'm adding an option to this function that will run that fix IF and ONLY IF the script knows you're trying to run this as part of the "run_a" option. If you're reading this because you've imported the script as a function somewhere else, ignore this.
        Returns: an array with the results of the integral at the x values in np.arange(0, x, stepsize).
    """
    if x == 0:
        raise ValueError('You cannot integrate this from 0 to 0 with this function! Please enter a non-zero x.')
    # remember to add stepsize to the x so because np.arange is exclusive (ie running the below line with just np.arange(0, 3, 1) would return an array ending with the integral for 0 -> 2.9)
    if special_hwpartab_case == True:
        print('APPLYING ADJUSTMENT TO NP.ARANGE SO IT GOES LIKE NP.ARANGE(0, x+stepsize, stepsize)')
        return(np.array([integ_fn(inpfn, lowerlim, x, nslices) for x in np.arange(lowerlim, x+stepsize, stepsize)]))
    else:
        return(np.array([integ_fn(inpfn, lowerlim, x, nslices) for x in np.arange(lowerlim, x, stepsize)]))

# printif
def printif(thingtoprint, printit = True):
    """simple space saving fn - conditional print that only prints the first argument given if the second argument == True.
       I use this for debug messages and as an easy way to set verbosity in long functions!
       Args:
           thingtoprint - the object to actually print
           printit = True - if True, print thingtoprint, if False, don't print it. Default True so it doesn't throw an error when nothing is given as the second argument."""
    if printit:
        print(thingtoprint)

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

# update and return dict
def updateandreturndict(ogdict, updatewith):
    """A really, really simple function that basically runs dict.update() without returning None, but rather the updated dictionary. made this to make custom kwarg variables dictionaries for nested functions less annoying, ages ago, and now i use it a lot. might be a better alternative out there?
    Runs:
        ogdict.update(updatewith)
        return(ogdict)
    """
    ogdict.update(updatewith)
    return(ogdict)
    
# test convergence
def test_to_convergence(inpfn, integ_fn, ll, ul,  
                       nslices=1, nstep=1, nslicesmax=1000, 
                       tolerance = 1e-8, 
                       bonusprints = True, unconvergedval = None, printtime = True, 
                       timeprecision = 0.4):
    """Test how many slices it takes to get an integral to converge on an answer. 
    
    Args:
        inpfn - the function you're integrating over.
        integ_fn - the function you're using to integrate.
        ll - the lower limit of integration.
        ul - the upper limit of integration.
        nslices = 1 - the number of slices to start with. (I know 1 is very much too few!)
        nstep = 1 - the amount to increase the nslices by every iteration.
        nslicesmax = 1000 - the number of slices to go up to before stopping.
        tolerance = 1e-8 - how close the integral with the current nslices and the integral with the current nslices - nstep have to be before this is considered converged.
        unconvergedval = None - what to return if the integral doesn't converge.
        printtime = True - whether or not to  print the time elapsed (in seconds) during this function.
        timeprecision = 0.4 - how far out to round the time (following f"{time:{timeprecision}f}" format). Also used for the difference in the current and last values when they're printed.
    """
    t1 = time.time()
    nsl = nslices
    lastval = None
    currentval = 0 
    while nsl < nslicesmax:
        currentval = integ_fn(inpfn, ll, ul, nsl)
        if lastval != None:
            if abs(currentval-lastval) >= tolerance:
                nsl += nstep
                lastval = currentval
            else:
                printif(f"Converged to within tolerance of {tolerance} at {nsl} slices; \n current val - previous: abs({currentval:{timeprecision}f} - {lastval:{timeprecision}f}) = {abs(currentval - lastval):{timeprecision}e}, \n note system limit {sys.float_info.epsilon:{timeprecision}e}",
                       bonusprints)
                return(currentval)
        else:
            nsl += nstep
            lastval = currentval
            
                
    printif(f'DID NOT CONVERGE, ended at {currentval:{timeprecision}f} and nslices {nsl} with abs(current - last value) of {abs(currentval-lastval):{timeprecision}e}',
               bonusprints)
    printif(f'Time to run: {(time.time()-t1):{timeprecision}f} s')
    return(unconvergedval)
    
# plot varying nslices
def plot_varying_nslices(lowerlim, upperlim, inpfn, nslices_arange_args, ax_kwargs = {'xscale':'log', 'yscale':'log', 'xlabel':r'N$_{slices}$', 'ylabel':'DEFAULT','title':'DEFAULT'}, simp_line_kwargs={'label':"simpson's"}, trap_line_kwargs = {'ls':'--', 'label':"trapezoidal"}, figkwargs = {}, inpfn_name = 'DEFAULT', gridon = True, pltshow = True):
    """
    Plots integrals from lowerlim-upperlim of some function inpfn as a function of the number of slices used in the integral function. 
    Args:
        lowerlim - lower limit of integration.
        upperlim - upper limit of integration.
        inpfn - function to integrate.
        nslices_arange_args - args to be passed to np.arange() to generate the array of nslices to try.
        ax_kwargs = {'xscale':'log', 'yscale':'log', 'xlabel':r'N$_{slices}$', 'ylabel':'DEFAULT', 'title':'DEFAULT'} - kwargs for ax.set(). leaving ylabel as default will fill it in with 'ylabel':r'$ int_{'+f'{lowerlim}'+'}^{'+f'{upperlim}'+'}$'+inpfn_name} . The same thing + 'as a function of Nslices' will be filled in for the title if title is left DEFAULT.
        simp_line_kwargs = {'label':"simpson's"} - kwargs for the line for simpson's rule integration.
        trap_line_kwargs = {'ls':'--', 'label':"trapezoidal"} - kwargs for the line for trapezoidal integration.
        fig_kwargs = {} - kwargs to pass to plt.figure(), if desired.
        inpfn_name = 'DEFAULT' - name to use for the input fn in labels. if left default, it'll read it in from the properties of the function itself.
        gridon = True - whether to turn on plt.grid()
        pltshow = True - whether to run plt.show(). Will also run plt.tight_layout if you do.
    """
    if inpfn_name == 'DEFAULT':
        inpfn_name = inpfn.__name__
    fig = plt.figure(**figkwargs)
    ax_kwargs = updateandreturndict({'xscale':'log', 'yscale':'log', 'xlabel':r'N$_{slices}$', 'ylabel':'DEFAULT','title':'DEFAULT'}, ax_kwargs)
    if ax_kwargs['ylabel'] == 'DEFAULT':
        ax_kwargs.update({'ylabel':r'$\int_{'+f'{lowerlim}'+'}^{'+f'{upperlim}'+'}$'+inpfn_name})
    if ax_kwargs['title'] == 'DEFAULT':
        ax_kwargs.update({'title':r'$\int_{'+f'{lowerlim}'+'}^{'+f'{upperlim}'+'}$'+inpfn_name+r' as a function of N$_{slices}$'})
    simp_line_kwargs = updateandreturndict({'label':"simpson's"}, simp_line_kwargs)
    trap_line_kwargs = updateandreturndict({'ls':'--', 'label':'trapezoidal'}, trap_line_kwargs)
    
    ax = fig.add_subplot()
    ax.set(**ax_kwargs)
    nslices_arange_arr = np.arange(nslices_arange_args[0], nslices_arange_args[1], nslices_arange_args[2])
    ax.plot(nslices_arange_arr, [(lambda nsl:simpsons_rule(inpfn, lowerlim, upperlim, nsl))(nsl) for nsl in nslices_arange_arr], **simp_line_kwargs)
    ax.plot(nslices_arange_arr, [(lambda nsl:trap_integral(inpfn, lowerlim, upperlim, nsl))(nsl) for nsl in nslices_arange_arr], **trap_line_kwargs)
    if gridon:
        plt.grid('on')
    plt.legend()
    if pltshow:
        plt.tight_layout()
        plt.show()
        
# split integrals fn
def split_integral(inpfn, llarr, ularr, nslices=100, integ_fn=simpsons_rule):
    """Function for calculating an integral split up into two intervals - like how:

        integral a->b = integral a->c + integral c->b

       args:
           inpfn - function to integrate
           llarr - array of lower limits (a la [a, c] for the above example)
           ularr - array of upper limits (a la [c, b] for the above example)
           nslices = 100 - number of slices
           integ_fn = simpsons_rule - integral function to use
    
    """
    outpt=0
    for ll, ul in zip(llarr, ularr):
        outpt+=(integ_fn(inpfn, ll, ul, nslices=nslices))
    return(outpt)
    
# plot varying nslices - with discont integral
def plot_varying_nslices_vs_split(llarr, ularr, inpfn, nslices_arange_args, ax_kwargs = {'xscale':'log', 'yscale':'log', 'xlabel':r'N$_{slices}$', 'ylabel':'DEFAULT', 'title':'DEFAULT'}, integ_fn = simpsons_rule, integ_line_kwargs={'label':"DEFAULT"}, split_line_kwargs = {'ls':'--', 'label':"split integral"}, figkwargs = {}, inpfn_name = 'DEFAULT', gridon = True, pltshow = True):
    """
    Plots two lines - one being a split integral along the rule:
        integral a->b = integral a->c + integral c->b
    And the other being simply integral a->b, versus the number of slices used in integration. Assuming you put in some continuous interval for the a, b, c values, they should converge, but they do so at a different rate depending on slices 
    Args:
        llarr - array of lower limits (a la [a, c] for the above example)
        ularr - array of upper limits (a la [c, b] for the above example)
        inpfn - function to integrate.
        nslices_arange_args - args to be passed to np.arange() to generate the array of nslices to try.
        ax_kwargs = {'xscale':'log', 'yscale':'log', 'xlabel':r'N$_{slices}$', 'ylabel':'DEFAULT', 'title':'DEFAULT'} - kwargs for ax.set(). leaving ylabel as default will fill it in with 'ylabel':r'$ int_{'+f'{lowerlim}'+'}^{'+f'{upperlim}'+'}$'+inpfn_name}. The same thing + 'as a function of Nslices' will be filled in for the title if title is left DEFAULT.
        integ_fn = simpsons_rule - integral function used.
        integ_line_kwargs = {'label':"DEFAULT"} - kwargs for the line for the straightforward integral a->b. DEFAULT label will assign itself the name of the input function. 
        split_line_kwargs = {'ls':'--', 'label':"split integral"} - kwargs for the line for split integration.
        fig_kwargs = {} - kwargs to pass to plt.figure(), if desired.
        inpfn_name = 'DEFAULT' - name to use for the input fn in labels. if left default, it'll read it in from the properties of the function itself.
        gridon = True - whether to turn on plt.grid()
        pltshow = True - whether to run plt.show(). Will also run plt.tight_layout if you do.
    """
    if inpfn_name == 'DEFAULT':
        inpfn_name = inpfn.__name__
    if integ_line_kwargs['label'] == 'DEFAULT':
        integ_line_kwargs['label'] = integ_fn.__name__.replace('_', ' ')
    fig = plt.figure(**figkwargs)
    if ax_kwargs['ylabel'] == 'DEFAULT':
        ax_kwargs.update({'ylabel':r'$\int_{'+f'{llarr[0]}'+'}^{'+f'{ularr[1]}'+'}$'+inpfn_name})
    if ax_kwargs['title'] == 'DEFAULT':
        ax_kwargs.update({'title':r'$\int_{'+f'{llarr[0]}'+'}^{'+f'{ularr[1]}'+'}$'+inpfn_name+r' as a function of N$_{slices}$'})
    ax = fig.add_subplot()
    ax.set(**ax_kwargs)
    nslices_arange = np.arange(nslices_arange_args[0], nslices_arange_args[1], nslices_arange_args[2])
    ax.plot(nslices_arange, [(lambda nsl:integ_fn(inpfn, llarr[0], ularr[1], nsl))(nsl) for nsl in nslices_arange], **integ_line_kwargs)
    ax.plot(nslices_arange, [(lambda nsl:split_integral(inpfn, llarr, ularr, nsl))(nsl) for nsl in nslices_arange], **split_line_kwargs)
    if gridon:
        plt.grid('on')
    plt.legend()
    if pltshow:
        plt.tight_layout()
        plt.show()
        
if __name__ == "__main__":

    # set up arguments with argparse
    parser = argparse.ArgumentParser(description="Script for calculating the integral of e**(-t**2) and plotting various interesting things about the integral and the numerical methods used to integrate it.")
    # read in the run mode
    parser.add_argument('--runmode', default='part a, part b', type = str, help='What mode to run this script in. For instance, the default of "part a, part b" will run through just the two parts of the problem that was assigned as homework.')
    # read in the basic arguments, needed for parts a and b
    parser.add_argument('-x', default = 3, help='x value out to which to calculate the integral e**(-t**2).', type = float)
    parser.add_argument('--lowerlim', default = 0, help = 'lower limit to calculate the integral from lowerlimit -> x.', type = float)
    parser.add_argument('--stepsize', default = 0.1, type = float, help = 'stepsize over which to increase x, following np.arange(lowerlim, x, stepsize)')
    parser.add_argument('--nslices', default = 1e3, type = float, help = 'number of slices to use in integration.')
    parser.add_argument('--inpfn', default = 'DEFAULT', help = 'input function. left as DEFAULT, will fill in with the e**-t**2 function meant to be used on this homework, but also can be set to "poly" to run one of the polynomials we have occasionally used in exercises.')
    parser.add_argument('--integ_fn', default = 'simps', help = 'input integral function. enter simps for simpsons rule, trap for trapezoidal. defaults to simps' )
    parser.add_argument('--compare_methods', default = True, help = 'bool controlling if both simpsons and trapezoidal integration are shown in the first plot made in this function.')
    #parser.add_argument
    # more arguments, as needed for plotting vs. a range of slices
    parser.add_argument('--nslices_arange', nargs=3, type = float, default = (1, 1e4, 1), help = 'for if you plot E(x) as a function of nslices, the values defining the np.arange() that generates slices; ie the default (1, 1e4, 1) will be used in an np.arange(1, 1e4, 1).')
    # more arguments, as needed for running the split integral
    parser.add_argument('--llim_arr', nargs=2, type = float, default = (0, 1.5), help= 'array of lower limits for a split integral following the rule integral a->b = integral a-c + integral c-b. in that example, llim_arr would be [a, c]. Will adjust itself automatically according to input to x and lowerlimits, if you change those arguments.') # note: i want to add something to make this adjust automatically, but also this script has a lot of moving parts already, so for now i'm leaving it off.
    parser.add_argument('--ulim_arr', nargs=2, type = float, default = (1.5, 3), help= 'array of upper limits for a split integral following the rule integral a->b = integral a-c + integral c-b. in that example, ulim_arr would be [c, b]. Will adjust itself automatically according to input to x and lowerlimits, if you change those arguments.') # note: i want to add something to make this adjust automatically, but also this script has a lot of moving parts already, so for now i'm leaving it off.
    # more arguments, as needed for running the test to convergence
    parser.add_argument('--nslices_step', default=1, type = float, help = 'step to increase slices by when testing an integral to convergence.')
    parser.add_argument('--nslices_max', default=1e4, type = float, help = 'maximum number of slices to go up to when testing to convergence.')
    parser.add_argument('--tolerance', default=1e-8, type = float, help = 'how close the integral with the current nslices and the integral with the current nslices - nstep have to be before this is considered converged.')
    parser.add_argument('--time_precision', default=0.4, type = float, help = 'how far out to round the time (following f"{time:{timeprecision}f}" format). Also used for the difference in the current and last values when they are printed by the test_to_convergence function.')
    # single extra bonus for handling the arange thing that was annoying me:
    parser.add_argument('--x_plus_stepsize', default = 'auto', help = "fixes the np.arange being a closed interval FOR 0 to 3 SPECIFICALLY by making the E(x) function go over a range of x given by np.arange(0, x+stepsize, stepsize). if left as 'auto' it'll set itself true IF AND ONLY IF lowerlim = 0 and x = 3. you can also set it to False and then it'll never do this, or True and then it'll ALWAYS go over np.arange(0, x+stepsize, stepsize) for the E(x) function.")

    args = parser.parse_args()
    
    # define args
    runmode = args.runmode
    x= args.x
    stepsize=args.stepsize
    nslices = args.nslices
    lowerlim = args.lowerlim
             
    inpfn = args.inpfn
    if inpfn.lower() in ['default', 'def', 'e**(-t**2)']:
        inpfn = e_to_minus_tsq
        fn_rstring = r"$e^{-t^2}$"
    elif inpfn.lower() in ['poly', 'polynomial', 'basic_polynomial']:
        inpfn = basic_polynomial
        fn_rstring = r"$x^{4} -2x + 1$"
    else:
        print("I didn't recognize your input function (it's not either the default or basic_polynomial setting) so I'm resetting it to the default (e**(-t**2)). Sorry!")
        inpfn = e_to_minus_tsq
    integ_fn = args.integ_fn
    if integ_fn in ['simps', 'simpsons', 'simpsons rule', 'simpsons_rule']:
        integ_fn = simpsons_rule
    if integ_fn in ['trap', 'trapezoid', 'trapezoidal rule', 'trap_integral']:
        integ_fn = trap_integral
             
    compare_methods = args.compare_methods
    nslices_arange = [float(x) for x in args.nslices_arange]
    if nslices_arange[1] > 1e5:
        print("WARNING: THIS MIGHT TAKE A REALLY LONG TIME, BE READY TO KILL THIS SCRIPT WITH CONTROL - C!!")
    llim_arr = [float(x) for x in args.llim_arr]
    ulim_arr = [float(x) for x in args.ulim_arr]
    nslices_step = args.nslices_step
    nslices_max = args.nslices_max
    tolerance = args.tolerance
    time_precision = args.time_precision
    x_plus_stepsize = args.x_plus_stepsize
        
    if x_plus_stepsize == 'auto':
        if x==3:
            if lowerlim == 0:
                print("APPLYING THE SPECIAL FIX FOR PART A AND B THAT MAKES NP.ARANGE PRETEND TO BE A CLOSED INTERVAL FOR 0 -> 3.")
                x_plus_stepsize=True
    if x_plus_stepsize == 'auto':
        x_plus_stepsize = False
    if x_plus_stepsize == 'False':
        x_plus_stepsize = False
    if x_plus_stepsize == 'True':
        x_plus_stepsize = True
            
    # handle different runmode cases
    run_a = False
    run_b = False
    if ifanyofthesein(['parta', 'part_a', 'part a', 'pa', 'runa', 'runab', 'run b', 'run_a', 'run_ab'], runmode.lower()):
        run_a = True
    if ifanyofthesein(['partb', 'part_b', 'part b', 'pb', 'runb', 'runab', 'run b', 'run_b', 'run_ab'], runmode.lower()):
        run_b = True
            
    plot_vs_slices = False
    if ifanyofthesein(['plotvs', 'plotslices', 'slicesplot', 'plot_vs_slices', 
                    'plotvs_slices', 'plot_vsslices', 'plot vs slices', 'vsslices'], runmode.lower()):
        plot_vs_slices = True
            
    plot_split_integral = False
    if ifanyofthesein(['split', 'split_integral', 'plotsplit', 'plot_split_integral', 'plot split integral', 'split integral'], runmode.lower()):
        plot_split_integral = True

    test_convergence = False
    if ifanyofthesein(['testconv', 'convergence', 'converge', 'test_convergence', 'test convergence', 'conv'], runmode.lower()):
        test_convergence = True
    # will run all of the options
    explore = False
    if ifanyofthesein(['explore', 'exp'], runmode.lower()):
        explore = True
        run_a = True
        run_b = True
        plot_vs_slices = True
        plot_split_integral =True
        test_convergence = True

    print('Arguments:')
    print(args)

    if run_a:
        print(f'\n\nPart A: Calculating the integral of {inpfn.__name__} from {lowerlim} to {x}, for values of {x} from {lowerlim} to {x}, in steps of {stepsize} (ie stepping over x in steps of {stepsize}).\n\n')
        Eofx_0to3 = integral_over_x(x, lowerlim = lowerlim, stepsize = stepsize, nslices = nslices, inpfn=inpfn, integ_fn=integ_fn, special_hwpartab_case = x_plus_stepsize)
        print(Eofx_0to3)

    if run_b:
        print(f'Part B: Plot this function integral of {inpfn.__name__} from {lowerlim} to {x}, for values of {x} from {lowerlim} to {x}, in steps of {stepsize} (ie stepping over x in steps of {stepsize}), as a function of x\n\n')
        
        fig = plt.figure()
        ax = fig.add_subplot()
        if x_plus_stepsize:
            xarange = np.arange(lowerlim, x+stepsize, stepsize)
        else:
            xarange = np.arange(lowerlim, x, stepsize)
        
        ax.plot(xarange, integral_over_x(x, stepsize = stepsize, nslices = nslices, inpfn=inpfn, integ_fn=integ_fn, lowerlim=lowerlim, special_hwpartab_case= x_plus_stepsize), 
                    label = integ_fn.__name__.replace('_', ' '))
            
        if compare_methods:
            othermethod = {simpsons_rule:trap_integral, trap_integral:simpsons_rule}[integ_fn]
            ax.plot(xarange, integral_over_x(x, stepsize = stepsize, nslices = nslices, inpfn=inpfn, integ_fn=othermethod, lowerlim=lowerlim, special_hwpartab_case=x_plus_stepsize), ls = '--', label = othermethod.__name__.replace('_', ' '))
            
        plt.grid('on')
        if inpfn == simpsons_rule:
            ax.set_title(r'E(x) = $\int_{'+f"{lowerlim}"+'}^{x} $'+fn_rstring+f', stepsize = {stepsize}')
        else:
            ax.set_title(r'E(x) = $\int_{'+f"{lowerlim}"+'}^{x} $'+fn_rstring+f', stepsize = {stepsize}')
        ax.set_xlabel('x')
        ax.set_ylabel('E(x)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    printif(f"Part A and B done, let's move on to some interesting playing around!\n\n\n", explore)

    if plot_vs_slices:
        print(f'Plot E(x) using both simpson and trapezoidal integration, with a constant integration limits {lowerlim} -> {x}, but varying the number of nslices along an arange like np.arange{nslices_arange}.\n\n')
        plot_varying_nslices(lowerlim, x, inpfn=inpfn, nslices_arange_args=nslices_arange, )

    if plot_split_integral:
        print(f'Plot E(x) from {llim_arr[0]} to {ulim_arr[1]} versus varying nslices like np.arange{nslices_arange}, using {integ_fn.__name__}, both with calculating the integral directly over {llim_arr[0]} to {ulim_arr[1]}, and with an integral split up into the sum of integral {llim_arr[0]} to {ulim_arr[0]} and {llim_arr[1]} to {ulim_arr[1]}.\n\n')
        plot_varying_nslices_vs_split(llim_arr, ulim_arr, inpfn=inpfn, integ_fn=integ_fn, nslices_arange_args=nslices_arange)

    if test_convergence:
        print(f"Test how many slices it takes, and how much time it takes to run, to calculate the integral of {inpfn.__name__} to convergence with some tolerance {tolerance}.\n\n")
        test_to_convergence(inpfn, integ_fn, ll=lowerlim, ul=x, nslices=nslices, nstep=nslices_step, nslicesmax=nslices_max, tolerance = tolerance, timeprecision= time_precision)

    if True not in [explore, test_convergence, plot_split_integral, plot_vs_slices, run_a, run_b]:
        print("Oops!! Whatever you entered for runmode, I didn't understand. Please see the options below!")

    if not explore:
        print("\n\nIf you haven't yet, please explore the other run modes of this script!")
        print('Try:')
        print(f"  'run_a' - will run part 'a' of the homework (calculate the integral of the input function {inpfn.__name__} from {lowerlim} to {x} and return the values for some range of {x} ({lowerlim}, {x}, {stepsize})).")
        print(f"  'run_b' - will run part 'b' of the homework (plot the integral of the input function {inpfn.__name__} for values of {x} from {lowerlim} to {x}, in steps of {stepsize}).")
        print("  'explore' - will run through both 'extra' features of this script.")
        print("  'plot_vs_slices' - will plot both the result of E(x) using both simpson's rule and trapezoidal rule to integrate, as a function of the number of slices, to show how many slices it takes for them to (roughly) agree on an answer.")
        print("  'split_integral' - illustrates something I noticed while playing around with the integral functions. Applies the rule of integration from a -> b = integration from a -> c + c -> b , but I noticed that the error is reasonably noticeable between the two if you don't use plenty of slices in the integration.")
        print("  'test_convergence' - test how many slices / how long it takes to converge on an answer (within a given tolerance) for either simpson's or trapezoidal integration.")

    