"""
Computational Astronomy Week 2 Homework

Use argparse in a script that can calculate the time it takes for a ball to hit the ground given height h and gravity g.

Written by Miranda McCarthy, Tuesday Sept. 2nd, August 2025
"""

# imports
import matplotlib.pyplot as plt
import numpy as np
import argparse

def ballheight(h, g = 9.8):
    """Function for time it takes for a ball to hit the ground given height h and kwarg g (default 9.8 m/s)"""
    t = ((2*h)/g)**0.5
    return(t)

class BallClass:

    def __init__(self, h, g = 9.8):
        self.h = h
        self.g = g
        self.t = ballheight(h, g)

    def plot_vs_height(self, heightarr, g = 'default', 
                       grid = True, passfig = 'None', passax = 'None', pltshow = True):
        if passfig != 'None':
            fig = passfig
        else:
            fig = plt.figure()
        if passax != 'None':
            ax = passax
        else:
            ax = fig.add_subplot(111)
        if g == 'default':
            g = self.g
        ax.grid('on')
        ax.plot(heightarr, ballheight(heightarr, g))
        ax.set_xlabel('Height (m)')
        ax.set_ylabel('Time (s)')
        ax.set_title(f'Height vs time with g = {g:.2f}')
        if pltshow:
            plt.show()
        
    def plot_vs_g(self, garr, h = 'default', 
                  grid = True, passfig = 'None', passax = 'None', pltshow = True):
        if passfig != 'None':
            fig = passfig
        else:
            fig = plt.figure()
        if passax != 'None':
            ax = passax
        else:
            ax = fig.add_subplot(111)
        if h == 'default':
            h = self.h
        ax.grid('on')
        ax.plot(garr, ballheight(h, garr))
        ax.set_xlabel(r'g ($\frac{m}{s^2}$)')
        ax.set_ylabel('Time (s)')
        ax.set_title(f"g vs time with h = {h:.2f}")
        if pltshow:
            plt.show()

if __name__ == "__main__":

    
    # set up arguments with argparse
    parser = argparse.ArgumentParser(description="Function for calculating the time it takes for a ball to hit the ground given a certain height.")
    
    # note, NOT forcing type on h or g, because then I can let it take strings as the arg.
    parser.add_argument('h', help='Height (in meters) above the ground from which a ball is dropped.')
    parser.add_argument('-g', default=9.8, help='Acceleration (in m/s^2) due to gravity. Default 9.8')
    parser.add_argument('--make_plots', type=str, default='None', help = 'String that controls if you want to make a plot showing time as a function of either height or gravity. By default, is None. If you enter this as some variation of True or Both but do not pass any arguments to plot_range_h or plot_range_g, it will make both a plot of h vs. time and g vs. time (with range of 0-10 m for h, and 9.8-10.8 m/s^2 for g). Else specify which you want to plot by passing "plot_h" or "plot_g".')
    parser.add_argument('--plot_range_h', nargs=2, type=float, default=(0, 10), help='Plot a basic graph showing the time it takes for the ball to hit the ground given a range of heights in meters. Default 0m-10m.')
    parser.add_argument('--plot_range_g', nargs=2, type=float, default=(9.8, 19.8), help='Plot a basic graph showing time it will take for a ball dropped from height h to hit the ground given a range of g values. Default 9.8-19.8 m/s^2.')
    
    args = parser.parse_args()


    # define args
    #print('Running with argparse:', args)
    h = args.h
    g = args.g
    make_plots = args.make_plots
    plot_range_h = tuple(args.plot_range_h)
    plot_range_g = tuple(args.plot_range_g)
    print('Running with args:',
                f"h = {h}", '\n',
                f"g = {h}", '\n',
                f"make_plots = {make_plots}", '\n',
                f"plot_range_h = {plot_range_h}", '\n',
                f"plot_range_g = {plot_range_g}")

    # handle string vers of h or g
    try:
        h = float(h)
    except ValueError:
        if type(h) == str:
            if h.lower() in ['empire', 'emp', 'esb']:
                print('Using the height of the Empire State Building (443 m, according to brittanica.com!)')
                h = 443
            elif h.lower() in ['cuny', 'grad center', 'gc', 'b altman']:
                print('Using the height of the CUNY graduate center, using appx. conversion of 1 story = 3.5 m (from reference.com) and the shorter side of cuny height of 8 stories (ie 28 m!)')
                h = 28
            else:
                raise ValueError('You have provided an invalid string for h.')
    try:
        g = float(g)
    except ValueError:
        if type(g) == str:
            # sourced from https://www.jpl.nasa.gov/images/pia24373-trappist1-and-solar-system-planet-stats/
            trappistg = {'b':1.10*9.8, 'c':1.09*9.8, 'd':0.62*9.8, 'e':0.82*9.8, 'f':0.95*9.8, 'g':1.04*9.8, 'h':0.57*9.8}
            # sourced from https://ssd.jpl.nasa.gov/planets/phys_par.html
            solarsysg = {'mercury':3.7, 'venus':8.87, 'earth':9.80, 'mars':3.71, 'jupiter':24.79, 'saturn':10.44, 'uranus':8.87, 'neptune':11.15}
            if g.lower() in trappistg:
                print(f"Using gravity for this planet trappist {g} from the trappist 1 from jpl!")
                g = trappistg[g.lower()]
            elif g.lower() in solarsysg:
                print(f"Using gravity for {g.lower()} from SSD at JPL!")
                g = solarsysg[g.lower()]
            else:
                raise ValueError('You have provided an invalid string for g.')
                
    
    print(f"Time to fall given height {h:.2f} m and acceleration {g:.2f} m/s^2")
    print(ballheight(h, g))

    if make_plots != 'None':
        tempball = BallClass(h, g)
        if make_plots.lower() in ['true', 'both', 'all']:
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            tempball.plot_vs_height(np.arange(plot_range_h[0], plot_range_h[1]), 
                                   passfig = fig, passax = ax1, pltshow=False)
            tempball.plot_vs_g(np.arange(plot_range_g[0], plot_range_g[1]), 
                              passfig = fig, passax = ax2, pltshow=False)
            plt.tight_layout()
            plt.show()
        elif make_plots.lower() in ['plot_h', 'ploth', 'height', 'h']:
            tempball.plot_vs_height(np.arange(plot_range_h[0], plot_range_h[1]))
        elif make_plots.lower() in ['plot_g', 'plotg', 'gravity', 'g']:
            tempball.plot_vs_g(np.arange(plot_range_g[0], plot_range_g[1]))
        else:
            raise ValueError('You have not provided a valid argument to make_plots! \n For reference, if you want to plot time as a function of height and time as a function of gravity enter "both", to plot just height "height", to plot just gravity "gravity".')
    