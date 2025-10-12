"""
Fourier Transform homework for Computational Methods
Script that explores various aspects of using Fourier transforms and inverse Fourier transforms on a TESS dataset.
-- Miranda McCarthy, 10/10/2025
"""

import argparse
import numpy as np
import pandas as pd
import pdb
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
import glob
from time import sleep
# this one takes a few more libraries!!
# I was gonna import warnings so i could repress the"complex values to real" message, but it created some other glitch I don't understand when I tried. Now I just have to actually code properly and make things real when they're being plotted...

# Functions

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
    """ Another simple space saving function I use a lot. Rather than checking if foo in [(list of things)], it checks if any item in a list is in a given object.
        Args:
            anyofthese - a list of things to check if they are in areinthis.
            areinthis - whatever you want to search for the items in anyofthese.
        Returns:
            True, if any item in anyofthese is in areinthis. Else False."""
    for itm in anyofthese:
        if itm in areinthis:
            return(True)

# specifics:
def manual_split_on_gap(inptab, axtosplit, selectbetween, returninds = False):
    """Splits an table of values with large gaps on a certain axis up into multiple tables, one for each 'section'. 
        Args:
            inptab - the table to split up.
            axtosplit - the axis to split up based upon (in our case, with TESS data, this is times)
            selectbetween - the values on the given axis axtosplit you want to use to break up the data into sections (ie, first table is the values between splitbetween[0][0] and splitbetween[0][1])
            returninds = False - whether to also return the indices of the values on which you split the data.
    """
    iterarr = inptab[axtosplit]
    splittabs = []

    for splithere in selectbetween:
        splittabs.append(inptab.loc[np.where(
            (splithere[0] < iterarr) & (iterarr < splithere[1])
        )])
    if returninds:
        indsofsplit = []
        for splithere in selectbetween:
            indsofsplit.append((
                np.where((splithere[0] < iterarr) & (iterarr < splithere[1]))[0][0],
                np.where((splithere[0] < iterarr) & (iterarr < splithere[1]))[0][-1]
            ))
        return(splittabs, indsofsplit)
    else:
        return(splittabs)

class FFT_Objects:
    """A class for performing fourier transforms on data and plotting various things like their power spectrum and inverse fourier transform.
        
        attributes:
        
            self.y - ydata
            self.x - xdata, if provided
            self.c - np.fft.fft(self.y)
            self.ifft_c - np.fft.ifft(self.c)
            self.c_sq - abs(fft_c)**2, with the zero term set = 0
            self.c_sq_og - abs(fft_c)**2, with the zero term left alone
            self.k = np.arange(0, len(fft_c))

        functions:
            plot_powerspec - plots power spectrum
            plot_ifft - plots inverse fourier transform
        """
    def __init__(self, yarr, xdat = 'NONE'):
        """yarr - ydata. 
           xdat = 'NONE' - corresponding x data. You don't technically need this unless you're going to plot things like the inverse fourier transform, so it's optional."""
        
        self.y = yarr
        if type(xdat) != str:
            self.x = xdat
        elif xdat != 'NONE':
            raise ValueError('Error: string provided for xdat; xdat must be array like. The only acceptable str argument for xdat is NONE, if you do not expect to ever use the xdata.')
        
        fft_c = np.fft.fft(yarr)
        ifft_c = np.fft.ifft(fft_c)
        fft_c_sq_og = abs(fft_c)**2
        fft_c_sq = copy.deepcopy(fft_c_sq_og)
        fft_c_sq[0] = 0

        self.c = fft_c
        self.ifft_c = ifft_c
        self.c_sq = fft_c_sq
        self.k = np.arange(0, len(fft_c))
        self.c_sq_og = fft_c_sq_og
        
    def plot_powerspec(self, include_zero_term = False, passfig = None, passax = None, scatter_kwargs = {},
                      figkwargs = {}, ax_kwargs = {'xlabel':'k', 'ylabel':r'$|c^2|$'}, plttight = True, pltshow = True, pltclose = True):
        """Plot the power spectrum. 
           Args:
               self (from class)
               include_zero_term = False - whether or not to use the version of |c|^2 that includes the zero term. 
               passfig = None, passax = None - figure and axis to use, if desired.
               scatter_kwargs = {} - kwargs to pass the scatter plot
               figkwargs = {} - kwargs for plt.figure, if generating figure within this function
               ax_kwargs = {'xlabel':'k', 'ylabel':r'$|c^2|$'} - kwargs for ax.set()
               plttight = True - whether to run plt.tight_layout()
               pltshow = True - whether to run plt.show()
               pltclose = True - whether to run plt.close() (mostly here for debugging)
        """
        if passfig != None:
            fig = passfig
        else:
            fig = plt.figure(**figkwargs)
        if passax != None:
            ax = passax
        else:
            ax = fig.add_subplot(111)

        if include_zero_term:
            scatterplot = ax.scatter(self.k, self.c_sq_og)
        else:
            scatterplot = ax.scatter(self.k, self.c_sq)
        ax_kwargs = updateandreturndict({'xlabel':'k', 'ylabel':r'$|c^2|$', 'title':'Power spectrum'}, ax_kwargs)
        ax.set(**ax_kwargs)
        if plttight:
            plt.tight_layout()
        if pltshow:
            plt.show()
        if pltclose:
            plt.close()
        
        return(scatterplot)

    def plot_ifft(self, xdat= 'NONE', passfig = None, passax = None, scatter_kwargs = {'label':'data'}, ifftline_kwargs = {'label':'IFFT', 'c':'orange'}, figkwargs = {}, 
                ax_kwargs = {}, pltlegend = True, plttight = True, pltshow = True, pltclose = True):
        """
        plots inverse fourier transform of the coefficients c, and overlays it on the original data.
        Args:
            self (from class)
            xdat = 'NONE' - if NONE will first try to grab it from the object's self.x, and if none was given to the object upon initialization it'll throw you a ValueError. 
            passfig = None, passax = None - figure and axes to use for plotting. If left None will make a fig and ax within the function.
            scatter_kwargs = {'label':'data'} - kwargs for the scatterplot of the original data.
            ifftline_kwargs = {'label':'IFFT'} - kwargs for the inverse fourier transform.
            ax_kwargs = {} - kwargs for ax.set()
            pltlegend = True - run plt.legend()
            plttight = True - run plt.tight_layout()
            pltshow = True - run plt.show()
            pltclose = True - run plt.close() (mostly for debugging)
        
        """

        if type(xdat) == str:
            if xdat == 'NONE':
                try:
                    xdat = self.x
                except:
                    raise ValueError('You left xdat as default except this object was never given an xdat array to start with. Either provide an x data array or make a new FFT_Objects that is given an xdat array to start with.')
            else:
                raise ValueError('xdat must be array like. the only acceptable string argument for xdat is NONE in which case this function will use whatever x data is already in the object.')
        
        if passfig != None:
            fig = passfig
        else:
            fig = plt.figure(**figkwargs)
        if passax != None:
            ax = passax
        else:
            ax = fig.add_subplot(111)

        ax_kwargs = updateandreturndict({'title':'IFFT'}, ax_kwargs)
        scatter_kwargs = updateandreturndict({'label':'data'}, scatter_kwargs)
        ifftline_kwargs = updateandreturndict({'label':'IFFT', 'c':'orange'}, ifftline_kwargs)
        
        scatterplot = ax.scatter(xdat, self.y, **scatter_kwargs)
        ifftline = ax.plot(xdat, self.ifft_c.real, **ifftline_kwargs)
        ax.set(**ax_kwargs)

        if pltlegend:
            plt.legend()
        if plttight:
            plt.tight_layout()
        if pltshow:
            plt.show()
        if pltclose:
            plt.close()
        return(scatterplot, ifftline)

# for plotting, masking, etc
def mask_carr_wlamb(carr, lambdafn):
    """A simple function for maskig arrays using a given function lambdafn.
       Runs: return(np.array([(lambdafn)(ind, val) for ind, val in enumerate(carr)]))"""
    return(np.array([(lambdafn)(ind, val) for ind, val in enumerate(carr)]))

# ANIMATION STATION
# for the stepping up in k ones
def anim_up_in_k_fft(fftobj, frameindices = 'default', framestepsize = 10, \
                passfig = None, passaxes = None, locklegend = ['lower center', 'upper center'], pltfigsize = (5, 7), pltfig_kwargs = {},
                ax1_set = {'xlabel':'Time', 'ylabel':'Flux'}, 
                ax2_set = {'xlabel':'c', 'ylabel':r'c$^2$'}, ogdata_kwargs = {}, npinvfft_kwargs = {'c':'orange'}, 
                changingfft_kwargs = {'ls':'-', 'c':'red'}, 
                powerspec_kwargs = {}, usedkscatter_kwargs = {}, kline_kwargs = {'c':'red'},
                anim_kwargs = {'repeat':True, 'blit':True}, animsave_kwargs = {},
                savefig = True, overwrite = False, clobber = False, savefilename = 'changingk.gif', 
                cushionframes = 10, pltclose = True, RUNANIM = True):

    """
    A function that makes an animation of the inverse fourier transform for a dataset given increasingly more coefficients in k.

    Args:
        fftobj - FFTObjects() object.
        frameindices = 'default' - array to call from with increasing frame (ie, frame 1 will call upon the first entry in this array, etc.) By default will be:
                np.arange(0, len(c) + framestepsize, framestepsize)
            where c is np.fft.fft(ydat). The extra + framestepsize is so that whatever you do you won't have a case where the last few coefficients arent counted because of any tricks of the frame step size (ie if framestepsize = 10 and c has data out to index 1056, those last few points would be skipped otherwise - that bug arose and i fixed it with this!)
        
        framestepsize = 10 - size of steps in np.arange(0, len(c), framestepsize) through which we'll step over k. ie, set this equal to ten, and each frame will be another 10 values farther in k.
        passfig = None - figure to pass if not using default
        passaxes = None - axes to use if not using default; if using, must pass both an ax1 and ax2 in format [ax1, ax2]
        locklegend = ['lower center', 'upper center'] - where to keep the legend for ax1 / ax2 (otherwise it may bounce around distractingly.)
        pltfigsize = (5, 7) - figure size if setting up default figure in this function
        pltfig_kwargs = {} - bonus kwargs for plt.figure()
        ax1_set = {'xlabel':'Time', 'ylabel':'Flux'} - kwargs to pass ax1.set(). Default, internally, is {'xlabel':'Time', 'ylabel':'Flux'}.
        ax2_set ={'xlabel':'k', 'ylabel':r'c$^2$'} - kwargs to pass ax2.set(). Default, internally, is {'xlabel':'k', 'ylabel':r'c$^2$'}.
        
        ogdata_kwargs = {}  - kwargs to pass the scatterplot of the original data.
        npinvfft_kwargs = {'c':'orange'} - kwargs to pass the line plot of the "correct" np.fft.ifft of the data.
        changingfft_kwargs = {'ls':'--', 'c':'red'} - kwargs to pass to the line of the fft that uses only a limited number of terms, that is changed with each frame of the animation (the line, not these kwargs, is changed!)

        powerspec_kwargs = {} - kwargs to pass the scatter plot that makes up the original power spectrum
        usedkscatter_kwargs = {} - kwargs to pass the scatter plot of the k values that we actually use in the line plotted with changingfft (also is updated every frame!)
        kline_kwargs = {'c':'red'} - kwargs to pass the line that shows up to where we are taking c values (changes with every frame!)
    
        anim_kwargs = {'repeat':False, 'blit':True} - kwargs to pass the animation.FuncAnimation() that we run to make our animation.
        animsave_kwargs = {} - kwargs to pass the saving of our animation.

        savefig = True - whether to save the figure at the end.
        overwrite = False - whether to allow overwriting. 
        clobber = False - same as overwrite, just using a term that I see more often used for this purpose. If either is set True, it'll override the other (ie, if you change from the default value, it'll go with the input value).
        savefilename = 'changingk.gif' - default filename to save to.
        cushionframes = 10 - number of frames to add to the end of the gif as a "pause" at the end. (If on mac, where the animation is openable in preview as a long series of stills, this isn't *super* necessary.)
        
        pltclose = True - closes the plt figure at the end, after saving, so that it doesn't stick around when the next figures are output in terminal. (That was an issue I was having, at least!)
        RUNANIM = True - all caps because you're not likely to change it!! If True, will run the animation. If False, it won't. You may set this = False if, say, you want to put the output of this function into another figure and save *that* as an animation.
    
    """

    # bookkeeping
    # check inputs
    if '.gif' not in savefilename:
        print('Hey!! You forgot to add .gif to the end of the output filename, so I will fix that for you.')
        savefilename = savefilename + '.gif'
    if clobber:
        overwrite = clobber
    if overwrite:
        clobber = overwrite
    # fig, axes
    if passfig != None:
        fig = passfig
    else:
        fig = plt.figure(figsize = pltfigsize, **pltfig_kwargs)
    if passaxes != None:
        ax1 = passaxes[0]
        ax2 = passaxes[1]
    else:
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
    # kwargs
    ogdata_kwargs = updateandreturndict({}, ogdata_kwargs)
    npinvfft_kwargs = updateandreturndict({'c':'orange'}, npinvfft_kwargs)
    changingfft_kwargs = updateandreturndict({'ls':'-', 'c':'red'}, changingfft_kwargs)
    ax1_set = updateandreturndict({'xlabel':'Time', 'ylabel':'Flux'}, ax1_set)

    powerspec_kwargs = updateandreturndict({}, powerspec_kwargs)
    usedkscatter_kwargs = updateandreturndict({}, usedkscatter_kwargs)
    kline_kwargs = updateandreturndict({'c':'red'}, kline_kwargs)
    ax2_set = updateandreturndict({'xlabel':'k', 'ylabel':r'c$^2$'}, ax2_set)

    anim_kwargs = updateandreturndict({'repeat':False, 'blit':True}, anim_kwargs)
    animsave_kwargs = updateandreturndict({}, animsave_kwargs)

    # get the fourier transform
    xdat = fftobj.x
    ydat = fftobj.y
    c = fftobj.c
    csq = fftobj.c_sq
    k = fftobj.k
    

    # ax1 - the data and fourier transforms
    ax1.scatter(xdat, ydat, **ogdata_kwargs)
    ax1.plot(xdat, np.fft.ifft(c).real, **npinvfft_kwargs)
    # the line that we'll update with the limited invfft!
    changingline, = ax1.plot([], [], **changingfft_kwargs) # empty for now, bc not in animation yet!
    ax1.set(**ax1_set)

    # ax2 - the power spectrum
    ax2.scatter(k, csq, **powerspec_kwargs) 
    csqscatter = ax2.scatter([], [], **usedkscatter_kwargs) # again, empty for now bc not in animation yet!
    kline = ax2.axvline(0, **kline_kwargs)
    ax2.set(**ax2_set)

    # handle the frames - 
    if type(frameindices) == str:
        if frameindices.lower() != 'default':
            raise ValueError("Hey, the only string you can pass frameindices is 'default' to tell it to use the default value - anything else, and you need to pass it as an array!!")
        frameindices= np.arange(0, len(c) + framestepsize, framestepsize)
    else:
        frameindices = frameindices
    
    # def the update fn, that will exist within this, so add "internal" to the name there
    def update_internal(framenum):
        if framenum < len(frameindices):
            #print(len(frameindices), framenum)
            try:
                frame =frameindices[framenum]
            except:
                pdb.set_trace()
            changingline.set_xdata(xdat)
            changingline.set_ydata(np.fft.ifft(mask_carr_wlamb(c, lambda indx,x: x if indx <= frame else 0)).real)
            changingline.set_label(f"k <= {frame:0.1f}")
            ax1.legend(loc = locklegend[0])
            csqscatter.set_offsets(np.column_stack([
                k[0:frame], csq[0:frame]
            ]))
            csqscatter.set_label(f"k <= {frame:0.1f}")
            kline.set_xdata([frame])
            ax2.legend(loc = locklegend[1])
            plt.tight_layout()
        #else:
            #print(framenum) # debugging
        return(changingline, csqscatter)
            
    if RUNANIM:
        ani = animation.FuncAnimation(fig = fig, func = update_internal, frames = len(frameindices)+cushionframes,
                                      **anim_kwargs)
        # an interesting note - saving the figure seems to be what takes time, so make sure not to if you don't have to.
        # ...but yknow, since that's the only way to see the result thru the terminal afaik... I assume you're going to.
        if savefig:
            if not overwrite:
                if glob.glob(savefilename) != []:
                    print('File already exists at', savefilename, '; and overwrite = False, so no gif saved.')
                    if pltclose:
                        plt.close()
                else:
                    print(f'Saving to {savefilename}!')
                    pilwriter = animation.PillowWriter()
                    ani.save(savefilename, writer = pilwriter,  **animsave_kwargs)
                    if pltclose:
                        plt.close()
            else:
                print(f'Saving to {savefilename}!')
                pilwriter = animation.PillowWriter()
                ani.save(savefilename, writer = pilwriter,  **animsave_kwargs)
                if pltclose:
                    plt.close()
    else:
        if pltclose:
            plt.close()
        # if not making the animation with this function, I assume you're gonna use the update function elsewhere
        return(update_internal)

# for the stepping up in k ones
def anim_random_k_fft(fftobj, sizerandomks = 'default', rng_kwargs = {'replace':False}, framestepsize = 10, \
                passfig = None, passaxes = None, locklegend = ['lower center', 'upper center'], pltfigsize = (5, 7), pltfig_kwargs = {},
                ax1_set = {'xlabel':'Time', 'ylabel':'Flux'}, 
                ax2_set = {'xlabel':'c', 'ylabel':r'c$^2$'}, ogdata_kwargs = {}, npinvfft_kwargs = {'c':'orange'}, 
                changingfft_kwargs = {'ls':'-', 'c':'red'}, 
                powerspec_kwargs = {}, usedkscatter_kwargs = {}, kline_kwargs = {'c':'red'},
                anim_kwargs = {'repeat':True, 'blit':True}, animsave_kwargs = {},
                savefig = True, overwrite = False, clobber = False, savefilename = 'changingk.gif', 
                pltclose = True, force_include_zero = True,
                cushionframes = 10, RUNANIM = True, RANDOMSEED = 1234):

    """
    A function that makes an animation of the inverse fourier transform of a dataset, with only coefficients chosen by taking a random sample of k values (ie, indices of the coefficients). By default it'll take larger and larger samples, without replacement, until it's taking all the coefficients there are to sample from.
    Note that by default it'll forcibly include the k = 0 term coefficient in the random sample, or else the resulting inverse transform will be so offset it won't lie over the original data and comparing to the "correct" inverse fft (the orange line) is hard.
    

    Args:        
        fftobj - FFTObjects() object.
        sizerandomks = 'default' - array to call from with increasing frame (ie, frame 1 will call upon the first entry in this array, etc.) By default will be:
                np.arange(0, len(c) + framestepsize, framestepsize)
            where c is np.fft.fft(ydat). The extra + framestepsize is so that whatever you do you won't have a case where the last few coefficients arent counted because of any tricks of the frame step size (ie if framestepsize = 10 and c has data out to index 1056, those last few points would be skipped otherwise - that bug arose and i fixed it with this!)
            i have this fear that this is making something something about the statistics of randomness a bit weird in this case, but oh well for now - if this were for science I'd be more worried, but for now I'll leave it be.
        
        framestepsize = 10 - size of steps in np.arange(0, len(c), framestepsize) through which we'll step over k. ie, set this equal to ten, and each frame will be another 10 values farther in k.
        passfig = None - figure to pass if not using default
        passaxes = None - axes to use if not using default; if using, must pass both an ax1 and ax2 in format [ax1, ax2]
        locklegend = ['lower center', 'upper center'] - where to keep the legend for ax1 / ax2 (otherwise it may bounce around distractingly.)
        pltfigsize = (5, 7) - figure size if setting up default figure in this function
        pltfig_kwargs = {} - bonus kwargs for plt.figure()
        ax1_set = {'xlabel':'Time', 'ylabel':'Flux'} - kwargs to pass ax1.set(). Default, internally, is {'xlabel':'Time', 'ylabel':'Flux'}.
        ax2_set = {} - kwargs to pass ax2.set(). Default, internally, is {'xlabel':'k', 'ylabel':r'c$^2$'}.
        
        ogdata_kwargs = {}  - kwargs to pass the scatterplot of the original data.
        npinvfft_kwargs = {'c':'orange'} - kwargs to pass the line plot of the "correct" np.fft.ifft of the data.
        changingfft_kwargs = {'ls':'--', 'c':'red'} - kwargs to pass to the line of the fft that uses only a limited number of terms, that is changed with each frame of the animation (the line, not these kwargs, is changed!)

        powerspec_kwargs = {} - kwargs to pass the scatter plot that makes up the original power spectrum
        usedkscatter_kwargs = {} - kwargs to pass the scatter plot of the k values that we actually use in the line plotted with changingfft (also is updated every frame!)
        kline_kwargs = {'c':'red'} - kwargs to pass the line that shows up to where we are taking c values (changes with every frame!)
        ax2_set = updateandreturndict({'xlabel':'k', 'ylabel':r'c$^2$'}, ax2_set)
    
        anim_kwargs = {'repeat':False, 'blit':True} - kwargs to pass the animation.FuncAnimation() that we run to make our animation.
        animsave_kwargs = {} - kwargs to pass the saving of our animation.

        savefig = True - whether to save the figure at the end.
        overwrite = False - whether to allow overwriting. 
        clobber = False - same as overwrite, just using a term that I see more often used for this purpose. If either is set True, it'll override the other (ie, if you change from the default value, it'll go with the input value).
        savefilename = 'changingk.gif' - default filename to save to.

        force_include_zero = True - forces the k = 0 term to be one of the coefficients included in the OTHERWISE random sample so that the resulting inverse fourier transform is at least visible on the plot with the actual data.
        
        cushionframes = 10 - number of frames to add to the end of the gif as a "pause" at the end.
        RUNANIM = True - all caps because you're not likely to change it!! If True, will run the animation. If False, it won't. You may set this = False if, say, you want to put the output of this function into another figure and save *that* as an animation.
        RANDOMSEED = 1234 - all caps because not likely to change it - the seed for the np.random.default_rng() that makes the used rng.
    """

    # bookkeeping
    # check inputs
    if '.gif' not in savefilename:
        print('Hey!! You forgot to add .gif to the end of the output filename, so I will fix that for you.')
        savefilename = savefilename + '.gif'
    if clobber:
        overwrite = clobber
    if overwrite:
        clobber = overwrite
    # fig, axes
    if passfig != None:
        fig = passfig
    else:
        fig = plt.figure(figsize = pltfigsize, **pltfig_kwargs)
    if passaxes != None:
        ax1 = passaxes[0]
        ax2 = passaxes[1]
    else:
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
    # kwargs
    ogdata_kwargs = updateandreturndict({}, ogdata_kwargs)
    npinvfft_kwargs = updateandreturndict({'c':'orange'}, npinvfft_kwargs)
    changingfft_kwargs = updateandreturndict({'ls':'-', 'c':'red'}, changingfft_kwargs)
    ax1_set = updateandreturndict({'xlabel':'Time', 'ylabel':'Flux'}, ax1_set)

    powerspec_kwargs = updateandreturndict({}, powerspec_kwargs)
    usedkscatter_kwargs = updateandreturndict({}, usedkscatter_kwargs)
    kline_kwargs = updateandreturndict({'c':'red'}, kline_kwargs)
    ax2_set = updateandreturndict({'xlabel':'k', 'ylabel':r'c$^2$'}, ax2_set)

    anim_kwargs = updateandreturndict({'repeat':False, 'blit':True}, anim_kwargs)
    animsave_kwargs = updateandreturndict({}, animsave_kwargs)

    rng_kwargs = updateandreturndict({'replace':False}, rng_kwargs)
    rng = np.random.default_rng(seed)
    

    # get the fourier transform
    xdat = fftobj.x
    ydat = fftobj.y
    c = fftobj.c
    csq = fftobj.c_sq
    k = fftobj.k
    

    # ax1 - the data and fourier transforms
    ax1.scatter(xdat, ydat, **ogdata_kwargs)
    ax1.plot(xdat, np.fft.ifft(c).real, **npinvfft_kwargs)
    # the line that we'll update with the limited invfft!
    changingline, = ax1.plot([], [], **changingfft_kwargs) # empty for now, bc not in animation yet!
    ax1.set(**ax1_set)

    # ax2 - the power spectrum
    ax2.scatter(k, csq, **powerspec_kwargs) 
    csqscatter = ax2.scatter([], [], **usedkscatter_kwargs) # again, empty for now bc not in animation yet!
    kline = ax2.axvline(0, **kline_kwargs)
    ax2.set(**ax2_set)

    # handle the frames - 
    if type(sizerandomks) == str:
        if sizerandomks.lower() != 'default':
            raise ValueError("Hey, the only string you can pass sizerandomks is 'default' to tell it to use the default value - anything else, and you need to pass it as an array!!")
        sizerandomks= np.arange(0, len(c)+framestepsize, framestepsize)
    else:
        sizerandomks = sizerandomks

    
    # def the update fn, that will exist within this, so add "internal" to the name there
    def update_internal(framenum):
        if (framenum < len(sizerandomks)):
            #print(len(sizerandomks), framenum)
            try:
                frame = sizerandomks[framenum]
            except:
                pdb.set_trace()
            # if frame > size of the c array - ie, resulting array of random k values will be longer than c -
            # forcibly act like frame is still == size of the c array
            if frame > len(c):
                frame = len(c)
            # generate random sample of k's
            if force_include_zero:
                if frame > 1:
                    randomkvals = rng.choice(k, size = frame-1, **rng_kwargs)
                    randomkvals = np.append(randomkvals, 0)
                else:
                    randomkvals = np.array([0])
            else:
                randomkvals = rng.choice(k, size = frame, **rng_kwargs)

            if framenum > 0:
                changingline.set_xdata(xdat)
                changingline.set_ydata(np.fft.ifft(mask_carr_wlamb(c, lambda indx,x: x if indx in randomkvals else 0)).real)

            changingline.set_label(f"# of terms (including zero term): {len(randomkvals):0.1f}")
            ax1.legend(loc = locklegend[0])
            csqscatter.set_offsets(np.column_stack([
                k[randomkvals], csq[randomkvals]
            ]))
            csqscatter.set_label(f"# of terms (including zero term): {len(randomkvals):0.1f}")
            kline.set_xdata([frame])
            #fig.suptitle(f"Max of ifft: {np.max(np.fft.ifft(mask_carr_wlamb(c, lambda indx,x: x if indx in randomkvals else 0))).real:0.2e}")
            ax2.legend(loc = locklegend[1])
            plt.tight_layout()
        #else:
            #print(framenum) # debugging
        return(changingline, csqscatter)
            
    if RUNANIM:
        ani = animation.FuncAnimation(fig = fig, func = update_internal, frames = len(sizerandomks)+cushionframes,
                                      **anim_kwargs)
        # an interesting note - saving the figure seems to be what takes time, so make sure not to if you don't have to.
        # ...but yknow, since that's the only way to see the result thru the terminal afaik... I assume you're going to.
        if savefig:
            if not overwrite:
                if glob.glob(savefilename) != []:
                    print('File already exists at', savefilename, '; and overwrite = False, so no gif saved.')       
                    if pltclose:
                        plt.close()
                else:
                    print(f'Saving to {savefilename}!')
                    pilwriter = animation.PillowWriter()
                    ani.save(savefilename, writer = pilwriter, **animsave_kwargs)
                    if pltclose:
                        plt.close()
            else:
                print(f'Saving to {savefilename}!')
                pilwriter = animation.PillowWriter()
                ani.save(savefilename, writer = pilwriter, **animsave_kwargs)
                if pltclose:
                    plt.close()
    else:
        return(update_internal)
    
def arange_ofsize(ar_min, ar_max, ar_size):
    """A tiny function that returns an np.arange between ar_min and ar_max with size ar_size. Saves a visually crowded line later in this script.
       Returns:
            return(np.arange(ar_min, ar_max, abs(ar_max-ar_min)/ar_size))
    """
    return(np.arange(ar_min, ar_max, abs(ar_max-ar_min)/ar_size))

if __name__ == '__main__':

    #ARGPARSE
    parser = argparse.ArgumentParser(description = "Script for running / exploring Fourier transforms and related analyses on TESS data given a file from TESS.")
    parser.add_argument('--runmode', default = 'explore', type = str, help = f"The mode in which to run this script. Default is 'explore', which will run all parts relating to the homework and the bonus parts. These are: splittabs (splitting the TESS data into tables corresponding to different sectors of the light curve), p1 (Running the chosen table from the split up observation - ie the chosen sector of the light curve - through a fourier analysis, and plotting its power spectrum and inverse fourier transform. Note that it WILL skip the zero term in the power spectrum unless you tell it not to with ps_inczero = True), p2 (Seeing how many coefficients are needed to get a decent fitting inverse FFT, by animating how the IFFT changes with more and more included coefficients, and also checking where the maximum coeffficient falls), p3 (Doing the previous two parts again but with a different sector of the observation), p4 (doing this with interpolated data to account for the gaps in TESS observations), and random (animating how the IFFT changes when using only a random sample of the FFT coefficients - though note the random seed is an argument in this script- and increasing the size of the sample until all coefficents are used). Enter the names of whatever parts you want to run as a string. ")    
    parser.add_argument('--slowdown', default = 0, type = float, help=f"This script can use sleep(slowdown) to pause the script at certain points if desired to give you more time to read the printouts in terminal. Increase slowdown to pause longer.")
    parser.add_argument('--TICfile', default = 'tic0009727392.fits', help = f"the TESS observations file from which to pull data.")
    parser.add_argument('--splitat', default = "(0, 1500) (2000, 2250) (2260, 2550) (3000, 3250)", help = f"The places (indices along the time axis) to split up the TESS data into different sectors for analysis.")
    parser.add_argument('--ind_tabmain', type = int, default = 0, help = f"The index in the list of tables made by splitting up the TESS data to use for getting the 'main' table - the one that is used for p1/2 and p4.")
    parser.add_argument('--ps_inczero', default = 0,type = int,  help = f"Default of 0 = False (to protect against argparse acting weird about boolean values and save potential typos). Whether or not to include the k = 0 term (first coefficient in abs(c)**2) in the power spectrum plots generated by this function. Will only apply to the power spectrum plots that are generated automatically and not the ones that are part of the animations; otherwise the bottom graph of the animations wouldn't be very useful!")
    parser.add_argument('--allthestops', default = 1, type = int, help = 'Default of 1 = True. determining if this script will "pull out all the stops" and step through everything including extra animations.')
    parser.add_argument('--fnameprefix', default = '', help = 'Prefix to append to the filenames of any files saved by this script.')
    parser.add_argument('--ind_alttab', default = 3, type = int, help = 'Index of the tab to use when checking a different region of the light curve for comparison.')
    parser.add_argument('--ind_bonustab', default = 'default', help = "Index of the tab to use in the 'bonus' part of this script (runmode random), showing how the IFFT evolves with a larger and larger random sample of coefficients (sampled without replacement).")
    parser.add_argument('--interp_bonustab', default = 0, type = int, help = "Default of 0 = False - Whether or not to use the interpolation of the bonus table in the 'bonus' part of this script (runmode random)")
    parser.add_argument('--max_randomterms', default = 'default', help = "Maximum number of random terms to draw in the bonus 'random' runmode.")
    parser.add_argument('--num_randomframes', default = 50, help = "Number of frames to use in the gif generated by the bonus 'random' runmode. Will  Minimum will be 1 - for context, max_randomterms/num_randomframes is used as the stepsize in the array that is drawn from within the 'random' runmode animation function, and it cannot be less than 1. So if you have a mxa_randomterms of 5, it'll only make 5 frames. This is to prevent overlarge gifs from being unneccessarily generated.")
    parser.add_argument('--seed', default = 1234, help = "The random seed number to be used in the bonus 'random' runmode.")
    parser.add_argument('--overwrite', default = 1, type = int, help = "Default of 1 = True - Whether or not to overwrite the .gifs generated by this script.")

    args = parser.parse_args()
    print('Args:')
    print(args)
    print('\n\n')
    runmode = args.runmode
    slowdown = args.slowdown
    seed = args.seed
    TICfile = args.TICfile
    splitat_inp = args.splitat
    splitat = []
    splitat_inp = splitat_inp.strip("()").split(') (')
    for itm in splitat_inp:
        itm = itm.split(',')
        splitat.append(tuple((float(itm[0]), float(itm[1]))))
    ind_tabmain = args.ind_tabmain
    ps_inczero = args.ps_inczero
    allthestops = args.allthestops
    fnameprefix = args.fnameprefix
    ind_alttab = args.ind_alttab
    ind_bonustab = args.ind_bonustab
    if ind_bonustab == 'default':
        ind_bonustab = ind_tabmain
    else:
        ind_bonustab = int(ind_bonustab)
    interp_bonustab = args.interp_bonustab
    max_randomterms = args.max_randomterms
    num_randomframes = args.num_randomframes
    gif_overwrite = args.overwrite

    # handle the 'bools'
    def boolify(argval):
        # I bet this is something that can be done in one line with just bool(argval) to get the same effect.
        # but also, I don't want to take chances with how this is handled being different across different versions of any package, so I'm hardcoding this.
        if argval == 0:
            argval = False
        else:
            argval = True
        return(argval)

    ps_inczero = boolify(ps_inczero)
    allthestops = boolify(allthestops)
    interp_bonustab = boolify(interp_bonustab)
    gif_overwrite = boolify(gif_overwrite)

    tempfits = fits.open(TICfile)
    times = tempfits[1].data['times']
    fluxes = tempfits[1].data['fluxes']
    ferrs = tempfits[1].data['ferrs']
    TICtab = Table.read(tempfits).to_pandas()
    tempfits.close()
    

    runopts_dict = {'explore':ifanyofthesein(['all', 'explore', 'explor', 'ex', 'run all'], runmode.lower()), 
                    'run_splittabs':ifanyofthesein(['splittabs', 'split', 'run_splittabs', 'run splittabs'], runmode.lower()),
                    'run_p1':ifanyofthesein(['p1', 'runp1', 'run p1', 'run_p1', 'p1_run'], runmode.lower()), 
                    'run_p2':ifanyofthesein(['p2', 'runp2', 'run p2', 'run_p2', 'p2_run'], runmode.lower()), 
                    'run_p3':ifanyofthesein(['p3', 'runp3', 'run p3', 'run_p3', 'p3_run'], runmode.lower()), 
                    'run_p4':ifanyofthesein(['p4', 'runp4', 'run p4', 'run_p4', 'p4_run'], runmode.lower()), 
                    'random_k':ifanyofthesein(['random', 'rand', 'rand_k', 'random_k'], runmode.lower())
                   }
    
    if runopts_dict['explore']:
        for ky in runopts_dict:
            runopts_dict[ky] = True

    # these need to run regardless:
    TICtablist, TICtabsplits = manual_split_on_gap(TICtab, 'times', splitat, True)
    TAB_MAIN = TICtablist[ind_tabmain]
    TAB_ALT = TICtablist[ind_alttab]
    

    # print intro
    print('\n\n')
    print('***'*10)
    print("Welcome to my script for exploring Fourier Transforms! Fair warning - this script makes a number of animations, and the resulting .gifs can get large. To avoid giving the user the chance to accidentally freeze up the script by trying to make a gif with 10000 frames, I've hardcoded more things than I usually do in the script itself, so there are a number of options to customize the gifs in the functions that I made than I actually use. If you want to explore the functions used in this script with more flexibility, I suggest using this more like a module than a script and importing some of this into a Jupyter notebook to play around with.")
    print('***'*10)
    print('\n\n')
    sleep(slowdown)
    
    #run_splittabs:
    if runopts_dict['run_splittabs']:
        print('Split the observation up into different sections based on the splitat arg, then see what the results look like, and make tables for each section. I have already put in my choice of places to split this data but feel free to play around! You should see four different colored sections, with black edges. If there are any fully black dots, something has gone wrong with the table splitting! Note the blue dashed lines marking the indices where this got split up; they look like the dots are overlapping the lines, but it is just an effect of the marker sizes.')
        sleep(slowdown)
        fig = plt.figure(layout = 'constrained', figsize = (7, 6.5))
        sf1, sf2 = fig.subfigures(2, 1, height_ratios=[0.42, 1])
        sf1.set(facecolor='whitesmoke')
        sf2.set(facecolor='whitesmoke')
        colorsarr = ['c','m','y','lime']
        ax1 = sf1.add_subplot(111)
        ax1.scatter(times, fluxes, c = 'black', s = 36)
        for ind, ttab in enumerate(TICtablist):
            ax1.scatter(ttab['times'], ttab['fluxes'], marker = 'o', c = colorsarr[ind], s= 20, label= f'Table {ind}')
            ax1.set(xlabel='times', ylabel='fluxes')
        for indval in TICtabsplits:
            ax1.axvline(times[indval[0]], ls = '--')
            ax1.axvline(times[indval[1]], ls = '--')
            
        for ind, ttab in enumerate(TICtablist):
            ax = sf2.add_subplot(2, 2, ind+1)
            ax.scatter(ttab['times'], ttab['fluxes'],  c= colorsarr[ind])
            ax.set(xlabel='times', ylabel='fluxes', title = f'Table {ind}')
            if ind < len(TICtabsplits):
                ax.axvline(times[TICtabsplits[ind][0]])
                ax.axvline(times[TICtabsplits[ind][1]])
        
        ax1.set_title(f'{TICfile} observation')
        plt.show()
        plt.close()
        print("I like the look of the first section best - it doesn't have a super huge gap, or repeated gaps. You can change this with the ind_tabmain argument, but by default this script will use the first table (index 0) generated by the default splitting.")

        
    #run_p1
    if runopts_dict['run_p1']:
        sleep(slowdown)
        print(f'\n\nTask: having picked a section of the observation, run it through the Fourier transform, then plot its power spectrum. \n For our chosen table, get its Fourier transform and power spectrum:\n')
        FFT_MAIN = FFT_Objects(TAB_MAIN['fluxes'], TAB_MAIN['times'])
        if not ps_inczero:
            print("Plot the power spectrum - note, this is NOT including the first term in c!!")
        else:
            print("Plot the power spectrum - note, this IS including the first term in c, so that's pretty much all you're going to see!!")
        FFT_MAIN.plot_powerspec(ax_kwargs={'title':f'TIC {TICfile} Table {ind_tabmain} Power Spectrum'}, include_zero_term=ps_inczero)
        print("Plot the inverse fourier transform, just to see!!")
        FFT_MAIN.plot_ifft(ax_kwargs={'title':f'TIC {TICfile} Table {ind_tabmain} Inverse FFT', 'xlabel':'times', 'ylabel':'fluxes'})
        print('Cool!!')
        plt.close()
   
    #run_p2
    if runopts_dict['run_p2']:
        sleep(slowdown)
        if not runopts_dict['run_p1']:
            FFT_MAIN = FFT_Objects(TAB_MAIN['fluxes'], TAB_MAIN['times'])
        print(f"\n\nNow: see how many coefficients are needed to get a decent fit.")
        print(f"We can always just see what the argmax of the abs(c)**2 is (ignoring the zero term, of course!!):")
        print(f"         c_sq.argmax()  = {np.argmax(FFT_MAIN.c_sq)} (which would give us a period of about {2*np.argmax(FFT_MAIN.c_sq)*np.pi}).")
        print(f"But, that's not as fun as what we COULD do - check this with animation!!!\n\n")
        if allthestops:
            print(f"Let's see what this looks like running through ALL the k values, though of course using some step size so it doesn't take forever. These .gifs can get pretty big - also, on mac, you should be able to select the file in Finder and hit space and the .gif will play. You may need to wait a second for them to save!!\n\n")
            sleep(slowdown)
            allk_main_gifname = f"{fnameprefix}increase_k_full.gif"
            anim_up_in_k_fft(FFT_MAIN, savefilename = allk_main_gifname, framestepsize=int(np.round((len(FFT_MAIN.k)/100)-0.5)), savefig = True, overwrite = gif_overwrite) # make the frame step size a bit under or even with whatever it takes to put this at 100 frames
            print(f"Go look at that, even step through it frame by frame with preview if you're on mac, and see how the rate of improvement changes with which part of k we're stepping over. \n Note how it looks pretty alright fairly quickly, and interestingly, when it goes through the last chunk of k values it starts spiking wildly before settling back down to match the np.fft.ifft result!!\n\n\n\n")
            sleep(slowdown)
        print(f"Now, let's make an animation that looks over k just going a little past where the maximum c_squared coefficient was - let's say by 10 points in k.\n\n")
        if np.argmax(FFT_MAIN.c_sq) < 100:
            use_framestepsize = 1
        else:
            use_framestepsize = int(np.round((np.argmax(FFT_MAIN.c_sq)/100)-0.5))
        anim_up_in_k_fft(FFT_MAIN, savefilename = f"{fnameprefix}increase_k_cut.gif", frameindices=np.arange(0, np.argmax(FFT_MAIN.c_sq)+10, use_framestepsize), framestepsize=use_framestepsize, savefig=True, overwrite = gif_overwrite)
        plt.close() # just in case above fn didnt write a figure, need this
        print(f"How does that look? You should see that at the exact value where we had the peak c_sq, we can start to see the dips in the sinusoidal ifft sync up with the transits. However, as it increases the number of coefficients, it gets closer and closer to replicating the 'real' answer from np.fft.ifft in orange, with its sharp dips in brightness. If you let it go all the way to the end of k, it'll be a perfect match for the orange line.\n\n\n\n\n\n")
    
    # run_p3
    # IN PRINCIPLE, i am very very aware that I could save a ton of code space by making all of the run_1 and run_2 sections mini functions; however, it's late and that's a timesink of debugging that isn't necessary right now. Sometime maybe!!
    if runopts_dict['run_p3']:
        sleep(slowdown)
        print(f"Try this for a different region of the light curve - how about the fourth section (table index 3)??? You can change this secondary table index with ind_alttab (default 3, currently {ind_alttab}) but I like this one bc it also has small gaps, but more of them.\n\n")
        
        FFT_ALT = FFT_Objects(TAB_ALT['fluxes'], TAB_ALT['times'])
        if not ps_inczero:
            print("Plot the power spectrum - note, this is NOT including the first term in c!!")
        else:
            print("Plot the power spectrum - note, this IS including the first term in c, so that's pretty much all you're going to see!!")
        FFT_ALT.plot_powerspec(ax_kwargs={'title':f'TIC {TICfile} Table {ind_alttab} Power Spectrum'}, include_zero_term = ps_inczero)
        print("Plot the inverse fourier transform, just to see!!")
        FFT_ALT.plot_ifft(ax_kwargs={'title':f'TIC {TICfile} Table {ind_alttab} Inverse FFT', 'xlabel':'times', 'ylabel':'fluxes'})
        print(f'Cool!! \n\n Now, explore k:')
        print(f"We can always just see what the argmax of the abs(c)**2 is (ignoring the zero term, of course!!):")
        print(f"         c_sq.argmax()  = {np.argmax(FFT_ALT.c_sq)} (which would give us a period of about {2*np.argmax(FFT_ALT.c_sq)*np.pi}).")
        print(f"But, that's not as fun as what we COULD do - check this with animation!!!\n\n\n\n")
        if allthestops:
            print(f"Let's see what this looks like running through ALL the k values, though of course using some step size so it doesn't take forever. These .gifs can get pretty big - also, on mac, you should be able to select the file in Finder and hit space and the .gif will play. You may need to wait a second for them to save!!")
            sleep(slowdown)
            allk_main_gifname = f"{fnameprefix}increase_k_full_ALT.gif"
            anim_up_in_k_fft(FFT_ALT, savefilename = allk_main_gifname, framestepsize=int(np.round((len(FFT_ALT.k)/100)-0.5)), savefig = True, overwrite = gif_overwrite) # make the frame step size a bit under or even with whatever it takes to put this at 100 frames
            print(f"Go look at that, even step through it frame by frame with preview if you're on mac, and see how the rate of improvement changes with which part of k we're stepping over. \n Note how it looks pretty alright fairly quickly, and interestingly, when it goes through the last chunk of k values it starts spiking wildly before settling back down to match the np.fft.ifft result!!\n\n\n\n")
            sleep(slowdown)
        print(f"Now, let's make an animation that looks over k just going a little past where the maximum c_squared coefficient was - let's say by 10 points in k.\n\n")
        if np.argmax(FFT_ALT.c_sq) < 100:
            use_framestepsize = 1
        else:
            use_framestepsize = int(np.round((np.argmax(FFT_ALT.c_sq)/100)-0.5))
        anim_up_in_k_fft(FFT_ALT, savefilename = f"{fnameprefix}increase_k_cut_ALT.gif", frameindices=np.arange(0, np.argmax(FFT_ALT.c_sq)+10, use_framestepsize), framestepsize=use_framestepsize, savefig=True, overwrite = gif_overwrite)
        plt.close() # just in case above fn didnt write a figure, need this
        
        print(f"How does that look? You should see that at the exact value where we had the peak c_sq, we can start to see the dips in the sinusoidal ifft sync up with the transits. However, as it increases the number of coefficients, it gets closer and closer to replicating the 'real' answer from np.fft.ifft in orange, with its sharp dips in brightness. If you let it go all the way to the end of k, it'll be a perfect match for the orange line.\n\n")
        print("\n\n ...point being: in this case, it's pretty similar to the first example, EXCEPT that the k of the maximum abs(c)^2 was much higher. You can see in the animation that was just made how the behavior of the eclipsing binary (with the dips) is captured by the fourier analysis fairly quickly, it takes a lot of coefficients to match the actual amplitude of the TESS observation's fluctuations in flux. ")        

    # run_p4
    if runopts_dict['run_p4']:
        sleep(slowdown)
        print(f"Interpolation time!!! Fill in the missing timesteps (pretending that we could have gotten perfectly evenly spaced data across the sample region).\n\n")
        
        if True not in [runopts_dict['run_p1'], runopts_dict['run_p2']]:
            FFT_MAIN = FFT_Objects(TAB_MAIN['fluxes'], TAB_MAIN['times'])
        if not runopts_dict['run_p3']:
            FFT_ALT = FFT_Objects(TAB_ALT['fluxes'], TAB_ALT['times'])
            
        interp_arange = arange_ofsize(np.min(FFT_MAIN.x), np.max(FFT_MAIN.x), len(FFT_MAIN.x))
        INTERP_MAIN = np.interp(interp_arange, TAB_MAIN['times'], TAB_MAIN['fluxes'])
        INT_FFT_MAIN = FFT_Objects(INTERP_MAIN, interp_arange)
        if allthestops:
            print(f"You'll notice (at least I did with my default args) that the power spectrum looks a LOT different, with far fewer in the 'horns' at either side of the scatter plot.\n\n")
            sleep(slowdown)
            INT_FFT_MAIN.plot_powerspec(include_zero_term = ps_inczero)
            print(f"But, the actual fourier transform looks great when I put it back through the ifft!\n\n")
            sleep(slowdown)
            INT_FFT_MAIN.plot_ifft()
            sleep(slowdown)
        print(f"Let's put this side-by-side with the original:\n\n")
        # takes some doing.
        fig = plt.figure(figsize = (10, 7))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        use_framestepsize = int(np.round((len(FFT_MAIN.k)/100)-0.5))
        
        uf_1 = anim_up_in_k_fft(FFT_MAIN, RUNANIM = False, passfig = fig, passaxes=[ax1, ax3], framestepsize=use_framestepsize)
        uf_2 = anim_up_in_k_fft(INT_FFT_MAIN, RUNANIM=False, passfig = fig, passaxes=[ax2, ax4], ax1_set={'ylabel':'with numpy interp'}, framestepsize=use_framestepsize)
        
        def tempfn(framen):
            #print(framen)
            uf_1(framen)
            uf_2(framen)
        pilwriter = animation.PillowWriter()
        if glob.glob(f'{fnameprefix}runp4_compare_with_interp.gif') != []:
            print('File already exists at', f'{fnameprefix}runp4_compare_with_interp.gif', '; and overwrite = False, so no gif saved.')       
            plt.close()
        else:
            ani = animation.FuncAnimation(fig = fig,func = tempfn, frames = len(np.arange(0, len(FFT_MAIN.c) + use_framestepsize, use_framestepsize))+10)
            ani.save(f'{fnameprefix}runp4_compare_with_interp.gif', writer = pilwriter)
            plt.close() # just in case above fn didnt write a figure, need this

        print(f"Note the difference between the two - the one with interpolated data looks fairly similar, but eventualy an oscillation picks up at the right hand edge of where the gap was in the original data. If you go through frame by frame in preview, you might notice that the intense spiking of the inverse transform as we approach having all the coefficients included is a little stabler in the interpolated data.\n\n\n\n")
        sleep(slowdown)

    # BONUS - animate what happens if you get k values at random, without replacement, until you have all the coefficients.
    # random_k
    if runopts_dict['random_k']:
        print(f"Animate what happens if you include a larger and larger subset of the coefficients, picking coefficients randomly without replacement, until you have all the coefficients. You should see that in a lot of the cases it won't actually make a valid fourier transform at all (because it isn't getting tht \n\n")
        BONUS_TAB = TICtablist[ind_bonustab]
        BONUS_FFT = FFT_Objects(BONUS_TAB['fluxes'], BONUS_TAB['times'])
        if max_randomterms == 'default':
            max_randomterms = len(BONUS_FFT.c)
            stepsize = max_randomterms - len(BONUS_FFT.c)
        else:
            stepsize = len(max_randomterms)/num_randomframes
        if interp_bonustab:
            interp_arange = arange_ofsize(np.min(BONUS_FFT.x), np.max(BONUS_FFT.x), len(BONUS_FFT.x))
            BONUS_INTERP_MAIN = np.interp(interp_arange, TAB_MAIN['times'], TAB_MAIN['fluxes'])
            BONUS_INTERP_FFT = FFT_Objects(BONUS_INTERP_MAIN, interp_arange)
            if max_randomterms == 'default':
                max_randomterms = len(BONUS_INTERP_FFT.c)    
                stepsize = max_randomterms/num_randomframes
                anim_random_k_fft(BONUS_INTERP_FFT, sizerandomks = np.arange(0, max_randomterms, 1+(int(stepsize))), savefilename = f"{fnameprefix}interp_random_k.gif", framestepsize = 1, overwrite = gif_overwrite, RANDOMSEED = seed) # plus 1 because we are forcing the zero term to be there
            else:
                max_randomterms = int(int(max_randomterms)+0.5) # plus 0.5 to make sure it is rounded up
                stepsize = max_randomterms/num_randomframes
                anim_random_k_fft(BONUS_INTERP_FFT, sizerandomks = np.arange(0, max_randomterms, 1+(int(stepsize))), savefilename = f"{fnameprefix}interp_random_k.gif", framestepsize = 1, overwrite = gif_overwrite, RANDOMSEED = seed)
        else:
            if max_randomterms == 'default':
                max_randomterms = len(BONUS_FFT.c)    
                stepsize = max_randomterms/num_randomframes
                anim_random_k_fft(BONUS_FFT, sizerandomks = np.arange(0, max_randomterms, 1+(int(stepsize))), savefilename = f"{fnameprefix}random_k.gif", framestepsize = 1, overwrite = gif_overwrite, RANDOMSEED = seed)
            else:
                max_randomterms = int(int(max_randomterms)+0.5) # plus 0.5 to make sure it is rounded up
                stepsize = max_randomterms/num_randomframes
                anim_random_k_fft(BONUS_FFT, sizerandomks = np.arange(0, max_randomterms, 1+(int(stepsize))), savefilename = f"{fnameprefix}random_k.gif", framestepsize = 1, overwrite = gif_overwrite, RANDOMSEED = seed)
            #np.arange(0, len(c) + framestepsize, framestepsize)
        print(f"I suggest you go through and see what this looks like with interpolated data versus the original data!\n\n")
        plt.close() # just in case above fn didnt write a figure, need this


    if not runopts_dict['explore']:
        print("I suggest you try running this with runmode = explore to see all the possible things this script can do!")