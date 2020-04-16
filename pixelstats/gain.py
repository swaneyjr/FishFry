#!/usr/bin/env python3

#
# gain.py:  combines multiple files under different exposure and light conditions to determine the best fit gain for every pixel
#

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import sys
from unpack import *
from matplotlib.colors import LogNorm

# for stacking
sums       = []
ssqs       = []
means      = []
variances  = []

# pixel information
total_pixels   = 0
first_pixel    = 0
last_pixel     = 0

WIDTH   = 0
HEIGHT  = 0 

def process(filename, args, count):
    global pixels, first_pixel, last_pixel, \
           sums, ssqs, means, variances, \
           WIDTH, HEIGHT
    
    if (count == 0):
        total_pixels  = filename['pixels']
        first_pixel   = filename['firstpixel']
        last_pixel    = filename['lastpixel']
        WIDTH         = filename['width']
        HEIGHT        = filename['height']

        print('total pixels: ', total_pixels)
    
    print('processing file: ', filename['name'])
    
    # import
    sum = filename['sum']
    ssq = filename['ssq']
    num = filename['num']

    # compute means and variances, append to global lists
    mean      = sum / num
    variance  = (( ssq / num ) - ( (sum / num) **2 )) * ( num / (num - 1) )
    
    sums.append(sum)
    ssqs.append(ssq)
    means.append(mean)
    variances.append(variance)


# vertically stack arrays to condense information
def stack():
    sum_stack        = np.array([])
    ssq_stack        = np.array([])
    mean_stack       = np.array([])
    var_stack        = np.array([])

    sum_stack        = np.vstack(sums)
    ssq_stack        = np.vstack(ssqs)
    mean_stack       = np.vstack(means)
    var_stack        = np.vstack(variances) 
    
    print('      shapes of stacked numpy arrays:')
    print('sum:         ', sum_stack.shape)
    print('ssq:         ', ssq_stack.shape)
    print('mean:        ', mean_stack.shape)
    print('variance:    ', var_stack.shape)
    print('stacking completed')
    
    return sum_stack, ssq_stack, mean_stack, var_stack


# Best-fit line, assuming constant fractional uncertainty in the y value.
# --> uses a weighted regression TBDetermined
def fit_line(x, y):
    print('\nfitting line, divide by zero will occur')
    print('size of x: ', x.size)
    print('size of y: ', y.size)
   
    # weighted regression for calculating gain and intercept
    ex     = np.sum( np.divide(x, np.square(y), out=np.zeros_like(x), where=(x < 850)), axis=0)
    ey     = np.sum( np.divide(np.ones(y.shape), y, out=np.zeros_like(np.ones(y.shape)), where=(x < 850)), axis=0)
    exx    = np.sum( np.divide( np.square(x), np.square(y), out=np.zeros_like(x), where=(x < 850)), axis=0)
    exy    = np.sum( np.divide( x, y, out=np.zeros_like(x), where=(x < 850)), axis=0)
    ew     = np.sum( np.divide( np.ones(y.shape), np.square(y), out=np.zeros_like(np.ones(y.shape)), where=(x < 850)), axis=0)
    eyy = 1
    denom  = exx*ew - ex*ex
    
    a = np.divide(( exy*ew - ex*ey),  denom)
    b = np.divide(( exx*ey - exy*ex), denom)
    
    # using Pearson coefficient method to calculate R^2 values
    e_x  = np.mean(x, axis=0) 
    e_y  = np.mean(y, axis=0) 
    e_xx = np.mean(np.square(x), axis=0)
    e_yy = np.mean(np.square(y), axis=0)
    e_xy = np.mean((x*y), axis=0)
    
    r2_numer = (e_xy - e_x*e_y)**2
    r2_denom = (e_xx - e_x**2) * (e_yy - e_y**2) 
    r2 = np.divide(r2_numer, r2_denom)

    print('line fitted. \nall intermediate values computed:')
    print('ex:      ', ex)
    print('ex.shape ', ex.shape)
    print('ey:      ', ey)
    print('exx:     ', exx)
    print('exy:     ', exy)
    print('ew:      ', ew)
    print('denom:   ', denom)
    print('a:       ', a)
    print('b:       ', b)
    print('R^2:       ', r2)
    print('min: ', r2[np.logical_not(np.isnan(r2))].min())
    print('max: ', r2[np.logical_not(np.isnan(r2))].max())
    print()
    
    return a, b, r2



def fit_gain(args, sum_stack, ssq_stack, mean_stack, var_stack, count):    
    
    gain, intercept, r_squared = fit_line(mean_stack, var_stack) 
    print('\nobtained gain and intercept')
    
    if args.sandbox:
        print('========================================')
        print('-- in sandbox development environment -- ')
        print('========================================')
        print()

        attributes('gain', gain)
        attributes('intercept', intercept)
        attributes('R squared', r_squared)
        
        fig, ax1 = plt.subplots()
        img1 = ax1.imshow(r_squared.reshape(HEIGHT, WIDTH))
        plt.colorbar(img1, ax=ax1)
        
        fig, ax2 = plt.subplots()
        img2 = ax2.imshow(gain.reshape(HEIGHT, WIDTH))
        plt.colorbar(img2, ax=ax2)
        plt.show()

        print('plotting')


    # computing R^2 value
    if args.rsquared:
        print('\ncomputing R^2 value')
        ### more to come

    # commit computed information to .npz files
    if (args.commit):
        print('saving fit line results')
        full_gain       = np.array(gain)
        full_intercept  = np.array(intercept)
        full_means      = np.array(mean_stack)
        full_vars       = np.array(var_stack)
        full_count      = np.full(full_gain.size, count, dtype=int)
        
        print('done. \ndumping information to be committed:')
        print('full_gain.shape:      ', full_gain.shape)
        print('full_gain.size:       ', full_gain.size)
        print('full_intercept.shape: ', full_gain.shape)
        print('full_intercept.size:  ', full_gain.size)
        print('full_count.shape:     ', full_count.shape)
        print('full_count.size:      ', full_count.size)
        print('full_means.shape:     ', full_means.shape)
        print('full_means.size:      ', full_means.size)
        print('full_vars.shape:      ', full_vars.shape)
        print('full_vars.size:       ', full_vars.size)
        
        np.savez("calib/gain.npz", gain=full_gain, intercept=full_intercept)
        np.savez("calib/gain_points.npz", count=full_count, means=full_means, vars=full_vars)

    return gain, intercept


### in progress
def make_plots(args, gain, intercept, mean_stack, var_stack):
    print('making plots (in progress)')
    # find array of temporary max values, then find true maximums
    temp_xmax = np.max(mean_stack, axis=1)
    temp_ymax = np.max(var_stack,  axis=1)
    
    xmax = np.max(temp_xmax)
    ymax = np.max(temp_ymax)
    attributes('temp_xmax', temp_xmax)
    attributes('temp_ymax', temp_ymax)

    print('xmax: ', xmax)
    print('ymax: ', ymax)

    plt.close()
    f, axes = plt.subplots(3, 3, sharex='col', sharey='row')
    axes = axes.flatten()

    nonzero = mean_stack != 0
    attributes('nonzero', nonzero)
    plotted = 0
   
    '''
    Notes: 
    1) gets stuck after running through first plot
    2) perhaps just try to make one individual plot?
    3) plot means and variances of just one pixel?
    4) find the original s6 file that would work with Mike's previous gain script, 
    then individually print out values so you can figure it out from there
    '''

    for i in nonzero:
        mean = mean_stack[1][i]
        var  = var_stack[1][i]
        print('mean.shape: ', mean.shape)
        print('mean.size: ', mean.size) 
        print('var.shape: ', var.shape)
        print('var.size: ', var.size) 
        perm = np.argsort(mean)
        mean = mean[perm]
        var  = var[perm]

        a = gain[i]
        b = intercept[i]

        attributes('a', a)
        attributes('b', b)

        #fx = np.array([0, xmax])
        #attributes('fx', fx)
        #fy = a + b

        #axes[plotted].plot(fx,fy,"--k")

        axes[plotted].plot(mean,var,"ob")
        axes[plotted].set_xlim(0,xmax)        
        axes[plotted].set_ylim(0,ymax)    
        axes[plotted].xaxis.set_major_locator(ticker.LinearLocator(3))
        axes[plotted].xaxis.set_minor_locator(ticker.LinearLocator(xmax/20+1))
        axes[plotted].yaxis.set_major_locator(ticker.LinearLocator(3))
        axes[plotted].yaxis.set_minor_locator(ticker.LinearLocator(ymax/100+1))
        axes[plotted].text(0.3*xmax,0.8*ymax,"pixel "+str(first_pixel+i),horizontalalignment='center')

        plotted += 1
        print('plotted: ', plotted)
        if (plotted == 1):
            break
    axes[3].set_ylabel("variance")
    axes[7].set_xlabel("mean")
    plt.savefig("plots/gain.pdf")
    plt.show()
    return

def analysis(args, sum_stack, ssq_stack, mean_stack, var_stack, count):
    gain, intercept = fit_gain( args, sum_stack, ssq_stack, mean_stack, var_stack, count )
    print('fitted gain complete.')
    if args.plot:
        make_plots(args, gain, intercept, mean_stack, var_stack)
        
if __name__ == "__main__":
    example_text = '''examples:

    ...'''
    
    parser = argparse.ArgumentParser(description='Fit gain of each pixel from a series of runs at different exposures and light levels', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', metavar='FILE', nargs='+', help='file to process')
    parser.add_argument('--commit',action="store_true", help="save calibration results to calibration directory")
    parser.add_argument('--pixel_range',type=int,nargs=2, metavar=("MIN","MAX"),help="Only evalulate pixels with MIN <= index < MAX", default=[0,0])
    parser.add_argument('--max_var',type=float,metavar="X",help="maximum variance in plots",default=800)
    parser.add_argument('--max_mean',type=float,metavar="Y",help="maximum mean in plots",default=200)
    
    parser.add_argument('--plot', action="store_true", help="make gain plots")
    parser.add_argument('--rsquared', action="store_true", help="plot R^2 value")
    parser.add_argument('--sandbox', action="store_true", help="experimental code")
    
    args = parser.parse_args()
    
    count = 0
    print('arguments accepted \nbeginning to process: ', args.files)
    for f in args.files:
        filename = np.load(f)
        process(filename, args, count)
        count += 1
    
    # after completed processing, stack and plot
    print('processing complete. \ntotal files processed: ', count)
    print('initializing stacking...')
    sum_stack, ssq_stack, mean_stack, var_stack = stack()
    analysis(args, sum_stack, ssq_stack, mean_stack, var_stack, count)
