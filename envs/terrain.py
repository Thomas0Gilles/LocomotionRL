from scipy.stats import multivariate_normal
from scipy.signal import convolve2d
import matplotlib
try:
    matplotlib.pyplot.figure()
    matplotlib.pyplot.close()
except Exception:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os

# the colormap should assign light colors to low values
TERRAIN_CMAP = 'Greens'
DEFAULT_PATH = r'C:\Users\tgill\Documents\Scolarité\X\4A\GD_AI\MAP641\Project\height_fields'
STEP = 0.1

def generate_hills(width, height, nhills):
    '''
    @param width float, terrain width
    @param height float, terrain height
    @param nhills int, #hills to gen. #hills actually generted is sqrt(nhills)^2
    '''
    # setup coordinate grid
    xmin, xmax = -width/2.0, width/2.0
    ymin, ymax = -height/2.0, height/2.0
    x, y = np.mgrid[xmin:xmax:STEP, ymin:ymax:STEP]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    
    # generate hilltops
    xm, ym = np.mgrid[xmin:xmax:width/np.sqrt(nhills), ymin:ymax:height/np.sqrt(nhills)]
    mu = np.c_[xm.flat, ym.flat]
    sigma = float(width*height)/(nhills*8)
    for i in range(mu.shape[0]):
        mu[i] = multivariate_normal.rvs(mean=mu[i], cov=sigma)
    
    # generate hills
    sigma = sigma + sigma*np.random.rand(mu.shape[0])
    rvs = [ multivariate_normal(mu[i,:], cov=sigma[i]) for i in range(mu.shape[0]) ]
    hfield = np.max([ rv.pdf(pos) for rv in rvs ], axis=0)
    return x, y, hfield

def generate_holes(width, height, nholes, width_hole=5):
    '''
    @param width float, terrain width
    @param height float, terrain height
    @param nholes int, #holes to gen. 
    @param width_hole int, #width of holes to gen. 
    
    '''
    
    # setup coordinate grid
    xmin, xmax = -width/2.0, width/2.0
    ymin, ymax = -height/2.0, height/2.0
    x, y = np.mgrid[xmin:xmax:STEP, ymin:ymax:STEP]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    
    #generate holes
    hfield=np.zeros(x.shape)+100
    holes = np.random.randint(0,x.shape[0],nholes)
    
    for i in range(nholes):
        hole=holes[i]
        hfield[hole-width_hole:hole+width_hole,:]=0
    
    return x, y, hfield

#generate random terrain with holes, hurdles, slopes 
def generate_random_terrain(width, height, nholes, nhurdles, is_slope, nslopes, slope):
    '''
    @param width float, terrain width
    @param height float, terrain height
    @param nholes int, #holes to gen. 
    @param nhurdles int, #hurdles to gen.
    @param is_slope bool, whether the terrain is orientated or flat.
    @param nslopes int, #slopes to gen.
    @param slope float, value of the slopes.

    '''
    
    # setup coordinate grid
    xmin, xmax = -width/2.0, width/2.0
    ymin, ymax = -height/2.0, height/2.0
    x, y = np.mgrid[xmin:xmax:STEP, ymin:ymax:STEP]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    
    hfield=np.zeros(x.shape)+100
    
    #generate slope
    if is_slope:
        slope_limits = np.random.randint(0,x.shape[0],nslopes-1)
        np.sort(slope_limits)
        np.append(slope_limits,x.shape[0]-1)
        np.insert(slope_limits,0,0)
        
        slope_signs = np.random.randint(0,2,nslopes-1)
        for i in range(len(slope_signs)):
            if slope_signs[i]==0:
                slope_signs[i]=-1
        
        for i in range(len(slope_limits)-1):
            lim_inf=slope_limits[i]
            lim_sup=slope_limits[i+1]
            if lim_inf!=0:
                hfield[lim_inf,:]=hfield[lim_inf-1,:]
            for j in range(lim_inf+1,lim_sup):
                hfield[j,:]=slope_signs[i]*slope*(j-lim_inf)+hfield[lim_inf,:]
            
            lim_inf=slope_limits[i]
            lim_sup=slope_limits[i+1]
    
    #generate holes and hurdles
    holes = np.random.randint(0,x.shape[0],nholes)
    hurdles = np.random.randint(0,x.shape[0],nhurdles)
    
    for i in range(nholes):
        hole=holes[i]
        hfield[hole-1:hole+1,:]+=-20
    
    for i in range(nhurdles):
        hurdle=hurdles[i]
        hfield[hurdle-1:hurdle+1]+=5
    
    return x, y, hfield

def clear_patch(hfield, box):
    ''' Clears a patch shaped like box, assuming robot is placed in center of hfield
    @param box: rllab.spaces.Box-like
    '''
    if box.flat_dim > 2:
        raise ValueError("Provide 2dim box")
    
    # clear patch
    h_center = int(0.5 * hfield.shape[0])
    w_center = int(0.5 * hfield.shape[1])
    fromrow, torow = w_center + int(box.low[0]/STEP), w_center + int(box.high[0] / STEP)
    fromcol, tocol = h_center + int(box.low[1]/STEP), h_center + int(box.high[1] / STEP)
    hfield[fromrow:torow, fromcol:tocol] = 0.0
    
    # convolve to smoothen edges somewhat, in case hills were cut off
    K = np.ones((10,10)) / 100.0
    s = convolve2d(hfield[fromrow-9:torow+9, fromcol-9:tocol+9], K, mode='same', boundary='symm')
    hfield[fromrow-9:torow+9, fromcol-9:tocol+9] = s
    
    return hfield
    
def _checkpath(path_):
    if path_ is None:
        path_ = DEFAULT_PATH
    if not os.path.exists(path_):
        os.makedirs(path_)
    return path_
        
def save_heightfield(x, y, hfield, fname, path=None):
    '''
    @param path, str (optional). If not provided, DEFAULT_PATH is used. Make sure the path + fname match the <file> attribute
        of the <asset> element in the env XML where the height field is defined
    '''
    path = _checkpath(path)
    plt.figure()
    plt.contourf(x, y, -hfield, 100, cmap=TERRAIN_CMAP) # terrain_cmap is necessary to make sure tops get light color
    plt.axis('off')
    plt.savefig(os.path.join(path, fname), bbox_inches='tight', pad_inches = 0)
    plt.close()

def save_texture(x, y, hfield, fname, path=None):
    '''
    @param path, str (optional). If not provided, DEFAULT_PATH is used. Make sure this matches the <texturedir> of the
        <compiler> element in the env XML
    '''
    path = _checkpath(path)
    plt.figure()
    plt.contourf(x, y, -hfield, 100, cmap=TERRAIN_CMAP)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    # for some reason plt.grid does not work here, so generate gridlines manually
    for i in np.arange(xmin,xmax,0.5):
        plt.plot([i,i], [ymin,ymax], 'k', linewidth=0.1)
    for i in np.arange(ymin,ymax,0.5):
        plt.plot([xmin,xmax],[i,i], 'k', linewidth=0.1)
    plt.axis('off')
    plt.savefig(os.path.join(path, fname), bbox_inches='tight', pad_inches = 0)
    plt.close()
    
    

    
    
def demo():
    #x, y, hfield = generate_hills(60,60,10)
    #x, y, hfield = generate_holes(60,10,10,5)
    x, y, hfield = generate_random_terrain(60,60,10,10,True,10,0.1)
    save_heightfield(x,y,hfield,'random_terrain_test7.png')
    
demo()