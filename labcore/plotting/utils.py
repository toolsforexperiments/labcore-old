import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec, cm
from matplotlib.colors import rgb2hex


# color management tools
def get_color_cycle(n, colormap, start=0., stop=1., format='hex'):
    if type(colormap) == str:
        colormap = getattr(cm, colormap)

    pts = np.linspace(start, stop, n)
    if format == 'hex':
        colors = [rgb2hex(colormap(pt)) for pt in pts]
    return colors