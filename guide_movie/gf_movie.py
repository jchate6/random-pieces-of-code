"""
Creates a guide frame movie gif when given a series of guide frames
Based on code written by Adam Tedeschi for NeoExchange
Date: 8/10/2018

Edited 09/04 by Joey Chatelain
Edited 09/10 by Joey Chatelain -- fix Bounding boxes, clean bottom axis, add frame numbers
Edited 09/18 by Joey Chatelain -- print Request number rather than tracking number. Make Filename more specific.
Edited 10/09 by Joey Chatelain -- accomodate older guide frames (from May 2018)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs._wcs import InvalidTransformError
from astropy.visualization import ZScaleInterval
from datetime import datetime
import os
from glob import glob
import argparse


def make_gif(frames, title=None, sort=True, fr=333, init_fr=None):
    """
    takes in list of .fits guide frames and turns them into a moving gif.
    <frames> = list of .fits frame paths
    <title> = [optional] string containing gif title, set to empty string or False for no title
    <sort> = [optional] bool to sort frames by title (Which usually corresponds to date)
    <fr> = frame rate for output gif in ms/frame [default = 333 ms/frame or 3fps]
    output = savefile (path of gif)
    """
    if sort is True:
        fits_files = np.sort(frames)
    else:
        fits_files = frames
    path = os.path.dirname(frames[0]).lstrip(' ')

    start_frames = 5
    copies = 1
    if init_fr and init_fr > fr and len(fits_files) > start_frames:
        copies = init_fr // fr
        i = 0
        while i < start_frames * copies:
            c = 1
            while c < copies:
                fits_files = np.insert(fits_files, i, fits_files[i])
                i += 1
                c += 1
            i += 1

    # pull header information from first fits file
    with fits.open(fits_files[0], ignore_missing_end=True) as hdul:
        try:
            header = hdul['SCI'].header
        except KeyError:
            try:
                header = hdul['COMPRESSED_IMAGE'].header
            except KeyError:
                header = hdul[0].header
        # create title
        obj = header['OBJECT']
        try:
            rn = header['REQNUM'].lstrip('0')
        except KeyError:
            rn = 'UNKNOWN'
        try:
            site = header['SITEID'].upper()
        except KeyError:
            site = ' '
        try:
            inst = header['INSTRUME'].upper()
        except KeyError:
            inst = ' '

    if title is None:
        title = 'Request Number {} -- {} at {} ({})'.format(rn, obj, site, inst)

    fig = plt.figure()
    if title:
        fig.suptitle(title)

    def update(n):
        """ this method is required to build FuncAnimation
        <file> = frame currently being iterated
        output: return plot.
        """
        # get data/Header from Fits
        with fits.open(fits_files[n], ignore_missing_end=True) as hdul:
            try:
                header = hdul['SCI'].header
                data = hdul['SCI'].data
            except KeyError:
                try:
                    header = hdul['COMPRESSED_IMAGE'].header
                    data = hdul['COMPRESSED_IMAGE'].data
                except KeyError:
                    header = hdul[0].header
                    data = hdul[0].data
        # pull Date from Header
        try:
            date_obs = header['DATE-OBS']
        except KeyError:
            date_obs = header['DATE_OBS']
        date = datetime.strptime(date_obs, '%Y-%m-%dT%H:%M:%S.%f')
        # reset plot
        ax = plt.gca()
        ax.clear()
        ax.axis('off')
        z_interval = ZScaleInterval().get_limits(data)  # set z-scale
        try:
            # set wcs grid/axes
            wcs = WCS(header)  # get wcs transformation
            ax = plt.gca(projection=wcs)
            dec = ax.coords['dec']
            dec.set_major_formatter('dd:mm')
            dec.set_ticks_position('br')
            dec.set_ticklabel_position('br')
            dec.set_ticklabel(fontsize=10, exclude_overlapping=True)
            ra = ax.coords['ra']
            ra.set_major_formatter('hh:mm:ss')
            ra.set_ticks_position('lb')
            ra.set_ticklabel_position('lb')
            ra.set_ticklabel(fontsize=10, exclude_overlapping=True)
            ax.coords.grid(color='black', ls='solid', alpha=0.5)
        except InvalidTransformError:
            pass
        # finish up plot
        current_count = len(np.unique(fits_files[:n+1]))
        ax.set_title('UT Date: {} ({} of {})'.format(date.strftime('%x %X'), current_count, int(len(fits_files)-(copies-1)*start_frames)), pad=10)

        plt.imshow(data, cmap='gray', vmin=z_interval[0], vmax=z_interval[1])

        # If first few frames, add 5" and 15" reticle
        if current_count < 6 and fr != init_fr:
            circle_5arcsec = plt.Circle((header['CRPIX1'], header['CRPIX2']), 5/header['PIXSCALE'], fill=False, color='limegreen', linewidth=1.5)
            circle_15arcsec = plt.Circle((header['CRPIX1'], header['CRPIX2']), 15/header['PIXSCALE'], fill=False, color='lime', linewidth=1.5)
            ax.add_artist(circle_5arcsec)
            ax.add_artist(circle_15arcsec)
        return ax

    ax1 = update(0)
    plt.tight_layout(pad=4)

    # takes in fig, update function, and frame rate set to fr
    anim = FuncAnimation(fig, update, frames=len(fits_files), blit=False, interval=fr)

    savefile = os.path.join(path, obj.replace(' ', '_') + '_' + rn + '_guidemovie.gif')
    anim.save(savefile, dpi=90, writer='imagemagick')

    return savefile


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to directory containing .fits or .fits.fz files", type=str)
    parser.add_argument("--fr", help="Frame rate in ms/frame (Defaults to 100 ms/frame or 10 frames/second", default=100, type=float)
    parser.add_argument("--ir", help="Frame rate in ms/frame for first 5 frames (Defaults to 1000 ms/frame or 1 frames/second", default=1000, type=float)
    args = parser.parse_args()
    path = args.path
    fr = args.fr
    ir = args.ir
    print(fr)
    if path[-1] != '/':
        path = path + '/'
    files = np.sort(glob(path+'*.fits.fz'))
    if len(files) < 1:
        files = np.sort(glob(path+'*.fits'))
    if len(files) >= 1:
        gif_file = make_gif(files, fr=fr, init_fr=ir)
        print("New gif created: {}".format(gif_file))
    else:
        print("No files found.")


