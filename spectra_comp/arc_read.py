"""
This code is designed to convert a fits file arc to exctract line strengths/FWHM.
Date:03/07/22
"""
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.signal import find_peaks, peak_widths


def pull_data_from_spectrum(spectra):
    try:
        hdul = fits.open(spectra)
    except FileNotFoundError:
        print("Cannot find file {}".format(spectra))
        return None, None, None

    data = hdul[0].data
    hdr = hdul[0].header

    yyy = data[45]

    w = WCS(hdr, naxis=1, relax=False, fix=False)
    lam = w.wcs_pix2world(np.arange(len(yyy)), 0)[0]

    return lam, yyy, hdr


def spectrum_plot(spectra, ax, data_set):
    spec_x, spec_y, spec_header = pull_data_from_spectrum(spectra)
    if spec_y is None:
        return ax, None, None

    xxx = spec_x
    yyy = spec_y
    slit_width = spec_header['aperwid']

    ax.plot(xxx, yyy, label=data_set)
    return ax, yyy, xxx, slit_width


def get_peak_info(flux, wav):
    offset = -5
    peaks, peak_heights = find_peaks(flux, height=15, distance=10)
    pix_width, half_max, low_pix, high_pix = peak_widths(flux, peaks, rel_height=0.5)

    high_wav = np.interp(high_pix, np.arange(len(wav)), wav)
    low_wav = np.interp(low_pix, np.arange(len(wav)), wav)
    peak_data = []
    for i, peak in enumerate(peaks):
        print(wav[peak]+offset, flux[peak], high_wav[i] - low_wav[i])
        peak_dict = {'peak': wav[peak], 'max_flux': flux[peak], 'fwhm': high_wav[i] - low_wav[i]}
        peak_data.append(peak_dict)
    return peak_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--outpath", help="Output path for plots", type=str, default='')
    parser.add_argument("--path", help="base path to arc", type=str, default='')
    parser.add_argument("--title", help="Title for Plot", type=str, default='Normalized Spectra')
    args = parser.parse_args()
    path = args.path
    outpath = args.outpath
    title = args.title
    arc = 'test.fits'
    label = 'test'
    # if path and path[-1] != '/':
    #     path = path + '/'
    if outpath and outpath[-1] != '/':
        outpath = outpath + '/'

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, title=title)

    # arc = input("Arc Filename:")

    ax, flux, wav, slit_width = spectrum_plot(path, ax, data_set=label)

    peak_info = get_peak_info(flux, wav)
    print(f"slit width = {slit_width}")
    # for peak in peak_info:
    #     print(peak)
    # print(peak_info)

    ax.set_xlabel('Wavelength ($\AA$)')

    ax.legend()
    # plt.xlim([4000, 4500])
    plt.ylim([0, 150])
    plt.savefig(outpath+'temp.png')
    print('New spectroscopy plot saved to {}'.format(outpath+'temp.png'))
