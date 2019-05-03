"""
This code is designed to calculate extent of fringing in a trace.
Date:2019/04/19
"""

from astropy.io import fits
from astropy.wcs import WCS
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.signal import hilbert, butter, filtfilt
import os
import subprocess
from glob import iglob
import argparse
from datetime import datetime
from itertools import chain
import csv


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', time_in=None):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        time_in     - Optional  : if given, will estimate time remaining until completion (Datetime object)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    if time_in is not None:
        now = datetime.now()
        delta_t = now-time_in
        delta_t = delta_t.total_seconds()
        total_time = delta_t/iteration*(float(total)-iteration)
        # print(total_time, delta_t, iteration, float(total))
        if total_time > 90:
            time_left = '| {0:.1f} min remaining |'.format(total_time/60)
        elif total_time > 5400:
            time_left = '| {0:.1f} hrs remaining |'.format(total_time/60/60)
        else:
            time_left = '| {0:.1f} sec remaining |'.format(total_time)
    else:
        time_left = ' '
    print('\r%s |%s| %s%%%s%s' % (prefix, bar, percent, time_left, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def pull_data_from_spectrum(spectra):
    hdul = fits.open(spectra)
    # print(hdul.info())

    data = hdul[0].data
    hdr = hdul[0].header

    hdr['OBJECT'] = hdr['OBJECT'].replace(' ', '_')

    yyy = np.array(data[0][0])

    if "10^20" in hdr['BUNIT']:
        spec_y = yyy / (10**20)
    else:
        spec_y = yyy

    w = WCS(hdr, naxis=1, relax=False, fix=False)
    lam = w.wcs_pix2world(np.arange(len(spec_y)), 0)[0]
    return lam, spec_y, hdr


def smooth(x, window_len=10, window='hanning'):
    """smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    see also:
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if len(x) < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        x = np.asarray(x)
        return x

    if window_len % 2 != 0:
        window_len += 1

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')

    return y[(window_len // 2 - 1):-(window_len // 2)]


def get_best_y_pos(ax, avoid1=[], avoid2=[]):
    y_lim = ax.get_ylim()
    y_lim_lower = y_lim[0]
    y_lim_upper = y_lim[1]

    third_upper = y_lim_upper / 3
    third_lower = y_lim_lower / 3

    if abs(third_upper - np.mean(avoid1)) > third_upper / 2 and abs(third_upper - np.mean(avoid2)) > third_upper / 2:
        return third_upper
    elif abs(abs(third_lower) - np.mean(avoid1)) > abs(third_lower) / 2 and abs(abs(third_lower) - np.mean(avoid2)) > abs(third_lower) / 2:
        return third_lower
    else:
        return y_lim_lower / 2


def header_2_dict(header):
    keys = list(header.keys())
    values = list(header.values())
    hdr_dict = {}
    for i, key in enumerate(keys):
        hdr_dict[key] = values[i]

    return hdr_dict


def fringe_scale(spectra, pl, directory):
    spec_x, spec_y, header = pull_data_from_spectrum(spectra)
    windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    plot_lam = [5000, 10000]
    noise_lam = [5000, 6500]
    fringe_lam = [8000, 9500]
    xxx = spec_x
    plot_range = [j for j, x in enumerate(xxx) if plot_lam[0] < x < plot_lam[1]]
    noise_range = [j for j, x in enumerate(xxx) if noise_lam[0] < x < noise_lam[1]]
    fringe_range = [j for j, x in enumerate(xxx) if fringe_lam[0] < x < fringe_lam[1]]

    ref_blu = smooth(spec_y, 10, windows[1])
    ref_gre = smooth(spec_y, 20, windows[1])
    ref_yel = smooth(spec_y, 80, windows[1])
    ref_red = smooth(spec_y, 100, windows[1])
    ref_ird = smooth(spec_y, 150, windows[1])

    ref_y = np.concatenate((ref_blu[:int(noise_range[0])], ref_gre[noise_range], ref_yel[noise_range[-1]+1:fringe_range[0]], ref_red[fringe_range], ref_ird[fringe_range[-1]+1:]), axis=None)
    yyy = [(s - a) for s, a in zip(spec_y, ref_y)]
    xxx = np.array(xxx)

    box = 5
    smoothy = smooth(yyy, box, windows[1])

    find_g = [j for j, x in enumerate(xxx) if 5400 < x < 5600]
    smoothy = (smoothy / np.mean(spec_y[find_g]))

    env_reg = smoothy
    find_red_cut = [j for j, x in enumerate(xxx) if 10400 < x]
    env_reg[find_red_cut] = 0
    he = homomorphic_envelope(env_reg, fs=800)

    info_dict = header_2_dict(header)
    info_dict['FRINGE'] = np.mean(he[fringe_range])/np.mean(he[noise_range])
    info_dict['GRN-AVG'] = np.mean(spec_y[find_g])
    info_dict['SPCNOISE'] = np.mean(he[noise_range])

    if pl:
        plot_title = "Fringe Scale for {} on {}".format(header['OBJECT'], header['DAY-OBS'])
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1, title=plot_title)
        f, (ax1, ax2) = plt.subplots(2, figsize=(7, 7), sharex='col', gridspec_kw={'height_ratios': [1, 1]})
        f.subplots_adjust(hspace=0)
        ax1.set_title(plot_title)

        # plot Spectrum
        ax1.plot(xxx[plot_range], smooth(spec_y[plot_range], box, windows[1]), label='Spectrum')
        ax1.plot(xxx[plot_range], ref_y[plot_range], label='Smoothed')

        ax1.set_ylabel('Flux ($erg\; cm^{-2}\; s^{-1}\; \AA^{-1} $)')
        ax1.legend()

        # Plot Fringe
        ax2.plot(xxx[plot_range], xxx[plot_range] * 0, color='k')
        ax2.plot(xxx[plot_range], smoothy[plot_range], label="Residual")

        color = next(ax2._get_lines.prop_cycler)['color']
        ax2.plot(xxx[plot_range], he[plot_range], color=color, label="Envelope")
        ax2.plot(xxx[plot_range], -he[plot_range], color=color)

        y_annote = get_best_y_pos(ax2, he[noise_range], he[fringe_range])

        # annotate Noise Region
        ax2.text(np.mean(noise_lam), y_annote, 'Noise',
                 {'color': 'k', 'ha': 'center', 'va': 'bottom',
                 'bbox': dict(boxstyle="round", fc="w", ec="w", pad=0.1, alpha=0.75)})

        ax2.annotate("", xy=(noise_lam[0], y_annote), xycoords='data', xytext=(noise_lam[1], y_annote),
                     textcoords='data', arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))

        # annotate Fringe Region
        ax2.text(np.mean(fringe_lam), y_annote, 'Fringe',
                 {'color': 'k', 'ha': 'center', 'va': 'bottom',
                 'bbox': dict(boxstyle="round", fc="w", ec="w", pad=0.1, alpha=0.75)})

        ax2.annotate("", xy=(fringe_lam[0], y_annote), xycoords='data', xytext=(fringe_lam[1], y_annote),
                     textcoords='data', arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))

        # Print Fringe Ratio
        ax2.text(noise_lam[0], ax2.get_ylim()[0], 'Fringe/Noise Ratio: {:.2f}'.format(info_dict['FRINGE']),
                 {'color': 'k', 'ha': 'left', 'va': 'bottom'})

        ax2.set_ylabel('Residual Flux Contribution of Noise [(F+N)/S]')
        ax2.set_xlabel('Wavelength ($\AA$)')
        ax2.legend()

        plt.savefig('{}/{}_{}_{}.png'.format(directory, header['OBJECT'], header['DAY-OBS'], header['REQNUM'].lstrip('0')))
        plt.close('all')

    return info_dict


def homomorphic_envelope(x, fs=1000, f_lpf=1, order=3):
    """
    Computes the homomorphic envelope of x

    Args:
        x : array
        fs : float
            Sampling frequency. Defaults to 1000 Hz
        f_lpf : float
            Lowpass frequency, has to be f_LPF < fs/2. Defaults to 3
        order :  The order of the Butterworth filter
    Returns:
        time : numpy array
    """
    b, a = butter(order, 2 * f_lpf / fs, 'low')
    he = np.exp(filtfilt(b, a, np.log(np.abs(hilbert(x)))))
    return he


# def find_envelope(xxx, fringes, ax):
#     smth = 100
#     # stdev_out = stdev_run(fringes, smth)
#     # ax.plot(xxx, stdev_out+1, label=smth)
#     analytic_signal = hilbert(fringes-1)
#     amplitude_envelope = np.abs(analytic_signal)
#     ax.plot(xxx, amplitude_envelope+1)


def stdev_run(data, window_len=10):
    window_wid = int(window_len/2)
    i = 0
    end_len = len(data)

    if len(data) < window_wid:
        y = np.ones(end_len, 'd')*np.std(data)
        return y

    y = []
    for datum in data:
        window = [i - window_wid, i + window_wid]
        if window[0] < 0:
            window[0] = 0
        if window[1] > end_len - 1:
            window[1] = end_len - 1
        dat_block = data[window[0]:window[1]]
        y.append(np.std(dat_block))
        i += 1

    return np.array(y)


def dmstodegrees(value):
    if ":" not in str(value):
        raise ValueError
    el = value.split(":")
    deg = float(el[0])
    if deg < 0:
        sign = -1.
    else:
        sign = 1
    return deg + sign*float(el[1])/60. + sign*float(el[2])/3600.


def hmstodegrees(value):
    if ":" not in str(value):
        raise ValueError
    el = value.split(":")
    return float(el[0])*15 + float(el[1])/60. + float(el[2])/3600.


def listofdicts_to_dictoflists(ld, keys):
    dl = {}
    for key in keys:
        dl[key] = []
        for d in ld:
            try:
                value = float(d[key])
            except ValueError:
                try:
                    value = datetime.strptime(d[key], '%Y-%m-%dT%H:%M:%S.%f')
                except ValueError:
                    if key == 'RA':
                        value = hmstodegrees(d[key])
                    elif key == 'DEC':
                        value = dmstodegrees(d[key])
                    elif d[key] in ['UNKNOWN', 'Saturated']:
                        value = np.nan
                    else:
                        value = d[key]
            dl[key].append(value)
    return dl


def analyze_header_info(path, header_dict):
    comp_keys = ['exptime', 'altitude', 'date-obs', 'azimuth', 'RA', 'dec', 'siteid', 'rotangle', 'rotskypa', 'airmass',
                 'skymag', 'agfwhm', 'GRN-AVG', 'SPCNOISE', 'aperwid', 'FRINGE', 'SRCTYPE']
    other_keys = ['FRINGE', 'GRN-AVG', 'SPCNOISE', 'AMSTART', 'AMEND']
    comp_keys = [key.upper() for key in comp_keys]
    comp_lists = listofdicts_to_dictoflists(header_dict, comp_keys)
    other_lists = listofdicts_to_dictoflists(header_dict, other_keys)

    airmass_diff = [max([abs(e - s), abs(m - s), abs(e - m)]) for e, s, m in zip(other_lists['AMEND'], other_lists['AMSTART'], comp_lists['AIRMASS'])]
    comp_lists['AMDIFF'] = airmass_diff
    comp_keys += ['AMDIFF']

    color_trend = np.sqrt(np.array(comp_lists['FRINGE']))

    directory = path + 'fringe_analysis'
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, xkey in enumerate(comp_keys):
        for ykey in comp_keys[i+1:]:
            fig = plt.figure()

            plot_title = "{} vs {} ({} to {})".format(xkey, ykey, min(comp_lists['DATE-OBS']).date(), max(comp_lists['DATE-OBS']).date())
            print("Building plot for {}".format(plot_title))

            ax = fig.add_subplot(1, 1, 1, title=plot_title)
            cs = ax.scatter(comp_lists[xkey], comp_lists[ykey], c=color_trend)

            cbticks = range(int(max(color_trend)))
            cblabs = ['{}'.format(x ** 2) for x in cbticks]
            cbar = fig.colorbar(cs, ticks=cbticks)
            cbar.ax.set_ylabel('Fringe to Noise Ratio')
            cbar.ax.set_yticklabels(cblabs)


            ax.set_ylabel(ykey)
            ax.set_xlabel(xkey)
            x_min = min(comp_lists[xkey])
            x_max = max(comp_lists[xkey])
            if isinstance(x_min, float) and (abs(ax.get_xlim()[0] - x_min) > abs(x_min - 10*abs(x_min - x_max)) or abs(ax.get_xlim()[1] - x_max) > abs(x_max + 10*abs(x_max - x_min))):
                ax.set_xlim(min(comp_lists[xkey]) - .1 * abs(min(comp_lists[xkey]) - max(comp_lists[xkey])), max(comp_lists[xkey]) + .1 * abs(min(comp_lists[xkey]) - max(comp_lists[xkey])))

            y_min = min(comp_lists[ykey])
            y_max = max(comp_lists[ykey])
            if isinstance(y_min, float) and (abs(ax.get_ylim()[0] - y_min) > abs(y_min - 10*abs(y_min - y_max)) or abs(ax.get_ylim()[1] - y_max) > abs(y_max + 10*abs(y_max - y_min))):
                ax.set_ylim(min(comp_lists[ykey]) - .1 * abs(min(comp_lists[ykey]) - max(comp_lists[ykey])), max(comp_lists[ykey]) + .1 * abs(min(comp_lists[ykey]) - max(comp_lists[ykey])))


            if xkey == 'SPCNOISE':
                ax.set_xlim(-.01, .2)
                # ax.set_xscale('symlog')
            # ax.legend()

            plt.savefig('{}/{}_vs_{}.png'.format(directory, xkey, ykey))
            plt.close('all')
    subprocess.run(['montage', '-geometry', '+0+0', directory+'/*_*.png', directory+'/montage.png'], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to directory containing extracted fits traces of CSV files", type=str)
    parser.add_argument('-pl', '--plot', action="store_true", default=False, help='Make plots of individual traces in directory YYYY-MM-DD')
    parser.add_argument('-csv', '--analysis', action="store_true", default=False, help='Load CSV files and analyze results.')
    args = parser.parse_args()
    path = args.path
    pl = args.plot
    csv_test = args.analysis

    if path[-1] != '/':
        path = path + '/'

    print('Opening {} -- Searching for files...'.format(path))

    if csv_test:
        files = iglob(os.path.join(path, '*_object_fringe_data.csv'))

        header_dat = []
        for file in files:
            with open(file) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if float(row['FRINGE']) > 50 or float(row['FRINGE']) < 0.1:
                        row['FRINGE'] = 'Saturated'
                    header_dat.append(row)
        analyze_header_info(path, header_dat)

        for dat in header_dat:
            try:
                abs_fringe = float(dat['FRINGE']) * float(dat['SPCNOISE'])
                if abs_fringe > 0.25:
                    print(dat['ORIGNAME'], dat['OBJECT'], dat['DATE-OBS'], dat['FRINGE'], abs_fringe, dat['SPCNOISE'], dat['EXPTIME'])
            except ValueError:
                print(dat['ORIGNAME'], dat['OBJECT'], dat['DATE-OBS'], dat['FRINGE'], dat['SPCNOISE'], dat['EXPTIME'])
            # lst = datetime.strptime(dat['LST'], '%H:%M:%S.%f')
            # merid_time = datetime.strptime(dat['RA'], '%H:%M:%S.%f')
            # exptime = float(dat['EXPTIME'])
            # amstart = float(dat['AMSTART'])
            # amend = float(dat['AMEND'])
            # ammean = float(dat['AIRMASS'])
            # ha = lst - merid_time
            # if ha.total_seconds() < 0 and abs(ha.total_seconds()) < exptime:
            #     print(dat['OBJECT'], dat['ORIGNAME'])
            #     print(amstart, amend, ammean, np.mean([amstart, amend]), dat['ALTITUDE'], ha.total_seconds(), exptime)

    else:
        files1 = iglob(os.path.join(path, '**', '*_merge_*.fits'), recursive=True)
        files2 = iglob(os.path.join(path, '**', '*_redblu_*.fits'), recursive=True)
        files = chain(files1, files2)

        directory = str(datetime.utcnow().date())
        if not os.path.exists(directory):
            os.makedirs(directory)

        fringe_data = []
        n = 1
        for file in files:
            info = fringe_scale(file, pl, directory)
            fringe_data.append(info)
            print('\r--{} traces checked'.format(n), end='\r')
            n += 1

        if pl:
            subprocess.run(['montage', '-geometry', '+0+0', directory+'/*_*.png', directory+'/montage.png'], check=True)

        print("Analyzed {} traces.".format(n-1))

        if n > 1:
            csv_columns = list(fringe_data[0].keys())
            csv_file = '{}/{}_object_fringe_data.csv'.format(directory, n-1)
            with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns, restval='', extrasaction='ignore')
                writer.writeheader()
                for data in fringe_data:
                    writer.writerow(data)
            print('CSV file written: {}'.format(csv_file))
        else:
            print("No fits files found, no output made.")
