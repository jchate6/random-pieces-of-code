"""
Plots a light curve for all ALCDEF data in a directory
Date: 2019/03/12

Still to do:
1.) finish planning capability
2.) find way to remove outliers.
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.animation import FuncAnimation
import numpy as np
import os
from glob import glob
import argparse
from math import floor, log10, ceil
from cycler import cycler
from matplotlib import container
from scipy import optimize
from scipy import signal
from datetime import datetime


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


def import_alcdef(file, meta_list, lc_list):
    lc_file = open(file)
    lines = lc_file.readlines()

    metadata = {}
    dates = []
    mags = []
    mag_errs = []
    met_dat = False

    for line in lines:
        if line[0] == '#':
            continue
        if '=' in line:
            if 'DATA=' in line and met_dat is False:
                chunks = line[5:].split('|')
                jd = float(chunks[0])
                mag = float(chunks[1])
                mag_err = float(chunks[2])
                dates.append(jd)
                mags.append(mag)
                mag_errs.append(mag_err)
            else:
                chunks = line.split('=')
                metadata[chunks[0]] = chunks[1].replace('\n', '')
        elif 'ENDDATA' in line:
            meta_list.append(metadata)
            lc_data = {
                'date': dates,
                'mags': mags,
                'mag_errs': mag_errs,
                }
            lc_list.append(lc_data)
            dates = []
            mags = []
            mag_errs = []
            metadata = {}
        elif 'STARTMETADATA' in line:
            met_dat = True
        elif 'ENDMETADATA' in line:
            met_dat = False

    filt_list = []
    for meta in meta_list:
        filt_list.append(meta['FILTER'])

    filt_unique = list(set(filt_list))
    filt_sets = {}
    for filt_u in filt_unique:
        filt_sets[filt_u] = []
        for i, filt in enumerate(filt_list):
            if filt == filt_u:
                filt_sets[filt_u].append(i)

    return meta_list, lc_list, filt_list


def phase_lc(lc_data, period, base_date):
    phase_list = []
    for lc in lc_data:
        phase = [(x - base_date) * 24 / period for x in lc['date']]
        phase = [x - x // 1 for x in phase]
        lc['date'] = phase
        phase_list.append(lc)
    return phase_list


def set_color_marker_cycle(ax, color=None):
    cycle_color = []
    if not color:
        rainbow = ax._get_lines.prop_cycler
        for c, color in enumerate(rainbow):
            cycle_color.append(color['color'])
            if c > 8:
                break
    else:
        cycle_color = color
    cycle_marker = ['o', 'X', 'v', '^', 's', '*', 'd', 'P', '<', '>']

    custom_cycler = (cycler(marker=cycle_marker) * cycler(color=cycle_color))
    ax.set_prop_cycle(custom_cycler)
    return ax


def get_name(meta_dat):
    name = meta_dat['OBJECTNAME']
    number = meta_dat['OBJECTNUMBER']
    desig = meta_dat['MPCDESIG']
    if name != 'False' and (number != 'False' and number != '0'):
        out_string = '{} ({})'.format(name, number)
    elif name != 'False':
        out_string = '{}'.format(name)
    elif number != 'False' and desig != 'False' and number != '0':
        out_string = '{} ({})'.format(number, desig)
    elif number != 'False' and number != '0':
        out_string = '{}'.format(number)
    elif desig != 'False':
        out_string = '{}'.format(desig)
    else:
        out_string = 'UNKNOWN OBJECT'
    return out_string, name, number


def build_figure():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.set_figheight(5)
    fig.set_figwidth(12)
    fig.tight_layout(pad=3, rect=(0, 0, .85, 1))
    set_color_marker_cycle(ax)
    return fig, ax


def plot_lightcurve(files, period, ax, plan_rots=None):

    if period and period <= 0:
        period = None

    meta_list = []
    lc_list = []
    for file in files:
        meta_list, lc_list, filt_list = import_alcdef(file, meta_list, lc_list)

    filt_unique = list(set(filt_list))
    filt_sets = {}
    for filt_u in filt_unique:
        filt_sets[filt_u] = []
        for i, filt in enumerate(filt_list):
            if filt == filt_u:
                filt_sets[filt_u].append(i)

    obj, name, num = get_name(meta_list[0])
    date_range = meta_list[0]['SESSIONDATE'].replace('-', '')+'-'+meta_list[-1]['SESSIONDATE'].replace('-', '')
    base_date = floor(min(sorted([jd for lc in lc_list for jd in lc['date']])))

    plot_title = "Light Curve for {}".format(obj)
    ax.set_title(plot_title)

    # pull out common date elements
    date_unique = list(set([meta['SESSIONDATE'] for meta in meta_list]))
    yyyy = date_unique[0][0:5]
    mm = date_unique[0][5:8]
    for d in date_unique:
        if yyyy and d[0:5] != yyyy:
            yyyy = ''
        if mm and d[5:8] != mm:
            mm = ''
    common_date = yyyy + mm

    # Set up space for legend
    tot_plots = len(filt_unique) + len(lc_list)
    leg_cols = ceil(tot_plots / 20)
    if len(common_date) > 5:
        leg_wid = 1.5
    elif len(common_date) > 1:
        leg_wid = 1.7
    else:
        leg_wid = 2.05
    fig.set_figwidth(10 + leg_wid * leg_cols)
    alpha = leg_wid/(10+leg_wid*leg_cols)
    fig.tight_layout(pad=4, rect=(0, 0, 1 - alpha*leg_cols, 1))

    offset = [0, .15, .21, .25, .29, 0, 0, 0, 0, 0, 0, 0]
    cov_out = []
    rainbow = ax._get_lines.prop_cycler
    coverage = 0
    for f, filt in enumerate(filt_unique):
        lc_filt_list = [lc_list[k] for k in filt_sets[filt]]
        meta_filt_list = [meta_list[k] for k in filt_sets[filt]]
        if period:
            lc_filt_list = phase_lc(lc_filt_list, period, base_date)
            for lc in lc_filt_list:
                for i, phase in enumerate(lc['date']):
                    if 0 < phase < .25:
                        lc['date'].append(phase+1)
                        lc['mags'].append(lc['mags'][i])
                        lc['mag_errs'].append(lc['mag_errs'][i])
                    elif 0.75 < phase < 1:
                        lc['date'].append(phase-1)
                        lc['mags'].append(lc['mags'][i])
                        lc['mag_errs'].append(lc['mag_errs'][i])

        if period:
            date_out = [d for lc in lc_filt_list for d in lc['date']]
        else:
            date_out = [(d - base_date) * 24 for lc in lc_filt_list for d in lc['date']]
        mag_out = [m+offset[i] for i, lc in enumerate(lc_filt_list) for m in lc['mags']]
        merr_out = [e for lc in lc_filt_list for e in lc['mag_errs']]

        if period:
            # Only try a fit if most of phase data present
            full_times = sorted(date_out)
            if full_times[-1] < full_times[0]+1:
                full_times.append(full_times[0]+1)
            time_diffs = []
            ot = None
            for time in full_times:
                if time < 0:
                    continue
                if ot is None:
                    ot = time
                    continue
                time_diffs.append(time - ot)
                ot = time
                if time > 1:
                    break
            coverage = 1 - sum([time - np.median(time_diffs) for time in time_diffs if time > (np.median(time_diffs) + 3 * np.std(time_diffs))])
            cov_out.append(coverage)

            if coverage > 0.1:
                mag_out = [x for _, x in sorted(zip(date_out, mag_out))]
                date_out.sort()

                # adjust fit order based on coverage.
                order = 1
                if coverage > 0.3:
                    order = 2
                if coverage > 0.5:
                    order = 3
                if coverage > 0.7:
                    order = 4
                if coverage > 0.90:
                    order = 5
                if coverage > 0.95:
                    order = 6

                # date_out_fit = date_out + [d+1 for d in date_out] + [d-1 for d in date_out]
                # mag_out_fit = mag_out + mag_out + mag_out
                # print(date_out_fit)
                params, params_covariance = optimize.curve_fit(fourier, date_out, mag_out,
                                                               [np.mean(mag_out)]+[1.0, 1.0, 1.0]*order,
                                                               bounds=([-100]+[-np.inf, -np.inf, -np.inf]*order, [100]+[np.inf, np.inf, np.inf]*order),
                                                               maxfev=100000)
                order_suffix = 'th'
                if order == 3:
                    order_suffix = 'rd'
                elif order == 2:
                    order_suffix = 'nd'
                elif order == 1:
                    order_suffix = 'st'
                t = np.linspace(-.1, 1.1, 300)

        if len(filt_unique) > 1:
            if 'I' in filt.upper():
                set_color_marker_cycle(ax, color=['k'])
            elif 'G' in filt.upper():
                set_color_marker_cycle(ax, color=['limegreen'])
            elif 'R' in filt.upper():
                set_color_marker_cycle(ax, color=['firebrick'])
            else:
                set_color_marker_cycle(ax, color=[next(rainbow)['color']])

        fit_median = 0
        if period and coverage > 0.1:
            if len(filt_unique) > 1:
                fit_median = np.median(fourier(t, *params))
            x_fit = fourier(t, *params) - fit_median
            ax.plot(t, x_fit, marker=',', label='{}{} order fit ({})'.format(order, order_suffix, filt))

        # Build y-axis boundaries
        y_max = -1000
        y_min = 1000
        buffer = (max(mag_out) - min(mag_out)) * .05
        if buffer < np.mean(lc_list[0]['mag_errs']):
            buffer = np.mean(lc_list[0]['mag_errs'])
        if max(mag_out) + buffer - fit_median > y_max:
            y_max = max(mag_out) + buffer - fit_median
        if min(mag_out) - buffer - fit_median < y_min:
            y_min = min(mag_out) - buffer - fit_median

        for i, lc in enumerate(lc_filt_list):
            if period:
                xxx = lc['date']
            else:
                xxx = [(d - base_date) * 24 for d in lc['date']]
            yyy = [y - fit_median+offset[i] for y in lc['mags']]
            yerr = lc['mag_errs']

            obs_date = meta_filt_list[i]['SESSIONDATE']
            ax.errorbar(xxx, yyy, yerr, linestyle='None', markersize='5',
                        label="{} ({}) {}".format(obs_date[obs_date.startswith(common_date) and len(common_date):],
                                                  meta_filt_list[i]['MPCCODE'], filt))

    if period:
        ax.set_xlabel('Phase (Period = {}h, {}% coverage)'.format(period, round(np.mean(cov_out)*100, 1)))
        # ax.set_xlabel('Phase (Period = {}h)'.format(period))
        ax.set_xlim(0, 1.1)
    else:
        ax.set_xlabel('Date (Hours from {}.0)'.format(base_date))

    ax.set_ylabel('Magnitude ({})'.format(meta_list[0]['MAGBAND']))
    ax.set_ylim([y_min, y_max])
    ax.invert_yaxis()

    # remove error bars in legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), title=common_date.rstrip('-'), ncol=leg_cols)

    if num != 'False' and num != '0':
        fn = num.replace(' ', '_')
    elif name != 'False':
        fn = name.replace(' ', '_')
    else:
        fn = obj.replace(' ', '_')
    path = os.path.dirname(files[0]).lstrip(' ')
    savefile = os.path.join(path, fn + '_' + date_range + '_lightcurve.png')
    
    return ax, savefile


def fourier(x, *a):
    ret = a[1] * np.cos(a[2] * np.pi * x + a[3]) + a[0]
    ing = range(1, (len(a)-1)//3)
    ing = [x * 3 + 1 for x in ing]
    for deg in ing:
        ret += a[deg] * np.cos(a[deg+1] * np.pi * x + a[deg+2])
    return ret


def make_gif(files, period, p_range, p_steps, fig, ax):
    if p_steps < 2:
        p_steps = 2
    p_min = max(period - p_range, 0)
    p_interval = 2*p_range / p_steps
    precision = ceil(log10(1/p_interval)) + 1
    ax0 = ax
    time_in = datetime.now()

    def update(n):
        p_new = round(p_min + p_interval * n, int(precision))
        ax0.clear()
        set_color_marker_cycle(ax0)
        ax, png = plot_lightcurve(files, p_new, ax0)
        if n < p_steps:
            print_progress_bar(n+1, p_steps, prefix='Creating Gif: Frame {}'.format(n+1), time_in=time_in)
        return ax

    ax1 = update(0)
    anim = FuncAnimation(fig, update, frames=int(p_steps)+1, blit=False, interval=333)  # takes in fig, update function, and frame rate set to fr

    savefile = os.path.join(path, 'lc_search.gif')
    anim.save(savefile, dpi=90, writer='imagemagick')
    return savefile


def build_periodigram(files, ax):

    meta_list = []
    lc_list = []
    for file in files:
        meta_list, lc_list, filt_list = import_alcdef(file, meta_list, lc_list)

    filt_unique = list(set(filt_list))
    filt_sets = {}
    for filt_u in filt_unique:
        filt_sets[filt_u] = []
        for i, filt in enumerate(filt_list):
            if filt == filt_u:
                filt_sets[filt_u].append(i)

    base_date = floor(min(sorted([jd for lc in lc_list for jd in lc['date']])))
    obj, name, num = get_name(meta_list[0])
    nout = 100000
    freq = np.linspace(50, 100, nout)

    A = 2.
    w = 2 * np.pi / .5
    phi = 0.5 * np.pi

    for f, filt in enumerate(filt_unique):
        lc_filt_list = [lc_list[k] for k in filt_sets[filt]]
        meta_filt_list = [meta_list[k] for k in filt_sets[filt]]

        date_out = [(d - base_date) * 24 for lc in lc_filt_list for d in lc['date']]
        mag_out = [m for lc in lc_filt_list for m in lc['mags']]
        merr_out = [e for lc in lc_filt_list for e in lc['mag_errs']]

        mag_out = [x for _, x in sorted(zip(date_out, mag_out))]
        date_out.sort()

        y = A * np.sin(w*np.array(date_out)+phi)

        # pgram = signal.lombscargle(date_out, mag_out, freq)
        # pgram_norm = np.sqrt(4*(pgram/len(date_out)))
        freq, pgram_norm = signal.periodogram(mag_out, 1/(date_out[3]-date_out[2]), nfft=10000, scaling='spectrum')
        peak = 1/freq[np.where(pgram_norm == max(pgram_norm))]

        if len(filt_unique) > 1:
            if 'I' in filt.upper():
                set_color_marker_cycle(ax, color=['k'])
            elif 'G' in filt.upper():
                set_color_marker_cycle(ax, color=['limegreen'])
            elif 'R' in filt.upper():
                set_color_marker_cycle(ax, color=['firebrick'])
            else:
                set_color_marker_cycle(ax, color=[next(rainbow)['color']])

        # Build y-axis boundaries
        # y_max = -1000
        # y_min = 1000
        # buffer = (max(mag_out) - min(mag_out)) * .05
        # if max(mag_out) + buffer - fit_median > y_max:
        #     y_max = max(mag_out) + buffer - fit_median
        # if min(mag_out) - buffer - fit_median < y_min:
        #     y_min = min(mag_out) - buffer - fit_median

        p_t = 1/freq[1:]
        # ax.plot(freq, pgram_norm, markersize=0)
        newpgram = pgram_norm[1:]
        ax.plot(p_t, newpgram, markersize=0)
        # ax.plot(date_out,mag_out )

    plot_title = "Periodogram for {}".format(obj)
    ax.set_title(plot_title)

    ax2 = plt.axes([.45,.4, 0.3, 0.4])
    data_range=np.where(p_t < 0.11)
    ax2.plot(p_t[data_range], newpgram[data_range])
    # ax2 = plt.gca()
    ax2.set_xscale("log")
    # ax2.set_yscale("logit")
    # ax2.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax2.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax2.set_xlabel('Period (hrs)')
    ax2.set_ylabel('Power')
    ax2.set_title("Best fit: Single Peak:{}hr / Double Peak {}hr".format(round(peak[0], 3), round(peak[0], 3) * 2))

    ax.set_xlabel('Period (hrs)')
    ax.set_xscale("log")
    # ax.set_xlim(10, 22)
    ax.set_ylabel('Power')
    # ax.set_ylim([y_min, y_max])

    path = os.path.dirname(files[0]).lstrip(' ')
    if num != 'False' and num != '0':
        fn = num.replace(' ', '_')
    elif name != 'False':
        fn = name.replace(' ', '_')
    else:
        fn = obj.replace(' ', '_')
    date_range = meta_list[0]['SESSIONDATE'].replace('-', '') + '-' + meta_list[-1]['SESSIONDATE'].replace('-', '')
    savefile = os.path.join(path, fn + '_' + date_range + '_periodigram.png')

    return ax, savefile


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to directory containing _ALCDEF.txt files", type=str)
    parser.add_argument("--period", help="Best-guess Period (in hours) to which data should be phased.", default=None, type=float)
    parser.add_argument("--p_range", help="range around period (hours) to search with gif", default=None, type=float)
    parser.add_argument("--p_step", help="number of steps to search with gif", default=5, type=float)
    parser.add_argument("--p_gram", help="Create Periodigram", default=False, action="store_true")
    parser.add_argument("--plan", help="Phase next n rotations (Leave blank for 2)", default=None, nargs='?', const=2, type=float)
    args = parser.parse_args()
    path = args.path
    period = args.period
    p_range = args.p_range
    p_step = args.p_step
    plan_rots = args.plan
    p_gram = args.p_gram
    if path[-1] != '/':
        path = path + '/'
    files = glob(path+'*ALCDEF.txt')
    files.sort()
    if len(files) >= 1:
        fig, ax = build_figure()
        if p_gram:
            ax, lc_plot = build_periodigram(files, ax)
            plt.savefig(lc_plot)
        else:
            if p_range:
                lc_plot = make_gif(files, period, p_range, p_step, fig, ax)
            else:
                ax, lc_plot = plot_lightcurve(files, period, ax, plan_rots)
                plt.savefig(lc_plot)
        print("New figure created: {}".format(lc_plot))
    else:
        print("No files found.")
