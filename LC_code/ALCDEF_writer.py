"""
Create ALCDEF files from other input sources.
"""
import os
from glob import glob
import argparse
from datetime import datetime, timedelta
import pyslalib.slalib as S


def jd_utc2datetime(jd):
    """Converts a passed Modified Julian date to a Python datetime object. 'None' is
    returned if the conversion was not possible."""

    mjd = float(jd) - 2400000.5
    year, month, day, frac, status = S.sla_djcl(mjd)
    if status != 0:
        return None
    sign, hms = S.sla_dd2tf(0, frac)
    dt = datetime(year, month, day, hms[0], hms[1], hms[2])
    return dt


def output_alcdef(lightcurve_file, obj_name, site, dates, mags, mag_errors, filt, outmag):
    with open(lightcurve_file, 'w') as lc_out:
        startdate = jd_utc2datetime(dates[0])
        enddate = jd_utc2datetime(dates[-1])
        mid_time = (enddate - startdate)/2 + startdate
        metadata_dict = {'ObjectNumber': 0,
                         'ObjectName'  : obj_name,
                         'MPCDesig'    : obj_name,
                         'ReviseData'  : 'FALSE',
                         'AllowSharing': 'TRUE',
                         'MPCCode'     : site,
                         'Delimiter'   : 'PIPE',
                         'ContactInfo' : '[tlister@lcogt.net]',
                         'ContactName' : 'T. Lister',
                         'DifferMags'  : 'FALSE',
                         'Facility'    : 'Las Cumbres Observatory',
                         'Filter'      : filt,
                         'LTCApp'      : 'NONE',
                         'LTCType'     : 'NONE',
                         'MagBand'     : outmag,
                         'Observers'   : 'T. Lister; J. Chatelain; E. Gomez',
                         'ReducedMags' : 'NONE',
                         'SessionDate' : mid_time.strftime('%Y-%m-%d'),
                         'SessionTime' : mid_time.strftime('%H:%M:%S')
                        }
        if obj_name.isdigit():
            metadata_dict['ObjectNumber'] = obj_name
            metadata_dict['MPCDesig'] = obj_name
            metadata_dict['ObjectName'] = obj_name
        lc_out.write('STARTMETADATA\n')
        for key, value in metadata_dict.items():
            lc_out.write('{}={}\n'.format(key.upper(), value))
        lc_out.write('ENDMETADATA\n')
        for i, date in enumerate(dates):
            lc_out.write('DATA={}|{:+.3f}|{:+.3f}\n'.format(date, mags[i], mag_errors[i]))
        lc_out.write('ENDDATA\n')


def extract_phot_data(file):
    dates = []
    mags = []
    mag_errs = []

    with open(file) as lc_file:
        for line in lc_file:
            if line[0] == '#':
                continue
            chunks = line.split(' ')
            chunks = list(filter(None, chunks))
            dates.append(chunks[1])
            mags.append(float(chunks[2]))
            mag_errs.append(float(chunks[3]))
    return dates, mags, mag_errs


def extract_astrometrica_data(file):
    dates = []
    mags = []
    mag_errs = []

    with open(file) as lc_file:
        for line in lc_file:
            if line[0] == '#':
                continue
            chunks = line.split(' ')
            chunks = list(filter(None, chunks))
            jd = float(chunks[0]) + 2400000.5
            dates.append(jd)
            mags.append(float(chunks[1]))
            mag_errs.append(float(chunks[2]))
    return dates, mags, mag_errs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to directory containing photometry*.dat file", type=str)
    parser.add_argument("site", help="MPC site code for observations", type=str)
    parser.add_argument("out_path", help="output_path", type=str)
    args = parser.parse_args()
    path = args.path
    site = args.site
    out_path = args.out_path
    if path[-1] != '/':
        path = path + '/'
    if out_path[-1] != '/':
        out_path += '/'
    files = glob(path+'photometry*pp.dat')
    if not files:
        files = glob(path+'*_lc.dat')
    files.sort()
    for file in files:
        if '_lc.dat' in file:
            object_name = file.replace(path, '').split('_')[0]
            dates, mags, mag_errs = extract_astrometrica_data(file)
        else:
            object_name = file.split('_')[1]
            dates, mags, mag_errs = extract_phot_data(file)
        out_filename = out_path+object_name+'_'+site+'_ALCDEF.txt'
        output_alcdef(out_filename, object_name, site, dates, mags, mag_errs, 'W', 'r')
        print(out_filename)
