"""
Create damit input files from ALCDEF files.
"""
import argparse
from glob import glob
import os


def import_alcdef(file, meta_list, lc_list):
    """Pull LC data from ALCDEF text files."""

    lc_file = default_storage.open(file, 'rb')
    lines = lc_file.readlines()

    metadata = {}
    dates = []
    mags = []
    mag_errs = []
    met_dat = False

    for line in lines:
        line = str(line, 'utf-8')
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
            if metadata not in meta_list:
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

    return meta_list, lc_list


def write_input_lcs():
    """
    Create input lcs:
    :input:
    :return:
    JD (Light-time corrected)
    Brightness (intensity)
    XYZ coordinates of Sun (astrocentric cartesian coordinates) AU
    XYZ coordinates of Earth (astrocentric cartesian coordinates) AU
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to directory containing alcdef file", type=str)
    parser.add_argument("out_path", help="output_path", type=str)
    args = parser.parse_args()
    path = args.path
    out_path = args.out_path
    if path[-1] != '/':
        path = path + '/'
    if out_path[-1] != '/':
        out_path += '/'
    files = glob(path+'*ALCDEF.txt')
    files.sort()
    for file in files:
        print(file)
        basename = os.path.basename(file)
        out_filename = out_path + basename.rstrip('ALCDEF.txt') + 'damit.lcs'
        print(out_filename)

