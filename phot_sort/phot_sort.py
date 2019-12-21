"""
Sorts a series of fits files by Header Key words
Date: 12/20/2019

"""

import argparse
from glob import glob
from astropy.io import fits
import shutil
import os


def parse_header_keywords(h_keys_in):
    h_keys_out = h_keys_in.replace(' ', '').upper().split(",")
    return h_keys_out


def sort_fits(files, h_keys):
    for file in files:
        path = os.path.dirname(file)
        name = os.path.basename(file)
        with fits.open(file, ignore_missing_end=True) as hdul:
            try:
                header = hdul['SCI'].header
            except KeyError:
                try:
                    header = hdul['COMPRESSED_IMAGE'].header
                except KeyError:
                    header = hdul[0].header
        for hk in h_keys:
            path = os.path.join(path, header[hk])
            if not os.path.isdir(path):
                os.mkdir(path)
        shutil.move(file, os.path.join(path, name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to directory containing .fits or .fits.fz files", type=str, default='.')
    parser.add_argument("--keys", help="comma separated header Key words in sort order", type=str, default='Object, Filter, Instrume')
    args = parser.parse_args()
    path = args.path
    h_keys = parse_header_keywords(args.keys)
    if path[-1] != '/':
        path += '/'
    files = glob(path+'*.fits.fz')
    if len(files) < 1:
        files = glob(path+'*.fits')
    if len(files) >= 1:
        sort_fits(files, h_keys)
