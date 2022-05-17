import csv
import argparse
import pprint


def pull_data_from_text(spectra):
    with open(spectra, newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        key_list = ["use_line", "wavelength",
                    "flux_1.2", "relflux_1.2", "fwhm_1.2",
                    "flux_1.6", "relflux_1.6", "fwhm_1.6",
                    "flux_2.0", "relflux_2.0", "fwhm_2.0",
                    "flux_6.0", "relflux_6.0", "fwhm_6.0",
                    "line_source", "line_notes"]
        peak_data = []
        for row in csv_reader:
            if line_count > 1:
                peak_dict = {}
                for i, cell in enumerate(row):
                    if i == 0:
                        if "#" in cell:
                            cell = False
                        else:
                            cell = True
                    try:
                        peak_dict[key_list[i]] = float(cell)
                    except ValueError:
                        peak_dict[key_list[i]] = cell
                peak_data.append(peak_dict)
            line_count += 1
    return peak_data


def split_data(peak_data):
    peak_dict_1p2 = []
    peak_dict_1p6 = []
    peak_dict_2p0 = []
    peak_dict_6p0 = []
    blue_peaks = []
    red_peaks = []
    unused_peaks = []
    red_lines = False
    select_keys = ["wavelength", "relflux_2.0", "line_source", "line_notes"]
    for peak in peak_data:
        peak_dict = {}
        for key in select_keys:
            if key == "relflux_2.0":
                peak_dict["line_strength"] = peak[key]
            else:
                peak_dict[key] = peak[key]
        if "RED" in str(peak["wavelength"]):
            red_lines = True
            continue
        if not peak["use_line"]:
            unused_peaks.append(peak_dict)
        elif red_lines:
            red_peaks.append(peak_dict)
        else:
            blue_peaks.append(peak_dict)
    return unused_peaks, red_peaks, blue_peaks


def print_data(peak_data):
    print("[")
    for peak in peak_data:
        print(peak, ",")
    print("]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--outpath", help="Output path for plots", type=str, default='')
    parser.add_argument("--path", help="base path spectra", type=str, default='')
    args = parser.parse_args()
    path = args.path
    outpath = args.outpath

    data_out = pull_data_from_text(path)
    unused_peaks, red_peaks, blue_peaks = split_data(data_out)
    print(unused_peaks)
    print(blue_peaks)
    print(red_peaks)
