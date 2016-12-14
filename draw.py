import json
import numpy as np
from scipy import misc
import argparse


def convert_row(input_data):
    row = np.zeros((3, 224, 224))
    cluster_xy_data = input_data[0]
    for pixel, energy in cluster_xy_data.items():
        location = pixel.split(":")
        location_x = int(location[0])
        location_y = int(location[1])
        location_x += 224 / 2
        location_y += 224 / 2
        if not (0 <= location_x < 224 and 0 <= location_y < 224):
            continue
        row[0, location_x, location_y] = energy
    cluster_zy_data = input_data[1]
    for pixel, energy in cluster_zy_data.items():
        location = pixel.split(":")
        location_z = int(location[0])
        location_y = int(location[1])
        location_z += 224 / 2
        location_y += 224 / 2
        if not (0 <= location_z < 224 and 0 <= location_y < 224):
            continue
        row[1, location_z, location_y] = energy
    return row


def load_data(input_data):
    print("Load Data.")
    f = open(input_data, "r")
    data = json.loads(f.read())
    signal = data['signal']
    background = data['background']
    return signal, background


def draw_data(row, file_name):
    data = convert_row(row)
    misc.toimage(data).save(file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-t", "--type")
    parser.add_argument("-s", "--seed", type=int)
    parser.add_argument("-o", "--output")
    args = parser.parse_args()
    signal, background = load_data(args.input)
    if args.type == "s":
        draw_data(signal[args.seed], args.output)
    else:
        draw_data(background[args.seed], args.output)
