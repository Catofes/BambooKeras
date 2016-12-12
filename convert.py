import ROOT
import json
import math
import argparse

cluster_x_size = 3
cluster_y_size = 3
cluster_z_size = 3

def get_center(x, y, z, e):
    total_energy = 0
    total_x = 0
    total_y = 0
    total_z = 0
    for i in range(0, len(x)):
        total_energy += e[i]
        total_x += x[i] * e[i]
        total_y += y[i] * e[i]
        total_z += z[i] * e[i]

    center_x = total_x / total_energy
    center_y = total_y / total_energy
    center_z = total_z / total_energy
    return center_x, center_y, center_z


def cluster(x, y, z, e):
    center_x, center_y, center_z = get_center(x, y, z, e)
    cluster_xy_data = {}
    cluster_zy_data = {}
    for i in range(0, len(x)):
        cluster_x = int(math.floor((x[i] - center_x) / cluster_x_size))
        cluster_y = int(math.floor((y[i] - center_y) / cluster_y_size))
        cluster_z = int(math.floor((z[i] - center_z) / cluster_z_size))
        cluster_xy_id = "%s:%s" % (cluster_x, cluster_y)
        cluster_zy_id = "%s:%s" % (cluster_z, cluster_y)
        if cluster_xy_id in cluster_xy_data:
            cluster_xy_data[cluster_xy_id] += e[i]
        else:
            cluster_xy_data[cluster_xy_id] = e[i]
        if cluster_zy_id in cluster_zy_data:
            cluster_zy_data[cluster_zy_id] += e[i]
        else:
            cluster_zy_data[cluster_zy_id] = e[i]
    return cluster_xy_data, cluster_zy_data


def convert(signal_path, background_path, output_path):
    signal_chain = ROOT.TChain("MLData")
    signal_chain.Add(signal_path)

    background_chain = ROOT.TChain("MLData")
    background_chain.Add(background_path)

    print("Get %s Signal, %s Background." % (signal_chain.GetEntries(), background_chain.GetEntries()))

    output = {
        "signal": [],
        "background": []
    }
    output_signal = output['signal']
    output_background = output['background']

    count = 0
    total = signal_chain.GetEntries() + background_chain.GetEntries()
    for entry in signal_chain:
        if count % 1000 == 0:
            print("%s/%s" % (count, total))
        cluster_xy_data, cluster_zy_data = cluster(entry.x, entry.y, entry.z, entry.e)
        output_signal.append((cluster_xy_data, cluster_zy_data))
        count += 1

    for entry in background_chain:
        if count % 1000 == 0:
            print("%s/%s" % (count, total))
        cluster_xy_data, cluster_zy_data = cluster(entry.x, entry.y, entry.z, entry.e)
        output_background.append((cluster_xy_data, cluster_zy_data))
        count += 1

    f = open(output_path, "w")
    f.write(json.dumps(output))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--signal")
    parser.add_argument("-b", "--background")
    parser.add_argument("-o", "--output", default="output.json")

    args = parser.parse_args()
    convert(args.signal, args.background, args.output)
