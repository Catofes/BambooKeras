import ROOT
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    args = parser.parse_args()
    file = open(args.input)
    print("Load File: ", args.input)
    data = json.loads(file.read())
    signal = data['signal']
    background = data['background']
    hist = ROOT.TH1F("max_energy", "Max Energy", 500, 0, 1000)
    for s in signal:
        hist.Fill(s[2])
    for b in background:
        hist.Fill(b[2])
    hist.Draw()
    input()

