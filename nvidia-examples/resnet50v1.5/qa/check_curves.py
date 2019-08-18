import argparse
import json

parser = argparse.ArgumentParser(description='Convergence Tests')
parser.add_argument('baseline', metavar='DIR', help='path to baseline')
parser.add_argument('report', metavar='DIR', help='path to report')
args = parser.parse_args()

METRICS = ['train.loss', 'train.top1', 'train.top5', 'val.loss', 'val.top1', 'val.top5']

def check(baseline, report):

    allright = True

    for m in METRICS:
        for epoch in range(len(report['metrics'][m])):
            minv = baseline['metrics'][m][epoch][0]
            maxv = baseline['metrics'][m][epoch][1]
            r = report['metrics'][m][epoch]

            if not (r > minv and r < maxv):
                allright = False
                print("Result value doesn't match baseline: {} epoch {}, allowed min: {}, allowed max: {}, result: {}".format(
                    m, epoch, minv, maxv, r))

    return allright


with open(args.report, 'r') as f:
    report_json = json.load(f)

with open(args.baseline, 'r') as f:
    baseline_json = json.load(f)

if check(baseline_json, report_json):
    print("&&&& PASSED")
    exit(0)
else:
    print("&&&& FAILED")
    exit(1)
