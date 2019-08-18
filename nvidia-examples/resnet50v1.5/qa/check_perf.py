import argparse
import json

parser = argparse.ArgumentParser(description='Performace Tests')
parser.add_argument('baseline', metavar='DIR', help='path to baseline')
parser.add_argument('report', metavar='DIR', help='path to report')
args = parser.parse_args()

METRICS = ['train.total_ips', 'val.total_ips']

def check(baseline, report):

    allright = True

    for m in METRICS:
        for ngpus in report['metrics']:
            for bs in report['metrics'][ngpus]:
                minv = baseline['metrics'][ngpus][bs][m] * 0.9
                r = report['metrics'][ngpus][bs][m]

                if r < minv:
                    allright = False
                    print("Result value doesn't match baseline: {} ngpus {} batch-size {}, allowed min: {}, result: {}".format(
                          m, ngpus, bs, minv, r))

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
