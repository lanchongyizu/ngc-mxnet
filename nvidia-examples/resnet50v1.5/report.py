# Report JSON file structure:
# - "model"          : architecture of the model (e.g. "resnet50").
# - "ngpus"          : number of gpus on which training was performed.
# - "total_duration" : total duration of training in seconds.
# - "cmd"            : list of application arguments.
# - "metrics"        : per epoch metrics for train and validation
#                      (some of below metrics may not exist in the report,
#                       depending on application arguments)
#       - "train.top1"      : training top1 accuracy in epoch.
#       - "train.top5"      : training top5 accuracy in epoch.
#       - "train.loss"      : training loss in epoch.
#       - "train.time"      : average training time of iteration in seconds.
#       - "train.total_ips" : training speed (data and compute time taken into account) for epoch in images/sec.
#       - "val.top1", "val.top5", "val.loss", "val.time", "val.total_ips" : the same but for validation.

import json
from collections import defaultdict, OrderedDict

class Report:
    def __init__(self, model_name, ngpus, cmd):
        self.model_name = model_name
        self.ngpus = ngpus
        self.cmd = cmd
        self.total_duration = 0
        self.metrics = defaultdict(lambda: [])

    def add_value(self, metric, value):
        self.metrics[metric].append(value)

    def set_total_duration(self, duration):
        self.total_duration = duration

    def save(self, filename):
        report = OrderedDict([
            ('model', self.model_name),
            ('ngpus', self.ngpus),
            ('total_duration', self.total_duration),
            ('cmd', self.cmd),
            ('metrics', self.metrics),
        ])
        with open(filename, 'w') as f:
            json.dump(report, f, indent=4)
