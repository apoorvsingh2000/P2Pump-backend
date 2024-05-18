import io
weight_ls = []
import urllib.request

address_ls = ["3.91.82.181", "3.92.27.19"]

contents = None
for add in address_ls:
    contents = urllib.request.urlopen("http://" + add + "/uploadweights").read()
    print(contents)
    if contents == b'shutdown':
        for add in address_ls:
            res = None
            contents = urllib.request.urlopen("http://" + add + "/searchweights").read()
            if contents == b'shutdown':
                continue
        weights = torch.load(io.BytesIO(contents))
        weight_ls.append(weights)

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


import copy
avg_weights = average_weights(weight_ls)

model.load_state_dict(avg_weights)
model_weights = torch.save(model.state_dict(), 'avg_weights')

import requests

for ad in address_ls:
    print(add)
    print(requests.post("http://" + add + "/receiveweights", data = to_send, headers = {'Content-Type': 'application/octet-stream'}))
