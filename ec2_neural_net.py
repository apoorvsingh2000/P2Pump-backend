import torch
import torch.nn as nn
import torch.optim as optim
import io
from datetime import datetime
import py2p

model = nn.Sequential(
    nn.Linear(5, 1),
)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

sock = py2p.MeshSocket("0.0.0.0", 11000)
dev_id = 9999
bstring = None
msg = None

time.sleep(5)
print("STARTED")
while msg is None:
    msg = sock.recv()

print("MESSAGE: ", msg)
print(len(msg.packets))
dev_id = msg.packets[1]
bstring = msg.packets[2]
msg.reply("received")

print("RECEIVED: ")
print("FROM: ", dev_id)
now = datetime.now()

weights = torch.load(io.BytesIO(bstring))
print("CHECK IF WEIGHTS CAN BE LOADED")
model.load_state_dict(weights)
print("Weights successfully loaded. Saving")
torch.save(model.state_dict(), "stored_weights_device_" + str(dev_id) + "_" + str(now))
print("SAVED")
