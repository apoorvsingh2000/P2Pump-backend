import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from flask import Flask, request, jsonify


df = pd.read_csv('25.csv')

from google.cloud import storage


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )


class CustomDataset(Dataset):
    def __init__(self, file):
        self.data = pd.read_csv(file)
        self.data = self.data.drop(['date'], axis = 1)
        self.data = self.data.astype(float)
        self.data = (self.data - self.data.min())/(self.data.max() - self.data.min())
        self.labels = self.data['calories_burned']
        self.feat = self.data.drop(['calories_burned'], axis = 1)
        self.labels = torch.from_numpy(self.labels.values)
        self.feat = torch.from_numpy(self.feat.values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.feat[idx].float(), self.labels[idx].float().reshape(-1, 1)
    


dataset = CustomDataset('25.csv')
training_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)


model = nn.Sequential(
    nn.Linear(5, 1),
)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        last_loss = running_loss/1000
        tb_x = epoch_index * len(training_loader) + i + 1
        tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        running_loss = 0.
    return last_loss

def train_and_store():
    for epoch in range(10):
        print('EPOCH {}:'.format(epoch + 1))
        model.train(True)
        avg_loss = train_one_epoch(epoch, writer)
        torch.save(model.state_dict(), 'stored_weights')
        return jsonify('Trained and stored')

def transfer_weights(self_addr, weights): # weights must be a binary string
    for add in address_ls:
        if add != self_addr:
            sock.connect(add, 12000)
            time.sleep(1)
            if(sock.routing_table):
                print("Sending weights")
                sock.send(DEVICE_ID, weights)
                time.sleep(1)
                msg = sock.recv()
                if(msg.packets[0] == 'received'):
                    print("Successfully sent. Shutting down")

def receive_weights():
    bstr = request.get_data()
    weights = torch.load(io.BytesIO(bstr))
    print("Local model succesfully updated")
    upload_blob('p2pfed_back', 'stored_weights', 'stored_weights_back')
    return 'Local Model successfully updated'

def convert_to_string(model):
    model_weights = model.state_dict()
    to_send = io.BytesIO()
    torch.save(model_weights, to_send, _use_new_zipfile_serialization=False)
    to_send.seek(0)
    to_send = to_send.getvalue()
    return to_send


def load_received_weights(bstring):
    weights = torch.load(io.BytesIO(bstring))
    return weights



app = Flask(__name__)

@app.route('/uploadweights', methods=['GET'])
def upload_weights():
    return convert_to_string()


@app.route('/receiveweights', methods=['GET'])
def receive_weights():
    return receive_weights()

@app.route('/shutdown', methods=['GET'])
def shutdown():
    transfer_weights()
    shut_down = 1
    return 'Shut down successfully'

@app.route('/searchweights', methods=['GET'])
def search_weights():
    if(shut_down == 1):
        return b'shutdown'
    files = os.listdir('/home/ubuntu/helloworld')
    for file in files:
        if("device_" in file):
            return convert_to_string('/home/ubuntu/helloworld/' + file)
    return None