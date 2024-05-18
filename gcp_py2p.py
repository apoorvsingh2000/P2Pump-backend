import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import io
import py2p
from flask import request, jsonify  # Assuming you're using Flask for web handling
from google.cloud import storage  # For GCS handling


def transfer_weights(self_addr, weights, sock):  # weights must be a binary string
    for add in address_ls:
        if add != self_addr:
            sock.connect(add, 12000)
            time.sleep(1)
            if sock.routing_table:
                print("Sending weights")
                sock.send(DEVICE_ID, weights)
                time.sleep(1)
                msg = sock.recv()
                print("MESSAGE RECEIVED: ", msg)
                if msg.packets[0] == "received":
                    print("Successfully sent. Shutting down")


def load_received_weights(bstring):
    # bstring.getvalue()
    weights = torch.load(io.BytesIO(bstring))
    return weights


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

    blob.upload_from_filename(
        source_file_name, if_generation_match=generation_match_precondition
    )

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def receive_weights():
    bstr = request.get_data()
    weights = torch.load(io.BytesIO(bstr))
    print("Local Model successfully updated")
    upload_blob("p2pfed_back", "stored_weights", "stored_weights_back")
    return "Local Model successfully updated"


def train_and_store():
    for epoch in range(10):
        print("EPOCH {}:".format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch, writer)
        # print('LOSS train {}:'.format(avg_loss))
        torch.save(model.state_dict(), "stored_weights")
    return jsonify("Trained and stored")


def convert_to_string(path="stored_weights"):
    model_weights = torch.load(path)
    to_send = io.BytesIO()
    torch.save(model_weights, to_send, _use_new_zipfile_serialization=False)
    to_send.seek(0)
    to_send = to_send.getvalue()
    return to_send


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs.float())

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        last_loss = running_loss / 1000  # loss per batch
        # print('  batch {} loss: {}'.format(i + 1, last_loss))
        tb_x = epoch_index * len(training_loader) + i + 1
        tb_writer.add_scalar("loss/train", last_loss, tb_x)
        running_loss = 0.0

    return last_loss
