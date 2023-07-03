"""
Prepare MoSeq2 data for language training.
Will insert special token for interruptions between sessions.
"""
import os
import pickle
import requests
import numpy as np
import pandas

# parse the df_csv
input_file_path = os.path.join(os.path.dirname(__file__), 'input.csv')
if not os.path.exists(input_file_path):
    raise Exception("input.csv not found.")

moseq_df = pandas.read_csv(input_file_path)
print(moseq_df.info)
session_changes = moseq_df["SessionName"].shift() != moseq_df["SessionName"]
session_changes = np.array(session_changes.index[session_changes])
print(session_changes)

label_data = moseq_df['labels (usage sort)'].to_numpy()
SEP_TOKEN = np.max(label_data)+1

data = []

for i in range(len(session_changes)-1):
    data.append(np.insert(np.array(label_data[session_changes[i]+3:session_changes[i+1]]), 0, SEP_TOKEN)) # exclude first 3 frames, due to the AR-HMM init
data.append(np.insert(np.array(label_data[session_changes[-1]+3:]), 0, SEP_TOKEN))
data = np.concatenate(data)

print(f"length of dataset: {len(data):,}")

# get all the unique syllables that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique syllables:", ''.join(str(chars)))
print(f"vocab size: {vocab_size:,}")

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# export to bin files
train_ids = np.array(train_data, dtype=np.uint16)
val_ids = np.array(val_data, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

