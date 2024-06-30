import mne
import numpy as np
#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,  LSTM,  TimeDistributed
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pdb

filepath = 'C:\ImaginationPerceptionDataset\DataSet\perceptionImaginationEEG\preprocessed\sub-03\ses-03\eeg\sub3_sess3_50_ica_eeg.fif'
data = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
events = mne.events_from_annotations(data)
eventId = {f'Imagination': list(range(1, 101)), f'Perception': list(range(101, 200))}

def cleanEvents(events):
    cleanedEvents = []
    ignoreEvents = [100, 200, 201]
    for event in events[0]:
        if event[2] in ignoreEvents:
            continue
        else:
            cleanedEvents.append(event)
    cleanedEvents = np.array(cleanedEvents)
    return cleanedEvents

def perceptionAndImaginationEvents(events): 
    cleanedEvents = cleanEvents(events)
    perceptionKeys = []
    imaginationKeys = []
    for item, key in events[1].items():
        if 'Perception' in item:
            perceptionKeys.append(key)
        if 'Imagination' in item:
            imaginationKeys.append(key)

    for index in range(cleanedEvents.shape[0]):
        key = cleanedEvents[index][2]
        if key in perceptionKeys:
            cleanedEvents[index][2] = 1
        elif key in imaginationKeys:
            cleanedEvents[index][2] = 2

    return cleanedEvents
    
events = mne.events_from_annotations(data)
cleanedEvents = perceptionAndImaginationEvents(events)
newEventId = {'Perception':1, 'Imagination':2}
epochs = mne.Epochs(data.copy(), cleanedEvents, event_id=newEventId, tmin=-0.2, tmax=1.0, preload=True)
perception = epochs['Perception'].get_data()
imagination = epochs['Imagination'].get_data()
perception.shape, imagination.shape


imagLables= np.array([0 for i in range(imagination.shape[0])])
perceptionLabels = np.array([1 for i in range(imagination.shape[0])])
labels = np.concatenate((imagLables, perceptionLabels), axis=0)
X = np.concatenate((imagination, perception), axis=0)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2],1)
labels.shape, X.shape

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        pdb.set_trace()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Model parameters
input_size = 50  # Number of features
hidden_size = 128
num_layers = 2
num_classes = 2

model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device='cpu'
# Training loop
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')



input_shape = (X.shape[1], X.shape[2], X.shape[3]) 
