import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import subprocess
import os

# ______________________________________________________________________________
source = "dronesim"
epochs = 300
marker_count = 9
path_desc = "zigzag"
layer_count = 6
# ____________________________________________________________________________


# Define the path to your NPZ file
folder_path = 'calibration_data/' + source + "/path_" + path_desc
file_path = folder_path + '/traveller_training_' + source + '_' + path_desc + '.npz'

# Load data from the NPZ file
with np.load(file_path) as data_file:
  data = data_file['data']


def play_this(say):
    audio_file = "Sounds/" + say + ".mp3"
    null_output = open(os.devnull, 'w')
    fpath = "ffmpeg/bin/ffplay.exe"
    subprocess.Popen([fpath, "-nodisp", "-autoexit", audio_file], stdout=null_output, stderr=null_output)

print(f"data: {data.shape}")
# Combine input features (Xp, Rp, Zp) and outputs (Xc, Rc)
#inputs = np.empty((0, (marker_count * 2) + 4), dtype=float)
#outputs = np.empty((0, 3), dtype=float)

#inputs = data[:19]
#outputs = data[19:]
# Initialize lists to store inputs and outputs
inputs = data[:, :19]
outputs = data[:, 19:]


print(f"input shape: {inputs.shape}")
print(f"output shape: {outputs.shape}")
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

# Define the model
unit_count = 64
act = 'tanh'

model = keras.Sequential()

# Input layer
model.add(keras.layers.Input(shape=((marker_count * 2) + 1), name='input_layer'))

# Hidden layers
for i in range(1, layer_count + 1):
  layer_name = 'hidden_layer_' + str(i)
  model.add(keras.layers.Dense(unit_count, activation=act, name=layer_name))

# Output layer
model.add(keras.layers.Dense(3, activation=act, name='output_layer'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Make predictions
#predictions = model.predict(X_test)

model.save(folder_path + '/traveller_model_' + source + '_' + path_desc + '.h5')

play_this("training complete")