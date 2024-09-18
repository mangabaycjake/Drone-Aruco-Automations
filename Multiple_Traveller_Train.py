import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# ______________________________________________________________________________
source = "dronesim"
epochs_set: int = [300, 1000, 2000]
unit_count_set: int = [64]
layer_count_set: int = [5, 6, 7, 8, 9, 10]
act_set = ["tanh"]
marker_count = 9
path_desc = "s"
# ____________________________________________________________________________
#epochs = 10
#unit_count = 64
#act = 'tanh'
#layer_count = 2

# Define the path to your NPZ file
folder_path = 'calibration_data/' + source + "/path_" + path_desc
file_path = folder_path + '/traveller_training_' + source + '.npz'

total_iterations = len(epochs_set) * len(unit_count_set) * len(act_set) * len(layer_count_set)
current_iteration = 0

for epochs in epochs_set:
  for unit_count in unit_count_set:
    for  act in act_set:
      for layer_count in layer_count_set:

        current_iteration += 1

        print("=================================================================")
        print("-----------------------------------------------------------------")
        print(f"{current_iteration} / {total_iterations}")
        print("_________________________________________________________________")
        print("-----------------------------------------------------------------")
        print(f"Layer: {layer_count} Epoch: {epochs} Unit: {unit_count} Activation: {act}")


        #print(iter_info)

        # Load data from the NPZ file
        with np.load(file_path) as data_file:
          data = data_file['data']

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

        model = keras.Sequential()

        # Input layer
        model.add(keras.layers.Input(shape=((marker_count * 2) + 1), name='input_layer'))

        # Hidden layers
        model.add(keras.layers.Dense(unit_count, activation=act, name='hidden_layer_1'))
        model.add(keras.layers.Dense(unit_count, activation=act, name='hidden_layer_2'))
        if layer_count > 2:
          for i in range(3, layer_count + 1):
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

        model.save(
            folder_path + '/multiple/traveller_model_' +
            source + '_L' + str(layer_count) +
            '_E' + str(epochs) +
            '_U' + str(unit_count) +
            '_' + act + "_" + str(round(loss, 4)) + '.h5'
        )