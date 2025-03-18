import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import sklearn.metrics
import itertools
import os
from matplotlib.colors import LogNorm  # Add this import


### Step 1 Get Some Data
# Load the speckle images
speckle_imgs = np.load('./FINALspeckle_imgs.npy')  # pwd should look like "/home/aaron/Documents/AFIT/SpeckleResearch"

# Flag to run a correlation test to see what the baseline looks like
run_correlation_test = False
if run_correlation_test:
    known_data = speckle_imgs[:19]
    num_iterations = known_data.shape[0]
    num_speckles = known_data.shape[1]
    sweep_correlations = np.zeros(num_speckles)
    for unkn_idx in range(num_speckles):
        known_data = speckle_imgs[:19]
        unknown_speckle = (speckle_imgs[19, unkn_idx] - np.mean(speckle_imgs[19, unkn_idx])) / np.std(speckle_imgs[19, unkn_idx])

        summed_correlations = np.zeros(num_speckles)
        wavelengths = np.zeros(num_speckles)

        for iter_idx in range(num_iterations):
            correlations = np.zeros(num_speckles)
            for lam in range(num_speckles):
                speckle = (known_data[iter_idx, lam] - np.mean(known_data[iter_idx, lam])) / np.std(known_data[iter_idx, lam])
                correlations[lam] = np.mean(np.multiply(unknown_speckle, speckle))

            summed_correlations += correlations

        sweep_correlations[unkn_idx] = np.argmax(summed_correlations)

    wavelengths = np.arange(num_speckles) + 400
    sweep_correlations = sweep_correlations + 400

    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, sweep_correlations)
    plt.title(f'Confusion Matrix for Speckle Pattern Correlation Test')
    plt.xlabel('Wavelength of Unknown Speckle Pattern (nm)')
    plt.ylabel('Highest Correlation Value')
    plt.gca().invert_yaxis()  # Invert the y-axis
    plt.grid(True)
    plt.savefig('confusion_matrix_correlation_test.png')  # Save the plot to a file
    plt.close()

### Step 3 Prepare our Data

# Get the shape of the data
num_iterations, num_speckles, img_height, img_width = speckle_imgs.shape

# Flatten the data structure
X = speckle_imgs.reshape(-1, img_height, img_width, 1)  # Add a channel dimension for CNN

# Normalize the data to the range [0, 1]
X = X / 255.0

# Create labels based on the second dimension index
wavelengths = np.arange(0, 256)  # Wavelengths from 0 to 255
labels = np.tile(wavelengths, num_iterations)  # Repeat for each iteration

### Step 4 Determine an Evaluation Method 

# Create a stratified split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, temp_index = next(sss.split(X, labels))

X_train, X_temp = X[train_index], X[temp_index]
y_train, y_temp = labels[train_index], labels[temp_index]

# Further split the temp set into validation and test sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_index, test_index = next(sss.split(X_temp, y_temp))

X_val, X_test = X_temp[val_index], X_temp[test_index]
y_val, y_test = y_temp[val_index], y_temp[test_index]

# Print the shape of the training, validation, and testing sets to verify
print(f'Training data shape: {X_train.shape}')
print(f'Training labels shape: {y_train.shape}')
print(f'Validation data shape: {X_val.shape}')
print(f'Validation labels shape: {y_val.shape}')
print(f'Testing data shape: {X_test.shape}')
print(f'Testing labels shape: {y_test.shape}')

# Function to build the model
def build_model(model_flag):
    if model_flag == 1:  # basic
        ### Step 5 Develop a Model
        model = Sequential([
            Input(shape=(img_height, img_width, 1)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(1)
        ])
    elif model_flag == 2:  # overfit
        ### Step 6 Overfit Model
        model = Sequential([
            Input(shape=(img_height, img_width, 1)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dense(1)
        ])
    elif model_flag == 3:  # regularized
        ### Step 7 Regularized Model
        model = Sequential([
            Input(shape=(img_height, img_width, 1)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1)
        ])
    else:
        raise ValueError("Invalid model_flag value. Choose 1, 2 or 3 for basic, overfit, or regularized.")
    return model

# Flag to use a saved model or train a new one
use_saved_model = False  # Set to False to train a new model

if use_saved_model and os.path.exists('best_model_regularized4.h5'):
    # Load the saved model
    best_model = load_model('best_model_regularized4.h5', compile=False)
    best_model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
    print('Loaded saved model.')
    
else:
    # Initialize variables to keep track of the best model
    best_model = None
    best_val_mae = float('inf')

    # Store the training history of each model
    histories = []

    # Train multiple models and keep track of the best one
    num_models = 10
    for i in range(num_models):
        print(f'Training model {i+1}/{num_models}')
        
        # Build the model
        model = build_model(model_flag=3)  # Change model_flag to 2 for overfit model or 3 for regularized model
        
        # Compile the model
        ### Step 2 Measures of Success
        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
        
        # Callback for learning rate reduction
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        
        # Train the model
        history = model.fit(X_train, y_train, batch_size=32, epochs=200, validation_data=(X_val, y_val),
                            callbacks=[reduce_lr])
        
        # Store the training history
        histories.append(history)
        
        # Evaluate the model on the validation set
        val_loss, val_mae = model.evaluate(X_val, y_val)
        print(f'Validation MAE: {val_mae}')
        
        # Update the best model if the current model is better
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model = model

    # Save the best model
    best_model.save('best_model.h5')
    print(f'Best model saved with validation MAE: {best_val_mae}')

# Evaluate the model on the test set
test_loss, test_mae = best_model.evaluate(X_test, y_test)
print(f'Test MAE: {test_mae}')
print(f'Test Loss: {test_loss}')

# Evaluate the model on the training set
train_loss, train_mae = best_model.evaluate(X_train, y_train)
print(f'Training MAE: {train_mae}')
print(f'Training Loss: {train_loss}')

# Evaluate the model on the validation set
val_loss, val_mae = best_model.evaluate(X_val, y_val)
print(f'Validation MAE: {val_mae}')
print(f'Validation Loss: {val_loss}')

# Print the model summary
#best_model.summary()

if not use_saved_model:
    # Plot the training progress of all models
    plt.figure(figsize=(12, 8))
    colors = plt.get_cmap('tab10', num_models)  # Get a colormap with unique colors

    for i, history in enumerate(histories):
        #plt.plot(history.history['mean_absolute_error'], color=colors(i), linestyle='-.', label=f'Model {i+1} Training MAE')
        plt.plot(history.history['val_mean_absolute_error'], color=colors(i), linestyle='-', label=f'Model {i+1} Validation MAE')

    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.title('Validation MAE for of Regularized Models')
    plt.xlabel('Epoch')
    plt.ylabel('Validation MAE (log scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_progress_all_models_regularized4.png')
    plt.close()

    # Plot the validation MAE and training MAE of the best model
    best_model_index = np.argmin([h.history['val_mean_absolute_error'][-1] for h in histories])
    best_history = histories[best_model_index]

    plt.figure(figsize=(12, 8))
    plt.plot(best_history.history['val_mean_absolute_error'], label='Validation MAE of Best Model', linestyle='-')
    plt.plot(best_history.history['mean_absolute_error'], label='Training MAE of Best Model', linestyle='--')
    #plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.title('Validation MAE and Training MAE of Best Regularized Model')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (log scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig('best_model_validation_training_mae_regularized4_temp.png')
    plt.close()

# Make predictions
predictions = best_model.predict(X_test)

# Print the first 10 sets of predictions and truths
for i in range(10):
    print(f'Prediction: {predictions[i]+400}, Truth: [{y_test[i]+400}]') # Add 400 to account for real life labels

# Make predictions for the confusion matrix
y_pred = np.round(best_model.predict(X_test)).astype(int)

# Ensure predictions are within the valid range
y_pred = np.clip(y_pred, 0, 255)

# Define class names and figure path
class_names = [str(i) for i in range(256)]
fig_path = './figures'

# Create the directory if it doesn't exist
os.makedirs(fig_path, exist_ok=True)

# Create confusion matrix
confusion_matrix = sklearn.metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='Greys',
                          label_interval=25,
                          figsize=(20, 20),
                          title_fontsize=36,
                          label_fontsize=30,
                          tick_fontsize=24,
                          colorbar_fontsize=24,
                          test_mae=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Increment all values by 400 to account for real life labels
    classes = [str(int(cls) + 400) for cls in classes]
    
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=title_fontsize)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=colorbar_fontsize)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks[::label_interval], classes[::label_interval], rotation=45, fontsize=tick_fontsize)
    plt.yticks(tick_marks[::label_interval], classes[::label_interval], fontsize=tick_fontsize)

    plt.ylabel('True label (nm)', fontsize=label_fontsize)
    plt.xlabel('Predicted label (nm)', fontsize=label_fontsize)
    plt.grid(False)  # Remove grid lines

    # Add Test MAE and Test Loss annotation in the upper right corner
    if test_mae is not None and test_loss is not None:
        plt.text(len(classes) - 1, -10, f'Test MAE: {test_mae:.4f}\nTest Loss: {test_loss:.4f}', fontsize=label_fontsize, ha='right')
    elif test_mae is not None:
        plt.text(len(classes) - 1, -10, f'Test MAE: {test_mae:.4f}', fontsize=label_fontsize, ha='right')
    elif test_loss is not None:
        plt.text(len(classes) - 1, -10, f'Test Loss: {test_loss:.4f}', fontsize=label_fontsize, ha='right')

    plt.tight_layout(pad=2.0)  # Adjust padding to ensure labels are not cut off

# Plot normalized confusion matrix
plot_confusion_matrix(confusion_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix', test_mae=test_mae)

plt.savefig(f'{fig_path}/normalized_confusion_matrix_temp.png')
# plt.show()