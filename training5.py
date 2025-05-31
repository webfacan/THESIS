from models import AtzoriNet
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import keras
from keras.callbacks import ModelCheckpoint
from data_loader import get_data_generators
from tensorflow import keras

train_generator, val_generator = get_data_generators()

input_shape = (15, 10, 1)
classes = 25
n_pool = 'max'
n_dropout = 0.2
n_l2 = 0.005
n_init = 'glorot_normal'
batch_norm = True

model = AtzoriNet(input_shape=input_shape, 
                  classes=classes, 
                  n_pool=n_pool, 
                  n_dropout=n_dropout, 
                  n_l2=n_l2, 
                  n_init=n_init, 
                  batch_norm=batch_norm)

from tensorflow.keras.optimizers import SGD

initial_lr = 0.0005
optimizer = SGD(learning_rate=initial_lr, momentum=0.9)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])



class EpochLogger(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting Epoch {epoch+1}")
        print("==========================================")

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss', 0)
        train_acc = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)

        print(f"End of Epoch {epoch+1}:")
        print(f"  Training: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")
        print(f"  Validation: Loss = {val_loss:.4f}, Accuracy = {val_acc:.4f}")
        print("---------------------------------------------------")
        
        from sklearn.metrics import accuracy_score
        from keras.losses import CategoricalCrossentropy
        
        val_predictions = []
        val_true_classes = []

        from tqdm import tqdm

        for i in tqdm(range(len(val_generator)), desc="Validation batches"):
            val_data, val_labels = val_generator[i]
            if val_data.ndim == 3:
                val_data = np.expand_dims(val_data, axis=-1)  

            val_pred = self.model.predict(val_data, verbose=0)
            val_predictions.append(val_pred)
            val_true_classes.append(val_labels)

        #Concatenate all predictions and true labels
        val_predictions = np.concatenate(val_predictions, axis=0)
        val_true_classes = np.concatenate(val_true_classes, axis=0)


        #True Labels from One-Hot Encoding
        val_pred_classes = np.argmax(val_predictions, axis=1)
        val_true_classes = np.argmax(val_true_classes, axis=1)

        print("Classification Report:")
        #print(classification_report(val_true_classes, val_pred_classes))

        from sklearn.metrics import accuracy_score

        manual_accuracy = accuracy_score(val_true_classes, val_pred_classes)
        print(f"Manual Accuracy: {manual_accuracy:.4f}")

        from keras.losses import CategoricalCrossentropy
        from tensorflow.keras.utils import to_categorical

        val_true_classes_onehot = to_categorical(val_true_classes, num_classes=classes)
        loss_fn = CategoricalCrossentropy()
        manual_loss = loss_fn(val_true_classes_onehot, val_predictions).numpy()
        #manual_loss = loss_fn(val_true_classes, val_predictions).numpy()
        print(f"Manual Loss: {manual_loss:.4f}")

        #print("Confusion Matrix:")
        #print(confusion_matrix(val_true_classes, val_pred_classes))
        print("---------------------------------------------------")

from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
import sys

def step_decay(epoch):
    drop_rate = 0.5  
    epochs_drop = 15
    lr = initial_lr * (drop_rate ** (epoch // epochs_drop))
    print(f"Epoch {epoch+1}: Learning Rate = {lr}")
    return lr

callback_dir = './callback_dir'
tensorboard_callback = TensorBoard(log_dir=callback_dir, histogram_freq=1)


#lr_scheduler = LearningRateScheduler(step_decay)
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True, verbose=1)

epochs = 50
history = model.fit_generator(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[checkpoint, EpochLogger(), tensorboard_callback]
)

print("Training completed.")
model.save('atzori_model.h5')
