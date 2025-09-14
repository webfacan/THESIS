from models_google import AtzoriNet
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import accuracy_score
import keras
import tensorflow
from keras.callbacks import ModelCheckpoint
from data_loader_cut import get_data_generators
from tensorflow import keras
import os
import random
import tensorflow as tf


SEED = 42  
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

input_shape = (15, 10, 1)
classes = 25
initial_lr = 0.0005
SUBJECTS = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def step_decay(epoch):
    drop_rate = 0.5  
    epochs_drop = 25
    lr = initial_lr * (drop_rate ** (epoch // epochs_drop))
    print(f"Epoch {epoch+1}: Learning Rate = {lr}")
    return lr

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

metrics = [] 

for subject_id in SUBJECTS:

    print(f"Training for Subject: {subject_id}")
    
    
    batch_size=32
    train_generator, val_generator = get_data_generators(subject_id,batch_size= batch_size)

    epochs = 50
    n_pool = 'max'
    n_dropout = 0.3
    #n_l2 = 0.005
    n_l2 = 0.0005
    #n_l2 = 0.001
    n_init = 'glorot_normal'
    batch_norm = True

    model = AtzoriNet(input_shape=input_shape, 
                    classes=classes, 
                    n_pool=n_pool, 
                    n_dropout=n_dropout, 
                    n_l2=n_l2, 
                    n_init=n_init, 
                    batch_norm=batch_norm)

    optimizer = SGD(learning_rate=initial_lr, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    
    checkpoint = ModelCheckpoint(f"Subject_{subject_id}_best_model.h5", monitor='val_loss', save_best_only=True, verbose=1)

    callback_dir = './callback_dir'
    tensorboard_callback = TensorBoard(log_dir=callback_dir, histogram_freq=1)
    lr_scheduler = LearningRateScheduler(step_decay)

    history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[checkpoint, EpochLogger(), tensorboard_callback,lr_scheduler])

    # Predictions and Accuracy Calculation
    y_true = []  
    y_pred = []  

    for X, y in val_generator:
        y_true.append(np.argmax(y, axis=-1))  
        predictions = model.predict(X)  
        y_pred.append(np.argmax(predictions, axis=-1))  

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    manual_accuracy = accuracy_score(y_true, y_pred)
    print(f'Subject {subject_id} Manual Accuracy: {manual_accuracy:.4f}')

    # Evaluate
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
    print(f"Subject {subject_id} -> Val Accuracy: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}")
    
    metrics.append({
        "subject": subject_id,
        "val_accuracy": val_accuracy,
        "val_loss": val_loss
    })

    model.save(f"subject_{subject_id}_final_model.h5")


average_acc = np.mean([m['val_accuracy'] for m in metrics])
average_loss = np.mean([m['val_loss'] for m in metrics])

for m in metrics:
    print(f"Subject {m['subject']}: Accuracy = {m['val_accuracy']:.4f}, Loss = {m['val_loss']:.4f}")

print(f"\n Mean Validation Accuracy: {average_acc:.4f}")
print(f" Mean Validation Loss: {average_loss:.4f}")
