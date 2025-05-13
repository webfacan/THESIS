from models import AtzoriNet
from keras.models import Model
from keras.layers import Input, Dropout, Flatten, Dense, BatchNormalization
import keras
from keras.callbacks import ModelCheckpoint
from mine import get_data_generators

train_generator, val_generator = get_data_generators()

input_shape = (15, 10, 1)
classes = 10
n_pool = 'average'
n_dropout = 0.15
n_l2 = 0.0002
n_init = 'glorot_normal'
batch_norm = False

model = AtzoriNet(input_shape=input_shape, 
                  classes=classes, 
                  n_pool=n_pool, 
                  n_dropout=n_dropout, 
                  n_l2=n_l2, 
                  n_init=n_init, 
                  batch_norm=batch_norm)

from keras.optimizers import SGD
initial_lr = 0.05
optimizer = SGD(lr=initial_lr)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

class EpochLogger(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print("==========================================")
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

from keras.callbacks import LearningRateScheduler

def step_decay(epoch):
    drop_rate = 0.5  
    epochs_drop = 30
    lr = initial_lr * (drop_rate ** (epoch // epochs_drop))
    return lr

lr_scheduler = LearningRateScheduler(step_decay)
checkpoint = ModelCheckpoint("best_model.h5", monitor='loss', save_best_only=True, verbose=1)

epochs = 50
history = model.fit_generator(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[lr_scheduler, checkpoint, EpochLogger()]
)

print("Training completed.")
model.save('atzori_model.h5')

