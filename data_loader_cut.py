import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import keras
import warnings
import random
from tensorflow import keras
import tensorflow as tf


SEED = 42
random.seed(SEED)        
np.random.seed(SEED)     
tf.random.set_seed(SEED)

FS = 100
SUBSAMPLE_FACTOR = 20

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

# Butterworth 1 Hz low-pass filter
def lpf(x, f=1., fs=FS):
    f = f / (fs / 2)
    x = np.abs(x)
    b, a = scipy.signal.butter(1, f, 'low')
    output = scipy.signal.filtfilt(b, a, x, axis=0, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))
    return output

def trim_data(data, target_length):
    if len(data) > target_length:
        cut = (len(data) - target_length) // 2
        return data[cut:cut + target_length]
    else:
        return data

#Add Gaussian Noise
def jitter(x, snr_db=25):
    if isinstance(snr_db, list):
        snr_db_low = snr_db[0]
        snr_db_up = snr_db[1]
    else:
        snr_db_low = snr_db
        snr_db_up = 45
    snr_db = np.random.randint(snr_db_low, snr_db_up, (1,))[0]
    snr = 10 ** (snr_db/10)
    Xp = np.sum(x**2, axis=0, keepdims=True) / x.shape[0]
    Np = Xp / snr
    n = np.random.normal(size=x.shape, scale=np.sqrt(Np), loc=0.0)
    xn = x + n
    return xn

def augmentation(x, y):
    x_aug = []
    y_aug = []
    for i in range(len(x)):
        x_aug.append(x[i])
        y_aug.append(y[i])
        
        jittered = jitter(x[i])  
        x_aug.append(jittered)
        y_aug.append(y[i])  
    
    return np.array(x_aug, dtype=object), np.array(y_aug, dtype=object)


#Data Generator
class DataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size=64 , dim=(15,10,1), classes=5, window_size=15, window_step=6, shuffle=True):
        self.x = x  
        self.y = y  
        self.batch_size = batch_size
        self.shuffle = shuffle
        #self.min_max_norm = min_max_norm
        self.dim = dim
        self.window_size = window_size
        self.window_step = window_step
        self.classes = list(range(0,25)) 

        #print("Before",np.unique(self.y))
        LE = LabelEncoder()
        LE.fit(self.classes) 
        self.classes = list(LE.fit_transform(self.classes))
        self.y = LE.transform(self.y)
        #print("After", np.unique(self.y)) 

        self.indexes = np.arange(len(self.x))
        self.__make_segments()
        self.__make_class_index()
        #self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.x_offsets) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        output = self.__data_generation(indexes)
        return output
    
    def __make_segments(self):
        x_offsets = []
        for i in range(len(self.x)):
            for j in range(0, len(self.x[i]) - self.window_size, self.window_step):
                x_offsets.append((i, j))
                #if i< 5:
                    #print(f"Sample {i} | Window starting at {j}")
        
        self.x_offsets = x_offsets
        self.indexes = np.arange(len(self.x_offsets))
        #print(f"\nTotal windows: {len(self.x_offsets)}")
        #print(f"First 20 windows: {self.x_offsets[:20]}")


    def __make_class_index(self):
        
        self.n_classes = len(self.classes)
        self.classes.sort()
        self.class_index = np.zeros(np.max(self.classes)+1, dtype=int) 
        for i, j in enumerate(self.classes):
            self.class_index[j] = i

    
    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, *self.dim))            
        y = np.empty((self.batch_size), dtype=int)

        
        for k, index in enumerate(indexes):
            i, j = self.x_offsets[index]
            
            x = self.x[i][j:j + self.window_size]  

            #print(f"Sample {i} : Window {k}: Starting at index {j}")

            #if self.min_max_norm:
                #max_x = x.max()
                #min_x = x.min()
                #x = (x - min_x) / (max_x - min_x)  

            if np.prod(x.shape) == np.prod(self.dim):
                x = np.reshape(x, self.dim)  
            else:
                raise Exception(f'Generated sample dimension mismatch. Found {x.shape}, expected {self.dim}.')

            X[k, ] = x
            #print("self.indexes[i]:", self.indexes[i])
            #print("self.y shape:", self.y.shape)
            
            stimulus = int(self.y[i])
            mapped_class = self.class_index[stimulus]
            y[k] = mapped_class
            one_hot = to_categorical([mapped_class], num_classes=len(self.classes))[0]
            #print(f"Window {k} | Original stimulus: {stimulus}, Mapped class: {mapped_class}, One-hot: {one_hot}")


        #print(f"Labels before {y}")
        y = to_categorical(y, num_classes=len(self.classes)) 
        #print(f"labels after: {(y)}")
            
        #y = keras.utils.to_categorical(y, num_classes=self.n_classes+1)
        #print(f"One-hot-encoded {y.tolist()}")

        output = (X, y)
        return output

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)



#Load EMG data
def load_emg_data(data_dir, subjects, gestures, reps):
    x, y = [], []

    #l = float('inf')
    l = 180
    for subject in subjects:
        for gesture in gestures:
            gesture_dir = os.path.join(data_dir, f"subject-{subject:02d}", f"gesture-{gesture:02d}", "rms")
            if gesture == 0:
                for rep in reps:
                    for subrep in range(1, 53):
                        file_path = os.path.join(gesture_dir, f"rep-{rep:02d}_{subrep:02d}.mat")
                        if os.path.exists(file_path):
                            data = loadmat(file_path)['emg']
                            if len(data) < l:
                                l = len(data)
            else:
                for rep in reps:
                    file_path = os.path.join(gesture_dir, f"rep-{rep:02d}.mat")
                    if os.path.exists(file_path):
                        data = loadmat(file_path)['emg']
                        if len(data) < l:
                            l = len(data)
                            
    FS = 100 
    print(f"Minimum length across all trials: {l} samples")
    print(f"Central seconds that I keep: {l / FS:.2f} s")

    print(f"Minimum length across all samples: {l}")

    # Load, trim & lpf
    for subject in subjects:
        for gesture in gestures:
            gesture_dir = os.path.join(data_dir, f"subject-{subject:02d}", f"gesture-{gesture:02d}", "rms")

            if gesture == 0:
                rest_data = []
                for rep in reps:
                    for subrep in range(1, 53):
                        file_path = os.path.join(gesture_dir, f"rep-{rep:02d}_{subrep:02d}.mat")
                        if os.path.exists(file_path):
                            data = loadmat(file_path)['emg']
                            data = trim_data(data, l)
                            data = lpf(data)
                            rest_data.append(data)
                            gesture_stim = int(loadmat(file_path)['stimulus'][0][0])

                rest_data = rest_data[:len(reps)]  

                for data in rest_data:
                    x.append(data)
                    y.append(gesture_stim)

            else:
                for rep in reps:
                    file_path = os.path.join(gesture_dir, f"rep-{rep:02d}.mat")
                    if os.path.exists(file_path):
                        data = loadmat(file_path)['emg']
                        data = trim_data(data, l)
                        data = lpf(data)
                        gesture_stim = int(loadmat(file_path)['stimulus'][0][0])
                        x.append(data)
                        y.append(gesture_stim)

    x, y = augmentation(x, y)
    print("Distribution", np.unique(y, return_counts=True))

    max_x = max([arr.max() for arr in x])
    min_x = min([arr.min() for arr in x])
    x = [(arr - min_x) / (max_x - min_x) for arr in x]

    return np.array(x, dtype=object), np.array(y, dtype=object)


def get_data_generators(subject_id, batch_size=32):
    DATA_DIR = "/content/drive/MyDrive/THESIS2/Ninapro-DB1-Proc"
    GESTURES = list(range(0,25))  
    REPS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Data split (80% training, 20% validation)
    random_state = 42
    train_ratio = 0.8
    val_ratio = 1 - train_ratio
    
    REPS = np.array(REPS)
    
    #ng = np.random.default_rng()  # χωρίς 42
    #rng.shuffle(REPS)
    
    #SEED 
    np.random.shuffle(REPS)
    #rng = np.random.default_rng(42)  
    #rng.shuffle(REPS)

    n_train = int(len(REPS) * train_ratio)
    TRAIN_REPS = REPS[:n_train].tolist()
    VAL_REPS = REPS[n_train:].tolist()

    #random.seed(random_state)
    #random.sample(REPS, len(REPS))
    #random.shuffle(REPS)

    #n_train = int(len(REPS) * train_ratio)
    #TRAIN_REPS = REPS[:n_train]
    #VAL_REPS = REPS[n_train:]
    print(f"Training Reps: {TRAIN_REPS}")
    print(f"Validation Reps: {VAL_REPS}")
    print(f"Subject ID: {subject_id}")
    
    x_train, y_train = load_emg_data(DATA_DIR,[subject_id], GESTURES, TRAIN_REPS)
    x_val, y_val = load_emg_data(DATA_DIR, [subject_id], GESTURES, VAL_REPS)
    print(f"Loaded EMG data shape (train): {x_train.shape}")
    print(f"Loaded EMG data shape (val): {x_val.shape}")

    batch_size = 32
    train_generator = DataGenerator(x_train, y_train, batch_size=batch_size, window_size=15, window_step=6)
    X, y = next(iter(train_generator))
    print("Batch X shape:", X.shape)
    print("Batch y shape:", y.shape)

    val_generator = DataGenerator(x_val, y_val, batch_size=batch_size, window_size=15, window_step=6, shuffle=False)
    
    return train_generator, val_generator

if __name__ == "__main__":
    train_generator, val_generator = get_data_generators()
    

   
        
    


        
