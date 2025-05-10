import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.io import loadmat
import keras

FS = 2000
SUBSAMPLE_FACTOR = 20

#Warning suppression
warnings.filterwarnings('ignore', category=FutureWarning)

# Butterworth 1 Hz low-pass filter
def lpf(x, f=1., fs=FS):
    f = f / (fs / 2)
    x = np.abs(x)
    b, a = scipy.signal.butter(1, f, 'low')
    output = scipy.signal.filtfilt(b, a, x, axis=0, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))
    return output

#Subsample rest data
def subsample_rest_data(rest_data, rest_reps=10):
    rest_data = rest_data[:rest_reps]
    return rest_data

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

def augmentation(x):
    x_aug = []
    for i in range(len(x)):
         x_aug.append(x[i])
         x_aug.append(jitter(x[i]))
    return np.array(x_aug, dtype=object)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, x_data, y_data, batch_size=32, window_size=15, window_step=6, dim=(15,10,1), classes=10, shuffle=True):
        
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.window_size = window_size
        self.window_step = window_step
        self.classes = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.shuffle = shuffle

        self.__make_segments()

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        #Batches per epoch
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        #Creates a batch of data
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation(indexes)
    
    def on_epoch_end(self):
        #Updates indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        #Creates data for a batch
        X = np.empty((self.batch_size, self.window_size, 1))  # Σχήμα εισόδου για 1D δεδομένα
        y = np.empty((self.batch_size), dtype=int)

        for k, index in enumerate(indexes):
            i, j = self.x_offsets[index]
            x_aug = np.copy(self.x_data[i][j:j + self.window_size])  #Sliding window
            X[k, ] = np.reshape(x_aug, (self.window_size, 1))  #Reshape for the model
            y[k] = self.y_data[i]

        y = keras.utils.to_categorical(y, num_classes=len(self.classes))  #One-hot encoding
        return X, y

    def __make_segments(self):
        #Windowing for all data
        x_offsets = []
        for i in range(len(self.x_data)):
            for j in range(0, len(self.x_data[i]) - self.window_size, self.window_step):
                x_offsets.append((i, j))
        self.x_offsets = x_offsets
        self.indexes = np.arange(len(self.x_offsets))


#Load EMG data
def load_emg_data(data_dir, subjects, gestures, reps):
    x, y = [], []

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
                            data = lpf(data)
                            rest_data.append(data)

                rest_data = subsample_rest_data(rest_data, rest_reps=10)           

                for data in rest_data:
                    x.append(data)
                    y.append(gesture)

            else:
                for rep in reps:
                    file_path = os.path.join(gesture_dir, f"rep-{rep:02d}.mat")
                    if os.path.exists(file_path):
                        data = loadmat(file_path)['emg']
                        data = lpf(data)
                        x.append(data)
                        y.append(gesture)
      
    x = augmentation(x)

    return np.array(x, dtype=object), np.array(y, dtype=object)


# Example usage
if __name__ == "__main__":
    DATA_DIR = "C:/Users/User/Desktop/THESIS2/Ninapro-DB1-Proc"
    SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    GESTURES = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    REPS = list(range(1, 11))

    x, y = load_emg_data(DATA_DIR, SUBJECTS, GESTURES, REPS)
    print(f"Loaded EMG data shape: {x.shape}")
    
    data_generator = DataGenerator(x_data=x, y_data=y, batch_size=32, window_size=15, window_step=6, classes=52, shuffle=True)

    # Example of using the DataGenerator
    for X_batch, y_batch in data_generator:
        print(X_batch.shape, y_batch.shape)
        break  # Για να δούμε μόνο το πρώτο batch

    
