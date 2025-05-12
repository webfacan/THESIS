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
warnings.filterwarnings('ignore')


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

def augmentation(x, y):
    x_aug = []
    y_aug = []
    for i in range(len(x)):
        x_aug.append(x[i])
        y_aug.append(y[i])
        
        # Jittered version
        jittered = jitter(x[i])  
        x_aug.append(jittered)
        y_aug.append(y[i])  
    
    return np.array(x_aug, dtype=object), np.array(y_aug, dtype=object)

#Data Generator
class DataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size=32,min_max_norm=True, dim=(15,10,1), classes=10, window_size=15, window_step=6, shuffle=True):
        self.x = x  
        self.y = y  
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_max_norm = min_max_norm
        self.dim = dim
        self.window_size = window_size
        self.window_step = window_step
        self.classes = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.indexes = np.arange(len(self.x))
        self.__make_segments()
        self.__make_class_index()
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        output = self.__data_generation(indexes)
        return output
    
    def __make_segments(self):
        x_offsets = []
        for i in range(len(self.x)):
            for j in range(0, len(self.x[i]) - self.window_size, self.window_step):
                x_offsets.append((i, j))
        
        self.x_offsets = x_offsets
        self.indexes = np.arange(len(self.x_offsets))

    def __make_class_index(self):
        self.n_classes = len(self.classes)
        self.classes.sort()
        self.class_index = np.zeros(np.max(self.classes) + 1, dtype=int)
        for i, j in enumerate(self.classes):
            self.class_index[j] = i 
    
    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, *self.dim))            
        y = np.empty((self.batch_size), dtype=int)

        # Δημιουργία δεδομένων
        for k, index in enumerate(indexes):
            i, j = self.x_offsets[index]
            # Αποθήκευση δείγματος
            x = self.x[i][j:j + self.window_size]  

            if self.min_max_norm:
                max_x = x.max()
                min_x = x.min()
                x = (x - min_x) / (max_x - min_x)  

            if np.prod(x.shape) == np.prod(self.dim):
                x = np.reshape(x, self.dim)  # Αναδιάταξη αν είναι απαραίτητο
            else:
                raise Exception(f'Generated sample dimension mismatch. Found {x.shape}, expected {self.dim}.')

            X[k, ] = x

            # Αποθήκευση κλάσης
            y[k] = self.class_index[int(self.y[i])]

        y = keras.utils.to_categorical(y, num_classes=self.n_classes)

        output = (X, y)
        return output

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


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
                            gesture_stim = int(loadmat(file_path)['stimulus'][0][0])
                                 
                rest_data = subsample_rest_data(rest_data, rest_reps=10)           

                for data in rest_data:
                    x.append(data)
                    y.append(gesture_stim)
                
            else:
                for rep in reps:
                    file_path = os.path.join(gesture_dir, f"rep-{rep:02d}.mat")
                    if os.path.exists(file_path):
                        data = loadmat(file_path)['emg']
                        data = lpf(data)
                        gesture_stim = int(loadmat(file_path)['stimulus'][0][0])
                        x.append(data)
                        y.append(gesture_stim)
    #print(x)
    #print(y)  
    x,y = augmentation(x,y)

    return np.array(x, dtype=object), np.array(y, dtype=object)


# Example usage
if __name__ == "__main__":
    DATA_DIR = "C:/Users/User/Desktop/THESIS2/Ninapro-DB1-Proc"
    SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    GESTURES = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    REPS = list(range(1, 11))

    x, y = load_emg_data(DATA_DIR, SUBJECTS, GESTURES, REPS)
    print(f"Loaded EMG data shape: {x.shape}")
    
    # Initialize the data generator
    data_generator = DataGenerator(x, y, batch_size=32, window_size=15, window_step=6)

    # Example of how to get a batch of data
    x_batch, y_batch = data_generator[0]
    print(f"x_batch shape: {x_batch.shape}, y_batch shape: {y_batch.shape}")
   

    
