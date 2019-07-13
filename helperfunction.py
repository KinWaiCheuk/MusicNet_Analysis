import numpy as np                                       # fast vectors and matrices
import matplotlib.pyplot as plt                          # plotting
from scipy import fft, ifft, fftpack

from IPython.display import Audio

from intervaltree import Interval,IntervalTree

class Waveform:
    ''' This Waveform class can do all kinds of processing on waveform
        e.g. Getting spectrograms
    '''
    def __init__(self, X, window_size, stride, length, fs=44100, colorbar=True):
        if X.shape[0] < window_size:
            print("Check window size and ensure that there's enough sample")
        
        wps = fs/stride
        self.stride = stride
        self.fs = fs
        # Check if the audio is long enough
        if length > X.shape[0]/fs:
            print("The audio has only {0:.3f} seconds ".format(X.shape[0]/fs))
            print("Setting the lenght to the maximum audio lenght")
            length = (X.shape[0]-window_size)//fs
        
        self.n = int(length*wps) # Calulate the total number of windows
        self.X = X[:length*fs] # Will be used in show_waveform
        
        self.Xfft = np.empty((self.n,window_size),dtype='complex128' ) # Storing audio info for inverse fft       
        self.Xs = np.empty((self.n,window_size)) # This container of spectrogram will be used in inverse fft  
        self.colorbar = colorbar
        
        for i in range(self.Xs.shape[0]):
                self.Xfft[i] = fft(X[i*stride:i*stride+window_size])
                self.Xs[i] = np.abs(self.Xfft[i]) # Getting Amplitude from fft

    def show_spectrogram(self, figsize=(20, 5), frequency_range=(0,1024)):
        fig = plt.figure(figsize=figsize)
        img = plt.imshow(self.Xs.T[frequency_range[0]:frequency_range[1]],aspect='auto', cmap='jet')
        plt.gca().invert_yaxis()
        fig.axes[0].set_xlabel('windows', size=26)
        fig.axes[0].set_ylabel('frequency bins', size=26)   
        fig.axes[0].tick_params(axis='x',labelsize=24)
        fig.axes[0].tick_params(axis='y',labelsize=24)
        if self.colorbar:
            fig.colorbar(img)
            
    def show_waveform(self, figsize=(20, 5)):
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(self.X)
        ax.set_xlabel('sample', size=26)
        ax.set_ylabel('Signal amplitude', size=26);       
        ax.tick_params(axis="both",labelsize=24)
            
    def get_spectrogram(self):
        return self.Xs
    
    def get_waveform(self):
        return self.X
    
    def get_Xfft(self):
        '''This information can be used to do inverse FFT'''
        return self.Xfft
    
    def play_audio(self):
        return Audio(self.X,rate=self.fs, autoplay=True)
        
        
def reconstruct_waveform(Xfft, window_size, stride):
    counter = 0
    jump = window_size//stride # select frames every {jump} number of index
    Overlap = window_size % stride
    LastFrame_idx = (len(Xfft)*stride)//stride
    
#     # Adjusting the last frame
#     if LastFrame_idx%jump==0:
#         pass
#     else:
#         LastFrame_idx = LastFrame_idx + jump-(LastFrame_idx%jump) # Fix index if the last frame is not the last appened frame
    
#     discarded = ((LastFrame_idx)*stride+window_size)-(len(Xfft)*stride+window_size) # Calculate how many frames are discarded
    reconstruction = []
#     print("{} samples are discarded".format(discarded))
    for frame_idx, frame_data in enumerate(Xfft):
#         print("Idx = {}, End sample = {}".format(frame_idx,frame_idx*stride+window_size))
        inversed_data = ifft(frame_data).real
        if frame_idx % jump==0: # Choose the frames with no or leeast overlapping with the last appended frame
            if Overlap==0: # Check if these any overlapping with the last appended frame
                reconstruction.append(inversed_data)
            else: # If there's overlapping, remove the overlapping frames
                print("Case 2")
                if frame_idx == len(Xfft)-1:
                    lastwindow = inversed_data

                else:
                    reconstruction.append(inversed_data[:-Overlap])
            counter +=2048
#             print("Appended total lenght",counter)
#             print("")
    if 'lastwindow' in locals():
        output = np.append(np.array(reconstruction).reshape(-1), lastwindow)
    else:
        output = np.array(reconstruction).reshape(-1)
    return output