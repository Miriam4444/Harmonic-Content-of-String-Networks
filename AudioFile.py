import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
import librosa as lib
import scipy as sci
import statistics as stat
import os
from pathlib import Path
from collections import Counter
from typing import Any
import re
from collections import defaultdict
from DataAnalysis import DataAnalysis
from AudioFilesArray import AudioFilesArray
from Constants import Constants

NDArray = np.ndarray[Any, np.dtype[np.float64]]

class AudioFile:

    def __init__(self, file: str):

        ##############################################
        # attributes of the audiofile object
        ##############################################
        # load wav file as array and store sample rate
        self.source, self.sr =  lib.load(file, sr=None) 

        # file name without path
        self.file: str = os.path.basename(file)

        # compute and store real fft of audio array
        self.fourier: NDArray = np.fft.rfft(self.source)

        # store number of samples in original file
        self.N: int = len(self.source)

        # store time array for original signal
        self.time: NDArray = np.arange(self.N)/self.sr
        self.lengthinseconds: float = self.N/self.sr

        # store original sample's frequency bins
        self.bins: NDArray = np.arange(len(self.fourier))

        # compute the magnitude spectrum of the rfft
        self.magspec: NDArray = np.abs(self.fourier)

        # compute the corresponding frequencies of the rfft in Hz
        # first uses the built-in rfft frequency calculator from numpy
        # third calculates by hand accounting for the rfft cutting the frequency bins in half
        self.freq: NDArray = np.fft.rfftfreq(len(self.source), 1/self.sr)

        # identify fundamental as freq of largest coefficient in the power spectrum 
        # -- this can lead to erroneous results, we need to be careful in how we define this!
        self.dummyfundamental: int = self.getfund()

        self.unfilteredpeaks, self.peakproperties = sci.signal.find_peaks(self.magspec, height = 2, distance = self.dummyfundamental*self.N/self.sr//4)
        self.prominences = sci.signal.peak_prominences(self.magspec, self.unfilteredpeaks)
        self.meanProminence = stat.mean(self.prominences[0])
        

    ##########################################
    # methods
    ##########################################
    def printN(self) -> None:
        print("the length (in samples) of the original file is", self.N)

    def printmagspec(self) -> None:
        print("the magnitude spectrum of the RFFT is", self.magspec)

    def printfreq(self) -> None:
        print("the frequencies in Hz of the magnitude spectrum are", self.freq)

    def printfundamental(self) -> None:
        print("the fundamental frequency of the signal is", self.dummyfundamental)

    def printpeaks(self) -> None:
        print(f"the unfiltered peaks of {self.file} are", self.unfilteredpeaks) 

    def printratios(self, percentile: float) -> None:
        print(f"the ratio array of {self.file} is\n", self.findratioArray(percentile=percentile))

    def printError(self, percentile: float) -> None:
        mean, stdev = self.findAbsoluteError(percentile=percentile)
        print(f"{self.file} has mean error {mean}\n and stdev of error {stdev}\n from {len(self.ratioArray)} datapoints")

    # class method to find the array of ratios: peak frequency/fundamental frequency
    def findratioArray(self, percentile: float) -> NDArray:

        P = np.zeros(len(self.findpeaks(percentile=percentile)))

        for i in range(len(P)):
            P[i] = self.freq[self.findpeaks(percentile=percentile)[i]]

        fund = self.dummyfundamental

        P = P/fund

        return P
    
    # returns the mean and stdev of the absolute error array corresponding to input percentile of prominences
    def findAbsoluteError(self, percentile: float) -> tuple[float,float]:
        E = self.findAbsoluteErrorArray(percentile=percentile)
        return stat.mean(E), stat.stdev(E)
    
    def findAbsoluteErrorArray(self,percentile: float) -> NDArray:
        # create an array of the correct length
        E = np.zeros(len(self.findratioArray(percentile=percentile)))

        for i in range(len(self.findratioArray(percentile=percentile))):
            E[i] = np.abs(self.findratioArray(percentile=percentile) - np.rint(self.findratioArray(percentile=percentile)))[i]

        return E
    
    def getfund(self) -> int:
        # initialize min and max indices to 0
        min = 0
        max = 0

        # find first index where the frequency exceeds 200
        for i in range(0, len(self.freq)-1):
            if self.freq[i] >= Constants.fundamentalLowerBound:
                min = i
                break

        # find first index where the frequency exceeds 400
        for j in range(min,len(self.freq)-1):
            if self.freq[j] >= Constants.fundamentalUpperBound:
                max = j
                break

        # search for loudest frequency only between 200 and 400 Hz.  Will return relative to min=0.
        F = np.argmax(self.magspec[min:max])

        # convert magspec index back to Hz
        F = self.freq[F+min]

        return F
    
    
    def windowedPeaks(self, numberFundamentalsInWindow: int, percentile: float) -> NDArray:
        # number of suspected harmonics we want to be present in our window (integer multiple of fundamental freq)
        numberFundamentalsInWindow = int(numberFundamentalsInWindow)

        # ratio to convert from Hz to bins
        R = self.N/self.sr

        # fundamental freq in bins
        fund = self.dummyfundamental*R

        loPass = Constants.numOfHarmonics*self.dummyfundamental*R

        # hiPass the magnitude spectrum to cut room noise
        signal = AudioFile.filtersignal(self.magspec, fund - Constants.harmonicFinderStartingPoint, loPass, 0)

        # initial minimal index for our window
        minIndex = round(fund/2)

        # total number of windows we will consider
        numberWindows = round((loPass-fund)/(fund*numberFundamentalsInWindow))

        # initialize an empty array of peaks to be populated later
        peaks = np.array([], dtype=int)

        for i in range(numberWindows):

            # width of the window that will slide through the signal to ID peaks
            windowWidth = round(numberFundamentalsInWindow*fund)

            # window indices in bins
            window = self.bins[minIndex + i*windowWidth: minIndex + (i+1)*windowWidth]

            #run through original array, and determine the threshold to accept frequencies
            tempPeaks = AudioFile.staticfindpeaks(signal[window], percentile=Constants.percentileOfLowestSelectedPeak, height=0, distance=fund//Constants.distance)
            
            tempPeakHeights = signal[tempPeaks + minIndex + i*windowWidth]

            # sort the indices of the peaks from shortest to tallest
            tempPeakHeightIndices = np.argsort(tempPeakHeights)

            # rearrange the peak heights in ascending order
            tempPeakHeights = tempPeakHeights[tempPeakHeightIndices]

            # set the threshold to the percentile/100 * (shortest suspected harmonic spike in window)
            threshold = (percentile/100)*tempPeakHeights[len(tempPeaks)-numberFundamentalsInWindow]

            tempPeaks = AudioFile.staticfindpeaks(signal[window], percentile=1, height=threshold, distance=fund//6)

            #windowedRatioArray = (tempPeaks + minIndex + i*windowWidth)/fund

            #E = round(stat.mean(np.abs(windowedRatioArray - np.rint(windowedRatioArray))),3)
            #StD = round(stat.stdev(np.abs(windowedRatioArray - np.rint(windowedRatioArray))),3)
            
            peaks = np.concatenate((peaks,tempPeaks + minIndex + i*windowWidth))
        return peaks
    
    @staticmethod
    def filtersignal(array: NDArray, loFthresh:float, hiFthresh:float, Athresh:float) -> NDArray:
        loFthresh = int(loFthresh)
        hiFthresh = int(hiFthresh)

        # create an array of zeroes followed by ones to filter below frequency threshold (Fthresh)
        Z = np.zeros(loFthresh)
        oneslength = len(array)-loFthresh
        Arr01 = np.ones(oneslength)
        Arr01 = np.concatenate((Z,Arr01))

        # zero out all array entries below the frequency threshold
        filteredArr = Arr01*array

        if hiFthresh==None:
            # zero out all array entries below the amplitude threshold (Athresh)
            for i in range(len(array)):
                if np.abs(array[i]) < Athresh:
                    filteredArr[i] = 0

            return filteredArr

    @staticmethod
    # static version of the above method for finding peaks in a given array 
    # with a certain prominence that is above lowest percentile of prominences
    def staticfindpeaks(array: NDArray, percentile: float, height: float = None, distance: float = None) -> NDArray:
        if height == None:
            height = 0
        if distance == None:
            distance = 1

        #peaks, peakproperties = sci.signal.find_peaks(array, height = height, distance = distance)

        peaks = sci.signal.find_peaks(array, height = height, distance = distance)[0]

        if peaks.shape[0] == 0:
            meanProminence = 0
        else:
            meanProminence = stat.mean(sci.signal.peak_prominences(array, peaks)[0])

        percentile = percentile/100*meanProminence
        
        peaks, peakproperties = sci.signal.find_peaks(array, height = height, prominence = percentile, distance = distance)

        return peaks

    # method for plotting the magnitude spectrum of a given audiofile object with 
    # the peaks (computed using the windowedPeaks method) visually illustrated 
    def graph_magspec_withWindowedPeaks(self, percentile: float, numberFundamentalsInWindow: int = 5) -> None:
        windowedPeaks = self.windowedPeaks(percentile=percentile, numberFundamentalsInWindow=numberFundamentalsInWindow)
        
        peakHeight = np.zeros(len(windowedPeaks))
        
        for i in range(len(windowedPeaks)):
            peakHeight[i] = self.magspec[windowedPeaks[i]]

        R = self.sr/self.N

        plt.figure(figsize=(8,8))

        plt.plot(self.freq, self.magspec)
        plt.scatter(windowedPeaks*R,peakHeight,c='orange',s=12)
        plt.xlabel('frequency (Hz)')
        plt.ylabel('Magnitude of RFFT')
        plt.title(f'Magnitude spectrum of {self.file}, window width {numberFundamentalsInWindow}, percentile {percentile}')
        plt.savefig(f'windowpeaks-{percentile}perc-{self.file}.png')
        #plt.clf()
        plt.show()
    
    @staticmethod
    def printAggregateError(directory: str, numberOfFundamentalsInWindow: int, percentile: float, badData: list = None, SpecificType: str = None) -> dict:
        nameArray = AudioFilesArray(Path(directory))

        if SpecificType != None:
            namelist = nameArray.getSpecificType(SpecificType)
        else:
            namelist = nameArray.getSpecificType("1S")
            print("No additional type information was given (e.g. 1S, 2S, 2S9, 2SC, etc.) so default of 1S was used.")

        # initialize an empty |audiofiles| array to be populated with audiofile arrays
        objArray = np.empty(len(namelist), dtype=AudioFile)
        for i in range(len(namelist)):
            objArray[i] = AudioFile(namelist[i])

        # initialize an empty |audiofiles| array to be populated with the meanerror of each audiofile
        M = np.empty(len(namelist))

        meanofmeans = list()
        datapointsArray = list()
        fundamentals = np.empty(len(namelist))

        #open(f"AggError-{SpecificType}-{numberOfFundamentalsInWindow}-{percentile}.txt", "w").close()
        
        fundamentalVsNonInt = {}
        for i in range(len(objArray)):
            fundamentals[i] = round(objArray[i].dummyfundamental)

            R = objArray[i].N/objArray[i].sr
            fund = objArray[i].dummyfundamental*R
            
            windowedPeaks = objArray[i].windowedPeaks(numberOfFundamentalsInWindow, percentile)

            windowedRatioArray = windowedPeaks/fund

            counter = 0

            if badData != None:
                DA = DataAnalysis(windowedRatioArray)
                for value in badData:
                    counter = counter + DA.checkIfDecimalClose(decimal= value, roundingPlace=1)
                windowedRatioArray = DA.array
            

            E = round(stat.mean(np.abs(windowedRatioArray - np.rint(windowedRatioArray))),3)

            StD = round(stat.stdev(np.abs(windowedRatioArray - np.rint(windowedRatioArray))),3)

            M[i] = E

            datapointsArray.append(len(windowedRatioArray))



            #print(f'{objArray[i].file}, mean error = {M[i]}, # datapoints = {datapointsArray[i]}, # removed = {counter}')
            DA = DataAnalysis(windowedRatioArray)
            #DA.checkDataTextFile(sampleValue=0.2, fileName=f"AggError-{SpecificType}-{numberOfFundamentalsInWindow}-{percentile}.txt")
            
            with open(f"AggError-{SpecificType}-{numberOfFundamentalsInWindow}-{percentile}.txt", "a") as f:
                f.write(f'{objArray[i].file}, fundamental = {round(objArray[i].dummyfundamental)}, mean error = {M[i]}, # datapoints = {datapointsArray[i]}, # removed = {counter}\n')
                #f.write(f'{DA.checkData(sampleValue=0.2)}\n')
            nonInt = DA.checkDataTextFile(sampleValue=0.2, fileName=f"AggError-{SpecificType}-{numberOfFundamentalsInWindow}-{percentile}.txt")
            for i in enumerate(nonInt):
                fundamentalVsNonInt[round(objArray[i].dummyfundamental)]= i

        m = stat.mean(M)

        with open(f"AggError-{SpecificType}-{numberOfFundamentalsInWindow}-{percentile}.txt", "a") as f:
            f.write(f'mean of mean absolute errors = {m}\n')
        
        return fundamentalVsNonInt
    
    @staticmethod
    def roundEntries(fileName : str, roundingValue : int) -> list:
        all_entries = AudioFile.analyzeTextFile(fileName)
        roundedEntries = []
        duplicates = []
        for i in all_entries:
            if all_entries.count(i) != 0:
                duplicates.append([f'{i} , number of times: {all_entries.count(i)}'])
        return duplicates
    
    @staticmethod
    def analyzeTextFile(file_name : str) -> NDArray:
        file_entries = []
        current_entries = []
        with open(file_name, "r") as file:
            lines = file.readlines()
        
        current_file = None
        for line in lines:
            # Check if line starts with a file name (e.g., 1SCD01.wav)
            match = re.match(r"^([\w\d]+\.wav),", line)
            if match:
                if current_entries:
                    file_entries.extend(current_entries)
                current_file = match.group(1)
                current_entries = []
            elif current_file:
                entry_match = re.search(r'Entry: "([\d.]+)"', line)
                if entry_match:
                    entry_value = entry_match.group(1)
                    current_entries.append(entry_value)
        if current_entries:
            file_entries.extend(current_entries)
        
        return file_entries
    
    @staticmethod
    def roundEntries(fileName : str, roundingValue : int) -> list:
        all_entries = AudioFile.analyzeTextFile(fileName)
        roundedEntries = []
        duplicates = []
        for i in all_entries:
            if all_entries.count(i) != 0:
                duplicates.append([f'{i} , number of times: {all_entries.count(i)}'])
        return duplicates
    
    # method to plot the actual harmonic ratio array of the signal against the predicted ratio array
    # also saves the figure to a file with all relevant info in the file name
    def graphRatioArray(self, percentile: float) -> None:
        idealRatioArray = np.rint(self.findratioArray(percentile=percentile))

        # the following is logic to parse the error array into positive and negative errors
        # so that the error bars can be plotted appropriately on the figure
        errorArray = idealRatioArray - self.findratioArray(percentile)

        positiveErrors = np.ones(len(idealRatioArray))
        negativeErrors = np.zeros(len(idealRatioArray))

        for i in range(len(idealRatioArray)):
            if errorArray[i] < 0:
                positiveErrors[i] = 0
                negativeErrors[i] = -1

        yerr = [errorArray, errorArray]

        yerr[1] = yerr[0]*positiveErrors
        yerr[0] = yerr[1]*negativeErrors

        # plot the ideal ratio array

        plt.figure(figsize=(Constants.ratioArrayGraphSize,Constants.ratioArrayGraphSize))
        
        plt.plot(idealRatioArray,idealRatioArray, label='theoretical')

        # plot the mean error for this sample in the bottom right 
        plt.text(np.max(idealRatioArray)-0.01, 1, f'mean abs. error = {round(self.findAbsoluteError(percentile=percentile)[0],3)}\n # datapoints = {len(self.findratioArray(percentile=percentile))}', ha='right', va='bottom')

        # plot the actual ratio array values including error bars
        plt.errorbar(idealRatioArray, self.findratioArray(percentile), yerr=yerr,
                     label='actual', c='orange', marker='d', markersize=6, 
                     linestyle='dotted', capsize=2)
        plt.xticks(idealRatioArray)
        plt.xlabel('harmonic number')
        plt.ylabel('harmonic ratio')
        plt.title(f'Actual vs Th. harmonic ratios - {self.file}, P={percentile}% prominence threshold')
        plt.legend()
        plt.savefig(f'ratioarray-prom{percentile}-{self.file}.png')

        # clears the figure to avoid overlays from successive iterations
        plt.clf()

        def hanningWindow(self) -> NDArray:
            H = np.hanning(len(self.source))
            return self.source*H
        

    
