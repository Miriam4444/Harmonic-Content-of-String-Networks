import os
from pathlib import Path

class Constants:

    def __init__(self):

        # How far away from an integer does a harmonic need to be to be considered an integer
        self.integerThreshold = .1

        # path name of the directory with the audio files
        self.pathName = Path(r"C:\Users\abeca\OneDrive\ICUNJ_grant_stuff\ICUNJ-grant-audiofiles")

        # file type to be added to the list of files
        self.fileSuffix = ".wav"

        # the lower bound to check for the fundamental frequency
        self.fundamentalLowerBound = 200

        # the upper bound to check for the fundamental frequency
        self.fundamentalUpperBound = 400

        # number of harmonics to search for
        self.numOfHarmonics = 16

        # where should we start looking for harmonics (fundamental - x, what is x?)
        self.harmonicFinderStartingPoint = 50 #meaning, we are beginning to look for harmonics 50 Hz below the fundamental frequency

        # we find the highest x amount of peaks where x is the number of frequencies in the window (determined in the windowedPeaks method's arguments)
        # the percentile that we're using is the percentile of the lowest frequency in that list
        self.percentileOfLowestSelectedPeak = 80

        # how close a peak needs to be to another peak to be rejected in terms of what the fundamental should be divided by
        self.distance = 10 #meaning the distance is the fundamental/10

        #size of ratio array graph (it's a square)
        self.ratioArrayGraphSize = 8


