import os
from pathlib import Path

class Constants:
        
    #AudioFile Constants:
    #########################################################

    # the lower bound to check for the fundamental frequency
    FUNDAMENTAL_LOWER_BOUND = 200

    # the upper bound to check for the fundamental frequency
    FUNDAMENTAL_UPPER_BOUND = 400

     # number of harmonics to search for
    NUM_OF_HARMONICS = 16

    # where should we start looking for harmonics (fundamental - x, what is x?)
    HARMONIC_FINDER_STARTING_POINT = 50 #meaning, we are beginning to look for harmonics 50 Hz below the fundamental frequency

    # we find the highest x amount of peaks where x is the number of frequencies in the window (determined in the windowedPeaks method's arguments)
    # the percentile that we're using is the percentile of the lowest frequency in that list
    PERCENTILE_OF_LOWEST_SELECTED_PEAK = 80

    # how close a peak needs to be to another peak to be rejected in terms of what the fundamental should be divided by
    DISTANCE = 10 #meaning the distance is the fundamental/10

    #size of ratio array graph and magSpec graph (it's a square)
    GRAPH_SIZE = 8
    
    # AudioFilesArray Constants:
    #########################################################

    # path name of the directory with the audio files
    PATH_NAME = Path(r"C:\Users\abeca\OneDrive\ICUNJ_grant_stuff\ICUNJ-grant-audiofiles")

    # file type to be added to the list of files
    FILE_SUFFIX = ".wav"

    # selected samples (e.g. "1SR")
    SELECTED_SAMPLES = "1SR"

    # DataAnalysis Constants:
    #########################################################
    
    # How far away from an integer does a harmonic need to be to be considered an integer
    INTEGER_THRESHOLD = .1



