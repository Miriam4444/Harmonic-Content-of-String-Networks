#This file is where im going to figure out what i actually need to be running
#Date: 12/11/2025
#link to the right github branch https://github.com/Miriam4444/ICUNJ-grant/tree/october-2025-updates
from AudioFile import AudioFile as af
from AudioFilesList import AudioFilesList
from pathlib import Path
import Constants

if __name__ == "__main__":
    #I know this print aggregate error is correct
    af.printAggregateError(directory= Constants.directoryName, numberOfFundamentalsInWindow = Constants.numFundInWindow, percentile = Constants.aggErrorPercentile, SpecificType = Constants.fileType)

    #this is for printing the mag spec i tested it in testfilelist and it was printing the correct number of harmonics
    DirectoryName = Path(Constants.directoryName)
    nameArray = AudioFilesList(DirectoryName)
    namelist = nameArray.getSpecificType(Constants.fileType)

    #for printing the amount of files in nameList 
    #print("number of audiofiles of specified type in specified directory" , len(namelist))
    
    #for printing one magspec
    S = af(namelist[5])

    #this graphs the magspec with no identified peaks
    #S.graph_magspec()

    #This graphs the ration array
    #S.graphRatioArray(percentile=25)

    
    '''
    #this is just printing the peaks
    windowedPeaks = S.windowedPeaks(percentile=Constants.aggErrorPercentile, numberFundamentalsInWindow=Constants.numFundInWindow)
        R = S.sr/S.N
    print(namelist[1])
    for peak in windowedPeaks:
        print(peak*R)
    '''

    #this graphs the magspec with the peaks
    #S.graph_magspec_withWindowedPeaks(percentile=80, numberFundamentalsInWindow=5)
    

    #this iterates through all the files in namelist and prints all of their peaks
    for i in range(len(namelist)):
        S=af(namelist[i])
        print (" ")
        print(namelist[i])

        windowedPeaks = S.windowedPeaks(percentile=Constants.aggErrorPercentile, numberFundamentalsInWindow=Constants.numFundInWindow)
        R = S.sr/S.N

        for peak in windowedPeaks:
            print(peak*R)





