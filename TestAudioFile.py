from AudioFilesArray import AudioFilesArray
from AudioFile import AudioFile

class TestAudioFile:

    if __name__ == "__main__":

        # initialize an array to house the desired samples
        nameArray = AudioFilesArray()

        # choose the sample type you want to analyze
        nameList = nameArray.getSpecificType() 

        # print length of nameList:
        print("amount of files in nameList: ", len(nameList))

        # iterate through each element of nameList 
        for i in range(len(nameList)):
            sample = AudioFile(nameList[i])

            # graph the harmonic spectrum
            sample.graph_magspec_withWindowedPeaks(80)

            # graph the correlation between experimentally determined harmonic data to theoretical model
            sample.graphRatioArray(80)

        # Choose a specific sample
        sample = AudioFile(nameList[0]) # If you would like to analyze a different sample, change the index appropriately

        # Graph the harmonic spectrum
        sample.graph_magspec_withWindowedPeaks(80)

        # Graph the correlation between experimentally determined harmonic data to theoretical model
        sample.graphRatioArray(80)

        # Calculate the aggregate error for the samples in the array
        print(AudioFile.printAggregateError())
