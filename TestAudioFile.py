from AudioFilesArray import AudioFilesArray
from AudioFile import AudioFile

class TestAudioFile:

    if __name__ == "__main__":
        nameArray = AudioFilesArray()
        nameList = nameArray.getSpecificType()

        # print length of nameList:
        print("amount of files in nameList: ", nameList)

        #iterate through each element of nameList or do one at a time
        sample = AudioFile(nameList[0])

        # graph the harmonic spectrum
        sample.graph_magspec_withWindowedPeaks(80)

        #graph the correlation between experimentally determined harmonic data to theoretical model
        sample.graphRatioArray(80)
