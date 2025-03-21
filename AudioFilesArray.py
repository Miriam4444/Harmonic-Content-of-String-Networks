import os
from Constants import Constants
from pathlib import Path

class AudioFilesArray:
    directoryPath = ()

    def  __init__(self):
        #We're taking in a directory path and assigning it to the instance variable directory_path
        directoryPathName = Constants.PATH_NAME
        self.directoryPath = directoryPathName

    #This function goes through all of the files in a directory and adds them to a list
    def makeFilePathList(self):
        fileNames = [] #instantiates a list that we're going to use to store the file names
        #Iterate through all of the files in the directory
        for filePath in self.directoryPath.iterdir():
            #If the file is a file and ends in ".wav" we're going to add it to the list of fileNames
            if (filePath.is_file()) and (filePath.suffix.lower() == Constants.FILE_SUFFIX):
                #If we want the files in the array to just be the file name instead of the whole path just take away the # from the next line
                fileNames.append(filePath)
        return fileNames
    
    def makeFileNameList(self):
        fileNames = [] #instantiates a list that we're going to use to store the file names
        #Iterate through all of the files in the directory
        for filePath in self.directoryPath.iterdir():
            #If the file is a file and ends in ".wav" we're going to add it to the list of fileNames
            if (filePath.is_file()) and (filePath.suffix.lower() == Constants.FILE_SUFFIX):
                filePath = os.path.basename(filePath)
                fileNames.append(filePath)
        return fileNames
    
    #choose what string of letters the file needs to contain
    def getSpecificType(self):
        type = Constants.SELECTED_SAMPLES
        fileNames = self.makeFileNameList()
        filePaths = self.makeFilePathList()
        rightType = []
        for i in range(len(fileNames)):
            if type in fileNames[i]:
                rightType.append(filePaths[i])
        return rightType