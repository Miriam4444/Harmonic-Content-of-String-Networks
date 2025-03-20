import os
from pathlib import Path
from Constants import Constants

class AudioFilesArray:

    def  __init__(self, directory_path):
        #We're taking in a directory path and assigning it to the instance variable directory_path
        self.directory_path = Constants.pathName
        #Path(r"C:\Users\abeca\OneDrive\ICUNJ_grant_stuff\ICUNJ-grant-audiofiles")

    #This function goes through all of the files in a directory and adds them to a list
    def makeFilePathList(self):
        fileNames = [] #instantiates a list that we're going to use to store the file names
        #Iterate through all of the files in the directory
        for file_path in self.directory_path.iterdir():
            #If the file is a file and ends in ".wav" we're going to add it to the list of fileNames
            if (file_path.is_file()) and (file_path.suffix.lower() == Constants.fileSuffix):
                #If we want the files in the array to just be the file name instead of the whole path just take away the # from the next line
                #file_path = os.path.basename(file_path)
                fileNames.append(file_path)
        return fileNames
    
    def makeFileNameList(self):
        fileNames = [] #instantiates a list that we're going to use to store the file names
        #Iterate through all of the files in the directory
        for file_path in self.directory_path.iterdir():
            #If the file is a file and ends in ".wav" we're going to add it to the list of fileNames
            if (file_path.is_file()) and (file_path.suffix.lower() == Constants.fileSuffix):
                file_path = os.path.basename(file_path)
                fileNames.append(file_path)
        return fileNames
    
    def getSpecificType(self, type):
        fileNames = self.makeFileNameList()
        filePaths = self.makeFilePathList()
        rightType = []
        for i in range(len(fileNames)):
            if type in fileNames[i]:
                rightType.append(filePaths[i])
        return rightType