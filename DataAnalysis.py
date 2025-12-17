from collections import Counter


class DataAnalysis:
    def __init__(self, array):
        self.array = array
        self.valid_data = True
        self.max_difference = .1 #max_difference is how far away from an integer multiple we're considering to be a good value

    def checkIfDecimalClose(self, decimal, roundingPlace = None, threshold = None):
            if threshold == None:
                threshold = 10**(-(roundingPlace))
            if roundingPlace == None:
                roundingPlace = 1
                
            for i, value in enumerate(self.array):
                if abs(round(value, roundingPlace) - decimal) <= threshold :
                    print(value)
                    self.array.remove(value)

    def checkIfClose(self, number):
        closest_int = round(number)
        if abs(closest_int - number) <= self.max_difference:
            number = closest_int
        return number
    
    @staticmethod
    def findDuplicates(array):
        counts = Counter(array)  # Count the amount of occurrences of each value
        duplicates = 0
        seen = set()  # Track already seen values in set()

        for i, value in enumerate(array):
            if value in seen:  # If it's already in seen then it's a duplicate
                duplicates += 1
                print(f'There\'s a duplicate at entry #{i + 1}. Entry: "{value}"')
            elif counts[value] > 1:  # Mark values with more than one occurrence
                seen.add(value)

        return duplicates

                
    def checkDataTextFile(self, sampleValue, fileName):
        #Function checks if data is integer multiples and if there are no duplicates
        badData = 0
        closest_values = []
        for i, value in enumerate(self.array):
            #This part checks if it's less than whatever value you set the sampleValue to be 
            if abs(value - round(value)) >= sampleValue:
                closest = self.checkIfClose(value)
                closest_values.append(closest)
                
                if not isinstance(closest, int):  # Not an integer multiple
                    with open(f"{fileName}", "a") as f:
                        f.write(f'There\'s a non-integer multiple at entry #{i + 1}. Entry: "{value}"\n')
                    badData += 1
            else:
                pass
        #we're going to run the findDuplicates function with the parameter closest_values which is the list of all of the whole numbers or bad decimal data
        duplicates = self.findDuplicates(closest_values)
        if duplicates > 0:
            badData += duplicates
        else:
            with open(f"{fileName}", "a") as f:
                f.write("No duplicates in converted values.\n")

        if badData == 0:
            with open(f"{fileName}", "a") as f:
                f.write("Array is integer multiples with no repeating values.\n")
        else:
            with open(f"{fileName}", "a") as f:
                f.write(f"There are {badData} entries that either aren't integer multiples of the fundamental or are duplicates.\n")
