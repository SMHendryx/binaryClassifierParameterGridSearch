# File reads in csv and writes to feather
# Authored by Sean M. Hendryx while working at the University of Arizona
# contact: seanmhendryx@email.arizona.edu
# Add licencse and copyright

#Example usage:
#python convertCSVToFeather.py /Users/seanmhendryx/reach_context-balancing/reach features.csv

import sys
import os
import pandas
import feather


def main():
    #get args:
    # first argument (index 1 bc 0 is the file name) is the directory in which to read AND write, second is the input csv file (inFile)
    args = sys.argv
    # Set working directory:
    dir = args[1]
    inFile = args[2]

    os.chdir(dir)
    
    df = pandas.read_csv(inFile)

    #write feather:
    outFileName = os.path.splitext(inFile)[0]
    outFileName += ".feather"
    feather.write_dataframe(df, outFileName)



if __name__ == "__main__":
    main()