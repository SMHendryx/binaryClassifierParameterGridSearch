# File reads in csv and writes to feather
# Author: Sean Hendryx


#Example usage:
#python convertCSVToFeather.py /Users/seanmhendryx/reach_context-balancing/reach/features.csv
#python convertCSVToFeather.py /your/path/to.csv

import sys
import os
import pandas
import feather


def main():
    #get args:
    args = sys.argv
    in_file = args[1]
    
    checkExists(in_file)

    print("Reading in file: ", in_file)
    df = pandas.read_csv(in_file)
    print("Successfully read in file: ", in_file)

    #write feather:
    out_file_name = os.path.splitext(in_file)[0]
    out_file_name += ".feather"
    print("Writing file: ", out_file_name)
    try:
        feather.write_dataframe(df, out_file_name)
        print("Successfully wrote file: ", out_file_name)
    except:
        print("File not written.")

def checkExists(file):
    """
    Check if file exists
    """
    if not os.path.isfile(file):
        raise FileNotFoundError("Input file %s does not exist." % file)


if __name__ == "__main__":
    main()