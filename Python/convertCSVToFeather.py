# File reads in csv and writes to feather
# Authored by Sean M. Hendryx while working at the University of Arizona
# contact: seanmhendryx@email.arizona.edu //github.com/SMHendryx/binaryClassifierParameterGridSearch
# Copyright (c)  2017 Sean Hendryx
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
####################################################################################################################################################################################


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