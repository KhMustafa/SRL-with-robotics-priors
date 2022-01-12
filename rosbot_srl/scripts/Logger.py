"""
This class is responsible for logging the random samples collected from the simulation environment into a pickle file.
This pickle file is then used to train the state representation network. An object of this class is created in
start_DDPG.py file.
"""

import pickle
import os


class Logger:
    uLog = {}  # this is the universal log, that is shared across all class instances

    def __init__(self, folder, filename='results.pkl'):
        self.logDict = {}  # this log file is unique

        self.directory = folder
        self.filename = filename
        # make sure the folder exists. If not, create a file in the specified directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def ulog(self, var, value):
        if var in self.uLog.keys():
            self.uLog[var].append(value)
        else:
            self.uLog[var] = [value]

    def save_ulog(self):
        with open(self.directory + '/' + self.filename, 'wb') as f:
            print("Saved experiment data")
            pickle.dump(self.uLog, f, pickle.HIGHEST_PROTOCOL)
