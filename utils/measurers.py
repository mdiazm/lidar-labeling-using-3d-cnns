"""
Script with functions to measure several quantities such as time of exec, precision, recall, etc.
"""

from time import time

class TimeMeasurer(object):
    """
    Measure time of execution using with ... statement.
    """
    def __init__(self, unit='seconds', reason=None):
        self.begin = None
        self.end = None
        self.unit = unit
        self.reason = reason

    def __enter__(self):
        self.begin = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        diff_time = self.end - self.begin

        if self.unit == 'milliseconds':
            diff_time *= 1000

        if self.reason is None:
            print("Time of execution: {} {}".format(diff_time, self.unit))
        else:
            print("Doing: {}. Time of execution: {} {}".format(self.reason, diff_time, self.unit))


def measure_time(callback, unit='seconds'):
    """
    Function to measure the time of execution of a function passed as a callback.

    :param unit: unit of time. default: seconds
    :param callback: callback of the function to execute.
    :return:
    """
    begin = time()
    callback()
    end = time()

    diff_time = end - begin

    if unit == 'milliseconds':
        diff_time *= 1000

    print("Time of execution: {} {}".format(diff_time, unit))

