
import pandas as pd

class streaming_peak_finder_cusum:
    def __init__(self, window_size, threshold, mu=1.44):
        """ Creates a streaming peak finder
        Args: 
            window_size (int): how many datapoints to consider in the moving window

        Returns: 
            A new streaming_peak_finder object.

        """
        self.data_in_moving_window = []
        self.window_size = window_size
        self.currentSum = 0
        self.mu = mu
        self.threshold = threshold
        pass


    def input_report(self, report):
        """
        Args: 
            report (float): The count of reports for the last timeperiod

        Returns:
            tuple(currentSum, signal)
            currentSum (float): the current running sum in this CUSUM measurement 
                 given the previous input reports
            signal (bool): whether there is flooding- that is, the running mean is over the threshold
        """

        self.currentSum += ( report - self.mu)

        return (self.currentSum, self.currentSum > self.threshold)
