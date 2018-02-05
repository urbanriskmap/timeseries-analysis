
import pandas as pd

class streaming_peak_finder_cusum:
    def __init__(self, window_size, threshhold, mu=1.44):
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
        pass


    def input_report(self, report):
        """
        Args: 
            report (timeSeries): 
                index:
                name | count

        Returns:
            tuple(mean, median, std_deviation)
            mean (float)
            median (float) 
            std_deviation (float): of the report that this was called on 
        """

        self.currentSum += ( report - self.mu)

        return (self.currentSum,)
