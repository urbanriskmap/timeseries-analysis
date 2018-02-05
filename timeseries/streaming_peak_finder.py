
import pandas as pd

class streaming_peak_finder:
    def __init__(self, window_size, threshold):
        """ Creates a streaming peak finder
        Args: 
            window_size (int): how many datapoints to consider in the moving window

        Returns: 
            A new streaming_peak_finder object.

        """
        self.data_in_moving_window = []
        self.window_size = window_size
        self.threshold = threshold
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
            signal (bool): whether there is flooding- that is, the running mean is over the threshold
        """

        self.data_in_moving_window.append(report)
        if len(self.data_in_moving_window) > self.window_size:
            self.data_in_moving_window.pop(0)

        temp = pd.Series(self.data_in_moving_window)

        #mean = self.data_in_moving_window.mean()
        #median = self.data_in_moving_window.median()
        #std = self.data_in_moving_window.std()

        mean = temp.mean()
        median = temp.median()
        std = temp.std()
        signal = mean > self.threshold

        return (mean, median, std, signal)
