"""
CUSUM proof of concept
Abraham Quintero, MIT 2018
v0.0.1
"""

__author__ = "Abraham Quintero"
__created__ = "2018-1-8"
__version__ = "0.0.1"
__copyright__ = "Copyright 2017 MIT Urban Risk Lab"
__license__ = "MIT"
__email__ = "abrahamq@mit.edu"
__status__ = "Development"
__url__ = "https://github.com/urbanriskmap/timeseries-analysis"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import datetime
import streaming_peak_finder as spf
from sqlalchemy import create_engine

global engine
engine = create_engine("postgresql://postgres:postgres@localhost:5432/petabencana")

#From: https://matplotlib.org/users/event_handling.html
def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

def onpick(event):
    artist = event.artist
    x = artist.get_xdata()
    y = artist.get_ydata()
    index = event.ind
    print("datapoint: ")
    print(x[index[0]])
    print(y[index[0]])

#takes in a dataframe with index 'date' and column 'count'
#returns the CUSUM at every index 'date'
def gen_CUSUM(series, expected_mean=0):
    #TODO: do this in place, just adding a column so we don't have to get a deep copy
    ret = series.copy(deep=True)
    sumSoFar = 0
    for index, row in series.iterrows():
        sumSoFar += row['count'] - expected_mean
        ret.loc[index]['count'] = sumSoFar
    return ret


def get_data_bin_by_minute(start_date, end_date, interval):
    """ Gets data from sql database between start_date and end_date
    Args: 
        start_date (str): the start date and time as a ISO8601 string
        end_date (str):  the end date and time as an ISO8601 string
        interval (str): a postgresql interval string

    Returns: 
        Pandas dataframe, with the index being a date and the 'count' column 
        saying how many flood reports were received on that interval
        Zero values are included for intervals that do not have any reports
    """
    date_trunc_to = "minute"

    num_reports_with_zeros = pd.read_sql_query('''
        SELECT date, COALESCE(count, NULL, 0) as count FROM 
                (SELECT date_trunc(%(date_trunc_to)s, offs) as date FROM 
        		generate_series(
                            %(start_date)s::timestamptz,
                            %(end_date)s::timestamptz,
                            %(interval)s::interval
        		    ) as offs ORDER BY date ASC) empty_hours
        LEFT JOIN 
                (select date_trunc(%(date_trunc_to)s, created_at), count(pkey) 
                   from archive.reports 
                     WHERE text  NOT SIMILAR To '%%(T|t)(E|e)(S|s)(T|t)%%' 
                     GROUP BY date_trunc(%(date_trunc_to)s, created_at)
                   ) no_test 
                   ON date = date_trunc
    ''', params={"start_date":start_date, "end_date":end_date, "interval":interval, "date_trunc_to":date_trunc_to}, con=engine, index_col="date", parse_dates=["date"])

    return num_reports_with_zeros

def get_data(start_date, end_date, interval):
    """ Gets data from sql database between start_date and end_date
    Args: 
        start_date (str): the start date and time as a ISO8601 string
        end_date (str):  the end date and time as an ISO8601 string
        interval (str): a postgresql interval string

    Returns: 
        Pandas dataframe, with the index being a date and the 'count' column 
        saying how many flood reports were received on that interval
        Zero values are included for intervals that do not have any reports
    """
    num_reports_with_zeros = pd.read_sql_query('''
        SELECT date, COALESCE(count, NULL, 0) as count FROM 
		(SELECT date_trunc('hour', offs) as date FROM 
			generate_series(
                            %(start_date)s::timestamptz,
                            %(end_date)s::timestamptz,
                            %(interval)s::interval
			    ) as offs ORDER BY date ASC) empty_hours
        LEFT JOIN 
                (select date_trunc('hour', created_at), count(pkey) 
                   from archive.reports 
                     WHERE text  NOT SIMILAR To '%%(T|t)(E|e)(S|s)(T|t)%%' 
                     GROUP BY date_trunc('hour', created_at)
                   ) no_test 
                   ON date = date_trunc
    ''', params={"start_date":start_date, "end_date":end_date, "interval":interval}, con=engine, index_col="date", parse_dates=["date"])
    return num_reports_with_zeros

def test_harness(alg, known_flooding, flood_reports):
    """ Plays each flood report in turn to the streaming algorithm, creating an animation
    Args: 
        alg: the streaming algorithm class. Has a input_report() function that takes in a report
            input_report returns

        known_flooding: 
            pd dataframe that has all reports during known flooding in a continuous interval

        flood_reports: 
            All flood reports within the continuous interval. Also includes known_flooding reports

    Returns: 
        None, displays the graph of the alg over the interval
    """
    print("shape")
    print(flood_reports.shape)

    t = pd.DataFrame(index=flood_reports.index)

    for index, row in flood_reports.itertuples(name=None):
        (mean, median, std) = alg.input_report(row)
        t.loc[index, 'mean'] = mean
        t.loc[index, 'median'] = median
        t.loc[index, 'std'] = std
        


    print(t)
    fig, ax = plt.subplots()
    ax.set_xlim(t.index.values[0], t.index.values[-1])
    ax.plot(t.index.values, t['mean'], color='g', alpha=0.5, label="mean")
    #ax.plot(t.index.values, t['median'], color='b', alpha=0.5, label="median")
    ax.scatter(flood_reports.index.values, flood_reports['count'], color='k', alpha=0.5)
    ax.legend()

    def update(frame):
        ax.set_xlabel(t.index.values[frame])
        ax.plot(t.index.values[0:frame], t[0:frame]['mean'], color='g', alpha=0.5, label="mean")
        ax.plot(t.index.values[0:frame], t[0:frame]['std'], color='r', alpha=0.5, label="std")
        ax.scatter(flood_reports.index.values[0:frame], flood_reports.iloc[0:frame]['count'], color='k', alpha=0.5)
        return ax

    anim = FuncAnimation(fig, update, frames=np.arange(0,flood_reports.shape[0]), interval=50)
    plt.show()
    #anim.save('../bin_by_hour.gif', writer='imagemagick', fps=60)



    return

def scatterPlot():
    limit = 3000
    
    df = pd.read_sql_query('''
            SELECT created_at, report_data->>'flood_depth' 
                FROM all_reports 
                    WHERE source = 'grasp' 
                    AND report_data->>'flood_depth' IS NOT NULL
                    AND (report_data->>'flood_depth')::int != 50::int
                    AND text NOT SIMILAR TO '%%test%%'
                    AND database_time > '2017-02-18 01:00:35.630000-05:00'
                    AND database_time < '2017-02-23 01:00:35.630000-05:00'
                ORDER BY database_time ASC
            LIMIT ''' + str(limit), con=engine, index_col="created_at")
    
    df.columns.values[0] = "flood_depth"
    
    def remove_none_turn_to_int(num):
        print(num)
        if num == None:
            return 0
        return float(num)
    
    df['flood_depth'] = df['flood_depth'].apply(remove_none_turn_to_int)

#    num_reports_by_hour = pd.read_sql_query('''
#            SELECT * 
#            FROM (
#                SELECT date_trunc('hour', offs) as date 
#                    FROM generate_series('2017-02-18 01:00:35.630000-05:00'::timestamp, 
#                    '2017-02-24 01:00:35.630000-05:00'::timestamp,
#                    '1 hour'::interval) as offs
#                ) h
#            LEFT JOIN(
#                SELECT date_trunc('hour', created_at), count(pkey)
#                    FROM (all_reports 
#                        WHERE text NOT SIMILAR TO '%%test%%'
#                        GROUP BY date_trunc('hour', created_at)
#            LIMIT ''' + str(limit) + '''))
#            
#            ''', con=engine, index_col="date_trunc")

# 
#              (select date_trunc('hour', created_at), count(pkey) 
#                 from all_reports 
#                   WHERE text  NOT SIMILAR To '%(T|t)(E|e)(S|s)(T|t)%' 
#                   GROUP BY date_trunc('hour', created_at)
#                 );


    num_reports_with_zeros = pd.read_sql_query('''
        SELECT date, COALESCE(count, NULL, 0) as count FROM 
		(SELECT date_trunc('hour', offs) as date FROM 
			generate_series(
                            '2017-02-10 00:00:35.630000-05:00'::timestamptz, 
			    '2017-02-27 00:00:35.630000-05:00'::timestamptz,
			    '1 hour'::interval) as offs ORDER BY date ASC) empty_hours
        LEFT JOIN 
                (select date_trunc('hour', created_at), count(pkey) 
                   from archive.reports 
                     WHERE text  NOT SIMILAR To '%%(T|t)(E|e)(S|s)(T|t)%%' 
                     GROUP BY date_trunc('hour', created_at)
                   ) no_test 
                   ON date = date_trunc
    ''', con=engine, index_col="date")

    print(num_reports_with_zeros)
    
    print(df.shape)

    # for non heavy flooding 
    #		generate_series('2014-12-29 01:00:35.630000-05:00'::timestamp, 
    #		'2015-01-20 01:00:35.630000-05:00'::timestamp,
    #standard dev:
    #0.4227586389603458
    #mean
    #0.14177693761814744
    #mu = 0.14177693761814744
    #sigma = 0.4227586389603458

    # 2017-02-10 
    # 2017-02-17
    mu = 1.4497041420118344
    sigma = 2.4444086893395007

    std_dev = num_reports_with_zeros.std()["count"]
    mean = num_reports_with_zeros.mean()["count"]
    print("standard dev:")
    print(std_dev)

    print("mean")
    print(mean)

    std_error = std_dev/np.sqrt(num_reports_with_zeros.shape[0])
    print("std error")
    print(std_error)

    CUSUM = gen_CUSUM(num_reports_with_zeros, mu)
    print(CUSUM)

    fig, ax = plt.subplots()

    cid = fig.canvas.callbacks.connect('pick_event', onpick)

    #ax.axhline(y=mean, color='r')
    #ax.axhline(y=mean+3*std_error, color='r')
    #ax.axhline(y=mean-3*std_error, color='r')
    #ax.scatter(df.index.values, df['flood_depth'])
    ax.axhline(y=mu+3*sigma, color='r')
    ax.plot(CUSUM.index.values, CUSUM['count'], color='orange')
    ax.scatter(num_reports_with_zeros.index.values, num_reports_with_zeros['count'], color='g')
    plt.show()

if __name__ == "__main__":
    #scatterPlot()
    alg = spf.streaming_peak_finder(10, 1)
    start = "'2017-02-20 00:00:35.630000-05:00'"
    end = "'2017-02-23 00:00:35.630000-05:00'"
    interval = "'1 minute'"
    known = get_data_bin_by_minute( start, end, interval)

    start_all_reports = "'2017-02-10 00:00:35.630000-05:00'"
    end_all_reports = "'2017-02-27 00:00:35.630000-05:00'"
    reports = get_data_bin_by_minute( start_all_reports, end_all_reports, interval) 

    #fig, ax = plt.subplots()
    #ax.scatter(known.index.values, known['count'], alpha=0.5)
    #ax.scatter(reports.index.values, reports['count'], color='g', alpha=0.5)
    #plt.show()

    test_harness(alg, known, reports)
