# Jakarta config: where only those reports with images included
# import this file and then overwrite whatever you need in
# the default_config object
import logging
import pandas as pd
from sqlalchemy import create_engine
DATABASE = "cognicity"
engine = create_engine(
        "postgresql://postgres:postgres@localhost:5432/"
        + DATABASE)

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

LOG_FILENAME = ".default_jakarta.log"
fh = logging.FileHandler(LOG_FILENAME)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
LOGGER.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

start_period = "'2017-01-01 00:00:35.630000-05:00'"
end_period = "'2017-03-10 00:00:35.630000-05:00'"

start_known_flood = "'2017-02-20 00:00:35.630000-05:00'"
end_known_flood = "'2017-02-23 00:00:35.630000-05:00'"


def __get_flood_pkeys(start_date, end_date, engine):
    # gets the pkeys of reports during flood dates

    pkeys = pd.read_sql_query(
        '''
        SELECT pkey, created_at FROM ''' + DATABASE + '''.all_reports WHERE
            created_at > %(start_date)s::timestamptz
                AND
            created_at < %(end_date)s::timestamptz
                AND
            image_url IS NOT NULL
                AND
            text IS NOT null
                AND
            LENGTH(text) > 0
        ''',
        params={"start_date": start_date, "end_date": end_date},
        con=engine, index_col="pkey")

    path = "/home/abrahamq/transfer_learning_trial/flood/train/heavy_flood"
    import os
    files = os.listdir(path)
    path = "/home/abrahamq/transfer_learning_trial/flood/val/heavy_flood"
    files.extend(os.listdir(path))
    keep = set()
    for each in files:
        num = each.split(".")[0]
        keep.add(int(num))
    res = set(pkeys.index).intersection(keep)
    return res


def __get_no_flood_pkeys(start_period,
                         start_flood_date,
                         end_flood_date,
                         end_period,
                         engine):
    # gets the pkeys of reports outside dates

    pkeys = pd.read_sql_query(
        '''
        SELECT pkey,
               created_at
        FROM ''' + DATABASE + '''.all_reports
        WHERE (
                created_at > %(start_period)s::timestamptz
            AND created_at < %(start_flood_date)s::timestamptz)
        OR (
                created_at > %(end_flood_date)s::timestamptz
            AND created_at < %(end_period)s::timestamptz)
        AND
            image_url IS NOT NULL
        AND
            text IS NOT null
        AND
            LENGTH(text) > 0
        ''',
        params={
            "start_period": start_period,
            "start_flood_date": start_flood_date,
            "end_flood_date": end_flood_date,
            "end_period": end_period
            },
        con=engine, index_col="pkey")

    import os
    path = "/home/abrahamq/transfer_learning_trial/flood/train/no_flood"
    files = os.listdir(path)
    path = "/home/abrahamq/transfer_learning_trial/flood/val/no_flood"
    files.extend(os.listdir(path))
    keep = set()
    for each in files:
        num = each.split(".")[0]
        keep.add(int(num))
    res = set(pkeys.index).intersection(keep)
    return res


flood_pkeys = __get_flood_pkeys(
    start_known_flood,
    end_known_flood,
    engine)

no_flood_pkeys = __get_no_flood_pkeys(
    start_period,
    start_known_flood,
    end_known_flood,
    end_period,
    engine)


config = {
    "flood_pkeys": flood_pkeys,
    "no_flood_pkeys": no_flood_pkeys,
    "all_pkeys": flood_pkeys.union(no_flood_pkeys),
    "database_engine": engine,
    "database_name": DATABASE,
    "location": "id",
    "data_folder_prefix": "default_jakarta_data",
    "logger": LOGGER
}
