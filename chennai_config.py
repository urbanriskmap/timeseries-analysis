# import this file and then overwrite whatever you need in
# the default_config object
import logging
import pandas as pd
from sqlalchemy import create_engine
DATABASE = "riskmap"
engine = create_engine(
        "postgresql://postgres:postgres@localhost:5432/"
        + DATABASE)

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

TEST_LOG_FILENAME = ".log_filename.log"
fh = logging.FileHandler(TEST_LOG_FILENAME)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
LOGGER.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

start_known_flood = "'2017-11-01 00:00:35.630000+05:30'"
end_known_flood = "'2017-11-07 00:00:35.630000+05:30'"


def __get_flood_pkeys(start_date, end_date, engine):
    # gets the pkeys of reports during flood dates

    pkeys = pd.read_sql_query(
        '''
        SELECT pkey, created_at FROM ''' + DATABASE + '''.all_reports WHERE
            created_at > %(start_date)s::timestamptz
                AND
            created_at < %(end_date)s::timestamptz
        ''',
        params={"start_date": start_date, "end_date": end_date},
        con=engine, index_col="pkey")
    return set(pkeys.index)


def __get_no_flood_pkeys(start_flood_date, end_flood_date, engine):
    # gets the pkeys of reports outside dates

    pkeys = pd.read_sql_query(
        '''
        SELECT pkey, created_at FROM ''' + DATABASE + '''.all_reports WHERE
            created_at < %(start_date)s::timestamptz
                OR
            created_at > %(end_date)s::timestamptz
        ''',
        params={"start_date": start_flood_date, "end_date": end_flood_date},
        con=engine, index_col="pkey")
    return set(pkeys.index)


flood_pkeys = __get_flood_pkeys(
    start_known_flood,
    end_known_flood,
    engine)

no_flood_pkeys = __get_no_flood_pkeys(
    start_known_flood,
    end_known_flood,
    engine)


config = {
    "flood_pkeys": flood_pkeys,
    "no_flood_pkeys": no_flood_pkeys,
    "all_pkeys": flood_pkeys.union(no_flood_pkeys),
    "database_engine": engine,
    "database_name": DATABASE,
    "location": "ch",
    "data_folder_prefix": "default_chennai_data",
    "logger": LOGGER
}
