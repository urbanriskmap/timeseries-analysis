import pandas as pd
import numpy as np
from sqlalchemy import create_engine


global engine
# for chennai, where I've taken the database dump and
# renamed cognicity -> riskmap by search replacing in the .sql file.
CH_DATABASE = "riskmap"
CH_ENGINE = create_engine("postgresql://postgres:postgres@localhost:5432/"+ CH_DATABASE)

IN_DATABASE = "cognicity"
IN_ENGINE = create_engine("postgresql://postgres:postgres@localhost:5432/"+ IN_DATABASE)

def get_flood_depth_chennai():
    pkeys_and_flood_depth = pd.read_sql_query('''
	SELECT pkey, CAST(report_data ->> 'flood_depth' AS INTEGER) AS flood_depth 
            FROM riskmap.all_reports 
	    WHERE report_data->>'flood_depth' IS NOT NULL
    ''', con=CH_ENGINE, index_col="pkey")

    print(pkeys_and_flood_depth)
    return pkeys_and_flood_depth

def get_flood_depth_jakarta():
    pkeys_and_flood_depth = pd.read_sql_query('''
	SELECT pkey, CAST(report_data ->> 'flood_depth' AS INTEGER) AS flood_depth 
            FROM cognicity.all_reports 
	    WHERE report_data->>'flood_depth' IS NOT NULL
    ''', con=IN_ENGINE, index_col="pkey")

    print(pkeys_and_flood_depth)
    return pkeys_and_flood_depth

def make_matrix(flood_depth):
    '''
        Returns numpy matrix of 
        pkey         | pkey  
        flood_depth  | flood_depth

    '''
    v = flood_depth['flood_depth'].to_numpy()
    i = flood_depth.index.to_numpy()
    return np.vstack((i,v))

if __name__=='__main__':
    ch = get_flood_depth_chennai()
    m = make_matrix(ch)
    print(m)
