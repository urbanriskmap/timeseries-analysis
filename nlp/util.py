import pandas as pd

def get_flood_pkeys(start_date, end_date, engine):
    # gets the pkeys of reports during flood dates

    pkeys = pd.read_sql_query('''
        SELECT pkey, created_at FROM riskmap.all_reports WHERE
            created_at > %(start_date)s::timestamptz
                AND 
            created_at < %(end_date)s::timestamptz
    ''', params={"start_date": start_date, "end_date": end_date}, con=engine, index_col="pkey")

    return pkeys


def get_no_flood_pkeys(start_flood_date, end_flood_date, engine):
    # gets the pkeys of reports during flood dates

    pkeys = pd.read_sql_query('''
        SELECT pkey, created_at FROM riskmap.all_reports WHERE
            created_at < %(start_date)s::timestamptz
                AND 
            created_at > %(end_date)s::timestamptz
    ''', params={"start_date": start_flood_date, "end_date": end_flood_date}, con=engine, index_col="pkey")

    return pkeys

#    num_reports_with_zeros = pd.read_sql_query('''
#        SELECT date, COALESCE(count, NULL, 0) as count FROM 
#		(SELECT date_trunc('hour', offs) as date FROM 
#			generate_series(
#                            %(start_date)s::timestamptz at time zone 'UTC',
#                            %(end_date)s::timestamptz at time zone 'UTC',
#                            %(interval)s::interval
#			    ) as offs ORDER BY date ASC) empty_hours
#        LEFT JOIN 
#                (select date_trunc('hour', created_at), count(pkey) 
#                   from cognicity.all_reports 
#                     WHERE text  NOT SIMILAR To '%%(T|t)(E|e)(S|s)(T|t)%%' 
#                     GROUP BY date_trunc('hour', created_at)
#                   ) no_test 
#                   ON date = date_trunc
#    ''', params={"start_date":start_date, "end_date":end_date, "interval":interval}, con=engine, index_col="date", parse_dates=["date"])
#


