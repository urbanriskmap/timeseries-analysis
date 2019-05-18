# Set the environment variables correctly
# or use the defaults if running locally

export PGHOST=localhost
export PROD=cognicity # Upgrade so this exists
export PGUSER=postgres
export PGPORT=5432

psql -d $PROD -h $PGHOST -U $PGUSER -p $PGPORT -c "SELECT count(report_data -> 'flood_depth'), created_at  FROM cognicity.all_reports WHERE report_data is not null and CAST (report_data->>'flood_depth' AS INTEGER) = 50;"
