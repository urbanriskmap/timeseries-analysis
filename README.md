[![DOI](https://zenodo.org/badge/116693707.svg)](https://zenodo.org/badge/latestdoi/116693707)



# timeseries-analysis
Used as a PoC that CUSUM can accurately
predict when to send alerts

#virtual env
To use python virtual environments:

```
python3 -m venv _python
source _python/bin/activate
pip install -r requirements/dev.txt
pip install -r requirements/common.txt
```

In order to make gifs, install imagemagick 
```
brew install imagemagick 
```
or 

```
sudo apt-get install imagemagick 
```

For jupyter notebooks first register the virtual env then run
```
source _python/bin/activate
python -m ipykernel install --user --name=time_series
jupyter notebook
```
Then make sure that you're using the time_series virtual environment.

If you want vim keybindings then use the jupyter-vim-binding plugin:
```
# Create required directory in case (optional)
mkdir -p $(jupyter --data-dir)/nbextensions
# Clone the repository
cd $(jupyter --data-dir)/nbextensions
git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding
# Activate the extension
jupyter nbextension enable vim_binding/vim_binding

```

## Google
If you'd like to run the google cloud vision labeling code, you must
first create a [google cloud project](https://cloud.google.com/vision)
and download credentials as json.

In order to set google credentials, set the path to credentials in the
GOOGLE_APPLICATION_CREDENTIALS env var:
```
export GOOGLE_APPLICATION_CREDENTIALS=<path to creds>
```

## AWS:
For an IAM user with access to Rekognition is needed then set
```
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
```



# Testing
Tests are written using the built in unit testing library
unittest. Each directory includes unit tests for its own 
functionality. All tests can be run at once by using 
the convenience script run_tests.sh after setting up 
the python virtual environment:
```
./run_tests
```

# Data sources
Data sources are configured via the config dictionary that is passed
into every data loader. See default_config.py for an example.
In general, the cognicity loaders expect the database schema to
follow: https://github.com/urbanriskmap/cognicity-schema
and to be clean of test reports.
A SQL command like this will remove all reports with 'test' in the report
text, but it will not remove these reports from the grasp table, so beware of
that (or remove them from the grasp table as well if you like). 
```
DELETE FROM riskmap.all_reports WHERE text SIMILAR TO '(%(T|t)(E|e)(S|s)(T|t)%)'
```
