


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
python -m ipykernel install --user --name=_python
jupyter notebook
```

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


# Testing
Tests are written using the built in unit testing library
unittest. Each directory includes unit tests for its own 
functionality. All tests can be run at once by using 
the convenience script run_tests.sh after setting up 
the python virtual environment:
```
./run_tests
```
