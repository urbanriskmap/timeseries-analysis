
# Project organization
img_util.py exists in the parent folder 
to easily run specific classes (such as goog_recog or aws_recog). 
the CognicityImageLoader class serves as a common data loader for 
images:
	- Fetches URLs from the supplied database
	- Downloads those images into a local folder
	- Given a pkey: { {label:confidence} } nested Dictionary can create matrix representations
	- Can create a label matrix given

# Testing
In order to run all tests do
` python -m unittest discover `

All test files must start with test_ as per
the Python unittest specs: https://docs.python.org/3/library/unittest.html#command-line-interface
