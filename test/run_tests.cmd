@ECHO Off

rem You can run the tests by using the unittest test runner, 
rem which can be invoked from the command line by running:
rem python -m unittest <module_name>

ECHO/
python -c "import test"
python ptk_test.py


PAUSE