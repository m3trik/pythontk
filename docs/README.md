###### \*A collection of backend utilities for Python.

---

## Installation:

###### 

To install:
Add the pythontk folder to a directory on your python path, or
install via pip in a command line window using:
```
python -m pip install pythontk
```

example use-case:
```
# import a class from the package.
from pythontk import Iter
# the class `Iter` holds all the iterable related functions.
Iter.filterList([0, 1, 2, 3, 2], [1, 2, 3], 2) #returns: [1, 3]
```
```
# or you can import the function directly.
from pythontk.itertk import filterDict
Iter.filterDict({1:'1', 'two':2, 3:'three'}, exc='t*', keys=True) #returns: {1: '1', 3: 'three'}
```