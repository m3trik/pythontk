###### \*A collection of backend utilities for Python.

---

## Installation:

###### 

To install:
Add the tentacle folder to a directory on your python path, or
install via pip in a command line window using:
```
python -m pip install tentacletk
```

example use-case:
```
from pythontk import Iter #import a class from the package.
Iter.filterList([0, 1, 2, 3, 2], [1, 2, 3], 2) #returns: [1, 3]
```
```
from pythontk.itertk import filterDict #or import the function directly.
Iter.filterDict(dct, exc='t*', keys=True) #returns: {1: '1', 3: 'three'}
```