### PYTHONTK (Python Toolkit)

---
<!-- short_description_start -->
*pythontk is a collection of backend utilities for Python.*
<!-- short_description_end -->

### Installation:

###### 

To install:
Add the `pythontk` folder to a directory on your python path, or
install via pip in a command line window using:
```
python -m pip install pythontk
```

### Example use-case:
###### Import the class `Iter` from the package.
###### As the name suggests, the class `Iter` holds the package's iterable related functions.
```python
from pythontk import Iter
Iter.filter_list([0, 1, 2, 3, 2], [1, 2, 3], 2) #returns: [1, 3]
```
###### You can also import any function directly.
```python
from pythontk import filter_dict
filter_dict({1:'1', 'two':2, 3:'three'}, exc='t*', keys=True) #returns: {1: '1', 3: 'three'}
```