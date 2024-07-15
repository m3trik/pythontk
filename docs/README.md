[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/Version-0.7.19-blue.svg)](https://pypi.org/project/pythontk/)
[![CoreUtils Tests](https://img.shields.io/badge/CoreUtils-Passing-brightgreen.svg)](../test/ptk_test.py#CoreUtilsTest)
[![FileUtils Tests](https://img.shields.io/badge/FileUtils-Passing-brightgreen.svg)](../test/ptk_test.py#FileUtilsTest)
[![ImgUtils Tests](https://img.shields.io/badge/ImgUtils-Passing-brightgreen.svg)](../test/ptk_test.py#ImgUtilsTest)
[![MathUtils Tests](https://img.shields.io/badge/MathUtils-Passing-brightgreen.svg)](../test/ptk_test.py#MathUtilsTest)
[![IterUtils Tests](https://img.shields.io/badge/IterUtils-Passing-brightgreen.svg)](../test/ptk_test.py#IterUtilsTest)
[![StrUtils Tests](https://img.shields.io/badge/StrUtils-Passing-brightgreen.svg)](../test/ptk_test.py#StrUtilsTest)

### PYTHONTK (Python Toolkit)

---
<!-- short_description_start -->
*pythontk is a collection of backend utilities for Python.*
<!-- short_description_end -->

### Installation:

To install:
Add the `pythontk` folder to a directory on your python path, or
install via pip in a command line window using:
```
python -m pip install pythontk
```

### Example use-case:
```python
import pythontk as ptk
ptk.filter_list([0, 1, 2, 3, 2], [1, 2, 3], 2)
# Returns: [1, 3]

ptk.filter_dict({1:'1', 'two':2, 3:'three'}, exc='t*', keys=True)
# Returns: {1: '1', 3: 'three'}
```
