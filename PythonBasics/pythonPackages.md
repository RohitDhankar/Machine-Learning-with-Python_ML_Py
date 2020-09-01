##### Creating own Python Packages 
#
- Python package DIR structure 
```
$ tree
.
├── __init__.py
├── __pycache__
│   ├── __init__.cpython-38.pyc
│   ├── testModule_1.cpython-38.pyc
│   └── testModule_2.cpython-38.pyc
├── testModule_1.py
└── testModule_2.py

1 directory, 6 files

```
- The DIR = ```__pycache__``` gets created on its own - as we execute the code from ```testModule_2.py```
