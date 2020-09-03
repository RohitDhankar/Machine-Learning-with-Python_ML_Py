from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('cyt.pyx'))
# python setup.py build_ext --inplace

"""
Compiling cyt.pyx because it changed.
[1/1] Cythonizing cyt.pyx
/home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages/Cython/Compiler/Main.py:369: 
FutureWarning: Cython directive 'language_level' not set, using 2 for now (Py2). 
This will change in a later release! 

File: /home/..../PythonBasics/cython/cyt.pyx
  tree = Parsing.p_module(s, pxd, full_module_name)
running build_ext
building 'cyt' extension
creating build
creating build/temp.linux-x86_64-3.8
gcc -pthread -B /home/dhankar/anaconda3/envs/pytorch_venv/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/home/dhankar/anaconda3/envs/pytorch_venv/include/python3.8 -c cyt.c -o build/temp.linux-x86_64-3.8/cyt.o
gcc -pthread -shared -B /home/dhankar/anaconda3/envs/pytorch_venv/compiler_compat -L/home/dhankar/anaconda3/envs/pytorch_venv/lib -Wl,-rpath=/home/dhankar/anaconda3/envs/pytorch_venv/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.8/cyt.o -o /home/dhankar/_dc_all/20_8/learn_ml/ML_Py_Basics/PythonBasics/cython/cyt.cpython-38-x86_64-linux-gnu.so
"""