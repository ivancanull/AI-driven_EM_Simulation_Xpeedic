from multiprocessing import Value
import pandas 

if pandas.__version__ != '1.3.0':
    raise ValueError(f"Pandas version need to be 1.3.0, current version is {pandas.__version__}")