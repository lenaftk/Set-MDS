### How to run bootstrapped with gentlemandata
- set up conda environment
- The folders that you will need:
    1. data_loader
    2. gentlemandata
        pip install -r requirements.txt OR pip install numpy, cython, scikit-learn, scipy, matplotlib
        pip install -e .
    3. bootstrapped_pattern_search
        python setup.py build_ext --inplace
        python setup.py install
        Now you are ready to run -----> python gentle_test_goal.py  
        
       
