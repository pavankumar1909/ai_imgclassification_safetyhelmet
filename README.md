### Dependency Management

This project uses `requirements.txt` to track Python dependencies.

The file is generated using **pipreqs**, which scans the project source code and lists only the libraries actually imported in the project. This keeps the dependency list minimal and avoids including unnecessary packages from the local environment.

To regenerate the file:

```bash
pipreqs ./ --ignore (any_name)_env
```

# 1.creating virtual environment 
** Python Version: python >=3.10 **  
-- Create a virtual environment  
     - python3.10 -m venv (any_name)_env  

# 2.activating virtual environment
      source (any_name)_env/bin/activate  
# 3.install required python libraries  
      pip install -r requirements.txt  

# deactivating virtual environment
     - whenever you are not required to use venv or before closing the project 'deactivate' it by using simple cmd "deactivate" 
     
# required version for project  
-- numpy        1.26.4  
-- ml-dtypes    0.3.1  
-- tensorflow   2.16.1  
-- mediapipe    0.10.x  
-- opencv       4.8.x  

# functionalities of each module  
  dataset         - contains datasets and other required data to train the model  
  models          - contains trained model in file format  
  src             - contains source file/s to run the models  
  requirement.txt - dependencies for project with versions
  .python-version - contains python version to install

  
