Repository contains experimentation and code related to training AI models for deepfake detection.
The code is written to work with UADFV dataset.
UADFV dataset is not very big in size (~155MB) and is ideal for running quick experiments.
Models trained on the dataset won't be perfect but gives the basic framework of how to train and test the deepfake detection AI networks, e.g, MesoNet or XceptionNet.
The experiments provided here are useful to work with other very large datasets: FaceForensics++, DeepFake Detection Challenge (DFDC), Celeb-DF, and DeeperForensics-1.0.

I strongly suggest creating a virtual environment with Python3.
Create a directory _venvs_ for better management of all the virtual environments. _cd venvs_ will take you in the venvs directory. 
_python3 -m venv deepFake_ should create virtual environment _deepFake_.
Activate the virtual environment: _source deepFake/bin/activate_. Now come out of _venv_ directory and create a new directory _deepFake-prototype_ (or other suitable name). This directory is at the same level of venv. Basically, we are creating a directory inside venv to maintain all the virtual environments. Then we are creating another directory at the same level of venv to maintain all the code / experimental scripts / data related to that particular project.
_cd deepFake-prototype_ will take you inside the directory.
After this step: You are inside the directory and also inside the virtual environment _deepFake_.
Now run : _python install-reqs.py_. This script will scan the requirements.txt file, install each package, if failed to install - it will keep going with the next, and then at the end will print a report on the success and failure.
