#!/bin/bash

echo "Starting lint on algorithms.py"
pylint algorithms.py --score=no --disable=R0902,R0903,R0904
pycodestyle algorithms.py
flake8 algorithms.py

echo "Starting lint on decay.py"
pylint decay.py --score=no --disable=R0902,R0903,R0904
pycodestyle decay.py
flake8 decay.py

echo "Starting lint on discrete.py"
pylint discrete.py --score=no --disable=R0902,R0903,R0904
pycodestyle discrete.py
flake8 discrete.py

echo "Starting lint on fitness.py"
pylint fitness.py --score=no --disable=R0902,R0903,R0904
pycodestyle fitness.py
flake8 fitness.py

echo "Starting lint on test.py"
pylint test.py --score=no --disable=R0902,R0903,R0904
pycodestyle test.py
flake8 test.py

echo "Finished linting all files"
