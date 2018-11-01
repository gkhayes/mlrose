#!/bin/bash

echo "Starting lint on algorithms.py"
pylint algorithms.py --score=no
pycodestyle algorithms.py
flake8 algorithms.py

echo "Starting lint on decay.py"
pylint decay.py --score=no
pycodestyle decay.py
flake8 decay.py

echo "Starting lint on discrete.py"
pylint discrete.py --score=no
pycodestyle discrete.py
flake8 discrete.py

echo "Starting lint on fitness.py"
pylint fitness.py --score=no
pycodestyle fitness.py
flake8 fitness.py

echo "Starting lint on test.py"
pylint test.py --score=no
pycodestyle test.py
flake8 test.py

echo "Finished linting all files"
