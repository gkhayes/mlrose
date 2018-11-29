#!/bin/bash

echo "Running tests on test_activation.py"
python test_activation.py

echo "Running tests on test_decay.py"
python test_decay.py

echo "Running tests on test_fitness.py"
python test_fitness.py

echo "Running tests on test_algorithms.py"
python test_algorithms.py

echo "Running tests on test_opt_probs.py"
python test_opt_probs.py

echo "Running tests on neural.py"
python neural.py

echo "Finished all tests"
