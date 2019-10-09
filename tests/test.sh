#!/bin/bash

echo "Running tests on test_activation.py"
python3 test_activation.py

echo "Running tests on test_decay.py"
python3 test_decay.py

echo "Running tests on test_fitness.py"
python3 test_fitness.py

echo "Running tests on test_algorithms.py"
python3 test_algorithms.py

echo "Running tests on test_opt_probs.py"
python3 test_opt_probs.py

echo "Running tests on test_neural.py"
python3 test_neural.py

echo "Finished all tests"
