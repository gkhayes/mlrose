#!/bin/bash

disable_options="R0201,R0902,R0903,R0904,R0912,R0913,R0914,R0915,C0103,C0200,C0302,C1801,W0612"
python_files=("mlrose/algorithms.py"
              "tests/test_algorithms.py"
              "mlrose/decay.py"
              "tests/test_decay.py"
              "mlrose/fitness.py"
              "tests/test_fitness.py"
              "mlrose/activation.py"
              "tests/test_activation.py"
              "mlrose/neural.py"
              "tests/test_neural.py"
              "mlrose/opt_probs.py"
              "tests/test_opt_probs.py"
              "setup.py")

for filename in "${python_files[@]}"
do
    echo "Starting lint on ${filename}"
    pylint ${filename} --score=no --disable=$disable_options
    pycodestyle ${filename}
    flake8 ${filename}
done

echo "Finished linting all files"
