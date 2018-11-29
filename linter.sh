#!/bin/bash

disable_options="R0201,R0902,R0903,R0904,R0912,R0913,R0914,R0915,C0103,C0200,C1801,W0612"
python_files=("algorithms.py"
              "test_algorithms.py"
              "decay.py"
              "test_decay.py"
              "fitness.py"
              "test_fitness.py"
              "activation.py"
              "test_activation.py"
              "neural.py"
              "test_neural.py"
              "opt_probs.py"
              "test_opt_probs.py")

for filename in "${python_files[@]}"
do
    echo "Starting lint on ${filename}"
    pylint ${filename} --score=no --disable=$disable_options
    pycodestyle ${filename}
    flake8 ${filename}
done

echo "Finished linting all files"
