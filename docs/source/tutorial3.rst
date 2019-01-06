Tutorial - Machine Learning Weight Optimization Problems
========================================================

What is a Machine Learning Weight Optimization Problem?
---------------------------------------------------------
For a number of different machine learning models, the process of fitting the model parameters involves finding the parameter values that minimize a pre-specified loss function for a given training dataset. 

Examples of such models include neural networks, linear regression models and logistic regression models, and the optimal model weights for such models are typically found using methods such as gradient descent.

However, the problem of fitting the parameters (or weights) of a machine learning model can also be viewed as a continuous-state optimization problem, where the loss function takes the role of the fitness function, and the goal is to minimize this function. 

By framing the problem this way, we can use any of the randomized optimization algorithms that are suited to continuous-state optimization problems to fit the model parameters. In this tutorial, we will work through an example of how this can be done with mlrose.

Solving Machine Learning Weight Optimization Problems with mlrose
-----------------------------------------------------------------
mlrose contains built-in functionality for solving the weight optimization problem for three types of machine learning models: (standard) neural networks, linear regression models and logistic regression models. This is done using the :code:`NeuralNetwork()`, :code:`LinearRegression()` and :code:`LogisticRegression()` classes respectively.

Each of these classes includes a :code:`fit` method, which implements the three steps for solving an optimization problem defined in the previous tutorials, for a given training dataset. 

However, when fitting a machine learning model, finding the optimal model weights is merely a means to an end. We want to find the optimal model weights so that we can use our fitted model to predict the labels of future observations as accurately as possible, not because we are actually interested in knowing the optimal weight values. 

As a result, the abovementioned classes also include a :code:`predict` method, which, if called after the :code:`fit` method, will predict the labels for a given test dataset using the fitted model.

The steps involved in solving a machine learning weight optimization problem with mlrose are typically:

1. Initialize a machine learning weight optimization problem object.
2. Find the optimal model weights for a given training dataset by calling the :code:`fit` method of the object initialized in step 1. 
3. Predict the labels for a test dataset by calling the :code:`predict` method of the object initialized in step 1. 

To fit the model weights, the user can choose between using either randomized hill climbing, simulated annealing, the genetic algorithm or gradient descent. In mlrose, the gradient descent algorithm is only available for use in solving the machine learning weight optimization problem and has been included primarily for benchmarking purposes, since this is one of the most common algorithm used in fitting neural networks and regression models.

We will now work through an example to illustrate how mlrose can be used to fit a neural network and a regression model to a given dataset.

**Example: the Iris Dataset**

The Iris dataset is a famous multivariate classification dataset first presented in a 1936 research paper by statistician and biologist Ronald Fisher. It contains 150 observations of three classes (species) of iris flowers (50 observations of each class), with each observation providing the sepal length, sepal width, petal length and petal width (i.e. the feature values), as well as the class label (i.e. the target value), of each flower under consideration.

The Iris dataset is included with the Python sklearn package. The feature values and label of the first observation in the dataset are shown below, along with the maximum and minimum values of each of the features and the unique label values:

.. highlight:: python
.. code-block:: python

    import numpy as np
    from sklearn.datasets import load_iris
    
    # Load the Iris dataset
    data = load_iris()

    # Get feature values
    print(data.data[0])
    [ 5.1  3.5  1.4  0.2]

    # Get feature names
    print(data.feature_names)
    ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    # Get target value of first observation
    print(data.target[0])
    0

    # Get target name of first observation
    print(data.target_names[data.target[0]])
    setosa
	
    # Get minimum feature values
    print(np.min(data.data, axis = 0))
    [ 4.3  2.   1.   0.1]
	
    # Get maximum feature values
    print(np.max(data.data, axis = 0))
    [ 7.9  4.4  6.9  2.5]
	
    # Get unique target values
    print(np.unique(data.target))
    [0 1 2]
	
From this we can see that all features in the Iris data set are numeric, albeit with different ranges, and that the class labels have been represented by integers.

In the next few sections we will show how mlrose can be used to fit a neural network and a logistic regression model to this dataset, to predict the species of an iris flower given its feature values.

Data Pre-Processing
-------------------
Before we can fit any sort of machine learning model to a dataset, it is necessary to manipulate our data into the form expected by mlrose. Each of the three machine learning models supported by mlrose expect to receive feature data in the form of a numpy array, with one row per observation and numeric features only (any categorical features must be one-hot encoded before passing to the machine learning models). 

The models also expect to receive the target values as either: a list of numeric values (for regression data); a list of 0-1 indicator values (for binary classification data); or as a numpy array of one-hot encoded labels, with one row per observation (for multi-class classification data). 

In the case of the Iris dataset, all of our features are numeric, so no one-hot encoding is required. However, it is necessary to one-hot encode the class labels.

In keeping with standard machine learning practice, it is also necessary to split the data into training and test subsets, and since the range of the Iris data varies considerably from feature to feature, to standardize the values of our feature variables.

These pre-processing steps are implemented below.

.. highlight:: python
.. code-block:: python

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
	
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, \
                                                        test_size = 0.2, random_state = 3)
	
    # Normalize feature data
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
	
    # One hot encode target values
    one_hot = OneHotEncoder()

    y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
    y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

Neural Networks
---------------
Once the data has been preprocessed, fitting a neural network in mlrose simply involves following the steps listed above. 

Suppose we wish to fit a neural network classifier to our Iris dataset with one hidden layer containing 2 nodes and a ReLU activation function (mlrose supports the ReLU, identity, sigmoid and tanh activation functions). 

For this example, we will use the Randomized Hill Climbing algorithm to find the optimal weights, with a maximum of 1000 iterations of the algorithm and 100 attempts to find a better set of weights at each step. We will also include a bias term; use a step size (learning rate) of 0.0001; and limit our weights to being in the range -5 to 5 (to reduce the landscape over which the algorithm must search in order to find the optimal weights).

This model is initialized and fitted to our preprocessed data below:

.. highlight:: python
.. code-block:: python

    # Initialize neural network object and fit object

    np.random.seed(3)

    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
                                     algorithm = 'random_hill_climb', max_iters = 1000, \
                                     bias = True, is_classifier = True, learning_rate = 0.0001, \
                                     early_stopping = True, clip_max = 5, max_attempts = 100)

    nn_model1.fit(X_train_scaled, y_train_hot)
	
Once the model is fitted, we can use it to predict the labels for our training and test datasets and use these prediction to assess the model's training and test accuracy.

.. highlight:: python
.. code-block:: python

    from sklearn.metrics import accuracy_score
	
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(X_train_scaled)

    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

    print(y_train_accuracy)
    0.45
	
    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model1.predict(X_test_scaled)

    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

    print(y_test_accuracy)
    0.533333333333
	
In this case, our model achieves training accuracy of 45% and test accuracy of 53.3%. These accuracy levels are better than if the labels were selected at random, but still leave room for improvement.

We can potentially improve on the accuracy of our model by tuning the parameters we set when initializing the neural network object. Suppose we decide to change the optimization algorithm to gradient descent, but leave all other model parameters unchanged.

.. highlight:: python
.. code-block:: python

    # Initialize neural network object and fit object
    np.random.seed(3)

    nn_model2 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
                                     algorithm = 'gradient_descent', max_iters = 1000, \
                                     bias = True, is_classifier = True, learning_rate = 0.0001, \
                                     early_stopping = True, clip_max = 5, max_attempts = 100)

    nn_model2.fit(X_train_scaled, y_train_hot)
	
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model2.predict(X_train_scaled)

    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

    print(y_train_accuracy)
    0.625

    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model2.predict(X_test_scaled)

    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

    print(y_test_accuracy)
    0.566666666667
	
This results in a 39% increase in training accuracy to 62.5%, but a much smaller increase in test accuracy to 56.7%.

Linear and Logistic Regression Models
-------------------------------------
Linear and logistic regression models are special cases of neural networks. A linear regression is a regression neural network with no hidden layers and an identity activation fuction, while a logistic regression is a classification neural network with no hidden layers and a sigmoid activation function. As a result, we could fit either of these models to our data using the :code:`NeuralNetwork()` class with parameters set appropriately.

For example, suppose we wished to fit a logistic regression to our Iris data using the randomized hill climbing algorithm and all other parameters set as for the example in the previous section. We could do this by initializing a :code:`NeuralNetwork()` object like so:

.. highlight:: python
.. code-block:: python

    lr_nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [], activation = 'sigmoid', \
                                        algorithm = 'random_hill_climb', max_iters = 1000, \
                                        bias = True, is_classifier = True, learning_rate = 0.0001, \
                                        early_stopping = True, clip_max = 5, max_attempts = 100)

However, for convenience, mlrose provides the :code:`LinearRegression()` and :code:`LogisticRegression()` wrapper classes, which simplify model initialization. 

In our Iris dataset example, we can, thus, initialize and fit our logistic regression model as follows:

.. highlight:: python
.. code-block:: python

    # Initialize logistic regression object and fit object

    np.random.seed(3)

    lr_model1 = mlrose.LogisticRegression(algorithm = 'random_hill_climb', max_iters = 1000, \
                                          bias = True, learning_rate = 0.0001, \
                                          early_stopping = True, clip_max = 5, max_attempts = 100)

    lr_model1.fit(X_train_scaled, y_train_hot)

    # Predict labels for train set and assess accuracy
    y_train_pred = lr_model1.predict(X_train_scaled)

    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

    print(y_train_accuracy)
    0.191666666667

    # Predict labels for test set and assess accuracy
    y_test_pred = lr_model1.predict(X_test_scaled)

    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

    print(y_test_accuracy)
    0.0666666666667
	
This model achieves 19.2% training accuracy and 6.7% test accuracy, which is worse than if we predicted the labels by selecting values at random.

Nevertheless, as in the previous section, we can potentially improve model accuracy by tuning the parameters set at initialization. 

Suppose we increase our learning rate to 0.01.

.. highlight:: python
.. code-block:: python

    # Initialize logistic regression object and fit object

    np.random.seed(3)

    lr_model2 = mlrose.LogisticRegression(algorithm = 'random_hill_climb', max_iters = 1000, \
                                          bias = True, learning_rate = 0.01, \
                                          early_stopping = True, clip_max = 5, max_attempts = 100)

    lr_model2.fit(X_train_scaled, y_train_hot)

    # Predict labels for train set and assess accuracy
    y_train_pred = lr_model2.predict(X_train_scaled)

    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

    print(y_train_accuracy)
    0.683333333333

    # Predict labels for test set and assess accuracy
    y_test_pred = lr_model2.predict(X_test_scaled)

    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

    print(y_test_accuracy)
    0.7

This results in signficant improvements to both training and test accuracy, with training accuracy levels now reaching 68.3% and test accuracy levels reaching 70%.

Summary
-------
In this tutorial we demonstrated how mlrose can be used to find the optimal weights of three types of machine learning models: neural networks, linear regression models and logistic regression models. 

Applying randomized optimization algorithms to the machine learning weight optimization problem is most certainly not the most common approach to solving this problem. However, it serves to demonstrate the versatility of the mlrose package and of randomized optimization algorithms in general.