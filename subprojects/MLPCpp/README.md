---
title: Multi-Layer Perceptron regression in C++ code
---

# Multi-Layer Perceptrons in C++
<img src="logo.png" alt="isolated" width="200"/>

Artificial neural networks like multi-layer perceptrons (MLP), are useful a variety of applications. From turbulence closure models in computational fluid dynamics to data regression applications, MLP's have been shown to be a valuable tool. This library was developed to easily allow for the evaluation of MLP's within a C++ code environment through a set of header files. To clarify: this library **DOES NOT** allow for the training or optimization of MLP architectures. It only allows for the evaluation of MLP outputs and first-order and second-order output gradients with respect to the network inputs for networks which already have been trained through an external tool like Python Tensorflow. 

# MLP class description
The MLP library can be downloaded from the git repository https://github.com/EvertBunschoten/MLPCpp.git. By including the header file CLookUp_ANN.hpp, it enables the use of multi-layer perceptrons for regression operations in C++ code. 
An MLP computes its outputs by having its inputs manipulated by a series of operations, depending on the architecture of the network. Interpreting the network architecture and its respective input and output variables is therefore crucial for the MLP functionality. Information regarding the network architecture, input and output variables, and activation functions has to be provided via a .mlp input file, of which two examples are provided in the main library folder ("MLP_1.mlp" and "MLP_2.mlp"). More information regarding the file structure is provided in a later section. 

The main class governing  which can be used for look-up operations is the CLookUp_ANN class. This class allows to load one or multiple networks given a list of input files. This librarly currently only supports **deep or shallow, dense, feed-forward** type neural networks. Each of the input files is read by the CReadNeuralNetwork class. This class reads the .mlp input file and stores the architectural information listed in it. It will also run some compatibility checks on the file format. For example, the total layer count should be provided before listing the activation functions. For every read .mlp file, an MLP class is generated using the CNeuralNetwork class. This class stores the input and output variables, network architecture, activation functions, and weights and biases of every synapse and neuron. Currently, the CNeuralNetwork class only supports simple, feed-forward, dense neural network types. Currently supported activation functions are:
1. linear (y = x)
2. relu
3. elu
4. swish
5. sigmoid
6. tanh
7. selu
8. gelu
9. exponential (y = exp(x))

It is possible to load multiple networks with different input and output variables. An error will be raised if none of the loaded MLP's contains all the input variables or if some of the desired outputs are missing from the MLP output variables.
In addition to loading multiple networks with different input and output variables, it is possible to load multple networks with the same input and output variables, but with different data ranges. When performing a regression operation, the CLookUp_ANN class will check which of the loaded MLPs with the right input and output variables has an input variable normalization range that includes the query point. The corresponding MLP will then be selected for regression. If the query point lies outside the data range of all loaded MLPs, extrapolation will be performed using the MLP with a data range close to the query point. 

# Regularization of inputs and outputs 
MLP's are often trained on regularized input and labeled data. MLPCpp supports three regularization methods: [min-max scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler), [standard deviation scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler), and [robust scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler). The default option is min-max scaling (feature range 0-1). 

Specify the regularization method in the header under [input regularization method] or [output regularization method] and list the following two normalization values under [input normalization] and [output normalization] respectively:
1. min-max ("minmax") : training data lower limit, training data upper limit
2. standard deviation ("standard") : mean value, standard deviation
3. robust ("robust") : mean value, quantile range 

# MLP Definition and Usage
The input files required for loading MLP's into C++ through the MLPCpp library are in ASCII format. A supporting script is provided with allows for the translation of an MLP trained through Tensorflow to a supported .mlp file. This script is named "Tensorflow_Translation.py" script, which can be found under "src". Details regarding the functionality of this translation script can be found in the code itself.

Using the MLPCpp library in your code is as simple as including the "CLookUp_ANN.hpp" file in your C++ code. An examplary script is provided through "main.cpp", which demonstrates the steps required for loading one or multiple .mlp files, preprocessing regression operations, and the evaluation of the network outputs and output derivatives. 

# Gradient Computation
The MLPCpp module allows for the evaluation of the analytical first-order and second-order derivatives of the network outputs with respect to the network inputs without the use of algorithmic differentiation. This can be useful in iterative Newton solvers for example. Gradient computation is enabled by supplying additional inputs to the "Predict_ANN" method, as is demonstrated in "main.cpp"

# Test Case

Under ```TestCase```, one can find a demonstration of the MLPCpp library. [Here](TestCase/test_problem.py), an MLP with two inputs and one output is trained using TensorFlow, converted to MLPCpp ASCII format, and evaluated using the functions in the MLPCpp library. 