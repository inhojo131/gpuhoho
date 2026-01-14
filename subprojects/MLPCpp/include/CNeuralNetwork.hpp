/*!
* \file CNeuralNetwork.hpp
* \brief Declaration of the CNeuralNetwork class.
* \author E.C.Bunschoten
* \version 1.2.0
*
* MLPCpp Project Website: https://github.com/EvertBunschoten/MLPCpp
*
* Copyright (c) 2023 Evert Bunschoten

* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.

* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#pragma once

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>

#include "CLayer.hpp"
#include "variable_def.hpp"

namespace MLPToolbox {
class CNeuralNetwork {
  /*!
   *\class CNeuralNetwork
   *\brief The CNeuralNetwork class allows for the evaluation of a loaded MLP
   *architecture for a given set of inputs. The class also contains a list of
   *the various supported activation function types (linear, relu, elu, gelu,
   *selu, sigmoid, swish, tanh, exp)which can be applied to the layers in the
   *network. Currently, only dense, feed-forward type neural nets are supported
   *in this implementation.
   */
private:
  std::vector<std::string> input_names, /*!< MLP input variable names. */
      output_names;                     /*!< MLP output variable names. */

  unsigned long n_hidden_layers = 0; /*!< Number of hidden layers (layers
                                        between input and output layer). */

  CLayer *inputLayer = nullptr, /*!< Pointer to network input layer. */
      *outputLayer = nullptr;   /*!< Pointer to network output layer. */

  std::vector<CLayer *> hiddenLayers; /*!< Hidden layer collection. */
  std::vector<CLayer *>
      total_layers; /*!< Hidden layers plus in/output layers */

  // std::vector<su2activematrix> weights_mat; /*!< Weights of synapses
  // connecting layers */
  std::vector<std::vector<std::vector<mlpdouble>>> weights_mat;

  std::vector<std::pair<mlpdouble, mlpdouble>>
      input_norm,  /*!< Normalization factors for network inputs */
      output_norm; /*!< Normalization factors for network outputs */

  mlpdouble *ANN_outputs; /*!< Pointer to network outputs */
  std::vector<std::vector<mlpdouble>>
      dOutputs_dInputs; /*!< Network output derivatives w.r.t inputs */
  std::vector<std::vector<std::vector<mlpdouble>>> d2Outputs_dInputs2;

  mlpdouble Phi,        /*!< Activation function output value. */
            Phi_prime,  /*!< Activation function derivative w.r.t. input. */
            Phi_dprime; /*!< Activation function second derivative w.r.t. input. */

  bool compute_gradient = false,        /*!< Evaluate network output gradients. */
       compute_second_gradient = false; /*!< Evaluate network output second order gradients. */
  /*!
   * \brief Available activation function enumeration.
   */
  enum class ENUM_ACTIVATION_FUNCTION {
    NONE = 0,
    LINEAR = 1,
    RELU = 2,
    ELU = 3,
    GELU = 4,
    SELU = 5,
    SIGMOID = 6,
    SWISH = 7,
    TANH = 8,
    EXPONENTIAL = 9
  };

  /*!
   * \brief Available activation function map.
   */
  std::map<std::string, ENUM_ACTIVATION_FUNCTION> activation_function_map{
      {"none", ENUM_ACTIVATION_FUNCTION::NONE},
      {"linear", ENUM_ACTIVATION_FUNCTION::LINEAR},
      {"elu", ENUM_ACTIVATION_FUNCTION::ELU},
      {"relu", ENUM_ACTIVATION_FUNCTION::RELU},
      {"gelu", ENUM_ACTIVATION_FUNCTION::GELU},
      {"selu", ENUM_ACTIVATION_FUNCTION::SELU},
      {"sigmoid", ENUM_ACTIVATION_FUNCTION::SIGMOID},
      {"swish", ENUM_ACTIVATION_FUNCTION::SWISH},
      {"tanh", ENUM_ACTIVATION_FUNCTION::TANH},
      {"exponential", ENUM_ACTIVATION_FUNCTION::EXPONENTIAL}};

  std::vector<ENUM_ACTIVATION_FUNCTION>
      activation_function_types; /*!< Activation function type for each layer in
                                    the network. */
  std::vector<std::string>
      activation_function_names; /*!< Activation function name for each layer in
                                    the network. */
  ENUM_SCALING_FUNCTIONS input_reg_method {ENUM_SCALING_FUNCTIONS::MINMAX},
                         output_reg_method {ENUM_SCALING_FUNCTIONS::MINMAX};
public:
  ~CNeuralNetwork() {
    delete inputLayer;
    delete outputLayer;
    for (std::size_t i = 1; i < total_layers.size() - 1; i++) {
      delete total_layers[i];
    }
    delete[] ANN_outputs;
  };
  /*!
   * \brief Set the input layer of the network.
   * \param[in] n_neurons - Number of inputs
   */
  void DefineInputLayer(unsigned long n_neurons) {
    /*--- Define the input layer of the network ---*/
    inputLayer = new CLayer(n_neurons);

    /* Mark layer as input layer */
    inputLayer->SetInput(true);
    input_norm.resize(n_neurons);
    input_names.resize(n_neurons);
  }

  /*!
   * \brief Set the output layer of the network.
   * \param[in] n_neurons - Number of outputs
   */
  void DefineOutputLayer(unsigned long n_neurons) {
    /*--- Define the output layer of the network ---*/
    outputLayer = new CLayer(n_neurons);
    output_norm.resize(n_neurons);
    output_names.resize(n_neurons);
  }

  /*!
   * \brief Add a hidden layer to the network
   * \param[in] n_neurons - Hidden layer size.
   */
  void PushHiddenLayer(unsigned long n_neurons) {
    /*--- Add a hidden layer to the network ---*/
    CLayer *newLayer = new CLayer(n_neurons);
    hiddenLayers.push_back(newLayer);
    n_hidden_layers++;
  }

  /*!
   * \brief Set the weight value of a specific synapse.
   * \param[in] i_layer - current layer.
   * \param[in] i_neuron - neuron index in current layer.
   * \param[in] j_neuron - neuron index of connecting neuron.
   * \param[in] value - weight value.
   */
  void SetWeight(unsigned long i_layer, unsigned long i_neuron,
                 unsigned long j_neuron, mlpdouble value) {
    weights_mat[i_layer][j_neuron][i_neuron] = value;
  };

  /*!
   * \brief Set bias value at a specific neuron.
   * \param[in] i_layer - Layer index.
   * \param[in] i_neuron - Neuron index of current layer.
   * \param[in] value - Bias value.
   */
  void SetBias(unsigned long i_layer, unsigned long i_neuron, mlpdouble value) {
    total_layers[i_layer]->SetBias(i_neuron, value);
  }

  /*!
   * \brief Set layer activation function.
   * \param[in] i_layer - Layer index.
   * \param[in] input - Activation function name.
   */
  void SetActivationFunction(unsigned long i_layer, std::string input) {
    /*--- Translate activation function name from input file to a number ---*/
    activation_function_names[i_layer] = input;

    // Set activation function type in current layer.
    activation_function_types[i_layer] =
        activation_function_map.find(input)->second;
    return;
  }

  /*!
   * \brief Define the regularization method used to normalize the inputs before feeding them to the network.
   * \param[in] reg_method_input - regularization method (minmax, standard, or robust).
   */
  void SetInputRegularization(ENUM_SCALING_FUNCTIONS reg_method_input) {
    input_reg_method = reg_method_input;
    return;
  }

  /*!
   * \brief Define the regularization method used to normalize the training data before training. 
   * \param[in] reg_method_input - regularization method (minmax, standard, or robust).
   */
  void SetOutputRegularization(ENUM_SCALING_FUNCTIONS reg_method_input) {
    output_reg_method = reg_method_input;
    return;
  }

  /*!
   * \brief Display the network architecture in the terminal.
   */
  void DisplayNetwork() const {
    /*--- Display information on the MLP architecture ---*/
    int display_width = 54;
    int column_width = int(display_width / 3.0) - 1;

    /*--- Input layer information ---*/
    std::cout << "+" << std::setfill('-') << std::setw(display_width)
              << std::right << "+" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "|" << std::left << std::setw(display_width - 1)
              << "Input Layer Information:"
              << "|" << std::endl;
    
    
    std::string val_disp_1, val_disp_2, label_disp_1, label_disp_2, norm_method;
    switch (input_reg_method)
    {
    case ENUM_SCALING_FUNCTIONS::MINMAX:
      label_disp_1 = "Lower limit";
      label_disp_2 = "Upper limit";
      norm_method = "minimum-maximum";
      break;
    case ENUM_SCALING_FUNCTIONS::STANDARD:
      label_disp_1 = "Mean";
      label_disp_2 = "std";
      norm_method = "mean-standard deviation";
      break;
    case ENUM_SCALING_FUNCTIONS::ROBUST:
      label_disp_1 = "Mean";
      label_disp_2 = "IQ range";
      norm_method = "Quantile range";
      break;
    default:
      break;
    }
    std::cout << "|" << std::left << std::setw(column_width)
              << "Input Normalization:";
    std::cout << std::setfill(' ') << std::right << std::setw(column_width) << norm_method
              << std::setfill(' ') << std::setw(column_width) << std::right << "|" << std::endl;
    std::cout << "+" << std::setfill('-') << std::setw(display_width)
              << std::right << "+" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "|" << std::left << std::setw(column_width)
              << "Input Variable:";
    std::cout << "|" << std::left << std::setw(column_width) << label_disp_1
              << "|" << std::left << std::setw(column_width) << label_disp_2
              << "|" << std::endl;      
    std::cout << "+" << std::setfill('-') << std::setw(display_width)
              << std::right << "+" << std::endl;
    std::cout << std::setfill(' ');

    /*--- Hidden layer information ---*/
    for (auto iInput = 0u; iInput < inputLayer->GetNNeurons(); iInput++)
      std::cout << "|" << std::left << std::setw(column_width)
                << std::to_string(iInput + 1) + ": " + input_names[iInput]
                << "|" << std::right << std::setw(column_width)
                << input_norm[iInput].first << "|" << std::right
                << std::setw(column_width) << input_norm[iInput].second << "|"
                << std::endl;
    std::cout << "+" << std::setfill('-') << std::setw(display_width)
              << std::right << "+" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "|" << std::left << std::setw(display_width - 1)
              << "Hidden Layers Information:"
              << "|" << std::endl;
    std::cout << "+" << std::setfill('-') << std::setw(display_width)
              << std::right << "+" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "|" << std::setw(column_width) << std::left << "Layer index"
              << "|" << std::setw(column_width) << std::left << "Neuron count"
              << "|" << std::setw(column_width) << std::left << "Function"
              << "|" << std::endl;
    std::cout << "+" << std::setfill('-') << std::setw(display_width)
              << std::right << "+" << std::endl;
    std::cout << std::setfill(' ');
    for (auto iLayer = 0u; iLayer < n_hidden_layers; iLayer++)
      std::cout << "|" << std::setw(column_width) << std::right << iLayer + 1
                << "|" << std::setw(column_width) << std::right
                << hiddenLayers[iLayer]->GetNNeurons() << "|"
                << std::setw(column_width) << std::right
                << activation_function_names[iLayer + 1] << "|" << std::endl;
    std::cout << "+" << std::setfill('-') << std::setw(display_width)
              << std::right << "+" << std::endl;
    std::cout << std::setfill(' ');

    /*--- Output layer information ---*/
    switch (output_reg_method)
    {
    case ENUM_SCALING_FUNCTIONS::MINMAX:
      label_disp_1 = "Lower limit";
      label_disp_2 = "Upper limit";
      norm_method = "minimum-maximum";
      break;
    case ENUM_SCALING_FUNCTIONS::STANDARD:
      label_disp_1 = "Mean";
      label_disp_2 = "std";
      norm_method = "mean-standard deviation";
      break;
    case ENUM_SCALING_FUNCTIONS::ROBUST:
      label_disp_1 = "Mean";
      label_disp_2 = "IQ range";
      norm_method = "Quantile range";
      break;
    default:
      break;
    }
    std::cout << "|" << std::left << std::setw(display_width - 1)
              << "Output Layer Information:"
              << "|" << std::endl;
    std::cout << "+" << std::setfill('-') << std::setw(display_width)
              << std::right << "+" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "|" << std::left << std::setw(column_width)
              << "Output Variable:";
    std::cout << "|" << std::left << std::setw(column_width) << label_disp_1
              << "|" << std::left << std::setw(column_width) << label_disp_2
              << "|" << std::endl;
    std::cout << "+" << std::setfill('-') << std::setw(display_width)
              << std::right << "+" << std::endl;
    std::cout << std::setfill(' ');
    for (auto iOutput = 0u; iOutput < outputLayer->GetNNeurons(); iOutput++)
      std::cout << "|" << std::left << std::setw(column_width)
                << std::to_string(iOutput + 1) + ": " + output_names[iOutput]
                << "|" << std::right << std::setw(column_width)
                << output_norm[iOutput].first << "|" << std::right
                << std::setw(column_width) << output_norm[iOutput].second << "|"
                << std::endl;
    std::cout << "+" << std::setfill('-') << std::setw(display_width)
              << std::right << "+" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << std::endl;
  }

  /*!
   * \brief Size the weight layers in the network according to its architecture.
   */
  void SizeWeights() {
    /*--- Size weight matrices based on neuron counts in each layer ---*/

    /* Generate std::vector containing input, output, and hidden layer
     * references
     */
    total_layers.resize(n_hidden_layers + 2);
    total_layers[0] = inputLayer;
    for (auto iLayer = 0u; iLayer < n_hidden_layers; iLayer++) {
      total_layers[iLayer + 1] = hiddenLayers[iLayer];
    }
    total_layers[total_layers.size() - 1] = outputLayer;

    weights_mat.resize(n_hidden_layers + 1);
    weights_mat[0].resize(hiddenLayers[0]->GetNNeurons());
    for (auto iNeuron = 0u; iNeuron < hiddenLayers[0]->GetNNeurons(); iNeuron++)
      weights_mat[0][iNeuron].resize(inputLayer->GetNNeurons());

    for (auto iLayer = 1u; iLayer < n_hidden_layers; iLayer++) {
      weights_mat[iLayer].resize(hiddenLayers[iLayer]->GetNNeurons());
      for (auto iNeuron = 0u; iNeuron < hiddenLayers[iLayer]->GetNNeurons();
           iNeuron++) {
        weights_mat[iLayer][iNeuron].resize(
            hiddenLayers[iLayer - 1]->GetNNeurons());
      }
    }
    weights_mat[n_hidden_layers].resize(outputLayer->GetNNeurons());
    for (auto iNeuron = 0u; iNeuron < outputLayer->GetNNeurons(); iNeuron++) {
      weights_mat[n_hidden_layers][iNeuron].resize(
          hiddenLayers[n_hidden_layers - 1]->GetNNeurons());
    }

    ANN_outputs = new mlpdouble[outputLayer->GetNNeurons()];

    /* Size data structures used for first and second order derivative
     * computation. */
    dOutputs_dInputs.resize(outputLayer->GetNNeurons());
    d2Outputs_dInputs2.resize(outputLayer->GetNNeurons());
    for (auto iOutput = 0u; iOutput < outputLayer->GetNNeurons(); iOutput++) {
      dOutputs_dInputs[iOutput].resize(inputLayer->GetNNeurons());
      d2Outputs_dInputs2[iOutput].resize(inputLayer->GetNNeurons());
      for (auto jInput = 0u; jInput < inputLayer->GetNNeurons(); jInput++) {
        d2Outputs_dInputs2[iOutput][jInput].resize(inputLayer->GetNNeurons());
      }
    }

    for (auto iLayer = 0u; iLayer < n_hidden_layers + 2; iLayer++) {
      total_layers[iLayer]->SizeGradients(inputLayer->GetNNeurons());
    }
  }

  /*!
   * \brief Get the number of connecting regions in the network.
   * \returns number of spaces in between layers.
   */
  unsigned long GetNWeightLayers() const { return total_layers.size() - 1; }

  /*!
   * \brief Get the total number of layers in the network
   * \returns number of netowork layers.
   */
  unsigned long GetNLayers() const { return total_layers.size(); }

  /*!
   * \brief Get neuron count in a layer.
   * \param[in] iLayer - Layer index.
   * \returns number of neurons in the layer.
   */
  unsigned long GetNNeurons(unsigned long iLayer) const {
    return total_layers[iLayer]->GetNNeurons();
  }

  void ComputeFirstOrderGradient(bool input) { compute_gradient = input; }

  void ComputeSecondOrderGradient(bool input) {
    compute_second_gradient = input;
  }

  /*!
   * \brief Set the neuron output values of the input layer.
   * \param[in] inputs - Vector containing non-normalized network inputs.
   * \param[in] iNeuron - Input neuron index.
   */
  void ComputeInputLayer(std::vector<mlpdouble> &inputs, std::size_t iNeuron) {

    /* Compute normalized input value according to regularizer. */
    mlpdouble x_norm = NormalizeInput(inputs[iNeuron], iNeuron);

    inputLayer->SetOutput(iNeuron, x_norm);
    if (compute_gradient) {
      for (auto jInput = 0u; jInput < inputLayer->GetNNeurons(); jInput++) {
        if (jInput == iNeuron) {
          inputLayer->SetdYdX(
              iNeuron, jInput,
              1 / GetRegularizationScale(iNeuron, true));
        } else {
          inputLayer->SetdYdX(iNeuron, jInput, 0.0);
        }
        if (compute_second_gradient)
          for (auto kInput = 0u; kInput < inputLayer->GetNNeurons(); kInput++) {
            inputLayer->Setd2YdX2(iNeuron, jInput, kInput, 0.0);
          }
      }
    }
  }

  /*!
   * \brief Evaluate activation function applied to the current layer.
   * \param[in] iLayer - Current layer index.
   * \param[in] input - Activation function input value.
   */
  void ActivationFunction(std::size_t iLayer, mlpdouble input) {
    /* Evaluating the activation function sets the value of the class parameter
     * Phi. In case of gradient computation, Phi_prime and/or Phi_dprime are set
     * as well, corresponding to the first and second analytical derivative of
     * the activation function w.r.t. its input respectively. */

    /* Constants used for various activation functions. */
    mlpdouble exp_x, tanh_x, alpha = 1.67326324, lambda = 1.05070098,
                             gelu_c = 1.702;

    switch (activation_function_types[iLayer]) {
    case ENUM_ACTIVATION_FUNCTION::ELU:
      if (input > 0) {
        Phi = input;
        if (compute_gradient) {
          Phi_prime = 1.0;
          if (compute_second_gradient)
            Phi_dprime = 0.0;
        }
      } else {
        exp_x = exp(input);
        Phi = exp_x - 1;
        if (compute_gradient) {
          Phi_prime = exp_x;
          if (compute_second_gradient)
            Phi_dprime = exp_x;
        }
      }
      break;
    case ENUM_ACTIVATION_FUNCTION::LINEAR:
      Phi = input;
      if (compute_gradient) {
        Phi_prime = 1.0;
        if (compute_second_gradient)
          Phi_dprime = 0.0;
      }
      break;
    case ENUM_ACTIVATION_FUNCTION::EXPONENTIAL:
      Phi = exp(input);
      if (compute_gradient) {
        Phi_prime = Phi;
        if (compute_second_gradient)
          Phi_dprime = Phi;
      }
      break;
    case ENUM_ACTIVATION_FUNCTION::RELU:
      if (input > 0) {
        Phi = input;
        if (compute_gradient)
          Phi_prime = 1.0;
      } else {
        Phi = 0.0;
        if (compute_gradient)
          Phi_prime = 0.0;
      }
      if (compute_second_gradient)
        Phi_dprime = 0.0;
      break;
    case ENUM_ACTIVATION_FUNCTION::SWISH:
      Phi = input / (1 + exp(-input));
      if (compute_gradient) {
        exp_x = exp(input);
        Phi_prime = exp_x * (input + exp_x + 1) / pow(exp_x + 1, 2);
        if (compute_second_gradient)
          Phi_dprime =
              exp_x * (-exp_x * (input - 2) + input + 2) / pow(exp_x + 1, 3);
      }
      break;
    case ENUM_ACTIVATION_FUNCTION::TANH:
      tanh_x = tanh(input);
      Phi = tanh_x;
      if (compute_gradient) {
        Phi_prime = 1 / pow(cosh(input), 2);
        if (compute_second_gradient)
          Phi_dprime = -2 * tanh_x * 1 / (pow(cosh(input), 2));
      }
      break;
    case ENUM_ACTIVATION_FUNCTION::SIGMOID:
      exp_x = exp(-input);
      Phi = 1.0 / (1 + exp_x);
      if (compute_gradient) {
        Phi_prime = exp_x / pow(exp_x + 1, 2);
        if (compute_second_gradient) {
          exp_x = exp(input);
          Phi_dprime = -(exp_x * (exp_x - 1)) / pow(exp_x + 1, 3);
        }
      }
      break;
    case ENUM_ACTIVATION_FUNCTION::SELU:
      if (input > 0) {
        Phi = lambda * input;
        if (compute_gradient) {
          Phi_prime = lambda;
          if (compute_second_gradient)
            Phi_dprime = 0.0;
        }
      } else {
        exp_x = exp(input);
        Phi = lambda * alpha * (exp_x - 1);
        if (compute_gradient) {
          Phi_prime = Phi + lambda * alpha;
          if (compute_second_gradient)
            Phi_dprime = Phi_prime;
        }
      }
      break;
    case ENUM_ACTIVATION_FUNCTION::GELU:
      Phi = 0.5 * input * (1 + erf(input / sqrt(2)));
      if (compute_gradient) {
        exp_x = exp(-gelu_c * input);
        Phi_prime = exp_x * (gelu_c * input + exp_x + 1) / pow(exp_x + 1, 2);
        if (compute_second_gradient)
          Phi_dprime = input * ((5.79361 * pow(exp_x, 2) / pow(exp_x + 1, 3)) -
                                (2.8968 * exp_x / pow(exp_x + 1, 2))) +
                       3.404 * exp_x / pow(exp_x + 1, 2);
      }
      break;
    case ENUM_ACTIVATION_FUNCTION::NONE:
      Phi = 0.0;
      if (compute_gradient) {
        Phi_prime = 0.0;
        if (compute_second_gradient)
          Phi_dprime = 0.0;
      }
      break;
    default:
      break;
    }
  }

  /*!
   * \brief De-normalize the network outputs based on output normalization
   * values.
   */
  void DeNormalizeOutputs() {
    /* Compute and de-normalize MLP output */
    for (auto iNeuron = 0u; iNeuron < outputLayer->GetNNeurons(); iNeuron++) {
      mlpdouble y_norm = outputLayer->GetOutput(iNeuron);
      mlpdouble output_scale = GetRegularizationScale(iNeuron, false);

      mlpdouble Y_out = DimensionalizeOutput(y_norm, iNeuron);

      /* Storing output value */
      ANN_outputs[iNeuron] = Y_out;

      /* Scale first and second derivatives */
      if (compute_gradient) {
        for (auto jInput = 0u; jInput < inputLayer->GetNNeurons(); jInput++) {
          outputLayer->SetdYdX(iNeuron, jInput,
                               output_scale *
                                   outputLayer->GetdYdX(iNeuron, jInput));
          dOutputs_dInputs[iNeuron][jInput] =
              outputLayer->GetdYdX(iNeuron, jInput);

          if (compute_second_gradient) {
            for (auto kInput = 0u; kInput < inputLayer->GetNNeurons();
                 kInput++) {
              outputLayer->Setd2YdX2(
                  iNeuron, jInput, kInput,
                  output_scale *
                      outputLayer->Getd2YdX2(iNeuron, jInput, kInput));
              d2Outputs_dInputs2[iNeuron][jInput][kInput] =
                  outputLayer->Getd2YdX2(iNeuron, jInput, kInput);
            }
          }
        }
      }
    }
  }

  /*!
   * \brief Evaluate the network based on dimensionalized inputs.
   * \param[in] inputs - Vector containing non-normalized network inputs.
   */
  void Predict(std::vector<mlpdouble> &inputs) {

    for (auto iLayer = 0u; iLayer < n_hidden_layers + 2; iLayer++) {
      for (auto iNeuron = 0u; iNeuron < total_layers[iLayer]->GetNNeurons();
           iNeuron++) {
        if (total_layers[iLayer]->IsInput()) {
          ComputeInputLayer(inputs, iNeuron);
        } else {
          /* Compute activation function input value. */
          mlpdouble X = ComputeX(iLayer, iNeuron);

          /* Evaluate activation function. */
          ActivationFunction(iLayer, X);

          /* Store activation function output in current neuron. */
          mlpdouble Yi = Phi;
          total_layers[iLayer]->SetOutput(iNeuron, Yi);

          /* Compute first and/or second order derivatives. */
          if (compute_gradient) {
            for (auto jInput = 0u; jInput < inputLayer->GetNNeurons();
                 jInput++) {
              mlpdouble psi_j = ComputePsi(iLayer, iNeuron, jInput);
              mlpdouble dYi_dIj = psi_j * Phi_prime;
              total_layers[iLayer]->SetdYdX(iNeuron, jInput, dYi_dIj);
              if (compute_second_gradient) {
                for (auto kInput = 0u; kInput < inputLayer->GetNNeurons();
                     kInput++) {
                  mlpdouble psi_k = ComputePsi(iLayer, iNeuron, kInput);
                  mlpdouble chi = ComputeChi(iLayer, iNeuron, jInput, kInput);
                  mlpdouble d2Yi_dIjdIk =
                      Phi_dprime * psi_j * psi_k + Phi_prime * chi;

                  total_layers[iLayer]->Setd2YdX2(iNeuron, jInput, kInput,
                                                  d2Yi_dIjdIk);
                } // kInput
              }   // compute_second_gradient
            }     // jInput
          }       // compute_gradient
        }         // is_input_layer
      }           // iNeuron
    }             // iLayer

    // De-normalize the network outputs and gradients.
    DeNormalizeOutputs();
  }

  /*!
   * \brief Set the normalization factors for the input layer
   * \param[in] iInput - Input index.
   * \param[in] input_min - Minimum input value.
   * \param[in] input_max - Maximum input value.
   */
  void SetInputNorm(unsigned long iInput, mlpdouble input_min,
                    mlpdouble input_max) {
    input_norm[iInput] = std::make_pair(input_min, input_max);
  }

  /*!
   * \brief Set the normalization factors for the output layer
   * \param[in] iOutput - Input index.
   * \param[in] input_min - Minimum output value.
   * \param[in] input_max - Maximum output value.
   */
  void SetOutputNorm(unsigned long iOutput, mlpdouble output_min,
                     mlpdouble output_max) {
    output_norm[iOutput] = std::make_pair(output_min, output_max);
  }

  std::pair<mlpdouble, mlpdouble> GetInputNorm(unsigned long iInput) const {
    return input_norm[iInput];
  }

  std::pair<mlpdouble, mlpdouble> GetOutputNorm(unsigned long iOutput) const {
    return output_norm[iOutput];
  }
  /*!
   * \brief Add an output variable name to the network.
   * \param[in] input - Input variable name.
   */
  void SetOutputName(size_t iOutput, std::string input) {
    output_names[iOutput] = input;
  }

  /*!
   * \brief Add an input variable name to the network.
   * \param[in] output - Output variable name.
   */
  void SetInputName(size_t iInput, std::string input) {
    input_names[iInput] = input;
  }

  /*!
   * \brief Get network input variable name.
   * \param[in] iInput - Input variable index.
   * \returns input variable name.
   */
  std::string GetInputName(std::size_t iInput) const {
    return input_names[iInput];
  }

  /*!
   * \brief Get network output variable name.
   * \param[in] iOutput - Output variable index.
   * \returns output variable name.
   */
  std::string GetOutputName(std::size_t iOutput) const {
    return output_names[iOutput];
  }

  /*!
   * \brief Get network number of inputs.
   * \returns Number of network inputs
   */
  std::size_t GetnInputs() const { return input_names.size(); }

  /*!
   * \brief Get network number of outputs.
   * \returns Number of network outputs
   */
  std::size_t GetnOutputs() const { return output_names.size(); }

  /*!
   * \brief Get network evaluation output.
   * \param[in] iOutput - output index.
   * \returns Prediction value.
   */
  mlpdouble GetANNOutput(std::size_t iOutput) const {
    return ANN_outputs[iOutput];
  }

  /*!
   * \brief Get network output derivative w.r.t specific input.
   * \param[in] iOutput - output variable index.
   * \param[in] iInput - input variable index.
   * \returns Output derivative w.r.t input.
   */
  mlpdouble GetdOutputdInput(std::size_t iOutput, std::size_t iInput) const {
    return dOutputs_dInputs[iOutput][iInput];
  }

  /*!
   * \brief Get network output second derivative w.r.t specific inputs.
   * \param[in] iOutput - output variable index.
   * \param[in] iInput - first input variable index.
   * \param[in] jInput - second input variable index.
   * \returns Output second derivative w.r.t inputs.
   */
  mlpdouble Getd2OutputdInput2(std::size_t iOutput, std::size_t iInput,
                               std::size_t jInput) const {
    return d2Outputs_dInputs2[iOutput][iInput][jInput];
  }
  /*!
   * \brief Set the activation function array size.
   * \param[in] n_layers - network layer count.
   */
  void SizeActivationFunctions(unsigned long n_layers) {
    activation_function_types.resize(n_layers);
    activation_function_names.resize(n_layers);
  }

  /*!
   * \brief Compute neuron activation function input.
   * \param[in] iLayer - Network layer index.
   * \param[in] iNeuron - Layer neuron index.
   * \returns Neuron activation function input.
   */
  mlpdouble ComputeX(std::size_t iLayer, std::size_t iNeuron) const {
    mlpdouble x;
    x = total_layers[iLayer]->GetBias(iNeuron);
    std::size_t nNeurons_previous = total_layers[iLayer - 1]->GetNNeurons();
    for (std::size_t jNeuron = 0; jNeuron < nNeurons_previous; jNeuron++) {
      x += weights_mat[iLayer - 1][iNeuron][jNeuron] *
           total_layers[iLayer - 1]->GetOutput(jNeuron);
    }
    return x;
  }

  /*!
   * \brief Compute the weighted sum of the neuron output derivatives of the
   * previous layer. \param[in] iLayer - Current network layer index. \param[in]
   * iNeuron - Layer neuron index. \param[in] jInput - Input variable index used
   * for derivative. \returns Weighted sum of prefious layer derivatives w.r.t.
   * input variable.
   */
  mlpdouble ComputePsi(std::size_t iLayer, std::size_t iNeuron,
                       std::size_t jInput) const {
    mlpdouble psi = 0;
    for (auto jNeuron = 0u; jNeuron < total_layers[iLayer - 1]->GetNNeurons();
         jNeuron++) {
      psi += weights_mat[iLayer - 1][iNeuron][jNeuron] *
             total_layers[iLayer - 1]->GetdYdX(jNeuron, jInput);
    }
    return psi;
  }

  /*!
   * \brief Compute the weighted sum of the neuron output second derivatives of
   * the previous layer. \param[in] iLayer - Current network layer index.
   * \param[in] iNeuron - Layer neuron index.
   * \param[in] jInput - First input variable index used for derivative.
   * \param[in] kInput - Second input variable index used for derivative.
   * \returns Weighted sum of previous layer second derivatives w.r.t. input
   * variables.
   */
  mlpdouble ComputeChi(std::size_t iLayer, std::size_t iNeuron,
                       std::size_t jInput, std::size_t kInput) const {
    mlpdouble chi = 0;
    for (auto jNeuron = 0u; jNeuron < total_layers[iLayer - 1]->GetNNeurons();
         jNeuron++) {
      chi += weights_mat[iLayer - 1][iNeuron][jNeuron] *
             total_layers[iLayer - 1]->Getd2YdX2(jNeuron, jInput, kInput);
    }
    return chi;
  }

  /*!
   * \brief Compute the weighted sum of the weighted output derivatives of the previous layer.
   * \param[in] iLayer - Current network layer index.
   * \param[in] iNeuron - Neuron index of current layer.
   * \param[in] iInput - Input index for which to compute derivative.
   * \returns Weighted sum of previous layer output derivatives.
   */
  mlpdouble ComputedOutputdInput(std::size_t iLayer, std::size_t iNeuron,
                                 std::size_t iInput) const {
    mlpdouble doutput_dinput = 0;
    for (auto jNeuron = 0u; jNeuron < total_layers[iLayer - 1]->GetNNeurons();
         jNeuron++) {
      doutput_dinput += weights_mat[iLayer - 1][iNeuron][jNeuron] *
                        total_layers[iLayer - 1]->GetdYdX(jNeuron, iInput);
    }
    return doutput_dinput;
  }

  /*!
   * \brief Normalize the network input.
   * \param[in] val_input_dim - Dimensional input value.
   * \param[in] iInput - Input index.
   * \returns Normalized network input value.
   */
  mlpdouble NormalizeInput(mlpdouble val_input_dim, std::size_t iInput) const {
    mlpdouble val_norm_input{0};
    switch(input_reg_method)
    {
    case ENUM_SCALING_FUNCTIONS::MINMAX:
      val_norm_input = (val_input_dim - input_norm[iInput].first) / (input_norm[iInput].second - input_norm[iInput].first);
      break;
    case ENUM_SCALING_FUNCTIONS::STANDARD:
    case ENUM_SCALING_FUNCTIONS::ROBUST:
    default:
      val_norm_input= (val_input_dim - input_norm[iInput].first) / input_norm[iInput].second;
      break;
    }
    return val_norm_input;
  }

  mlpdouble GetRegularizationScale(std::size_t iInput, bool is_input=true) const {
    switch(input_reg_method)
    {
    case ENUM_SCALING_FUNCTIONS::MINMAX:
      if (is_input) {
        return input_norm[iInput].second - input_norm[iInput].first;
      } else {
        return output_norm[iInput].second - output_norm[iInput].first;
      }
      break;
    case ENUM_SCALING_FUNCTIONS::STANDARD:
    case ENUM_SCALING_FUNCTIONS::ROBUST:
      if (is_input) {
        return input_norm[iInput].second;
      } else {
        return output_norm[iInput].second;
      }
      break;
    default:
      return 0;
      break;
    }
  }

  mlpdouble GetRegularizationOffset(std::size_t iInput, bool is_input=true) const {
    switch(input_reg_method)
    {
    case ENUM_SCALING_FUNCTIONS::MINMAX:
      if (is_input) {
        return 0.5*(input_norm[iInput].second + input_norm[iInput].first);
      } else {
        return 0.5*(output_norm[iInput].second + output_norm[iInput].first);
      }
      break;
    case ENUM_SCALING_FUNCTIONS::STANDARD:
    case ENUM_SCALING_FUNCTIONS::ROBUST:
      if (is_input) {
        return input_norm[iInput].first;
      } else {
        return output_norm[iInput].first;
      }
      break;
    default:
      return 0;
      break;
    }
  }

  mlpdouble DimensionalizeOutput(mlpdouble val_output_norm, std::size_t iOutput) const {
    mlpdouble val_dim_output{0};
    switch(input_reg_method)
    {
    case ENUM_SCALING_FUNCTIONS::MINMAX:
      val_dim_output = (output_norm[iOutput].second - output_norm[iOutput].first) * val_output_norm + output_norm[iOutput].first;
      break;
    case ENUM_SCALING_FUNCTIONS::STANDARD:
    case ENUM_SCALING_FUNCTIONS::ROBUST:
    default:
      val_dim_output = output_norm[iOutput].second * val_output_norm + output_norm[iOutput].first;
      break;
    }
    return val_dim_output;
  }


  bool CheckInputInclusion(mlpdouble val_input, size_t iInput) const {
    bool inside {true};
    mlpdouble val_input_norm;
    switch(input_reg_method)
    {
    case ENUM_SCALING_FUNCTIONS::MINMAX:
      if ((val_input < input_norm[iInput].first) || 
          (val_input > input_norm[iInput].second))
          inside = false;
      break;
    case ENUM_SCALING_FUNCTIONS::STANDARD:
      val_input_norm = (val_input - input_norm[iInput].first) / input_norm[iInput].second;
      if ((val_input_norm < -2.0) || (val_input_norm > 2.0))
        inside = false;
      break;
    case ENUM_SCALING_FUNCTIONS::ROBUST:
      val_input_norm = (val_input - input_norm[iInput].first) / input_norm[iInput].second;
      if ((val_input_norm < -10.0) || (val_input_norm > 10.0))
        inside = false;
      break;
    default:
      break;
    }
    
    return inside;
  }

};

} // namespace MLPToolbox
