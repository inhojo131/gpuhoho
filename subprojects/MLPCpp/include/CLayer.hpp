/*!
* \file CLayer.hpp
* \brief Dense layer class definition used within the CNeuralNetwork class.
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
#include <iostream>
#include <limits>
#include <vector>

#include "CNeuron.hpp"
#include "variable_def.hpp"

namespace MLPToolbox {
class CLayer {
  /*!
   *\class CLayer
   *\brief This class functions as one of the hidden, input, or output layers in
   *the multi-layer perceptron class. The CLayer class is used to communicate
   *information (activation function inputs and outputs and gradients) between
   *the CNeuralNetwork class and the CNeuron class. Currently, only a single
   *activation function can be applied to the neuron inputs within the layer.
   */
private:
  unsigned long number_of_neurons; /*!< Neuron count in current layer */
  std::vector<CNeuron> neurons;    /*!< Array of neurons in current layer */
  bool is_input;                   /*!< Input layer identifyer */
  std::string activation_type;     /*!< Activation function type applied to the
                                      current layer*/
public:
  CLayer() : CLayer(1) {}
  CLayer(unsigned long n_neurons)
      : number_of_neurons{n_neurons}, is_input{false} {
    neurons.resize(n_neurons);
    for (size_t i = 0; i < number_of_neurons; i++) {
      neurons[i].SetNumber(i + 1);
    }
  }
  /*!
   * \brief Set current layer neuron count
   * \param[in] n_neurons - Number of neurons in this layer
   */
  void SetNNeurons(unsigned long n_neurons) {
    if (number_of_neurons != n_neurons) {
      neurons.resize(n_neurons);
      for (size_t i = 0; i < number_of_neurons; i++) {
        neurons[i].SetNumber(i + 1);
      }
    }
  }

  /*!
   * \brief Get the current layer neuron count
   * \return Neuron count
   */
  unsigned long GetNNeurons() const { return number_of_neurons; }

  /*!
   * \brief Define current layer as input layer
   * \param[in] input - input layer identifyer
   */
  void SetInput(bool def) { is_input = def; }

  /*!
   * \brief Get input layer identifyer
   * \return input layer identifyer
   */
  bool IsInput() const { return is_input; }

  /*!
   * \brief Set the output value of a neuron in the layer
   * \param[in] i_neuron - Neuron index
   * \param[in] output_value - Activation function output
   */
  void SetOutput(std::size_t i_neuron, mlpdouble value) {
    neurons[i_neuron].SetOutput(value);
  }

  /*!
   * \brief Get the output value of a neuron in the layer
   * \param[in] i_neuron - Neuron index
   * \return Neuron output value
   */
  mlpdouble GetOutput(std::size_t i_neuron) const {
    return neurons[i_neuron].GetOutput();
  }

  /*!
   * \brief Set the input value of a neuron in the layer
   * \param[in] i_neuron - Neuron index
   * \param[in] input_value - Activation function input
   */
  void SetInput(std::size_t i_neuron, mlpdouble value) {
    neurons[i_neuron].SetInput(value);
  }

  /*!
   * \brief Get the input value of a neuron in the layer
   * \param[in] i_neuron - Neuron index
   * \return Neuron input value
   */
  mlpdouble GetInput(std::size_t i_neuron) const {
    return neurons[i_neuron].GetInput();
  }

  /*!
   * \brief Set the bias value of a neuron in the layer
   * \param[in] i_neuron - Neuron index
   * \param[in] bias_value - Bias value
   */
  void SetBias(std::size_t i_neuron, mlpdouble value) {
    neurons[i_neuron].SetBias(value);
  }

  /*!
   * \brief Get the bias value of a neuron in the layer
   * \param[in] i_neuron - Neuron index
   * \return Neuron bias value
   */
  mlpdouble GetBias(std::size_t i_neuron) const {
    return neurons[i_neuron].GetBias();
  }

  /*!
   * \brief Get the output-input gradient of a neuron in the layer
   * \param[in] i_neuron - Neuron index
   * \return Gradient of neuron output wrt input
   */
  mlpdouble GetdYdX(std::size_t i_neuron, std::size_t iInput) const {
    return neurons[i_neuron].GetGradient(iInput);
  }

  /*!
   * \brief Get the output-input gradient of a neuron in the layer
   * \param[in] i_neuron - Neuron index
   * \return Gradient of neuron output wrt input
   */
  void SetdYdX(std::size_t i_neuron, std::size_t iInput, mlpdouble dy_dx) {
    neurons[i_neuron].SetGradient(iInput, dy_dx);
  }

  mlpdouble Getd2YdX2(std::size_t iNeuron, std::size_t iInput, std::size_t jInput) {
    return neurons[iNeuron].GetSecondGradient(iInput, jInput);
  }
  
  void Setd2YdX2(std::size_t iNeuron, std::size_t iInput, std::size_t jInput, mlpdouble d2y_dx2) {
    neurons[iNeuron].SetSecondGradient(iInput, jInput, d2y_dx2);
  }
  /*!
   * \brief Size neuron output derivative wrt network inputs.
   * \param[in] nInputs - Number of network inputs.
   */
  void SizeGradients(std::size_t nInputs) {
    for (auto iNeuron = 0u; iNeuron < number_of_neurons; iNeuron++)
      neurons[iNeuron].SizeGradient(nInputs);
  }

  /*!
   * \brief Get the activation function name applied to this layer
   * \return name of the activation function
   */
  std::string GetActivationType() const { return activation_type; }
};

} // namespace MLPToolbox
