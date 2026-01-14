/*!
* \file CNeuron.hpp
* \brief Declaration of the CNeuron class within the CLayer class.
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

#include "variable_def.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>
#include "option_maps.hpp"

namespace MLPToolbox {
class CNeuron {
  /*!
   *\class CNeuron
   *\brief This class functions as a neuron within the CLayer class, making up
   *the CNeuralNetwork class. The CNeuron class functions as a location to store
   *activation function inputs and outputs, as well as gradients and biases.
   *These are accessed through the CLayer class for network evalution
   *operations.
   */
private:
  unsigned long i_neuron; /*!< Neuron identification number */
  mlpdouble output{0},    /*!< Output value of the current neuron */
      input{0},           /*!< Input value of the current neuron */
      doutput_dinput{0},  /*!< Gradient of output with respect to input */
      bias{0};            /*!< Bias value at current neuron */
  std::vector<mlpdouble> doutput_dinputs;
  std::vector<std::vector<mlpdouble>> d2output_d2inputs;

public:
  /*!
   * \brief Set neuron identification number
   * \param[in] input - Identification number
   */
  void SetNumber(unsigned long input) { i_neuron = input; }

  /*!
   * \brief Get neuron identification number
   * \return Identification number
   */
  unsigned long GetNumber() const { return i_neuron; }

  /*!
   * \brief Set neuron output value
   * \param[in] input - activation function output value
   */
  void SetOutput(mlpdouble input) { output = input; }

  /*!
   * \brief Get neuron output value
   * \return Output value
   */
  mlpdouble GetOutput() const { return output; }

  /*!
   * \brief Set neuron input value
   * \param[in] input - activation function input value
   */
  void SetInput(mlpdouble x) { input = x; }

  /*!
   * \brief Get neuron input value
   * \return input value
   */
  mlpdouble GetInput() const { return input; }

  /*!
   * \brief Set neuron bias
   * \param[in] input - bias value
   */
  void SetBias(mlpdouble input) { bias = input; }

  /*!
   * \brief Get neuron bias value
   * \return bias value
   */
  mlpdouble GetBias() const { return bias; }

  /*!
   * \brief Size the derivative of the neuron output wrt MLP inputs.
   * \param[in] nInputs - Number of MLP inputs.
   */
  void SizeGradient(std::size_t nInputs) { 
    doutput_dinputs.resize(nInputs); 
    d2output_d2inputs.resize(nInputs);
    for (auto iInput=0u; iInput<nInputs; iInput++) {
      d2output_d2inputs[iInput].resize(nInputs);
    }
  }
  /*!
   * \brief Set neuron output gradient with respect to its input value
   * \param[in] input - Derivative of activation function with respect to input
   */
  void SetGradient(std::size_t iInput, mlpdouble input) {
    doutput_dinputs[iInput] = input;
  }

  void SetSecondGradient(std::size_t iInput, std::size_t jInput, mlpdouble input) {
    d2output_d2inputs[iInput][jInput] = input;
  }
  /*!
   * \brief Get neuron output gradient with respect to input value
   * \return output gradient wrt input value
   */
  mlpdouble GetGradient(std::size_t iInput) const {
    return doutput_dinputs[iInput];
  }

  mlpdouble GetSecondGradient(std::size_t iInput, std::size_t jInput) const {
    return d2output_d2inputs[iInput][jInput];
  }
};

} // namespace MLPToolbox
