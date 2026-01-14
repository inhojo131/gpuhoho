/*!
* \file CReadNeuralNetwork.hpp
* \brief Read MLP input files.
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
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

namespace MLPToolbox {

class CReadNeuralNetwork {
private:
  std::vector<std::string> input_names, /*!< Input variable names. */
      output_names;                     /*!< Output variable names. */

  std::string filename; /*!< MLP input filename. */

  unsigned long n_layers; /*!< Network total layer count. */

  std::vector<unsigned long> n_neurons; /*!<  Neuron count per layer. */

  std::vector<std::vector<std::vector<mlpdouble>>>
      weights_mat; /*!< Network synapse weights. */

  std::vector<std::vector<mlpdouble>>
      biases_mat; /*!< Bias values per neuron. */

  std::vector<std::string>
      activation_functions; /*!< Activation function per layer. */

  std::vector<std::pair<mlpdouble, mlpdouble>>
      input_norm,  /*!< Input variable normalization values (min, max). */
      output_norm; /*!< Output variable normalization values (min, max). */
  
  /*!
  * \brief Available activation function map.
  */
  std::map<std::string, ENUM_SCALING_FUNCTIONS> scaling_map{
      {"minmax", ENUM_SCALING_FUNCTIONS::MINMAX},
      {"standard", ENUM_SCALING_FUNCTIONS::STANDARD},
      {"robust", ENUM_SCALING_FUNCTIONS::ROBUST},
  };

  ENUM_SCALING_FUNCTIONS input_reg_method {ENUM_SCALING_FUNCTIONS::MINMAX},
                         output_reg_method {ENUM_SCALING_FUNCTIONS::MINMAX};
public:
  /*!
   * \brief CReadNeuralNetwork class constructor
   * \param[in] filename_in - .mlp input file name containing network
   * information.
   */
  CReadNeuralNetwork(std::string filename_in) { filename = filename_in; }

  /*!
   * \brief Read input file and store necessary information
   */
  void ReadMLPFile() {
    std::ifstream file_stream;
    file_stream.open(filename.c_str(), std::ifstream::in);
    if (!file_stream.is_open()) {
      throw std::invalid_argument("There is no MLP file called " + filename);
    }

    std::string line, word;
    bool eoHeader = false, found_layercount = false, found_input_names = false,
         found_output_names = false;

    /* Read general architecture information from file header */

    line = SkipToFlag(&file_stream, "<header>");

    while (getline(file_stream, line) && !eoHeader) {
      /* Read layer count */
      if (line.compare("[number of layers]") == 0) {
        getline(file_stream, line);
        n_layers = stoul(line);
        n_neurons.resize(n_layers);
        biases_mat.resize(n_layers);
        weights_mat.resize(n_layers - 1);
        activation_functions.resize(n_layers);

        found_layercount = true;
      }

      /* Set number of neurons for each layer */
      if (line.compare("[neurons per layer]") == 0) {
        /* In case layer count was not yet provided, return an error */
        if (!found_layercount) {
          throw std::invalid_argument(
              "No layer count provided before defining neuron count per layer");
        }
        /* Loop over layer count and size neuron count and bias count per layer
         * accordingly */
        for (auto iLayer = 0u; iLayer < n_layers; iLayer++) {
          getline(file_stream, line);
          n_neurons[iLayer] = stoul(line);
          biases_mat[iLayer].resize(n_neurons[iLayer]);
        }
        /* Loop over spaces between layers and size the weight matrices
         * accordingly */
        for (auto iLayer = 0u; iLayer < n_layers - 1; iLayer++) {
          weights_mat[iLayer].resize(n_neurons[iLayer]);
          for (auto iNeuron = 0u; iNeuron < n_neurons[iLayer]; iNeuron++)
            weights_mat[iLayer][iNeuron].resize(n_neurons[iLayer + 1]);
        }
        /* Size input and output normalization and set default values */
        input_norm.resize(n_neurons[0]);
        for (auto iNeuron = 0u; iNeuron < n_neurons[0]; iNeuron++)
          input_norm[iNeuron] = std::make_pair(0, 1);

        output_norm.resize(n_neurons[n_neurons.size() - 1]);
        for (auto iNeuron = 0u; iNeuron < n_neurons[n_neurons.size() - 1];
             iNeuron++)
          output_norm[iNeuron] = std::make_pair(0, 1);
      }

      /* Read layer activation function types */
      if (line.compare("[activation function]") == 0) {
        if (!found_layercount) {
          throw std::invalid_argument(
              "No layer count provided before providing "
              "layer activation functions");
        }
        for (auto iLayer = 0u; iLayer < n_layers; iLayer++) {
          getline(file_stream, line);
          std::istringstream activation_stream(line);
          activation_stream >> word;
          activation_functions[iLayer] = word;
        }
      }

      /* Read MLP input variable names */
      if (line.compare("[input names]") == 0) {
        found_input_names = true;
        input_names.resize(n_neurons[0]);
        for (auto iInput = 0u; iInput < n_neurons[0]; iInput++) {
          getline(file_stream, line);
          input_names[iInput] = line;
        }
      }

      if (line.compare("[input regularization method]") == 0) {
        getline(file_stream, line);
        std::istringstream input_norm_stream(line);
        input_norm_stream >> word;
        input_reg_method = scaling_map.find(word)->second;
      }


      /* In case input normalization is applied, read upper and lower input
       * bounds
       */
      if (line.compare("[input normalization]") == 0) {
        for (auto iInput = 0u; iInput < input_norm.size(); iInput++) {
          getline(file_stream, line);
          if (line.compare("") != 0) {
            std::istringstream input_norm_stream(line);
            input_norm_stream >> word;
            mlpdouble input_min = stold(word);
            input_norm_stream >> word;
            mlpdouble input_max = stold(word);
            input_norm[iInput] = std::make_pair(input_min, input_max);
          }
        }
      }

      /* Read MLP output variable names */
      if (line.compare("[output names]") == 0) {
        found_output_names = true;
        auto n_outputs = n_neurons[n_neurons.size() - 1];
        output_names.resize(n_outputs);
        for (auto iOutput = 0u; iOutput < n_outputs; iOutput++) {
          getline(file_stream, line);
          output_names[iOutput] = line;
        }

        if (output_names.size() != (n_neurons[n_neurons.size() - 1])) {
          throw std::invalid_argument(
              "No layer count provided before providing "
              "layer activation functions");
        }
      }

      if (line.compare("[output regularization method]") == 0) {
        getline(file_stream, line);
        std::istringstream input_norm_stream(line);
        input_norm_stream >> word;
        output_reg_method = scaling_map.find(word)->second;
      }

      /* In case output normalization is applied, read upper and lower output
       * bounds */
      if (line.compare("[output normalization]") == 0) {
        for (auto iOutput = 0u; iOutput < output_norm.size(); iOutput++) {
          getline(file_stream, line);
          if (line.compare("") != 0) {
            std::istringstream output_norm_stream(line);
            output_norm_stream >> word;
            mlpdouble output_min = stold(word);
            output_norm_stream >> word;
            mlpdouble output_max = stold(word);
            output_norm[iOutput] = std::make_pair(output_min, output_max);
          }
        }
      }

      if (line.compare("</header>") == 0) {
        eoHeader = true;
      }
    } // eoHeader

    /* Error checking */
    if (!found_input_names) {
      throw std::invalid_argument("No MLP input variable names provided");
    }
    if (!found_output_names) {
      throw std::invalid_argument("No MLP input variable names provided");
    }

    /* Read weights for each layer */
    line = SkipToFlag(&file_stream, "[weights per layer]");
    for (auto iLayer = 0u; iLayer < n_layers - 1; iLayer++) {
      getline(file_stream, line);
      for (auto iNeuron = 0u; iNeuron < n_neurons[iLayer]; iNeuron++) {
        getline(file_stream, line);
        std::istringstream weight_stream(line);
        for (auto jNeuron = 0u; jNeuron < n_neurons[iLayer + 1]; jNeuron++) {
          weight_stream >> word;
          weights_mat[iLayer][iNeuron][jNeuron] = stold(word);
        }
      }
      getline(file_stream, line);
    }

    /* Read biases for each neuron */
    line = SkipToFlag(&file_stream, "[biases per layer]");
    for (auto iLayer = 0u; iLayer < n_layers; iLayer++) {
      getline(file_stream, line);
      std::istringstream bias_stream(line);
      for (auto iNeuron = 0u; iNeuron < n_neurons[iLayer]; iNeuron++) {
        bias_stream >> word;
        biases_mat[iLayer][iNeuron] = stold(word);
      }
    }
  }

  /*!
   * \brief Go to a specific line in file.
   * \param[in] file_stream - input file stream.
   * \param[in] flag - line to be skipped to.
   * \returns flag line
   */
  std::string SkipToFlag(std::ifstream *file_stream, std::string flag) {
    /*--- Search file for a line and set it as the current line in the file
     * stream
     * ---*/
    std::string line;
    getline(*file_stream, line);

    while (line.compare(flag) != 0 && !(*file_stream).eof()) {
      getline(*file_stream, line);
    }

    if ((*file_stream).eof())
      std::cout << "line not in file!" << std::endl;

    return line;
  }

  /*!
   * \brief Get number of read input variables.
   * \returns Number of neurons in the input layer.
   */
  unsigned long GetNInputs() const { return n_neurons[0]; }

  /*!
   * \brief Get number of read output variables.
   * \returns Number of neurons in the output layer.
   */
  unsigned long GetNOutputs() const { return n_neurons[n_layers - 1]; }

  /*!
   * \brief Get total number of layers in the network.
   * \returns Total layer count.
   */
  unsigned long GetNlayers() const { return n_layers; }

  /*!
   * \brief Get neuron count of a specific layer.
   * \param[in] iLayer - Total layer index.
   * \returns Number of neurons in the layer.
   */
  unsigned long GetNneurons(std::size_t iLayer) const {
    return n_neurons[iLayer];
  }

  /*!
   * \brief Get synapse weight between two neurons in subsequent layers.
   * \param[in] iLayer - Total layer index.
   * \param[in] iNeuron - Neuron index in layer with index iLayer.
   * \param[in] jNeuron - Neuron index in subsequent layer.
   * \returns Weight value
   */
  mlpdouble GetWeight(std::size_t iLayer, std::size_t iNeuron,
                      std::size_t jNeuron) const {
    return weights_mat[iLayer][iNeuron][jNeuron];
  }

  /*!
   * \brief Get bias value of specific neuron.
   * \param[in] iLayer - Total layer index.
   * \param[in] iNeuron - Neuron index.
   * \returns Bias value
   */
  mlpdouble GetBias(std::size_t iLayer, std::size_t iNeuron) const {
    return biases_mat[iLayer][iNeuron];
  }

  /*!
   * \brief Get input variable normalization values.
   * \param[in] iInput - Input variable index.
   * \returns Input normalization values (min first, max second)
   */
  std::pair<mlpdouble, mlpdouble> GetInputNorm(std::size_t iInput) const {
    return input_norm[iInput];
  }

  /*!
   * \brief Get output variable normalization values.
   * \param[in] iOutput - Input variable index.
   * \returns Output normalization values (min first, max second)
   */
  std::pair<mlpdouble, mlpdouble> GetOutputNorm(std::size_t iOutput) const {
    return output_norm[iOutput];
  }

  /*!
   * \brief Get layer activation function type.
   * \param[in] iLayer - Total layer index.
   * \returns Layer activation function type.
   */
  std::string GetActivationFunction(std::size_t iLayer) const {
    return activation_functions[iLayer];
  }

  /*!
   * \brief Get input variable name.
   * \param[in] iInput - Input variable index.
   * \returns Input variable name.
   */
  std::string GetInputName(std::size_t iInput) const {
    return input_names[iInput];
  }

  ENUM_SCALING_FUNCTIONS GetInputRegularization() const {
    return input_reg_method;
  }

  /*!
   * \brief Get output variable name.
   * \param[in] iOutput - Output variable index.
   * \returns Output variable name.
   */
  std::string GetOutputName(std::size_t iOutput) const {
    return output_names[iOutput];
  }

  ENUM_SCALING_FUNCTIONS GetOutputRegularization() const {
    return output_reg_method;
  }
};
} // namespace MLPToolbox
