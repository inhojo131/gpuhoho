/*!
* \file CLookUp_ANN.hpp
* \brief Declaration of the main MLP evaluation class.
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
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "CIOMap.hpp"
#include "CNeuralNetwork.hpp"
#include "CReadNeuralNetwork.hpp"
#include "variable_def.hpp"

namespace MLPToolbox {

class CLookUp_ANN {
  /*!
   *\class CLookUp_ANN
   *\brief This class allows for the evaluation of one or more multi-layer
   *perceptrons in for example thermodynamic state look-up operations. The
   *multi-layer perceptrons are loaded in the order listed in the MLP collection
   *file. Each multi-layer perceptron is generated based on the architecture
   *described in its respective input file. When evaluating the MLP collection,
   *an input-output map is used to find the correct MLP corresponding to the
   *call function inputs and outputs.
   */

private:
  std::vector<CNeuralNetwork> NeuralNetworks; /*!< std::std::vector containing
                                                 all loaded neural networks. */

  unsigned short number_of_variables; /*!< Number of loaded ANNs. */

  /*!
   * \brief Load ANN architecture
   * \param[in] ANN - pointer to target NeuralNetwork class
   * \param[in] filename - filename containing ANN architecture information
   */
  void GenerateANN(CNeuralNetwork &ANN, std::string filename) {
    /*--- Generate MLP architecture based on information in MLP input file ---*/

    /* Read MLP input file */
    CReadNeuralNetwork Reader = CReadNeuralNetwork(filename);

    /* Read MLP input file */
    Reader.ReadMLPFile();

    /* Set input and output regularization methods */
    ANN.SetInputRegularization(Reader.GetInputRegularization());
    ANN.SetOutputRegularization(Reader.GetOutputRegularization());

    /* Generate basic layer architectures */
    ANN.DefineInputLayer(Reader.GetNInputs());
    for (auto iInput = 0u; iInput < Reader.GetNInputs(); iInput++) {
      ANN.SetInputName(iInput, Reader.GetInputName(iInput));
    }
    for (auto iLayer = 1u; iLayer < Reader.GetNlayers() - 1; iLayer++) {
      ANN.PushHiddenLayer(Reader.GetNneurons(iLayer));
    }
    ANN.DefineOutputLayer(Reader.GetNOutputs());
    for (auto iOutput = 0u; iOutput < Reader.GetNOutputs(); iOutput++) {
      ANN.SetOutputName(iOutput, Reader.GetOutputName(iOutput));
    }

    /* Size weights of each layer */
    ANN.SizeWeights();

    /* Define weights and activation functions */
    ANN.SizeActivationFunctions(ANN.GetNWeightLayers() + 1);
    for (auto i_layer = 0u; i_layer < ANN.GetNWeightLayers(); i_layer++) {
      ANN.SetActivationFunction(i_layer, Reader.GetActivationFunction(i_layer));
      for (auto i_neuron = 0u; i_neuron < ANN.GetNNeurons(i_layer);
           i_neuron++) {
        for (auto j_neuron = 0u; j_neuron < ANN.GetNNeurons(i_layer + 1);
             j_neuron++) {
          ANN.SetWeight(i_layer, i_neuron, j_neuron,
                        Reader.GetWeight(i_layer, i_neuron, j_neuron));
        }
      }
    }
    ANN.SetActivationFunction(
        ANN.GetNWeightLayers(),
        Reader.GetActivationFunction(ANN.GetNWeightLayers()));

    /* Set neuron biases */
    for (auto i_layer = 0u; i_layer < ANN.GetNWeightLayers() + 1; i_layer++) {
      for (auto i_neuron = 0u; i_neuron < ANN.GetNNeurons(i_layer);
           i_neuron++) {
        ANN.SetBias(i_layer, i_neuron, Reader.GetBias(i_layer, i_neuron));
      }
    }

    /* Define input and output layer normalization values */
    for (auto iInput = 0u; iInput < Reader.GetNInputs(); iInput++) {
      ANN.SetInputNorm(iInput, Reader.GetInputNorm(iInput).first,
                       Reader.GetInputNorm(iInput).second);
    }
    for (auto iOutput = 0u; iOutput < Reader.GetNOutputs(); iOutput++) {
      ANN.SetOutputNorm(iOutput, Reader.GetOutputNorm(iOutput).first,
                        Reader.GetOutputNorm(iOutput).second);
    }
  }

public:
  /*!
   * \brief ANN collection class constructor
   * \param[in] n_inputs - Number of MLP files to be loaded.
   * \param[in] input_filenames - String array containing MLP input file names.
   */
  CLookUp_ANN(const unsigned short n_inputs,
              const std::string *input_filenames) {
    /*--- Define collection of MLPs for regression purposes ---*/
    number_of_variables = n_inputs;

    NeuralNetworks.resize(n_inputs);

    /*--- Generate an MLP for every filename provided ---*/
    for (auto i_MLP = 0u; i_MLP < n_inputs; i_MLP++) {
      GenerateANN(NeuralNetworks[i_MLP], input_filenames[i_MLP]);
    }
  }

  /*!
   * \brief Get average input variable bounds of the loaded MLPs for a specific
   * look-up operation. 
   * \param[in] input_output_map - Pointer to input-output
   * map for look-up operation. 
   * \param[in] input_index - Input variable index
   * for which to get the bounds.
   */
  std::pair<mlpdouble, mlpdouble>
  GetInputNorm(MLPToolbox::CIOMap *input_output_map,
               std::size_t input_index) const {
    mlpdouble CV_min{0.0}, CV_max{0.0};

    for (auto i_map = 0u; i_map < input_output_map->GetNMLPs(); i_map++) {
      auto i_ANN = input_output_map->GetMLPIndex(i_map);
      auto i_input = input_output_map->GetInputIndex(i_map, input_index);
      std::pair<mlpdouble, mlpdouble> ANN_input_limits =
          NeuralNetworks[i_ANN].GetInputNorm(i_input);
      CV_min += ANN_input_limits.first;
      CV_max += ANN_input_limits.second;
    }

    CV_min /= input_output_map->GetNMLPs();
    CV_max /= input_output_map->GetNMLPs();
    return std::make_pair(CV_min, CV_max);
  }

  
  /*!
   * \brief Get the median input regularization value for a specific look-up operation.
   * \param[in] input_output_map - Pointer to input-output
   * map for look-up operation. 
   * \param[in] input_index - Input variable index
   * for which to get the input median.
   * \returns Input median value.
   */
  mlpdouble GetInputOffset(MLPToolbox::CIOMap *input_output_map,
               std::size_t input_index) const {
    mlpdouble CV_center{0.0};

    for (auto i_map = 0u; i_map < input_output_map->GetNMLPs(); i_map++) {
      auto i_ANN = input_output_map->GetMLPIndex(i_map);
      auto i_input = input_output_map->GetInputIndex(i_map, input_index);
      mlpdouble center = NeuralNetworks[i_ANN].GetRegularizationOffset(i_input, true);
      CV_center += center;
    }

    CV_center /= input_output_map->GetNMLPs();
    return CV_center;
  }

  /*!
   * \brief Evaluate loaded ANNs for given inputs and outputs
   * \param[in] input_output_map - input-output map coupling desired inputs and
   * outputs to loaded ANNs.
   * \param[in] inputs - input values.
   * \param[in] outputs - pointers to output variables.
   * \param[in] doutputs_dinputs - pointers to output derivatives w.r.t. inputs.
   * \param[in] d2outputs_dinputs2 - pointers to output second order derivatives
   * w.r.t. inputs. \returns Within output normalization range.
   */
  unsigned long PredictANN(
      MLPToolbox::CIOMap *input_output_map, const std::vector<mlpdouble> &inputs,
      std::vector<mlpdouble *> &outputs,
      const std::vector<std::vector<mlpdouble *>> *doutputs_dinputs = nullptr,
      std::vector<std::vector<std::vector<mlpdouble *>>> *d2outputs_dinputs2 =
          nullptr) {
    /*--- Evaluate MLP based on target input and output variables ---*/
    bool within_range, // Within MLP training set range.
        MLP_was_evaluated =
            false; // MLP was evaluated within training set range.

    bool compute_firstorder_gradient = (doutputs_dinputs != nullptr),
         compute_secondorder_gradient = (d2outputs_dinputs2 != nullptr);

    /* If queries lie outside the training data set, the nearest MLP will be
     * evaluated through extrapolation. */
    mlpdouble distance_to_query = 1e20; // Overall smallest distance between
                                        // training data set middle and query.
    size_t i_ANN_nearest = 0,           // Index of nearest MLP.
        i_map_nearest = 0;              // Index of nearest iomap index.

    for (auto i_map = 0u; i_map < input_output_map->GetNMLPs(); i_map++) {
      within_range = true;
      auto i_ANN = input_output_map->GetMLPIndex(i_map);
      NeuralNetworks[i_ANN].ComputeFirstOrderGradient(
          compute_firstorder_gradient);
      NeuralNetworks[i_ANN].ComputeSecondOrderGradient(
          compute_secondorder_gradient);
      auto ANN_inputs = input_output_map->GetMLPInputs(i_map, inputs);

      mlpdouble distance_to_query_i = 0;
      for (auto i_input = 0u; i_input < ANN_inputs.size(); i_input++) {
        within_range = NeuralNetworks[i_ANN].CheckInputInclusion(ANN_inputs[i_input], i_input);
        

        /* Calculate distance between MLP training range center point and query
         */
        mlpdouble middle = NeuralNetworks[i_ANN].GetRegularizationOffset(i_input);
        distance_to_query_i +=
            pow(NeuralNetworks[i_ANN].NormalizeInput(ANN_inputs[i_input] - middle, i_input), 2);
      }

      /* Evaluate MLP when query inputs lie within training data range */
      if (within_range) {
        NeuralNetworks[i_ANN].Predict(ANN_inputs);
        MLP_was_evaluated = true;
        for (auto i = 0u; i < input_output_map->GetNMappedOutputs(i_map); i++) {
          *outputs[input_output_map->GetOutputIndex(i_map, i)] =
              NeuralNetworks[i_ANN].GetANNOutput(
                  input_output_map->GetMLPOutputIndex(i_map, i));
          if (compute_firstorder_gradient) {
            for (auto iInput = 0u; iInput < inputs.size(); iInput++) {
              *(doutputs_dinputs->at(input_output_map->GetOutputIndex(i_map, i))
                    .at(iInput)) =
                  NeuralNetworks[i_ANN].GetdOutputdInput(
                      input_output_map->GetMLPOutputIndex(i_map, i),
                      input_output_map->GetInputIndex(i_map, iInput));

              if (compute_secondorder_gradient) {
                for (auto jInput = 0u; jInput < inputs.size(); jInput++) {
                  *(d2outputs_dinputs2
                        ->at(input_output_map->GetOutputIndex(i_map, i))
                        .at(iInput)
                        .at(jInput)) =
                      NeuralNetworks[i_ANN].Getd2OutputdInput2(
                          input_output_map->GetMLPOutputIndex(i_map, i),
                          input_output_map->GetInputIndex(i_map, iInput),
                          input_output_map->GetInputIndex(i_map, jInput));
                }
              }
            }
          }
        }
      }

      /* Update minimum distance to query */
      if (distance_to_query_i < distance_to_query) {
        i_ANN_nearest = i_ANN;
        distance_to_query = distance_to_query_i;
        i_map_nearest = i_map;
      }
    }

    /* Evaluate nearest MLP in case no query data within range is found */
    if (!MLP_was_evaluated) {
      auto ANN_inputs = input_output_map->GetMLPInputs(i_map_nearest, inputs);
      NeuralNetworks[i_ANN_nearest].Predict(ANN_inputs);
      for (auto i = 0u; i < input_output_map->GetNMappedOutputs(i_map_nearest);
           i++) {
        *outputs[input_output_map->GetOutputIndex(i_map_nearest, i)] =
            NeuralNetworks[i_ANN_nearest].GetANNOutput(
                input_output_map->GetMLPOutputIndex(i_map_nearest, i));
      }
    }

    /* Return 1 if query data lies outside the range of any of the loaded MLPs
     */
    return MLP_was_evaluated ? 0 : 1;
  }

  /*!
   * \brief Pair inputs and outputs with look-up operations.
   * \param[in] ioMap - input-output map to pair variables with.
   */
  void PairVariableswithMLPs(MLPToolbox::CIOMap &ioMap) {
    /*
    In this function, the call inputs and outputs are matched to those within
    the MLP collection.
    */
    bool isInput, isOutput;

    auto inputVariables = ioMap.GetInputVars();
    auto outputVariables = ioMap.GetOutputVars();
    // Looping over the loaded MLPs to check wether the MLP inputs match with
    // the call inputs
    for (size_t iMLP = 0; iMLP < NeuralNetworks.size(); iMLP++) {
      // Mapped call inputs to MLP inputs
      std::vector<std::pair<size_t, size_t>> Input_Indices =
          FindVariableIndices(iMLP, inputVariables, true);
      isInput = Input_Indices.size() > 0;

      if (isInput) {
        // Only when the MLP inputs match with a portion of the call inputs are
        // the output variable checks performed

        std::vector<std::pair<size_t, size_t>> Output_Indices =
            FindVariableIndices(iMLP, outputVariables, false);
        isOutput = Output_Indices.size() > 0;

        if (isOutput) {
          // Update input and output mapping if both inputs and outputs match
          ioMap.PushMLPIndex(iMLP);
          ioMap.PushInputIndices(Input_Indices);
          ioMap.PushOutputIndices(Output_Indices);
        }
      }
    }

    CheckUseOfInputs(ioMap);
    CheckUseOfOutputs(ioMap);
  }

  /*!
   * \brief Get number of loaded ANNs
   * \return number of loaded ANNs
   */
  std::size_t GetNANNs() const { return NeuralNetworks.size(); }

  /*!
   * \brief Check if all output variables are present in the loaded ANNs
   * \param[in] output_names - output variable names to check
   * \param[in] input_output_map - pointer to input-output map to be checked
   */
  bool CheckUseOfOutputs(MLPToolbox::CIOMap &input_output_map) const {
    /*--- Check wether all output variables are in the loaded MLPs ---*/
    auto output_names = input_output_map.GetOutputVars();
    std::vector<std::string> missing_outputs;
    bool outputs_are_present{true};
    /* Looping over the target outputs */
    for (auto iOutput = 0u; iOutput < output_names.size(); iOutput++) {
      bool found_output{false};

      /* Looping over all the selected ANNs */
      for (auto i_map = 0u; i_map < input_output_map.GetNMLPs(); i_map++) {
        auto output_map = input_output_map.GetOutputMapping(i_map);

        /* Looping over the outputs of the output map of the current ANN */
        for (auto jOutput = 0u; jOutput < output_map.size(); jOutput++) {
          if (output_map[jOutput].first == iOutput)
            found_output = true;
        }
      }
      if (!found_output) {
        missing_outputs.push_back(output_names[iOutput]);
        outputs_are_present = false;
      };
    }
    /*--- Raise error if any outputs are missing ---*/
    if (missing_outputs.size() > 0) {
      std::string message{"Outputs "};
      for (size_t iVar = 0; iVar < missing_outputs.size(); iVar++)
        message += missing_outputs[iVar] + " ";
      throw std::invalid_argument(message +
                                  "are not present in any loaded ANN.");
    }
    return outputs_are_present;
  }

  /*!
   * \brief Check if all input variables are present in the loaded ANNs
   * \param[in] input_names - input variable names to check
   * \param[in] input_output_map - pointer to input-output map to be checked
   */
  bool CheckUseOfInputs(MLPToolbox::CIOMap &input_output_map) const {
    /*--- Check wether all input variables are in the loaded MLPs ---*/
    auto input_names = input_output_map.GetInputVars();
    std::vector<std::string> missing_inputs;
    bool inputs_are_present{true};
    for (auto iInput = 0u; iInput < input_names.size(); iInput++) {
      bool found_input = false;
      for (auto i_map = 0u; i_map < input_output_map.GetNMLPs(); i_map++) {
        auto input_map = input_output_map.GetInputMapping(i_map);
        for (auto jInput = 0u; jInput < input_map.size(); jInput++) {
          if (input_map[jInput].first == iInput) {
            found_input = true;
          }
        }
      }
      if (!found_input) {
        missing_inputs.push_back(input_names[iInput]);
        inputs_are_present = false;
      };
    }
    /*--- Raise error if input variables are missing ---*/
    if (missing_inputs.size() > 0) {
      std::string message{"Inputs "};
      for (size_t iVar = 0; iVar < missing_inputs.size(); iVar++)
        message += missing_inputs[iVar] + " ";
      throw std::invalid_argument(message +
                                  "are not present in any loaded ANN.");
    }
    return inputs_are_present;
  }

  /*!
   * \brief Map variable names to ANN inputs or outputs
   * \param[in] i_ANN - loaded ANN index
   * \param[in] variable_names - variable names to map to ANN inputs or outputs
   * \param[in] input - map to inputs (true) or outputs (false)
   */
  std::vector<std::pair<std::size_t, std::size_t>>
  FindVariableIndices(std::size_t i_ANN,
                      std::vector<std::string> variable_names,
                      bool input) const {
    /*--- Find loaded MLPs that have the same input variable names as the
     * variables listed in variable_names ---*/

    std::vector<std::pair<size_t, size_t>> variable_indices;
    auto nVar = input ? NeuralNetworks[i_ANN].GetnInputs()
                      : NeuralNetworks[i_ANN].GetnOutputs();

    for (auto iVar = 0u; iVar < nVar; iVar++) {
      for (auto jVar = 0u; jVar < variable_names.size(); jVar++) {
        std::string ANN_varname =
            input ? NeuralNetworks[i_ANN].GetInputName(iVar)
                  : NeuralNetworks[i_ANN].GetOutputName(iVar);
        if (variable_names[jVar].compare(ANN_varname) == 0) {
          variable_indices.push_back(std::make_pair(jVar, iVar));
        }
      }
    }
    return variable_indices;
  }

  /*!
   * \brief Display architectural information on the loaded MLPs
   */
  void DisplayNetworkInfo() const {
    /*--- Display network information on the loaded MLPs ---*/

    std::cout << std::setfill(' ');
    std::cout << std::endl;
    std::cout << "+------------------------------------------------------------"
                 "------+"
                 "\n";
    std::cout
        << "|                 Multi-Layer Perceptron (MLP) info                "
           "|\n";
    std::cout << "+------------------------------------------------------------"
                 "------+"
              << std::endl;

    /* For every loaded MLP, display the inputs, outputs, activation functions,
     * and architecture. */
    for (auto i_MLP = 0u; i_MLP < NeuralNetworks.size(); i_MLP++) {
      NeuralNetworks[i_MLP].DisplayNetwork();
    }
  }
};

} // namespace MLPToolbox
