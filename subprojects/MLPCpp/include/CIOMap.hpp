/*!
* \file CIOMap.hpp
* \brief Input-output map class definition for the definition of MLP look-up
operations.
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

#include "variable_def.hpp"
#include <string>
#include <vector>
namespace MLPToolbox {
class CIOMap
/*!
 *\class CIOMap
 *\brief This class is used by the CLookUp_ANN class to assign user-defined
 *inputs and outputs to loaded multi-layer perceptrons. When a look-up operation
 *is called with a specific CIOMap, the multi-layer perceptrons are evaluated
 *with input and output variables coinciding with the desired input and output
 *variable names.
 *
 *
 * For example, in a custom, data-driven fluid model, MLP's are used for
 *thermodynamic state definition. There are three MLP's loaded. MLP_1 predicts
 *temperature and specific heat based on density and energy. MLP_2 predicts
 *pressure and speed of sound based on density and energy as well. MLP_3
 *predicts density and energy based on pressure and temperature. During a
 *certain look-up operation in the CFluidModel, temperature, speed of sound and
 *pressure are needed for a given density and energy. What the CIOMap does is to
 *point to MLP_1 for temperature evalutation, and to MLP_2 for pressure and
 *speed of sound evaluation. MLP_3 is not considered, as the respective inputs
 *and outputs don't match with the function call inputs and outputs.
 *
 *  call variables:      MLP inputs:                     MLP outputs: call
 *outputs:
 *
 *                        2--> energy --|            |--> temperature --> 1
 *                                      |--> MLP_1 --|
 *  1:density            1--> density --|            |--> c_p 1:temperature
 *  2:energy 2:speed of sound 1--> density --|            |--> pressure --> 3
 *3:pressure
 *                                      |--> MLP_2 --|
 *                        2--> energy --|            |--> speed of sound --> 2
 *
 *                           pressure --|            |--> density
 *                                      |--> MLP_3 --|
 *                        temperature --|            |--> energy
 *
 *
 * \author E.Bunschoten
 */
{
private:
  std::vector<std::string> inputVariables, /*!< Input variable names for the
                                              current input-output map. */
      outputVariables; /*!< Output variable names for the current input-output
                          map. */

  std::vector<std::size_t> MLP_indices; /*!< Loaded MLP index */
  std::vector<std::vector<std::pair<std::size_t, std::size_t>>>
      Input_Map,  /*!< Mapping of call variable inputs to matching MLP inputs */
      Output_Map; /*!< Mapping of call variable outputs to matching MLP outputs
                   */
public:
  /*!
   * \brief Initiate input-output map with user-defined input and output
   * variables. \param[in] inputVariables_in - Vector containing input variable
   * names. \param[in] outputVariables_in - Vector containing output variable
   * names.
   */
  CIOMap(std::vector<std::string> &inputVariables_in,
         std::vector<std::string> &outputVariables_in) {
    inputVariables.resize(inputVariables_in.size());
    for (auto iVar = 0u; iVar < inputVariables_in.size(); iVar++) {
      inputVariables[iVar] = inputVariables_in[iVar];
    }
    outputVariables.resize(outputVariables_in.size());
    for (auto iVar = 0u; iVar < outputVariables_in.size(); iVar++) {
      outputVariables[iVar] = outputVariables_in[iVar];
    }
  }

  /*!
   * \brief Insert MLP index with stored input and output variables.
   * \param[in] iMLP - Loaded MLP index.
   */
  void PushMLPIndex(std::size_t iMLP) { MLP_indices.push_back(iMLP); }

  /*!
   * \brief Insert MLP input translation vector.
   * \param[in] inputIndices - Vector containing input variable index
   * translation.
   */
  void PushInputIndices(std::vector<std::pair<size_t, size_t>> inputIndices) {
    Input_Map.push_back(inputIndices);
  }

  /*!
   * \brief Insert MLP output translation vector.
   * \param[in] outputIndices - Vector containing output variable index
   * translation.
   */
  void PushOutputIndices(std::vector<std::pair<size_t, size_t>> outputIndices) {
    Output_Map.push_back(outputIndices);
  }

  /*!
   * \brief Get input variables of current input-output map.
   * \return Vector of input variables.
   */
  std::vector<std::string> GetInputVars() { return inputVariables; }

  /*!
   * \brief Get output variables of current input-output map.
   * \return Vector of output variables.
   */
  std::vector<std::string> GetOutputVars() { return outputVariables; }

  /*!
   * \brief Get the number of MLPs in the current IO map
   * \return number of MLPs with matching inputs and output(s)
   */
  std::size_t GetNMLPs() const { return MLP_indices.size(); }

  /*!
   * \brief Get the loaded MLP index
   * \return MLP index
   */
  std::size_t GetMLPIndex(std::size_t i_Map) const {
    return MLP_indices[i_Map];
  }

  /*!
   * \brief Get the call input variable index
   * \param[in] i_Map - input-output mapping index of the IO map
   * \param[in] iInput - input index of the call input variable
   * \return MLP input variable index
   */
  std::size_t GetInputIndex(std::size_t i_Map, std::size_t iInput) const {
    return Input_Map[i_Map][iInput].first;
  }

  /*!
   * \brief Get the call output variable index
   * \param[in] i_Map - input-output mapping index of the IO map
   * \param[in] iOutput - output index of the call input variable
   * \return call variable output index
   */
  std::size_t GetOutputIndex(std::size_t i_Map, std::size_t iOutput) const {
    return Output_Map[i_Map][iOutput].first;
  }

  /*!
   * \brief Get the MLP output variable index
   * \param[in] i_Map - input-output mapping index of the IO map
   * \param[in] iOutput - output index of the call input variable
   * \return MLP output variable index
   */
  std::size_t GetMLPOutputIndex(std::size_t i_Map, std::size_t iOutput) const {
    return Output_Map[i_Map][iOutput].second;
  }

  /*!
   * \brief Get the number of matching output variables between the call and MLP
   * outputs \param[in] i_Map - input-output mapping index of the IO map \return
   * Number of matching variables between the loaded MLP and call variables
   */
  std::size_t GetNMappedOutputs(std::size_t i_Map) const {
    return Output_Map[i_Map].size();
  }

  /*!
   * \brief Get the mapping of MLP outputs matching to call outputs
   * \param[in] i_Map - input-output mapping index of the IO map
   * \return Mapping of MLP output variables to call variables
   */
  std::vector<std::pair<std::size_t, std::size_t>>
  GetOutputMapping(std::size_t i_map) const {
    return Output_Map[i_map];
  }

  /*!
   * \brief Get the mapping of MLP inputs to call inputs
   * \param[in] i_Map - input-output mapping index of the IO map
   * \return Mapping of MLP input variables to call inputs
   */
  std::vector<std::pair<std::size_t, std::size_t>>
  GetInputMapping(std::size_t i_map) const {
    return Input_Map[i_map];
  }

  /*!
   * \brief Get the mapped inputs for the MLP at i_Map
   * \param[in] i_Map - input-output mapping index of the IO map
   * \param[in] inputs - call inputs
   * \return std::vector with call inputs in the correct order of the loaded MLP
   */
  std::vector<mlpdouble> GetMLPInputs(std::size_t i_Map,
                                      const std::vector<mlpdouble> &inputs) const {
    std::vector<mlpdouble> MLP_input;
    MLP_input.resize(Input_Map[i_Map].size());

    for (std::size_t iInput = 0; iInput < Input_Map[i_Map].size(); iInput++) {
      MLP_input[iInput] = inputs[GetInputIndex(i_Map, iInput)];
    }
    return MLP_input;
  }
};
} // namespace MLPToolbox