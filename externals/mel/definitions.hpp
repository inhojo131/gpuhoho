// Copyright 2021-2023, Pedro Gomes
//
// This file is part of MEL.
//
// MEL is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// MEL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with MEL.  If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <string>

namespace mel {

using str_t = std::string;

#ifndef MEL_ONLY_ARITHMETIC_OPS
// Define defaults if not provided.
#ifndef MEL_SUPPORTED_FUNCTIONS
/// Text representation of functions.
#define MEL_SUPPORTED_FUNCTIONS                                 \
  "sqrt", "cbrt", "pow", "hypot", "log", "exp", "fabs", "fmax", \
  "fmin", "cos", "sin", "tan", "acos", "asin", "atan", "atan2"
/// Enum representation of function (same order, must end with ",").
#define MEL_FUNCTION_CODES SQRT, CBRT, POW, HYPOT, LOG, EXP,  \
  FABS, FMAX, FMIN, COS, SIN, TAN, ACOS, ASIN, ATAN, ATAN2,
/// Handling of enum cases (call a function for each code).
#define MEL_FUNCTION_IMPLEMENTATIONS(LEFT, RIGHT)       \
  case OpCode::SQRT: v[i] = sqrt(LEFT); break;          \
  case OpCode::CBRT: v[i] = cbrt(LEFT); break;          \
  case OpCode::POW: v[i] = pow(LEFT, RIGHT); break;     \
  case OpCode::HYPOT: v[i] = hypot(LEFT, RIGHT); break; \
  case OpCode::LOG: v[i] = log(LEFT); break;            \
  case OpCode::EXP: v[i] = exp(LEFT); break;            \
  case OpCode::FABS: v[i] = fabs(LEFT); break;          \
  case OpCode::FMAX: v[i] = fmax(LEFT, RIGHT); break;   \
  case OpCode::FMIN: v[i] = fmin(LEFT, RIGHT); break;   \
  case OpCode::COS: v[i] = cos(LEFT); break;            \
  case OpCode::SIN: v[i] = sin(LEFT); break;            \
  case OpCode::TAN: v[i] = tan(LEFT); break;            \
  case OpCode::ACOS: v[i] = acos(LEFT); break;          \
  case OpCode::ASIN: v[i] = asin(LEFT); break;          \
  case OpCode::ATAN: v[i] = atan(LEFT); break;          \
  case OpCode::ATAN2: v[i] = atan2(LEFT, RIGHT); break;
#endif
#else
// Only simple operations, no math functions.
#define MEL_SUPPORTED_FUNCTIONS
#define MEL_FUNCTION_CODES
#define MEL_FUNCTION_IMPLEMENTATIONS
#endif

/// List of supported operations.
static const str_t supported_operations[] = {
  "+", "-", "*", "/", MEL_SUPPORTED_FUNCTIONS
};

namespace internal {

#ifndef MEL_MAX_TREE_SIZE
#define MEL_MAX_TREE_SIZE 255
#endif
/// Static size of evaluation trees, with the default value and scalars of
/// type double, each tree occupies ~4KB.
static constexpr int max_tree_size = MEL_MAX_TREE_SIZE;

/// Efficient representation of the supported operations and expression node
/// types, must match the order of "supported_operations".
enum class OpCode {
  ADD, SUB, MUL, DIV, MEL_FUNCTION_CODES
  NOOP,   // special value, keep if here.
  SYMBOL, // the node is leaf, and a symbol (e.g. "x").
  NUMBER, // the node is leaf, and a constant number (e.g. 42).
};

/// Least binding operations.
static const str_t type_one_ops_comm = "+";
static const str_t type_one_ops_non_comm = "-";
static const auto type_one_ops = type_one_ops_comm +
                                 type_one_ops_non_comm;
/// Most binding operations.
static const str_t type_two_ops_comm = "*";
static const str_t type_two_ops_non_comm = "/";
static const auto type_two_ops = type_two_ops_comm +
                                 type_two_ops_non_comm;
/// All operations.
static const auto all_ops = type_one_ops + type_two_ops;
/// No-ops when found at the start of an expression.
static const str_t no_ops = "+";
/// Unary ops, must also be listed as functions.
static const str_t unary_ops = "-";
/// Functions of the form f(...).
static const str_t funcs[] = {"-", MEL_SUPPORTED_FUNCTIONS};

/// Digits.
static const str_t valid_digits = ".0123456789";
/// Possible characters before starting a number.
static const auto possible_num_starts = all_ops + "(," + valid_digits;

/// Cleanup/simplifications applied to text before parsing into expressions.
static const str_t prep_rules[] = {" ", "++", "--", "+-", "-+"};
static const str_t prep_subs[] = {"", "", "", "-", "-"};

} // namespace internal
} // namespace mel
