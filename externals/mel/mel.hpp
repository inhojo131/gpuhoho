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

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <iterator>
#include <set>
#include <sstream>
#include <vector>

#include "definitions.hpp"

namespace mel {
namespace internal {

/// Converts a string to a number of the desired type,
/// returns <true> if the conversion is successful.
template<class StringType, class NumberType>
bool ToNumber(const StringType& s, NumberType& n) {
  std::stringstream ss;
  ss << s;
  return static_cast<bool>(ss >> n);
}

/// Returns <true> if a string is convertible to number.
template<class StringType>
bool IsNumber(const StringType& s) {
  double n{};
  return ToNumber(s, n);
}

/// Replace all occurrences of "target" in a string.
template<class StringType>
void ReplaceAll(const StringType& target, const StringType& repl,
                StringType& str) {
  typename StringType::size_type pos = 0;
  while ((pos = str.find(target)) != StringType::npos) {
    str.replace(pos, target.size(), repl);
  }
}

/// Returns <true> if parentheses are balanced. Examples:
/// (((a+b)*c)) -> True
/// a)*2*(c -> False
template<class StringType>
bool BalancedParentheses(const StringType& expr) {
  int nest_level = 0;
  for (const auto c : expr) {
    if (c == '(') {
      nest_level += 1;
    } else if (c == ')') {
      nest_level -= 1;
    }
    // if the nest level becomes negative we know there is an imbalance
    if (nest_level < 0) {
      return false;
    }
  }
  return nest_level == 0;
}

/// Remove redundant parentheses around an expression. Examples:
/// (((a+b)*c)) -> (a+b)*c
template<class StringType>
StringType RemoveParentheses(StringType expr) {
  if (expr.size() > 2) {
    auto new_expr = StringType(expr.begin()+1, expr.end()-1);
    while ((expr.front() == '(') && BalancedParentheses(new_expr)) {
      expr = new_expr;
      new_expr = StringType(expr.begin()+1, expr.end()-1);
    }
  }
  return expr;
}

/// Splits an expression into (op, rhs, lhs), where "op" is a character
/// in op_list. The split is not done if parentheses are not balanced,
/// or if the character preceeding "op" is in excl_chars. Examples:
/// "a+b" -> ("+", "a", "b")
/// "(a+b)*(a-b)" -> ("*", "(a+b)", "(a-b)")
template<class StringType>
std::array<StringType, 3> SplitAtOperation(const StringType& op_list,
    const StringType& excl_chars, const StringType& expr) {
  std::array<StringType, 3> ret;
  for (auto it = expr.end() - 1; it != expr.begin(); --it) {
    const auto rhs = StringType(expr.begin(), it+1);
    if (!BalancedParentheses(rhs)) {
      continue;
    }
    if (op_list.find(*it) != StringType::npos) {
      if (excl_chars.find(*(it-1)) == StringType::npos) {
        ret[0] = *it;
        ret[1] = StringType(expr.begin(), it);
        ret[2] = StringType(it+1, expr.end());
        return ret;
      }
    }
  }
  return ret;
}

/// Detects if a function from "func_list" is being applied to an expression.
/// Functions are of the form f() or f(,) (for binary functions).
/// The result is returned as (func, arg1, [arg2]). Examples:
/// "sqrt(pow(x,2)+1)" -> ("sqrt", "pow(x,2)+1")
template<class StringListType, class StringType>
std::array<StringType, 3> DetectFunction(const StringListType& func_list,
    const StringType& expr) {
  std::array<StringType, 3> ret;
  for (const auto& f : func_list) {
    if (expr.size() <= f.size() + 1) {
      continue;
    }
    // remove "f(" and ")"
    const auto it = expr.begin() + f.size() + 1;
    const auto start_expr = StringType(expr.begin(), it);
    const auto inner_expr = StringType(it, expr.end()-1);
    if (f+'(' == start_expr && BalancedParentheses(inner_expr)) {
      // consider "," an operation to detect binary functions
      const auto args = SplitAtOperation(StringType(","),
                                         StringType(), inner_expr);
      ret[0] = f;
      if (args[0].empty()) {
        ret[1] = inner_expr;
      } else {
        ret[1] = args[1];
        ret[2] = args[2];
      }
      return ret;
    }
  }
  return ret;
}

/// Modifies an expression to ensure that single character unary operations,
/// such as "-", look like a function, for example "-a" -> "-(a)". However,
/// literals are not modified, i.e. "-2" -> "-2". This also restores the
/// sign of scientific notation exponents.
template<class StringType>
StringType UnaryOpToUnaryFunc(const StringType& op_list,
                              const StringType& expr,
                              bool& is_number) {
  // recover the sign for scientific notation
  auto expr2 = expr;
  typename StringType::size_type pos;
  if ((pos = expr2.find('}')) != StringType::npos) expr2[pos] = '+';
  if ((pos = expr2.find('{')) != StringType::npos) expr2[pos] = '-';

  if (!IsNumber(expr2)) {
    is_number = false;
    if (op_list.find(expr.front()) != StringType::npos && expr[1] != '(') {
      return StringType({expr.front(), '('}) + StringType(expr.begin()+1,
             expr.end()) + ')';
    } else {
      return expr;
    }
  }
  is_number = true;
  return expr2;
}

/// Replaces the sign of scientific number exponents by }(+) or {(-) to avoid
/// splitting numbers when detecting operations.
template<class StringType>
void MarkScientificNotation(StringType& expr) {
  for (auto it = expr.begin(); it != expr.end(); ++it) {
    // Find where a number might start.
    if (possible_num_starts.find(*it) == str_t::npos) continue;
    // Advance to the next character,
    if (++it == expr.end()) break;
    // that can be a unary operation,
    if (unary_ops.find(*it) != str_t::npos) ++it;
    // or a sequence of valid digits,
    while (it != expr.end() && valid_digits.find(*it) != str_t::npos) ++it;
    if (it == expr.end()) break;
    // followed by "e" or "E",
    if (*it != 'e' && *it != 'E') continue;
    if (++it == expr.end()) break;
    // and then the sign we need to replace.
    if (*it == '+') *it = '}';
    if (*it == '-') *it = '{';
  }
}

/// Applies a series of substitution rules to an expression to make it
/// compatible with the parsing rules.
template<class StringListType, class StringType>
void Preprocess(const StringListType& rules, const StringListType& subs,
                StringType& expr) {
  auto old = StringType();
  while (old != expr) {
    old = expr;
    auto subs_it = std::begin(subs);
    for (const auto& rule : rules) {
      ReplaceAll(rule, *subs_it, expr);
      ++subs_it;
    }
  }
}

/// Finds string symbols, within quotation marks and possibly with spaces.
template <class StringType>
std::set<StringType> FindStrings(const StringType& expr) {
  std::set<StringType> strings;
  for (auto i = expr.find('"', 0); i < expr.size();) {
    const auto j = expr.find('"', i + 1);
    if (j < expr.size()) {
      strings.emplace(expr.begin() + i, expr.begin() + j + 1);
      i = expr.find('"', j + 1);
    } else {
      break;
    }
  }
  return strings;
}

/// Finds an applicable rule to an expression by trying all in the right order.
template<class StringType>
std::array<StringType, 3> ApplyRules(StringType expr) {
  if (no_ops.find(expr.front()) != str_t::npos) {
    expr = StringType(expr.begin()+1, expr.end());
  }
  expr = RemoveParentheses(expr);

  auto result = SplitAtOperation(type_one_ops_comm, type_two_ops, expr);
  if (result[0].empty()) {
    result = SplitAtOperation(type_one_ops_non_comm, type_two_ops, expr);
  }
  if (result[0].empty()) {
    result = SplitAtOperation(type_two_ops_comm, StringType(), expr);
  }
  if (result[0].empty()) {
    result = SplitAtOperation(type_two_ops_non_comm, StringType(), expr);
  }
  if (result[0].empty()) {
    bool is_number = false;
    expr = UnaryOpToUnaryFunc(unary_ops, expr, is_number);
    if (!is_number) {
      result = DetectFunction(funcs, expr);
    }
  }
  if (result[0].empty()) {
    // symbol or number
    result[0] = expr;
  }
  return result;
}

/// Converts the text representation of the operation into an operation code.
template<class StringType>
OpCode StringToOpCode(const StringType& str) {
  auto b = std::begin(supported_operations);
  return static_cast<OpCode>(std::find(b, std::end(supported_operations),
                                       str) - b);
}

/// Builds an expression tree by recursively extracting operations and building
/// subtrees for their lhs and rhs expressions. The symbols are also extracted.
template<class StringType, class StringListType, class TreeType>
void BuildExpressionTree(const StringType& expr, StringListType& symbols,
                         TreeType& tree) {
  auto& node = tree.nodes[tree.size];
  const auto result = internal::ApplyRules(expr);

  if (result[1].empty()) {
    typename TreeType::type value;
    if (ToNumber(result[0], value)) {
      node.type = OpCode::NUMBER;
      node.val = value;
    } else {
      node.type = OpCode::SYMBOL;
      // Find the index of the symbol, or append it to the list.
      const auto pos = std::find(symbols.begin(), symbols.end(), result[0]);
      node.symbol_id = static_cast<int>(pos - symbols.begin());
      if (pos == symbols.end()) {
        symbols.push_back(result[0]);
      }
    }
  } else {
    // The node is an expression with 1 or 2 children.
    node.type = StringToOpCode(result[0]);
    assert(node.type != OpCode::NOOP);

    node.child.left = ++tree.size;
    tree.nodes[node.child.left].level = static_cast<short>(node.level + 1);
    BuildExpressionTree(result[1], symbols, tree);

    if (!result[2].empty()) {
      node.child.right = ++tree.size;
      tree.nodes[node.child.right].level = static_cast<short>(node.level + 1);
      BuildExpressionTree(result[2], symbols, tree);
    } else {
      node.child.right = -1;
    }
  }
}

/// Prints a representation of a tree to a stream.
template<class TreeType, class StringListType, class StreamType>
void PrintExpressionTree(const TreeType& tree, int i,
                         const StringListType& symbols,
                         int level, StreamType& stream) {
  for (int k = 0; k < level; ++k) {
    stream << "  ";
  }
  const auto& node = tree.nodes[i];

  switch (node.type) {
  case OpCode::NUMBER:
    stream << node.val << '\n';
    break;
  case OpCode::SYMBOL:
    stream << symbols[node.symbol_id] << '\n';
    break;
  case OpCode::NOOP:
    assert(false);
    break;
  default:
    const auto& op = supported_operations[static_cast<int>(node.type)];
    stream << op << '\n';
    PrintExpressionTree(tree, node.child.left, symbols, level+1, stream);
    if (node.child.right >= 0) {
      PrintExpressionTree(tree, node.child.right, symbols, level+1, stream);
    }
    break;
  }
}

/// Evaluates an expression tree.
template<class ReturnType, class TreeType, class FunctionType>
ReturnType EvaluateExpressionTree(const TreeType& tree,
                                  const FunctionType& index_to_value) {
  // Because a node only depends on higher-level nodes, we can start evaluating
  // them from the highest level (bottom of the tree) until we arrive at the
  // top (final result). Note that there are no dependencies within each level.
  // This avoids recursion and thus is faster, at the expense of using stack
  // space for all possible intermediate results.
  std::array<ReturnType, max_tree_size> v;

  for (int i = tree.size - 1; i >= 0; --i) {
    const auto& node = tree.nodes[i];
    switch (node.type) {
    case OpCode::NUMBER:
      v[i] = static_cast<ReturnType>(node.val);
      break;
    case OpCode::SYMBOL:
      v[i] = index_to_value(node.symbol_id);
      break;
    case OpCode::ADD:
      v[i] = v[node.child.left] + v[node.child.right];
      break;
    case OpCode::SUB:
      if (node.child.right >= 0) {
        v[i] = v[node.child.left] - v[node.child.right];
      } else {
        v[i] = -v[node.child.left];
      }
      break;
    case OpCode::MUL:
      v[i] = v[node.child.left] * v[node.child.right];
      break;
    case OpCode::DIV:
      v[i] = v[node.child.left] / v[node.child.right];
      break;
      MEL_FUNCTION_IMPLEMENTATIONS(v[node.child.left], v[node.child.right])
    case OpCode::NOOP:
      assert(false);
    }
  }
  return v[0];
}

} // namespace internal

/// Type for an expression tree. The result of parsing expressions and used
/// to evaluate them.
template<class NumberType>
struct ExpressionTree {
  using type = NumberType;

  struct Node {
    // Type of the node, either as a leaf or as an operation.
    internal::OpCode type = internal::OpCode::NOOP;

    // Level of the node in the tree.
    short level = 0;

    // Index of the node in the tree.
    short index = 0;

    struct Children {
      int left, right;
    };

    // If the node is a leaf, it is either a number or a symbol,
    // otherwise is has one or two child nodes.
    union {
      NumberType val;
      int symbol_id;
      Children child;
    };

    bool operator<(const Node& other) const {
      return level != other.level ? level < other.level :
             static_cast<int>(type) < static_cast<int>(other.type);
    }
  };

  std::array<Node, internal::max_tree_size> nodes;
  int size = 0;
};

/// Preprocess an expression, create an expression tree for it, and extract its
/// symbols in the process. NumberType is the type used for stored constants
/// (i.e. literals).
template<class NumberType, class StringType, class StringListType>
ExpressionTree<NumberType> Parse(StringType expr, StringListType& symbols) {
  StringListType orig_strings, repl_strings;
  int i_repl = 0;
  for (const auto& str : internal::FindStrings(expr)) {
    const auto h = std::hash<StringType>{}(str) % 8192 + i_repl * 8192;
    orig_strings.emplace_back(str);
    repl_strings.emplace_back("mel_" + std::to_string(h));
    ++i_repl;
  }
  internal::Preprocess(orig_strings, repl_strings, expr);
  internal::Preprocess(internal::prep_rules, internal::prep_subs, expr);
  internal::MarkScientificNotation(expr);

  ExpressionTree<NumberType> tree{};
  for (int i = 0; i < internal::max_tree_size; ++i) {
    tree.nodes[i].index = static_cast<short>(i);
  }
  internal::BuildExpressionTree(expr, symbols, tree);
  tree.size++;

  // Undo the string replacements.
  for (auto& symbol : symbols) {
    const auto it =
        std::find(repl_strings.begin(), repl_strings.end(), symbol);
    if (it != repl_strings.end()) {
      symbol = orig_strings[std::distance(repl_strings.begin(), it)];
    }
  }

  // Sort nodes by their level in the tree, nodes in level i can be evaluated
  // with the values at level i+1. This is not strictly required, but it makes
  // the evaluation faster.
  std::sort(tree.nodes.begin(), tree.nodes.begin() + tree.size);

  // Renumber children.
  std::array<int, internal::max_tree_size> perm;
  for (int i = 0; i < internal::max_tree_size; ++i) {
    perm[tree.nodes[i].index] = i;
  }
  for (int i = 0; i < tree.size; ++i) {
    auto& node = tree.nodes[i];
    switch (node.type) {
    case internal::OpCode::NUMBER:
    case internal::OpCode::SYMBOL:
    case internal::OpCode::NOOP:
      break;
    default:
      node.child.left = perm[node.child.left];
      if (node.child.right >= 0) {
        node.child.right = perm[node.child.right];
      }
    }
  }
  return tree;
}

/// Prints a representation of a tree to a stream.
template<class TreeType, class StringListType, class StreamType>
void Print(const TreeType& tree, const StringListType& symbols,
           StreamType& stream) {
  internal::PrintExpressionTree(tree, 0, symbols, 0, stream);
}

/// Evaluates an expression. The functor "index_to_value" should map the index
/// of each symbol (order in the list produced by Parse) to its value
/// (int -> ReturnType). The return type does not need to be the same as the
/// type of number for the constants in the tree.
template<class ReturnType, class TreeType, class FunctionType>
ReturnType Eval(const TreeType& tree, const FunctionType& index_to_value) {
  return internal::EvaluateExpressionTree<ReturnType>(tree, index_to_value);
}

/// Overload of Eval, where the functor "symbol_to_value" should map each
/// symbol to its value (StringType -> ReturnType).
template<class ReturnType, class TreeType, class StringListType,
         class FunctionType>
ReturnType Eval(const TreeType& tree, const StringListType& symbols,
                const FunctionType& symbol_to_value) {
  auto index_to_value = [&](int i) {
    return symbol_to_value(symbols[i]);
  };
  return internal::EvaluateExpressionTree<ReturnType>(tree, index_to_value);
}

/// Overload of Eval, evaluates a raw expression (string) assuming it does not
/// contain symbols (provided for convenience).
template<class ReturnType, class StringType>
ReturnType Eval(const StringType& expr) {
  std::vector<str_t> s;
  auto f = [&](int) {
    assert(false && "Unexpected symbol");
    return ReturnType{};
  };
  return internal::EvaluateExpressionTree<ReturnType>(
           Parse<ReturnType>(str_t(expr), s), f);
}

} // namespace mel
