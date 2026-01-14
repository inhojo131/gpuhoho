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

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <map>
#include <vector>

#include "mel.hpp"

#define MEL_CHECK(VAL) if (!(VAL)) {assert(false); return 1;}

namespace mel {

inline int tests() {
  using namespace internal;
  std::cout << "\nTests\n\n";

  MEL_CHECK(sizeof(ExpressionTree<double>) == (max_tree_size*2+1)*sizeof(double))

  MEL_CHECK(BalancedParentheses(str_t("(((a+b)*c))")))
  MEL_CHECK(!BalancedParentheses(str_t("a)*2*(c")))

  MEL_CHECK(RemoveParentheses(str_t("(((a+b)*c))")) == "(a+b)*c")

  auto r = SplitAtOperation(type_two_ops, str_t(""),
                            str_t("(a+b)*(a-b)"));
  MEL_CHECK(r[0] == "*" && r[1] == "(a+b)" && r[2] == "(a-b)")

  auto r2 = DetectFunction(funcs, str_t("sqrt(pow(x,2)+1)"));
  MEL_CHECK(r2[0] == "sqrt" && r2[1] == "pow(x,2)+1" && r2[2] == "")

  bool is_n;
  MEL_CHECK(UnaryOpToUnaryFunc(unary_ops, str_t("-a"), is_n) == "-(a)")
  MEL_CHECK(UnaryOpToUnaryFunc(unary_ops, str_t("-(a)"), is_n) == "-(a)")
  MEL_CHECK(!is_n);
  MEL_CHECK(UnaryOpToUnaryFunc(unary_ops, str_t("-2.1"), is_n) == "-2.1")
  MEL_CHECK(is_n);

  str_t t = "a + -b";
  Preprocess(prep_rules, prep_subs, t);
  MEL_CHECK(t == "a-b")

  str_t t2 = "-2e-3";
  Preprocess(prep_rules, prep_subs, t2);
  MarkScientificNotation(t2);
  MEL_CHECK(t2 == "-2e{3")

  str_t t3 = "\"var 1\" * pow(\"var 0\", N) - \"var 1\"";
  const auto strings = FindStrings(t3);
  MEL_CHECK(strings.size() == 2)
  MEL_CHECK(strings.count("\"var 1\"") && strings.count("\"var 0\""))

  auto r3 = ApplyRules(str_t("((a+b)*c-d)"));
  MEL_CHECK(r3[0] == "-" && r3[1] == "(a+b)*c" && r3[2] == "d")

  std::vector<str_t> symb;
  Parse<double>(str_t("((a+b)*c - \"var 1\")"), symb);
  MEL_CHECK(symb[0] == "a" && symb[1] == "b" && symb[2] == "c" && symb[3] == "\"var 1\"")

#define MEL_CHECK_EXPR(EXPR) {  \
  auto v = Eval<double>(#EXPR); \
  std::cout << v << '\n';       \
  MEL_CHECK(v == (EXPR)) }

  MEL_CHECK_EXPR(2 + 2)
  MEL_CHECK_EXPR(1e-1)
  MEL_CHECK_EXPR(-2 - 2 + (1.5 / 2 + 1))
  MEL_CHECK_EXPR(-2e2 * -3e-2 + 1e1)
  MEL_CHECK_EXPR(-2e+2 * pow(-3e-2,2) + sqrt(.1E1) / -.2E-1 + exp(-0.32e0))
  MEL_CHECK_EXPR(-3 * -4 + pow(2, 3) / sqrt(9))
  MEL_CHECK_EXPR(-3 * (-4 + pow(2, 3)) / sqrt(9))
  MEL_CHECK_EXPR(fmax(3, fmin(5, pow(-2, -2))))
  MEL_CHECK_EXPR(sin(atan2(1, 1)) - 1/sqrt(2))
  MEL_CHECK_EXPR(1e1 - hypot(3,4) * log(exp(2)))

#undef MEL_CHECK_EXPR

#define MEL_CHECK_EXPR(EXPR, ...) {                               \
  std::vector<str_t> s;                                           \
  const double x[] = {__VA_ARGS__};                               \
  auto t = Parse<double>(str_t(#EXPR), s);                        \
  auto v = Eval<double>(t, [&x](int i) {return x[i];});           \
  std::cout << v << '\n';                                         \
  MEL_CHECK(v == (EXPR))                                          \
  std::map<str_t, double> m = {{"x[0]", x[0]}, {"x[1]", x[1]}};   \
  v = Eval<double>(t, s, [&m](const str_t& k) {return m.at(k);}); \
  MEL_CHECK(v == (EXPR)) }

  MEL_CHECK_EXPR(x[0] / x[1] - 1, 4.5, 2.25)
  MEL_CHECK_EXPR(sqrt(x[0]) / exp(x[1] - 1.1) / 3.14, 49, 3)

  return 0;
}

#undef MEL_CHECK_EXPR
#undef MEL_CHECK

namespace internal {

struct Timer {
  const clock_t c0;
  double time;
  Timer() : c0(clock()) {}
  double mark() {
    time = double(clock() - c0) * 1000 / CLOCKS_PER_SEC;
    return time;
  }
};

#define MEL_BENCHMARK(NAME, SIZE, SAMPLES, ...)                         \
int benchmark_##NAME(const double tol, const double allowed_ratio) {    \
  constexpr int samples = SAMPLES;                                      \
  constexpr int n = SIZE;                                               \
  std::vector<double> x(n), y(n), f(n);                                 \
                                                                        \
  for (int i=0; i<n; ++i) {                                             \
    x[i] = rand() / double(RAND_MAX);                                   \
    y[i] = rand() / double(RAND_MAX);                                   \
  }                                                                     \
                                                                        \
  const str_t expr = #__VA_ARGS__;                                      \
  std::vector<str_t> s;                                                 \
  const auto t = Parse<double>(expr, s);                                \
  std::cout << expr << '\n';                                            \
  Print(t, s, std::cout);                                               \
                                                                        \
  auto t0 = Timer();                                                    \
  auto* tree = new ExpressionTree<double>;                              \
  for (int k = 0; k < samples; ++k) {                                   \
    std::vector<str_t> s;                                               \
    *tree = Parse<double>(expr, s);                                     \
  }                                                                     \
  delete tree;                                                          \
  const auto t_parse = t0.mark() / samples;                             \
                                                                        \
  auto t1 = Timer();                                                    \
  for (int k = 0; k < samples; ++k) {                                   \
    for (int i = 0; i < n; ++i) {                                       \
      f[i] = __VA_ARGS__;                                               \
    }                                                                   \
  }                                                                     \
  const auto t_nat = t1.mark();                                         \
                                                                        \
  auto mel_func = [&](int i) {                                          \
    const double vals[] = {x[i], y[i]};                                 \
    return Eval<double>(t, [&vals](int j) {return vals[j];});           \
  };                                                                    \
                                                                        \
  auto t2 = Timer();                                                    \
  for (int k = 0; k < samples; ++k) {                                   \
    for (int i = 0; i < n; ++i) {                                       \
      f[i] = mel_func(i);                                               \
    }                                                                   \
  }                                                                     \
  const auto t_mel = t2.mark();                                         \
                                                                        \
  double diff = 0.0;                                                    \
  for (int i = 0; i < n; ++i) {                                         \
    const auto ref = fmax(1, fabs(f[i]));                               \
    diff = fmax(diff, fabs(f[i] - (__VA_ARGS__)) / ref);                \
  }                                                                     \
  std::cout << "Tree with " << t.size                                   \
            << " nodes, parsed in " << t_parse << "ms\n"                \
            << "Native " << t_nat << "ms, MEL " << t_mel                \
            << "ms, Ratio " << t_mel / t_nat                            \
            << ", Max diff. " << diff << "\n\n";                        \
  return (diff > tol || t_mel / t_nat > allowed_ratio) ? 1 : 0;         \
}

MEL_BENCHMARK(1, 8192, 4096, x[i] + y[i])
MEL_BENCHMARK(2, 8192, 2048, x[i]*x[i]*y[i] + (3*y[i]*y[i] - x[i] - 1) / y[i])
MEL_BENCHMARK(3, 8192, 1024, pow(x[i],
                                 3.1) + exp(y[i] * -1.4e-1) / sqrt(y[i] + x[i]))
MEL_BENCHMARK(4, 8192, 1024,
              (.5*x[i]+1)*(.7*x[i]-2)/(1.3*y[i]-1)-(1-.2*y[i])*(y[i]/x[i]+1))

#undef MEL_BENCHMARK

} // namespace internal

inline int benchmarks() {
  std::cout << "\nBenchmarks\n\n";
  if (internal::benchmark_1(0.0, 30)) return 1;
  if (internal::benchmark_2(1e-15, 55)) return 1;
  if (internal::benchmark_3(1e-16, 2.5)) return 1;
  if (internal::benchmark_4(1e-12, 50)) return 1;
  return 0;
}

} // namespace mel
