# MEL

Math Expression Library

```c++
const auto val = mel::Eval<double>("sqrt(pow(3,2) + pow(4,2)");
```

MEL is a small (~500 loc) header-only C++11 library to parse strings into math expression objects that can be evaluated at runtime, by substituting symbols (e.g. `x`) by runtime values.
It can be used, for example, to implement user-defined functions (UDF) in a larger code, in a self-contained way.

![Unit Tests](https://github.com/pcarruscag/mel/actions/workflows/c-cpp.yml/badge.svg)
![Code QL](https://github.com/pcarruscag/mel/actions/workflows/codeql-analysis.yml/badge.svg)

## Usage

The two main functions of the library are `Parse` and `Eval`.
 - **Parse** creates an expression object (a tree) for an input string and extracts the symbols into a container, a symbol is a sub-string that cannot be split into more math operations, or interpreted directly as a number. The type (`float`, `double`, etc.) for such constants is a template parameter of the function.
 - **Eval** evaluates an expression tree, the mapping from symbol to value can be specified in two ways, either via a `string -> value` functor (3 argument version), or more efficiently via a `index -> value` functor (2 argument version). In the latter, "index" is the position of a symbol in the list of symbols created by `Parse`. The return type for `Eval` is a template parameter, and can be different from the one used in `Parse` (provided the latter can be converted into the former).

**Note**: A single-argument overload of `Eval` is provided for expressions without symbols, it evaluates a string directly (i.e. parses and evaluates). This is a convenience function, it is not efficient when the expression needs to be evaluated multiple times.
An additional convenience function, `Print`, can be used to visualize the expression trees produced by `Parse`.

### Example 1

Symbols as strings.

```c++
#include <cmath>
#include <map>
#include "mel.hpp"

const std::string expr = "a + \"const b\" * x + c*pow(x,2)";
std::vector<std::string> symbols;
const auto tree = mel::Parse<double>(expr, symbols);

// Assign values to symbols.
std::map<std::string, double> values = {{"a", 1}, {"\"const b\"", -1}, {"c", 0.5}, {"x", 0}};
auto symbol_to_val = [&values](const std::string& s) {
  return values.at(s);
};

// Evaluate for different values of x.
for (double x = 0; x <= 10; x += 0.1) {
  values.at("x") = x;
  std::cout << mel::Eval<double>(tree, symbols, symbol_to_val) << '\n';
}
```

### Example 2

Symbols as indices.

```c++
#include <cmath>
#include <vector>
#include "mel.hpp"

const std::string expr = "a + \"const b\" * x + c*pow(x,2)";
std::vector<std::string> symbols;
const auto tree = mel::Parse<double>(expr, symbols);

// Assign values to symbol indices, here we know the order is "a", "const b", "x", "c",
// but in a real application this process of (efficiently) mapping symbol indices
// to indices of recognized symbols (by the app using MEL) can be more involved.
std::vector<double> values = {1, -1, 0, 0.5};
auto idx_to_val = [&values](int i) {
  return values[i];
};

// Evaluate for different values of x.
for (double x = 0; x <= 10; x += 0.1) {
  values[2] = x;
  std::cout << mel::Eval<double>(tree, idx_to_val) << '\n';
}
```

### Default math functions and customization

In the default configuration, MEL parses the four arithmetic operations (`+-*/`), power and trigonometric functions from [cmath](https://www.cplusplus.com/reference/cmath/), `log`, `exp`, `fabs`, `fmin`, and `fmax`.
If the type used for `Eval` only supports simple arithmetic, the macro `MEL_ONLY_ARITHMETIC_OPS` can be defined before including `mel.hpp`.
The macros `MEL_SUPPORTED_FUNCTIONS`, `MEL_FUNCTION_CODES`, and `MEL_FUNCTION_IMPLEMENTATIONS`, can be used to fully customize the supported math functions, see `definitions.hpp` for instructions.

**Note**: MEL can only handle functions of 1 or 2 arguments, with a single return value.

## Performance

MEL expressions are between 1 and 100 times slower to evaluate than native C++ (a few benchmarks are included in `tests.hpp`)
The best-case scenario is for expressions that use a large number of functions (`pow`, `exp`, etc.) relative to the number of symbols, or when the expression is evaluated in a bandwidth-bound context.
The worst-case scenario is for expressions with many repeated symbols (which implies common sub expressions that are not optimized in MEL) used in a compute-bound context.

**Note**: To achieve good performance, the expression trees have a maximum number of nodes (constants, symbols, or operations) defined at compile-time. The default value is 255 (~4KB trees with up to 64bit constants) and can be overriden with macro `MEL_MAX_TREE_SIZE`.

## License

[LGPL-3.0](https://www.gnu.org/licenses/lgpl-3.0.html)

