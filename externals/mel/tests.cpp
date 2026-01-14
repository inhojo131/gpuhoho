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

#include "tests.hpp"

int main(int, char* []) {

  if (mel::tests()) return 1;

  return mel::benchmarks();
}
