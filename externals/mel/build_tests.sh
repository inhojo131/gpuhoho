rm tests.o tests
g++ -pedantic -Wextra -Wall -Wconversion -Werror -std=c++11 -m64 -O2 -march=native -funroll-loops -ftree-vectorize -DNDEBUG -c ./tests.cpp -o ./tests.o
g++ -o tests ./tests.o -m64 -s
