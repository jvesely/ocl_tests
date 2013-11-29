CXX=clang++
#CXXFLAGS=-O3

test: blend.o
	clang++ $^ -o $@ -lOpenCL -Wall -Wextra

clean:
	rm -v *.o test
