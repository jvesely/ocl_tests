CXX=clang++

test: list.o
	clang++ $^ -o $@ -lOpenCL -Wall -Wextra

clean:
	rm -v *.o test
