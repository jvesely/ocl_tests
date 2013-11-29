CXX=clang++

test: arr.o
	clang++ $^ -o $@ -lOpenCL -Wall -Wextra

clean:
	rm -v *.o test
