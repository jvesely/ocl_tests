#include <cmath>
#include <iostream>


#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#define VECTOR

// Simple compute kernel which computes the square of an input array

const char kernelSource[] = "             \n" \
"__kernel void fmin_test(                 \n" \
"   __global float* input1,               \n" \
"   __global float* input2,               \n" \
"   __global float* output)               \n" \
"{                                        \n" \
"   int i = get_global_id(0);             \n" \
"   output[i] = fmin(input1[i], input2[i]);      \n" \
"}                                        \n" \
"__kernel void fmin_vec_test(             \n" \
"   __global float4* input1,              \n" \
"   __global float4* input2,              \n" \
"   __global float4* output)              \n" \
"{                                        \n" \
"   int i = get_global_id(0);             \n" \
"   output[i] = fmin(input1[i], input2[i]);      \n" \
"}                                        \n" \
"\n";

enum {
	DATA_SIZE = 64,
};

static float to_float(unsigned code)
{
	union {
		float f;
		unsigned u;
	} conv;
	conv.u = code;
	return conv.f;
}
static unsigned to_uint(float num)
{
	union {
		float f;
		unsigned u;
	} conv;
	conv.f = num;
	return conv.u;
}

int main(int argc, const char*argv[])
{
	(void) argc;
	(void) argv;

	float data1[DATA_SIZE];       // original data set given to device
	float data2[DATA_SIZE];       // original data set given to device
	float results[DATA_SIZE];    // results returned from device
	float results2[DATA_SIZE];   // results returned from device

        for(unsigned i = 0; i < DATA_SIZE; i++) {
	        data1[i] = rand() / (float)RAND_MAX;
	        data2[i] = rand() / (float)RAND_MAX;
	}

	// Special cases
	data1[0] = to_float(0x7fe7afae);
	data2[0] = to_float(0x134a752c);

	data1[1] = to_float(0x10108d49);
	data2[1] = to_float(0xffa66c5d);

	cl::vector< cl::Platform > platformList;
	cl::Platform::get(&platformList);
	std::cout << "Platform count is: " << platformList.size() << std::endl;
	if (platformList.size() < 1)
		return 1;

	cl::Platform platform = platformList[0];

	std::string vendor, name, version;
	platform.getInfo(CL_PLATFORM_VENDOR, &vendor);
	platform.getInfo(CL_PLATFORM_NAME, &name);
	platform.getInfo(CL_PLATFORM_VERSION, &version);
	std::cout << "Platform is `" << name << "' by: " << vendor
		<< " version: " << version << std::endl;

	cl::vector <cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	std::cout << devices.size() << " available device(s)\n";
	if (devices.size() == 0)
		return 1;

	devices[0].getInfo(CL_DEVICE_VENDOR, &vendor);
	devices[0].getInfo(CL_DEVICE_NAME, &name);
	devices[0].getInfo(CL_DEVICE_VERSION, &version);
	std::cout << "Platform is `" << name << "' by: " << vendor
		<< " version: " << version << std::endl;

	/* Create CL context */
	cl::Context ctx(devices);

	/* CL buffers to use as kernel arguments */
	cl::Buffer in1(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(data1), data1);
	cl::Buffer in2(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(data2), data2);
	cl::Buffer out(ctx, CL_MEM_WRITE_ONLY, sizeof(results));
	cl::Buffer out2(ctx, CL_MEM_WRITE_ONLY, sizeof(results2));

	/* Create program from source */
	cl::Program::Sources src(1, std::make_pair(kernelSource, std::strlen(kernelSource)));
	cl::Program prg(ctx, src);
	try {
		int ret = prg.build(devices);
		if (ret != CL_SUCCESS) {
			std::cout <<"BUILD FAIL" << std::endl;
		}
	} catch (cl::Error e) {
		std::cerr << "Build failed:\n" << e.what() << " "
			<< e.err() << std::endl;
		std::cerr << "BUILD LOG:\n" <<
			prg.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
			<< "\nLOG DONE\n";
		return 1;
	} catch (...) {
		return 1;
	}


	/* Create kernel and set arguments */
	try {
		cl::Kernel kernel(prg, "fmin_test");
		kernel.setArg(0, in1);
		kernel.setArg(1, in2);
		kernel.setArg(2, out);

		//todo: use this
		cl::size_t<3> local;                // local domain size for our calculation
		kernel.getWorkGroupInfo(devices[0], CL_KERNEL_COMPILE_WORK_GROUP_SIZE, &local);
		std::cout << "Local size is: " << local[2] << std::endl;

		/* Command queue */
		cl::CommandQueue cmd(ctx, devices[0]);

		cmd.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(DATA_SIZE), cl::NDRange(1));
		cmd.finish();
		cmd.enqueueReadBuffer(out, true, 0, sizeof(results), results, 0);
#ifdef VECTOR
		/* test vector fmin */
		cl::Kernel kernel2(prg, "fmin_vec_test");
		kernel2.setArg(0, in1);
		kernel2.setArg(1, in2);
		kernel2.setArg(2, out2);

		/* Command queue */
		cmd.enqueueNDRangeKernel(kernel2, cl::NDRange(0), cl::NDRange(DATA_SIZE / 4), cl::NDRange(1));
		cmd.finish();
		cmd.enqueueReadBuffer(out2, true, 0, sizeof(results2), results2, 0);
#endif
	} catch (cl::Error e) {
		std::cerr << "Kernel failed: " << e.what() << " "
			<< e.err() << std::endl;
		return 1;
	} catch (...) {
		return 1;
	}
	unsigned errors1 = 0, errors2 = 0;
	for (int i = 0; i < DATA_SIZE; ++i) {
		float result = fmin(data1[i], data2[i]);
		if (to_uint(result) != to_uint(results[i])) {
			++errors1;
			std::cerr << "Incorrect element(" << i << "): "
				<< data1[i] << ", " << data2[i] << " result: "
				<< results[i] << " correct: " << result
				<< std::endl;
		}
#ifdef VECTOR
		if (to_uint(result) != to_uint(results2[i])) {
			++errors2;
			std::cerr << "Incorrect element2(" << i << "): "
				<< data1[i] << ", " << data2[i] << " result: "
				<< results2[i] << " correct: " << result
				<< std::endl;
		}
#endif
	}

	std::cout << "Wrong1: " << errors1 << "/" << DATA_SIZE << std::endl;
#ifdef VECTOR
	std::cout << "Wrong2: " << errors2 << "/" << DATA_SIZE << std::endl;
#endif
	return 0;
}
