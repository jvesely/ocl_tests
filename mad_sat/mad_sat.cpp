#include <iostream>
#include <algorithm>
#include <climits>

#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

const char kernelSource[] = "            \n" \
"__kernel void mad_sat_test(             \n" \
"   __global uint3* input1,              \n" \
"   __global uint3* input2,              \n" \
"   __global uint3* input3,              \n" \
"   __global uint3* output)              \n" \
"{                                       \n" \
"   int i = get_global_id(0);            \n" \
"   output[i] = mad_sat(input1[i], input2[i], input3[i]);          \n" \
"}                                       \n" \
"\n";

enum {
	DATA_SIZE = 3,
};

int main(int argc, const char*argv[])
{
	(void) argc;
	(void) argv;

	cl_uint data1[DATA_SIZE];      // original data set given to device
	cl_uint data2[DATA_SIZE];      // original data set given to device
	cl_uint data3[DATA_SIZE];      // original data set given to device
	cl_uint results[DATA_SIZE];    // results returned from device

        for(unsigned i = 0; i < DATA_SIZE; i++) {
	        data1[i] = rand();
	        data2[i] = rand();
	        data3[i] = rand();
	}

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
	cl::Buffer in3(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(data3), data3);
	cl::Buffer out(ctx, CL_MEM_WRITE_ONLY, sizeof(results));

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
		cl::Kernel kernel(prg, "mad_sat_test");
		kernel.setArg(0, in1);
		kernel.setArg(1, in2);
		kernel.setArg(2, in3);
		kernel.setArg(3, out);

		//todo: use this
		cl::size_t<3> local;                // local domain size for our calculation
		kernel.getWorkGroupInfo(devices[0], CL_KERNEL_COMPILE_WORK_GROUP_SIZE, &local);
		std::cout << "Local size is: " << local[2] << std::endl;

		/* Command queue */
		cl::CommandQueue cmd(ctx, devices[0]);

		cmd.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(1), cl::NDRange(1));
		cmd.finish();
		cmd.enqueueReadBuffer(out, true, 0, sizeof(results), results, 0);
	} catch (cl::Error e) {
		std::cerr << "Kernel failed: " << e.what() << " "
			<< e.err() << std::endl;
		return 1;
	} catch (...) {
		return 1;
	}
	unsigned errors1 = 0;
	for (int i = 0; i < DATA_SIZE; ++i) {
		cl_uint result = ::std::min<uint64_t>(
		    (uint64_t) data1[i] * (uint64_t)data2[i] + (uint64_t)data3[i], UINT_MAX);
		if (result != results[i]) {
			++errors1;
			std::cerr << "Incorrect element(" << std::dec
				<< i << "): " << std::dec
				<< data1[i] << " result: " << results[i]
				<< " correct: " << result << std::endl;
		}
	}

	std::cout << "Wrong1: " << errors1 << "/" << DATA_SIZE << std::endl;
	return 0;
}
