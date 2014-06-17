#include <iostream>


#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#define VECTOR

// Simple compute kernel which computes the square of an input array

const char kernelSource[] = "            \n" \
"__kernel void shl_test(                 \n" \
"   __global ulong* input,               \n" \
"   __global ulong* output)              \n" \
"{                                       \n" \
"   int i = get_global_id(0);            \n" \
"   output[i] = input[i] >> i ;          \n" \
"}                                       \n" \
"__kernel void shl_vec_test(             \n" \
"   __global ulong4* input,              \n" \
"   __global ulong4* output)             \n" \
"{                                       \n" \
"   int i = get_global_id(0);            \n" \
"   output[i] = input[i] >> i;           \n" \
"}                                       \n" \
"\n";

enum {
	DATA_SIZE = 64,
};

int main(int argc, const char*argv[])
{
	(void) argc;
	(void) argv;

	uint64_t data[DATA_SIZE];       // original data set given to device
	uint64_t results[DATA_SIZE];    // results returned from device
	uint64_t results2[DATA_SIZE];   // results returned from device

        for(unsigned i = 0; i < DATA_SIZE; i++)
	        data[i] = rand();

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
	cl::Buffer in(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(data), data);
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
		cl::Kernel kernel(prg, "shl_test");
		kernel.setArg(0, in);
		kernel.setArg(1, out);

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
		/* test vector pow */
		cl::Kernel kernel2(prg, "shl_vec_test");
		kernel2.setArg(0, in);
		kernel2.setArg(1, out2);

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
	unsigned errors1 = 0;
#ifdef VECTOR
	unsigned errors2 = 0;
#endif
	for (int i = 0; i < DATA_SIZE; ++i) {
		uint64_t result = data[i] >> i;
		if (result != results[i]) {
			++errors1;
			std::cerr << "Incorrect element(" << std::dec
				<< i << "): " << std::hex
				<< data[i] << " result: " << results[i]
				<< " correct: " << result << std::endl;
		}
#ifdef VECTOR
		uint64_t result2 = data[i] >> (i / 4);
		if (result2 != results2[i]) {
			++errors2;
			std::cerr << "Incorrect element2(" << std::dec
				<< i << "): " << std::hex
				<< ::std::hex << data[i] << " result: "
                                << ::std::hex << results2[i] << " correct: "
                                << ::std::hex << result << std::endl;
		}
#endif
	}

	std::cout << "Wrong1: " << errors1 << "/" << DATA_SIZE << std::endl;
#ifdef VECTOR
	std::cout << "Wrong2: " << errors2 << "/" << DATA_SIZE << std::endl;
#endif
	return 0;
}
