#include <iostream>


#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

// Simple compute kernel which computes the square of an input array

const char kernelSource[] = "             \n" \
"__kernel void square(                   \n" \
"   __global float* input,               \n" \
"   __global float* output,              \n" \
"   unsigned int count)                  \n" \
"{                                       \n" \
"   int i = get_global_id(0);            \n" \
"   if(i < count)                        \n" \
"       output[i] = input[i] * input[i]; \n" \
"}                                       \n" \
"\n";

enum {
	DATA_SIZE = 64,
};

int main(int argc, const char*argv[])
{
	float data[DATA_SIZE];       // original data set given to device
	float results[DATA_SIZE];    // results returned from device

        for(unsigned i = 0; i < DATA_SIZE; i++)
	        data[i] = rand() / (float)RAND_MAX;

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
		cl::Kernel kernel(prg, "square");
		kernel.setArg(0, in);
		kernel.setArg(1, out);
		kernel.setArg(2, (unsigned)DATA_SIZE);

		//todo: use this
		cl::size_t<3> local;                // local domain size for our calculation
		kernel.getWorkGroupInfo(devices[0], CL_KERNEL_COMPILE_WORK_GROUP_SIZE, &local);
		std::cout << "Local size is: " << local[2] << std::endl;

		/* Command queue */
		cl::CommandQueue cmd(ctx, devices[0]);

		cmd.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(DATA_SIZE), cl::NDRange(1));
		cmd.finish();
		cmd.enqueueReadBuffer(out, true, 0, sizeof(results), results, 0);

	} catch (cl::Error e) {
		std::cerr << "Kernel failed: " << e.what() << " "
			<< e.err() << std::endl;
		return 1;
	} catch (...) {
		return 1;
	}
	unsigned errors = 0;
	for (int i = 0; i < DATA_SIZE; ++i) {
		float result = data[i] * data[i];
		if (result != results[i]) {
			++errors;
			std::cerr << "Incorrect element(" << i << "): "
				<< data[i] << " result: " << results[i]
				<< " correct: " << result << std::endl;
		}
	}

	std::cout << "Wrong: " << errors << "/" << DATA_SIZE << std::endl;

	return 0;
}
