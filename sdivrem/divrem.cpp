#include <iostream>
#include <climits>


#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

// Simple compute kernel which computes the square of an input array

const char kernelSource[] = "              \n" \
"__kernel void sdivrem(                    \n" \
"   __global char* inputa,                 \n" \
"   __global char* inputb,                 \n" \
"   __global char* divout,                 \n" \
"   __global char* remout,                 \n" \
"   unsigned int count)                    \n" \
"{                                         \n" \
"   int i = get_global_id(0);              \n" \
"   if(i < count) {                        \n" \
"       divout[i] = inputa[i] / inputb[i]; \n" \
"       remout[i] = inputa[i] % inputb[i]; \n" \
"   }                                      \n" \
"}                                         \n" \
"__kernel void udivrem(                    \n" \
"   __global char* inputa,        \n" \
"   __global char* inputb,        \n" \
"   __global char* divout,        \n" \
"   __global char* remout,        \n" \
"   unsigned int count)                    \n" \
"{                                         \n" \
"   int i = get_global_id(0);              \n" \
"   if(i < count) {                        \n" \
"       divout[i] = inputa[i] / inputb[i]; \n" \
"       remout[i] = inputa[i] % inputb[i]; \n" \
"   }                                      \n" \
"}                                         \n" \
"\n";

enum {
	DATA_SIZE = 254 * 256,
};

int main(void)
{
	char dataA[DATA_SIZE];       // original data set given to device
	char dataB[DATA_SIZE];       // original data set given to device
	char resD[DATA_SIZE];       // original data set given to device
	char resR[DATA_SIZE];       // original data set given to device

        for(unsigned i = 0; i < DATA_SIZE; i++) {
	        dataA[i] = i % UCHAR_MAX;
	        dataB[i] = (i / UCHAR_MAX) + 1;
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
	cl::Buffer inA(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(dataA), dataA);
	cl::Buffer inB(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(dataB), dataB);
	cl::Buffer outD(ctx, CL_MEM_WRITE_ONLY, sizeof(resD));
	cl::Buffer outR(ctx, CL_MEM_WRITE_ONLY, sizeof(resR));

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
		cl::Kernel kernel(prg, "sdivrem");
		kernel.setArg(0, inA);
		kernel.setArg(1, inB);
		kernel.setArg(2, outD);
		kernel.setArg(3, outR);
		kernel.setArg(4, (unsigned)DATA_SIZE);

		//todo: use this
		cl::size_t<3> local;                // local domain size for our calculation
		kernel.getWorkGroupInfo(devices[0], CL_KERNEL_COMPILE_WORK_GROUP_SIZE, &local);
		std::cout << "Local size is: " << local[2] << std::endl;

		/* Command queue */
		cl::CommandQueue cmd(ctx, devices[0]);

		cmd.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(DATA_SIZE), cl::NDRange(1));
		cmd.finish();
		cmd.enqueueReadBuffer(outD, true, 0, sizeof(resD), resD, 0);
		cmd.enqueueReadBuffer(outR, true, 0, sizeof(resR), resR, 0);

	} catch (cl::Error e) {
		std::cerr << "Kernel failed: " << e.what() << " "
			<< e.err() << std::endl;
		return 1;
	} catch (...) {
		return 1;
	}
	unsigned errors = 0;
	for (int i = 0; i < DATA_SIZE; ++i) {
		char resultD = dataB[i] != 0 ? dataA[i] / dataB[i] : 0;
		char resultR = dataB[i] != 0 ? dataA[i] % dataB[i] : 0;
		if (resultD != resD[i] || resultR != resR[i]) {
			++errors;
			std::cerr << "Incorrect element(" << i << "): "
				<< (int)dataA[i] << " /,% " << (int)dataB[i]
				<< " result: " << (int)resD[i] << ", " << (int)resR[i]
				<< " correct: " << (int)resultD << ", " << (int)resultR
				 << std::endl;
		}
	}

	std::cout << "Wrong: " << errors << "/" << DATA_SIZE << std::endl;

	return 0;
}
