#include <iostream>


#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#define LONG
#define SW

const char kernelSource[] = "             \n" \
"__kernel void test1(                   \n" \
"   unsigned x,                             \n" \
"   unsigned y,                             \n" \
"   __global unsigned *output)              \n" \
"{                                       \n" \
"   output[0] = x * y;                   \n" \
"   output[1] = x / y;                   \n" \
"   output[2] = x % y;                   \n" \
"}                                       \n" \
"                                        \n" \
"__kernel void test2(                   \n" \
"   ulong x,                             \n" \
"   ulong y,                             \n" \
"   __global ulong *output)              \n" \
"{                                       \n" \
"   output[0] = x * y;                   \n" \
"   output[1] = x / y;                   \n" \
"   output[2] = x % y;                   \n" \
"}                                       \n" \
"\n";

enum {
	X = 6,
	Y = 4,
};

int main(int argc, const char*argv[])
{
	(void) argv;
	(void) argc;

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

#ifdef SW
	cl_uint result1[3];
	cl::Buffer out1(ctx, CL_MEM_WRITE_ONLY, sizeof(result1));
#endif
#ifdef LONG
	cl_ulong result2[3];
	cl::Buffer out2(ctx, CL_MEM_WRITE_ONLY, sizeof(result2));
#endif

	/* Create kernel and set arguments */
	try {
		/* Command queue */
		cl::CommandQueue cmd(ctx, devices[0]);
#ifdef SW
		cl::Kernel kernel1(prg, "test1");
		kernel1.setArg(0, (cl_uint)X);
		kernel1.setArg(1, (cl_uint)Y);
		kernel1.setArg(2, out1);

		//todo: use this
		cl::size_t<3> local;                // local domain size for our calculation
		kernel1.getWorkGroupInfo(devices[0], CL_KERNEL_COMPILE_WORK_GROUP_SIZE, &local);
		std::cout << "Local size is: " << local[2] << std::endl;


		cmd.enqueueNDRangeKernel(kernel1, cl::NDRange(0), cl::NDRange(1), cl::NDRange(1));
		cmd.finish();
		cmd.enqueueReadBuffer(out1, true, 0, sizeof(result1), result1, 0);
#endif
#ifdef LONG
		/* test ulong */
		cl::Kernel kernel2(prg, "test2");
		kernel2.setArg(0, (cl_ulong)X);
		kernel2.setArg(1, (cl_ulong)Y);
		kernel2.setArg(2, out2);

		cmd.enqueueNDRangeKernel(kernel2, cl::NDRange(0), cl::NDRange(1), cl::NDRange(1));
		cmd.finish();
		cmd.enqueueReadBuffer(out2, true, 0, sizeof(result2), result2, 0);
#endif
	} catch (cl::Error e) {
		std::cerr << "Kernel failed: " << e.what() << " "
			<< e.err() << std::endl;
		return 1;
	} catch (...) {
		return 1;
	}
#ifdef SW
	std::cout << X << " MUL " << Y << " = " << result1[0] << std::endl;
	std::cout << X << " DIV " << Y << " = " << result1[1] << std::endl;
	std::cout << X << " MOD " << Y << " = " << result1[2] << std::endl;
#endif
#ifdef LONG
	std::cout << X << " MUL " << Y << " = " << result2[0] << std::endl;
	std::cout << X << " DIV " << Y << " = " << result2[1] << std::endl;
	std::cout << X << " MOD " << Y << " = " << result2[2] << std::endl;
#endif
	return 0;
}
