#include <iostream>


#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

const char kernelSource[] = "             \n" \
"__kernel void cl_weighted_blend(__global const float4 *in, \n"
"                                __global const float4 *aux, \n"
"                                __global       float4 *out) \n"
"{ \n"
"  int gid = get_global_id(0); \n"
"  float4 in_v = in[gid]; \n"
"  float4 aux_v = aux[gid]; \n"
"  float4 out_v; \n"
"  float in_weight; \n"
"  float aux_weight; \n"
"  float total_alpha = in_v.w + aux_v.w; \n"
" \n"
"  total_alpha = total_alpha == 0 ? 1 : total_alpha; \n"
" \n"
"  in_weight = in_v.w / total_alpha; \n"
"  aux_weight = 1.0f - in_weight; \n"
" \n"
"  out_v.xyz = in_weight * in_v.xyz + aux_weight * aux_v.xyz; \n"
"  out_v.w = total_alpha; \n"
"  out[gid] = out_v; \n"
"} \n";



enum {
	DATA_SIZE = 64,
};

int main(int argc, const char*argv[])
{
	float in[DATA_SIZE];       // original data set given to device
	float aux[DATA_SIZE];       // original data set given to device
	float results[DATA_SIZE];    // results returned from device

        for(unsigned i = 0; i < DATA_SIZE; i += 4) {
		in[i  ] = 0;
		in[i+1] = 1;
		in[i+2] = 0;
		in[i+3] = 1;

		aux[i  ] = 1;
		aux[i+1] = 0;
		aux[i+2] = 0;
		aux[i+3] = 0.5;
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
	cl::Buffer in1(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(in), in);
	cl::Buffer in2(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(aux), aux);
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
		cl::Kernel kernel(prg, "cl_weighted_blend");
		kernel.setArg(0, in1);
		kernel.setArg(1, in2);
		kernel.setArg(2, out);
//		kernel.setArg(3, (unsigned)DATA_SIZE);

		//todo: use this
		cl::size_t<3> local;                // local domain size for our calculation
		kernel.getWorkGroupInfo(devices[0], CL_KERNEL_COMPILE_WORK_GROUP_SIZE, &local);
		std::cout << "Local size is: " << local[0] << std::endl;

		/* Command queue */
		cl::CommandQueue cmd(ctx, devices[0]);

		std::cout << "Adding kernel" << std::endl;
		cmd.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(DATA_SIZE), cl::NDRange(1));
		std::cout << "Waiting for kernel" << std::endl;
		cmd.finish();
		std::cout << "Reading results" << std::endl;
		cmd.enqueueReadBuffer(out, true, 0, sizeof(results), results, 0);

	} catch (cl::Error e) {
		std::cerr << "Kernel failed: " << e.what() << " "
			<< e.err() << std::endl;
		return 1;
	} catch (...) {
		std::cerr << "Bailing out\n";
		return 1;
	}
	unsigned errors = 0;
	for (int i = 0; i < DATA_SIZE; i += 4) {
		float total_alpha = in[i + 3] + aux[i + 3];
		total_alpha = total_alpha == 0 ? 1 : total_alpha;

		float in_w = in[i+3] / total_alpha;
		float aux_w = aux[i+3] / total_alpha;

		float w_in[3] = {
			in[i] * in_w,  in[i+1] * in_w, in[i+2] * in_w,
		};
		float w_aux[3] = {
			aux[i] * aux_w, aux[i+1] * aux_w, aux[i+2] * aux_w,
		};

		float res[4] = {
			w_in[0] + w_aux[0], w_in[1] + w_aux[1],
			w_in[2] + w_aux[2], total_alpha,
		};
		if (res[0] != results[i] ||
		    res[1] != results[i+1] ||
		    res[2] != results[i+2] ||
		    res[3] != results[i+3]) {
			++errors;
			std::cerr << "Incorrect element(" << i/4 << "):\n";
			std::cerr << "\tIN(" << in[i] << ", " << in[i+1] << ", "
				<< in[i+2] << ", " << in[i+3] << ") ";
			std::cerr << "AUX(" << aux[i] << ", " << aux[i+1]
				<< ", "	<< aux[i+2] << ", " << aux[i+3] << ") ";
			std::cerr << "RES(" << results[i] << ", "
				<< results[i+1] << ", " << results[i+2] << ", "
				<< results[i+3] << ") ";
			std::cerr << "CORRECT(" << res[0] << ", " << res[1]
				<< ", "	<< res[2] << ", " << res[3] << ")\n";
			std::cerr << "\t\tDIFF(" << results[i] - res[0] << ", "
				<< results[i+1] - res[1] << ", "
				<< results[i+2] - res[2] << ", "
				<< results[i+3] - res[3] << ")\n";
				
		}
	}

	std::cout << "Wrong: " << errors << "/" << DATA_SIZE/4 << std::endl;

	return 0;
}
