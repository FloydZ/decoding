#ifndef CUM4RI_HELPER_CUH
#define CUM4RI_HELPER_CUH

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <thread>

#include "instance.h"

#include "m4ri/m4ri.h"
#include "m4ri/mzd.h"
#include "m4ri/strassen.h"

#ifndef checkCudaErrors
#define checkCudaErrors(x) x
#endif

#ifdef DEBUG
#include <cassert>
#define ASSERT(x) assert(x)
#define DEBUG_MACRO(x) (x)
#else
#define ASSERT(x)
#define DEBUG_MACRO(x) 
#endif

#include <iostream>
#include <algorithm>

//TODO merge this with helper.h: problem is that cuda does not know C++20
template <uint32_t ll>
using LogTypeTemplate =
typename std::conditional<(ll <= 8), uint8_t,
		typename std::conditional<(ll <= 16), uint16_t,
				typename std::conditional<(ll <= 32), uint32_t,
						typename std::conditional<(ll <= 64), uint64_t,
								__uint128_t
						>::type
				>::type
		>::type
>::type;


constexpr static uint64_t TESTLOOPS = 10;

#include "matrix/custom_m4ri.h"
#endif //CUM4RI_HELPER_CUH
