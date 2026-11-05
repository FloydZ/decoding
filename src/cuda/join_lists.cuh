// This file contains list join algorithms.
#include <stdint.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include <cuda/atomic>
#include "cuhelper.cuh"

// needs sm atleast 75
#ifdef __CUDA_SUBBYTE_IMMA__
#include <mma.h>
using namespace nvcuda;
using namespace nvcuda::wmma; //::experimental;
#endif


template<class T>
class ListJoinSimple {
	// This class allows to join two lists, whos `rows` are only limbs. So
	// its not allowed to insert multi limb sizes rows.

	// number of coordinates to check equality on
	const uint32_t l = 16;

	// Number of limbs in each input list
	const uint64_t listSize = 10;
	const uint64_t expOutListSize = (listSize * listSize) >> l;

	// cuda internal configuration
	const uint32_t blocks = 1;
	const uint32_t tpb = 32;

	// helper mask: identifies the elements which are equal
	const T mask = (1ull << l) - 1;

		
	/// return 0 if collision, else rubbish > 0
	__device__
	inline uint32_t check(const T a, const T b) {
		return ((a ^ b) & mask);
	}


//	/// IDEA: to the stupid quadratic search over both lists.
//	/// ASSUMES: that for every list element in the two input list, there is a
//	///          thread which computes the sum.
//	__global__
//	uint64_t simple_join(T *out, 
//					 T *in1, 
//					 T *in2) const noexcept {
//		assert((list_size == blocks * tpb) && "wrong number of threads");
//
//		const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;	
//		
//		// TODO: copy the list L1 into shared mem.
//		// do the stupid thing.
//		for (uint64_t loop = 0; loop < listSize; loop++) {
//			const T tmp = check(in1[idx], in2[idx + loop]);
//
//			if constexpr (expOutListSize > listSize) {
//				// easy case: we can store everything in or output list
//				
//				// TODO: quite difficult:
//				// 			we need to apply a exlusive_scan to the list
//				//          to know where to put each element in the outlist
//			} else {
//				// not implemented
//			}
//		}
//	}
//	
//	// In order to be able to use thrust (which is only able to operate and 
//	// filter only on a single input list), we need to zip the two input lists.
//	// And the type is this:
//	using SimpleJoinTuple = T[2];
//
//	// Helper struct for thrust.
//	struct SimpleJoinThrustFilter {
//		SimpleJoinThrustFilter() {}
//		
//		__device__
//		bool operator()(SimpleJoinTuple &a){
//		    return !check(a[0], a[1]);
//		}
//	};
//
//	/// zips the two input lists 
//	__global__
//	void SimpleJoinThrustFillZipList(SimpleJoinTuple *zipList, T *in1, T *in2) {
//		const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;	
//		uint32_t index = idx;
//		
//		while (index < listSize) {
//			zipList[index][0] = in1[index]; 
//			zipList[index][1] = in2[index]; 
//
//			index += idx;
//		}
//	}
//
//	/// IDEA: as above: so do the stupid quadratic search, but use thrust 
//	///       to do all the hard work.
//	/// NOTE: This is cheatted. It only workes on a single list, because we 
//	/// 	  we are using the given list as a input and output list. 
//	///       An example of creating a zipList is also shown in the code.
//	///       Currently we are not doing the full work. We are only doing the 
//	///       join on a single test, no the quadratic one. 
//	///
//	/// RETURN: number of elements 
//	__host__
//	uint64_t simple_join_thrust(T *zipList) const noexcept {
//		using IndexType = uint64_t;
//
//		// we first need to zip in1 and in2;
//		// T *zipList = (T *)malloc(sizeof(SimpleJoinTuple) * listSize);
//		uint32_t *predicates = (uint32_t *)malloc(sizeof(uint32_t) * listSize);
//		uint32_t *scatter_indices = (uint32_t *)malloc(sizeof(uint32_t) * listSize);
//
//		// SimpleJoinThrustFillZipList<<<blocks, tpb>>>(zipList, in1, in2);	
//  		
//		// calculate the 0/1 predicate
//		thrust::transform(thrust::host,
//                    zipList,
//                    zipList + listSize,
//                    predicates,
//                    SimpleJoinThrustFilter{});
//
//		// scan the predicate
//  		thrust::exclusive_scan(thrust::host,
//                         predicates,
//                         predicates + listSize,
//                         scatter_indices,
//                         static_cast<IndexType>(0),
//                         thrust::plus<IndexType>());
//
//  		// scatter the true elements
//  		thrust::scatter_if(thrust::host,
//                     zipList,
//                     zipList + listSize,
//                     scatter_indices,
//                     predicates,
//                     zipList, //result,
//                     thrust::identity<IndexType>());
//		
//		const uint64_t output_size = scatter_indices[listSize - 1] + predicates[listSize + 1];
//		free(zipList);
//		free(predicates);
//		free(scatter_indices);
//		
//		return output_size;
//	}

};




	/// selection function or hash function.
	template<typename T> 
	__device__
	inline uint32_t check(const T a, const T b) {
		// TODO currently l is hardcoded. But this should be passed as an 
		// compiletime argument.
		const uint32_t l = 5;
		const T mask = (1ull << l) - 1;
		
		return ((a ^ b) & mask);
	}


	/// Ok after some thinking, i may have a good approach:
	
	/// NOTE: currently assumes that the output list as atleast as big as the input lists.
	/// each thread writes into out[idx] if in1[idx] == in2[idx]
	/// NOTE: kernel assumes 1D thread address. Meaning we have multiple block
	///  		in which 1024 (or less) threads are.
	/// :param out selection bit. Set to 1 at pos idx iff in1[idx] == in2[idc]
	/// 		else set to 0
	/// :param in1 input list 1
	/// :param in2 input list 2
	/// :param stride: offset in which a collision should be searched in the 
	///       	the second list. This is usefull if you want to do the stupid 
	///  		quadratic search.
	template<typename T> 
	__global__
	void transform(uint64_t *out, T *in1, T *in2, const uint64_t stride) {
		// first get the index of the thread;
		const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
		out[idx] = not check<T>(in1[idx], in2[idx + stride]);
	}

#ifdef __CUDA_SUBBYTE_IMMA__
	/// NOTE: needs sm >= 75
	/// :param out selection bit. Set to 1 at pos idx iff in1[idx] == in2[idc]
	/// 		else set to 0
	/// :param in1 input list 1
	/// :param in2 input list 2
	/// :param stride: offset in which a collision should be searched in the 
	///       	the second list. This is usefull if you want to do the stupid 
	///  		quadratic search	template<typename T> 
	__global__
	void transform_tensor(uint64_t *out, T *in1, T *in2, const uint64_t loop) {
		// TODO student

		// Must be multiples of 16 for wmma code to work
		#define MATRIX_M 16384
		#define MATRIX_N 16384
		#define MATRIX_K 16384
		
		// The only dimensions currently supported by WMMA
		const int WMMA_M = 8;
		const int WMMA_N = 8;
		const int WMMA_K = 128;
		
		using TT = experimental::precision::b1;
		
		// Declare the fragments
		typedef wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major> A;
		typedef wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, wmma::col_major> B;
		typedef wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> ACC; 
		typedef wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> C;

		A a_frag;
		B b_frag;
		ACC acc_frag;
		C c_frag;
		wmma::fill_fragment(acc_frag, 0);
		wmma::fill_fragment(c_frag, 0);
		wmma::load_matrix_sync(a_frag, a, 128);
		wmma::load_matrix_sync(b_frag, b, 128);
	
		// Perform the matrix multiplication
		wmma::bmma_sync(c_frag, a_frag, b_frag, c_frag);
		
		// Store the output
		wmma::store_matrix_sync(c, c_frag, 128, wmma::mem_row_major);
	}
#endif

	/// NOTE: the exclusive_scan is computed inplace.
	/// NOTE: out __MUST__ be a device pointer.
	/// computes the exclusive scan over `out`
	/// :param out device pointer to a list
	/// :param listSize
	__host__
	void exclusive_scan(uint64_t *out, const uint64_t listSize) {
		thrust::exclusive_scan(thrust::device, out, out + listSize, out);
	}

	/// NOTE: for a good description and implementation of a scatter_if func
	/// see: https://thrust.github.io/doc/group__scattering_ga39d45ea8d32a2057e941c59abff6912b.html#ga39d45ea8d32a2057e941c59abff6912b
	/// NOTE: currently there is a buffer overflow
	/// The input pointer out contains the exclusive_scan over the selection
	/// bits
	/// :param out device pointer of a list of length the same as number of threads
	/// :param ret pointer to a single integer, counts the number of collisions
	/// selection_if function.
	__global__
	void scatter_if(uint64_t *out, uint64_t *ret) {
		const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
		
		// NOTE: buffer overflow.
		if (out[idx] != out[idx + 1]) {
			__syncthreads();

			// now we know that idx is a valid collison and should be moved to.
			// the position the exclusive_scan computed
			out[out[idx]] = idx;

			// TODO think of something smarter
			atomicAdd((uint32_t *) ret, 1);
		}
	}
	
	/// NOTE: currently this function assumes that we only have elements of 
	///  		size at most of one limb of type T.
	/// RETURN: output list size
	template<typename T> 
	__host__
	uint64_t join(uint64_t *out, T *in1, T *in2, const uint64_t listSize) {
		// first get the index of the thread;
		//const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
			
		uint64_t current_size = 0, *dcurrent_size, acc_size = 0;
		checkCudaErrors(cudaMalloc(&dcurrent_size, sizeof(uint64_t)));
		checkCudaErrors(cudaMemcpy(dcurrent_size, &current_size, sizeof(uint64_t), cudaMemcpyHostToDevice));

		const uint64_t blocks = 1;
		const uint64_t tpb = listSize;

		// because we are doing something stupid (quadratic search) we need to
		// enumerate over all elements.
		for (uint64_t loop = 0; loop < 1 /*listSize*/; loop++) {
			transform<T><<<blocks, tpb>>>(out+acc_size, in1, in2, loop);
			
			checkCudaErrors(cudaDeviceSynchronize());
			
			exclusive_scan(out+acc_size, listSize);
			checkCudaErrors(cudaDeviceSynchronize());
			
			scatter_if<<<blocks, tpb>>>(out+acc_size, dcurrent_size);
			checkCudaErrors(cudaDeviceSynchronize());


			checkCudaErrors(cudaMemcpy(&current_size, dcurrent_size, sizeof(uint64_t), cudaMemcpyDeviceToHost));
			acc_size += current_size;
		}

		std::cout << acc_size << "\n";
		return acc_size;
	}
