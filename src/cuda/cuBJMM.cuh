#pragma once

#include "cuhelper.cuh"
#include "cuMisc.cuh"
#include "cuStern.cuh"

#include <stdint.h>

template<typename T>
class ConfigBJMM {
public:
	/// Construction
	///          Ts      T     S               ▲
	///          ┌┐     ┌─┐   ┌─┐              │
	///          │┼──┐  │ │   │ │              │
	///          └┘  │  │ │   │ │              │              +T
	///              └─►│ ├───► ├──────────────┴───────────────┐
	///          ┌┐     │ │   │ │                              │
	///          │┼─────► ├───► │                              │
	///          └┘     └─┘   └─┘                              │
	///                        ▲                               │
	/// Ts      T       S      │                               │
	/// ┌┐     ┌─┐     ┌─┐     │                               │
	/// │┼────►│ ├────►│ │     │                               │
	/// └┘     │ │     │ │     │    +it                        │
	///     ┌─►│ ├───┐ │ ├─────┴──────┐                        │
	/// ┌┐  │  │ │   └►│ │            │                        │
	/// │┼──┘  │ │     │ │            │                        │
	/// └┘     └─┘     └▲┘            │                        │
	/// Te              │─────────────┼───────────┐            │
	///                 │             │           │            │
	///              ┌──┴──┐       ┌──┴──┐     ┌──┴──┐      ┌─ ┴─ ─┐
	///              │     │       │     │                  │
	///              │     │       │     │     │     │             │
	///              │     │       │     │                  │
	///              └─────┘       └─────┘     └── ──┘      └─ ── ─┘



	// instance parameters
	const uint32_t n;
	const uint32_t k;
	const uint32_t w;
	const uint32_t p;
	const uint32_t l;
	const uint32_t l1;

	// Additional parameters
	const uint32_t M4RI_factor      = sizeof(uint64_t) / sizeof(T);
	const uint32_t matrix_rows      = k+l+1;        // CPU Matrix rows of HT
	const uint32_t matrix_width     = (n-k+63)/64;  // CPU Matrix width in limbs of HT

	const uint32_t gpu_matrix_width = M4RI_factor * matrix_width;
	const uint32_t gpu_l_limbpos    = (n-k-1)/(sizeof(T)*8); //IMPORTANT: must be 0 aligned

	const uint64_t kl_size          = k + l;
	const uint64_t klpart_size      = cceil(float(kl_size) / 2.);
	const uint64_t numb_p_out_of_k  = bc(uint64_t(klpart_size), uint64_t(p));
	const uint64_t l_table_size     = uint32_t(1) << l;

	const uint32_t nr_streams = 1;
	const uint32_t nr_threads = 1;

	constexpr ConfigBJMM(const uint32_t n, const uint32_t k, const uint32_t l, const uint32_t l1,
	                         const uint32_t w, const uint32_t p) :
			n(n), k(k), l(l), l1(l1), w(w), p(p) {};

	constexpr ConfigBJMM(ConfigBJMM &c) = default;


	// return the number of loops needed in expectation
	constexpr uint64_t Loops() {
		return bc(n, w) / (bc(n-k-l, w-p) * bc(k+l, p));
	}

	/// print information about the configuration
	void print() const {
		std::cout << "n: " << n
		          << ", k: " << k
		          << ", l: " << l
		          <<", (k+l)/2: " << klpart_size
		          << ", n-k:" << n-k
		          << ", p: " << p
		          << ", M4RI_factor: " << M4RI_factor
		          << ", gpu_matrix_width: " << gpu_matrix_width
		          << ", M4RI width: " <<  matrix_width << "\n"
		          << ", kl_size: " << kl_size << "\n"
		          << ", klpart_size: " << klpart_size << "\n"
		          << ", numb_p_out_of_k: " << numb_p_out_of_k << "\n"
		          << ", l_table_size: " << l_table_size << "\n"
		          << ", gpu_l_limbpos: " << gpu_l_limbpos << "\n"
		          << ", n-k-l: " << n-k-l << "\n";
	}
};

//template<const ParameterDumer &param, const ConfigDumer<param> &config>
class ParamsBJMM {
public:
	// Types needed
	using CombinationsType  = uint32_t;

	// Cuda parameters
	thrust::host_vector<CombinationsType> *combinations;
	uint64_t blocks = 0;
	uint64_t tpb = 0;

	uint64_t tableSize = 0;
	uint64_t tableReps = 0;

	// CPU Parameters
	mzd_t *AT = nullptr;
	mzd_t *A = nullptr;
	mzd_t *syndrom = nullptr;
	mzd_t *syndromT = nullptr;

	ParamsBJMM (thrust::host_vector<CombinationsType> *b,
	           uint64_t blocks,
	           uint64_t tpb,
	           uint64_t tableSize,
	           uint64_t tableReps,
	           mzd_t *AT,
	           mzd_t *syndrome) :
			combinations(b), blocks(blocks), tpb(tpb), tableSize(tableSize), tableReps(tableReps), AT(AT), syndrom(syndrome) {};
};


template<typename T, typename BucketType,
		// instance parameters
		const uint32_t n, const uint32_t k, const uint32_t l, const uint32_t l1, const uint32_t w, const uint32_t p,
		const uint32_t gpu_width,      // width in limbs of the 'columns' of dH
		const uint32_t lpart_limbpos,  // limb position of the l part limb, zero indexed
		const size_t nr_combinations   // number of elements in dC
>
__device__ void bjmm_step1(const T *__restrict__ dHH,           // parity check matrix
                           const T *__restrict__ dss,           // syndrome
                           const uint16_t *__restrict__ dC,     // array containing bit pos of "1" in error vector
                           T *__restrict__ dS_1,                // output: sorting array, left join
                           BucketType *__restrict__ dTs_1,      // output: sorting array, left join
                           const uint64_t itarget               // input: left
) {
	constexpr uint32_t T_bitwidth   = sizeof(T)*8;
	constexpr bool target_singlelimb= ((n-k-l1)/T_bitwidth) == ((n-k)/T_bitwidth);
	const uint32_t block_number     = blockDim.x * gridDim.x;
	uint32_t index                  = threadIdx.x + blockIdx.x * blockDim.x;

	// copy the syndrome into the shared memory
	__shared__ T s[gpu_width],          // syndrome
	dH[gpu_width*(k+l)];    // parity check matrix
	if (threadIdx.x < gpu_width)
		s[threadIdx.x] = dss[threadIdx.x];
	// TODO probably not correct
	if (threadIdx.x < gpu_width*(k+l))
		dH[threadIdx.x] = dHH[threadIdx.x];
	__syncthreads();

	// base index of this thread
	const uint32_t f = index*(lpart_limbpos+1);

	// for every element in the change list
	while(index < nr_combinations) {
		// TODO den ganzen kram in ein unterfunktionen die das atom add beschleunigen
		// TODO not add the target
		// calculate H2*e2 + it
		matrix_vector_mult<T, gpu_width, p, target_singlelimb>(dS_1 + f, dH, dC, index, 0);

		// Update Bucket
		BucketType a = extract_l_bit<T, BucketType, n, k, l, w, p, gpu_width, lpart_limbpos>(dS_1, index);

		atomicAdd((uint32_t *)&dTs_1[a], 1);
		index += block_number;
	}
}


template<typename T, typename BucketType,
		// instance parameters
		const uint32_t n, const uint32_t k, const uint32_t l, const uint32_t l1, const uint32_t w, const uint32_t p,
		const uint32_t gpu_width,      // width in limbs of the 'columns' of dH
		const uint32_t lpart_limbpos,  // limb position of the l part limb, zero indexed
		const size_t nr_combinations   // number of elements in dC
>
__device__ void bjmm_step2(const T *__restrict__ dS_1,      // output: counting sort helper array
                           BucketType *__restrict__ dT_1,   // output: counting sort helper array
                           BucketType *__restrict__ dTe_1   // output: counting sort helper array
) {
	const uint32_t block_number     = blockDim.x * gridDim.x;
	uint32_t index                  = threadIdx.x + blockIdx.x * blockDim.x;

	while(index < nr_combinations) {
		// TODO  sub function die irgendwie die atomic adds beschleunigt
		// extract l bit
		BucketType a1 = extract_l_bit<T, BucketType, n, k, l, w, p, gpu_width, lpart_limbpos>(dS_1, index);

		// returns the old value.
		BucketType o1 = atomicAdd((uint32_t *)(dTe_1 + a1), 1);
		assert(o1 < nr_combinations);
		dT_1[o1] = index;

		index += block_number;
	}
}


template<typename T, typename BucketType,
         // instance parameters
         const uint32_t n, const uint32_t k, const uint32_t l, const uint32_t l1, const uint32_t w, const uint32_t p,
         const uint32_t gpu_width,      // width in limbs of the 'columns' of dH
         const uint32_t lpart_limbpos,  // limb position of the l part limb, zero indexed
         const size_t nr_combinations   // number of elements in dC
         >
__device__ void bjmm_step1_mutli(const T *__restrict__ dHH,              // parity check matrix
                                    const T *__restrict__ dss,           // syndrome
                                    const uint16_t *__restrict__ dC,     // array containing bit pos of "1" in error vector
                                    T *__restrict__ dS_1,                // output: sorting array, left join
                                    BucketType *__restrict__ dTs_1,      // output: sorting array, left join
                                    T *__restrict__ dS_2,                // output: sorting array, right join
                                    BucketType *__restrict__ dTs_2,      // output: sorting array, right join
                                    const uint64_t target,               // input: right target
                                    const uint64_t itarget               // input: left
							) {
	constexpr uint32_t T_bitwidth   = sizeof(T)*8;
	constexpr bool target_singlelimb= ((n-k-l1)/T_bitwidth) == ((n-k)/T_bitwidth);
	const uint32_t block_number     = blockDim.x * gridDim.x;
	uint32_t index                  = threadIdx.x + blockIdx.x * blockDim.x;

	// copy the syndrome into the shared memory
	__shared__ T s[gpu_width],          // syndrome
	            dH[gpu_width*(k+l)];    // parity check matrix
	if (threadIdx.x < gpu_width)
		s[threadIdx.x] = dss[threadIdx.x];
	// TODO probably not correct
	if (threadIdx.x < gpu_width*(k+l))
		dH[threadIdx.x] = dHH[threadIdx.x];
	__syncthreads();

	// base index of this thread
	const uint32_t f = index*(lpart_limbpos+1);

	// for every element in the change list
	while(index < nr_combinations) {
		// TODO den ganzen kram in ein unterfunktionen die das atom add beschleunigen
		// calculate H2*e2 + it
		matrix_vector_mult<T, gpu_width, p, target_singlelimb>(dS_1 + f, dH, dC, index, itarget);
		matrix_vector_mult<T, gpu_width, p, target_singlelimb>(dS_2 + f, dH, dC, index, target);

		// Update Bucket
		BucketType a = extract_l_bit<T, BucketType, n, k, l, w, p, gpu_width, lpart_limbpos>(dS_1, index);
		BucketType b = extract_l_bit<T, BucketType, n, k, l, w, p, gpu_width, lpart_limbpos>(dS_2, index);

		atomicAdd((uint32_t *)&dTs_1[a], 1);
		atomicAdd((uint32_t *)&dTs_2[b], 1);
		index += block_number;
	}
}


template<typename T, typename BucketType,
         // instance parameters
         const uint32_t n, const uint32_t k, const uint32_t l, const uint32_t l1, const uint32_t w, const uint32_t p,
         const uint32_t gpu_width,      // width in limbs of the 'columns' of dH
		 const uint32_t lpart_limbpos,  // limb position of the l part limb, zero indexed
		 const size_t nr_combinations   // number of elements in dC
         >
__device__ void bjmm_step2_multi(const T *__restrict__ dS_1,      // output: counting sort helper array
                                 BucketType *__restrict__ dT_1,   // output: counting sort helper array
                                 BucketType *__restrict__ dTe_1,  // output: counting sort helper array
                                 const T *__restrict__ dS_2,      // output: counting sort helper array
                                 BucketType *__restrict__ dT_2,   // output: counting sort helper array
                                 BucketType *__restrict__ dTe_2   // output: counting sort helper array
	) {
	const uint32_t block_number     = blockDim.x * gridDim.x;
	uint32_t index                  = threadIdx.x + blockIdx.x * blockDim.x;

	while(index < nr_combinations) {
		// TODO  sub function die irgendwie die atomic adds beschleunigt
		// extract l bit
		BucketType a1 = extract_l_bit<T, BucketType, n, k, l, w, p, gpu_width, lpart_limbpos>(dS_1, index);
		BucketType a2 = extract_l_bit<T, BucketType, n, k, l, w, p, gpu_width, lpart_limbpos>(dS_2, index);

		// returns the old value.
		BucketType o1 = atomicAdd((uint32_t *)(dTe_1 + a1), 1);
		BucketType o2 = atomicAdd((uint32_t *)(dTe_2 + a2), 1);
		assert(o1 < nr_combinations);
		assert(o2 < nr_combinations);
		dT_1[o1] = index;
		dT_2[o2] = index;

		index += block_number;
	}
}


template<typename T,                    // limb type of the syndrome/parity check matrix
         typename T2,                   // limb type of the datastruct
         typename BucketType,           //
         // instance parameters
         const uint32_t n, const uint32_t k, const uint32_t l, const uint32_t l1, const uint32_t w, const uint32_t p,
		 const uint32_t gpu_width,      // width in limbs of the 'columns' of dH
		 const uint32_t lpart_limbpos,  // limb position of the l part limb, zero indexed
		 const size_t nr_combinations   // number of elements in dC
         >
__device__ void bjmm_step3(uint16_t *__restrict__ de,               // output error positions
                           const T *__restrict__ dH,               // parity check matrix
                           const T *__restrict__ ds,               // syndrome
                           const uint16_t *__restrict__ dC,        // table
						   // lvl1 base list
                           const T *__restrict__ dS_1,             // radix/counting sort algorithm helper
                           const BucketType *__restrict__ dT_1,    // radix/counting sort algorithm helper
                           const BucketType *__restrict__ dTs_1,   // radix/counting sort algorithm helper
                           const BucketType *__restrict__ dTe_1,   // radix/counting sort algorithm helper
						   // lvl2 intermediate list
                           T2 *__restrict__ dS_2,                  // radix/counting sort algorithm helper
                           BucketType *__restrict__ dT_2,          // radix/counting sort algorithm helper
                           BucketType *__restrict__ dTs_2,         // radix/counting sort algorithm helper
                           BucketType *__restrict__ dTe_2,         // radix/counting sort algorithm helper
                           const BucketType itarget
                           ) {
	constexpr uint32_t T_bitwidth   = sizeof(T) * 8u;
	constexpr uint32_t nk           = n-k;
	constexpr uint32_t nkl          = n-k-l;
	constexpr uint32_t nkl1         = n-k-l1;
	ASSERT(l<T_bitwidth);

	// if set to `true`, then `l1` only spans across a single limb
	constexpr bool l1_singlelimb    = (nkl1/T_bitwidth) == (nk/T_bitwidth);
	constexpr bool l_singlelimb     = (nkl/T_bitwidth) == (nkl1/T_bitwidth);

	constexpr uint32_t l1_shift     = nkl1%T_bitwidth;
	constexpr uint32_t l_shift      = nkl%T_bitwidth;

	constexpr uint32_t l1_mask      = 1; // TODO
	constexpr uint32_t l_mask       = 1;

	// needed
	constexpr uint32_t klpart_size  = cceil(double(k+l)/2.);

	const uint32_t block_number     = blockDim.x * gridDim.x;
	uint32_t index                  = threadIdx.x + blockIdx.x * blockDim.x;

	// TODO copy syndrome and parity check into shared mem

	// for every element in L2
	while(index < nr_combinations) {
		T x = 0; // will hold l1 bits in the end
		if constexpr (l_singlelimb) {
			#pragma unroll
			for (uint32_t j = 0; j < p; j++) {
				x ^= dH[((klpart_size + dC[p*index + j]) * gpu_width) + lpart_limbpos];
			}

			// now shift down x, After that x contains the l part
			x >>= l1_shift;
			x ^= itarget;
		} else {
			// TODO
		}

		// for every match S1/L1
		for (uint32_t j = dTs_1[x]; j < dTe_1[x]; j++) {
			const BucketType index1 = dT_1[j];
			const T f = index1*gpu_width;

			// for every match save the result (k+l) bits in S_2
			#pragma unroll
			for(uint32_t i = 0; i < gpu_width; i++) {
				dS_2[index + i] = dS_1[f + i];

				#pragma unroll
				for (uint32_t u = 0; u < p; u++) {
					dS_2[f + i] ^= dH[((klpart_size + dC[p*index + j]) * gpu_width) + i];
				}
			}

			// increment the helper struct for the counting sort
			BucketType a = extract_l_bit<T, BucketType, n,k,l,l1>(dS_2 + f, index);
			atomicAdd((uint32_t *)&dTs_2[a], 1);
		}

		index += block_number;
	}
}

template<typename T,
		 typename T2,                   // limb type of the datastruct
		 typename BucketType,           //
		 // instance parameters
		 const uint32_t n, const uint32_t k, const uint32_t l, const uint32_t l1, const uint32_t w, const uint32_t p,
		 const uint32_t gpu_width,      // width in limbs of the 'columns' of dH
		 const uint32_t lpart_limbpos,  // limb position of the l part limb, zero indexed
		 const size_t nr_combinations   // number of elements in dC
         >
__global__ void BJMM_Step4(uint16_t *__restrict__ de,          // output bits. NOT the final error
                           const T *__restrict__ dH,           // parity check matrix
                           const T *__restrict__ ds,           // syndrome
                           const uint16_t *__restrict__ dC,    // change list
						   // base list L1 sorting helper
		                   T *__restrict__ dS_1,               //
		                   BucketType *__restrict__ dT_1,      //
		                   BucketType *__restrict__ dTs_1,     //
		                   BucketType *__restrict__ dTe_1,     //
						   // intermediate list sorting helper
		                   T *__restrict__ dS_2,               //
		                   BucketType *__restrict__ dT_2,      //
		                   BucketType *__restrict__ dTs_2,     //
		                   BucketType *__restrict__ dTe_2,     //

		                   const T2 itarget, const T2 target,  //
                           uint32_t *compare) {
	constexpr uint32_t T_bitwidth   = sizeof(T) * 8u;
	constexpr uint32_t nk           = n-k;
	constexpr uint32_t nkl          = n-k-l;
	constexpr uint32_t nkl1         = n-k-l1;
	ASSERT(l<T_bitwidth);

	// if set to `true`, then `l1` only spans across a single limb
	constexpr bool l1_singlelimb    = (nkl1/T_bitwidth) == (nk/T_bitwidth);
	constexpr bool l_singlelimb     = (nkl/T_bitwidth) == (nkl1/T_bitwidth);

	constexpr uint32_t l1_shift     = nkl1%T_bitwidth;
	constexpr uint32_t l_shift      = nkl%T_bitwidth;

	constexpr uint32_t l1_mask      = 1; // TODO
	constexpr uint32_t l_mask       = (BucketType(1u) << l) - 1u;

	// needed
	constexpr uint32_t klpart_size  = cceil(double(k+l)/2.);

	constexpr uint32_t weightt      = w - (2u*p);
	const uint32_t block_number     = blockDim.x * gridDim.x;
	uint32_t index                  = threadIdx.x + blockIdx.x * blockDim.x;

	// TODO copy syndrome and parity check into shared mem

	// for every element in L4
	while(index < nr_combinations) {
		T xl1 = 0; // will hold l1 bits
		T xl2 = 0; // will hold l-l1 bits
		if constexpr (l_singlelimb) {
			#pragma unroll
			for (uint32_t j = 0; j < p; j++) {
				xl1 ^= dH[((klpart_size + dC[p*index + j]) * gpu_width) + lpart_limbpos];
			}

			// now shift down x, After that x contains the l part
			xl2 = xl1 >> l_shift;
			xl1 >>= l1_shift;
			xl2 ^= target;
			xl2 &= l_mask;
		} else {
			// TODO
		}

		// for every match in S1/L1
		for (uint32_t i1 = dTs_1[xl1]; i1 < dTe_1[xl1]; i1++) {
			const BucketType index1 = dT_1[i1];
			const T f1              = index1*gpu_width;

			// for every match in the intermediate list
			for (uint32_t i2 = dTs_2[xl2]; i2 < dTe_1[xl2]; i2++) {
				const BucketType index2 = dT_2[i2];
				const T f2              = index2*gpu_width;

				// now we need to rebuild the whole thing
				uint32_t weight = 0;
				T esp = 0;

				// TODO early exit
				#pragma unroll
				for (uint32_t i = 0; i < lpart_limbpos; i++) {
					#pragma unroll
					for (uint32_t j = 0; j < p; j++) {
						esp ^= dH[((klpart_size + dC[p*index + j]) * gpu_width) + i];
					}

					// TODO correct indicies
					esp ^= dS_1[index1 + i1] ^ dS_2[index2 + i2];
					weight += __popc(esp);
				}

				// did we really found the solution
				if (weight <= weightt) {
					// atomic check/increment so no other thread overwrites the solution
					if( 0 != atomicCAS(compare, 0, 1))
						return;

					for (uint32_t i = 0; i < p; i++) {
						de[2*i + 0] = nkl + klpart_size + dC[p*index1  + i1];
						de[2*i + 1] = nkl +             + dC[p*index2  + i2];
					}
				}
			}
		}

		// exit the whole process if we found the solution
		if (*compare > 0)
			return;

		// if we did not found the solution, retry
		index += block_number;
	}
};





template<typename T,
		 typename T2,                   // limb type of the datastruct
		 typename BucketType,           //
		 // instance parameters
		 const uint32_t n, const uint32_t k, const uint32_t l, const uint32_t l1, const uint32_t w, const uint32_t p,
		 const uint32_t gpu_width,      // width in limbs of the 'columns' of dH
		 const uint32_t lpart_limbpos,  // limb position of the l part limb, zero indexed
		 const size_t nr_combinations,  // number of elements in dC

		typename Iterator>
__global__ void KernelBJMM(uint16_t *__restrict__ pde,          // output bits. NOT the final error
                           const T *__restrict__ pdH,          // parity check matrix
                           const T *__restrict__ pds,          // syndrome
                           const uint16_t *__restrict__ pdC,   // change list
                           // base list L1 sorting helper
                           T *__restrict__ pdS_1,              //
                           BucketType *__restrict__ pdT_1,     //
                           BucketType *__restrict__ pdTs_1,    //
                           BucketType *__restrict__ pdTe_1,    //
						   // intermediate list sorting helper
                           T *__restrict__ pdS_2,              //
                           BucketType *__restrict__ pdT_2,     //
                           BucketType *__restrict__ pdTs_2,    //
                           BucketType *__restrict__ pdTe_2,    //

                           const T2 itarget, const T2 arget,   // input: left
						   //
                           Iterator dTb,  Iterator dTe,        //
                           Iterator dTsb, Iterator dTse,       //
                           Iterator dTeb, Iterator dTee,       //
                           uint32_t *compare)     //
{
	thrust::fill(thrust::device, dTsb, dTse, 0);
	thrust::fill(thrust::device, dTeb, dTee, 0);
	thrust::fill(thrust::device, dTb, dTe, 0); // TODO maybe remove
	__syncthreads();

	bjmm_step1<T, BucketType, n, k, l, l1, w, p, gpu_width, lpart_limbpos, nr_combinations>
	        (pdH, pds, pdC, pdS_1, pdTs_1, 0);
	__syncthreads();

	// TODO
	thrust::copy(thrust::device, dTsb, dTse, dTeb);
	__syncthreads();
	thrust::exclusive_scan(thrust::device, dTsb, dTse, dTsb, 0);
	__syncthreads();

	bjmm_step2<T, BucketType, n, k, l, l1, w, p, gpu_width, lpart_limbpos, nr_combinations>
	        (pdS_1, pdT_1, pdTe_1);
	__syncthreads();

	bjmm_step3<T, T2, BucketType, n, k, l, l1, w, p, gpu_width, lpart_limbpos, nr_combinations>
	        (pde, pdH, pds, pdC, pdS_1, pdT_1, pdTs_1, pdTe_1, pdS_2, pdT_2, pdTs_2, pdTe_2);
	__syncthreads();

	// sort the intermediate list
	// TODO die intermdediate list muss nicht nr_combinations gross sein
	bjmm_step2<T, BucketType, n, k, l, l1, w, p, gpu_width, lpart_limbpos, nr_combinations>
			(pdS_2, pdT_2, pdTe_2);
	__syncthreads();


}


template<typename T, const ConfigBJMM<T> &config>
void* BJMM_Thread(void *par) {
	// const values
	constexpr static uint32_t n = config.n; // instance parameter
	constexpr static uint32_t k = config.k; // instance parameter
	constexpr static uint32_t w = config.w; // instance parameter
	constexpr static uint32_t l = config.l; // instance parameter
	constexpr static uint32_t l1= config.l1;// instance parameter
	constexpr static uint32_t p = config.p; // instance parameter

	constexpr static uint32_t gpu_width = config.gpu_matrix_width; // number of limbs for one column in the parity check matrix
	constexpr static uint32_t gpu_l_limbpos = config.gpu_l_limbpos;   //

	// internal Datatypes
	using BucketType            = LogTypeTemplate<MAX(l1, MAX(l, 32))>;
	using CombinationsType      = uint32_t;

	//
	ParamsBJMM *params = (ParamsBJMM *)par;

	// CPU stuff
	mzd_t *syndrom_tmp  = mzd_init(params->syndrom->ncols, params->syndrom->nrows);
	mzd_t *syndromT_tmp = mzd_init(params->syndrom->nrows, params->syndrom->ncols);
	mzd_t *e            = mzd_init(1, n);
	mzd_t *eT           = mzd_init(n, 1);
	mzd_t *wH           = matrix_init(n-k, n+1);
	mzd_t *wHT          = mzd_init(wH->ncols, wH->nrows);
	mzd_t *H           =  mzd_init(n - k, k+l+1);
	mzd_t *HT           = mzd_init(H->ncols, H->nrows);
	mzp_t *permutation  = mzp_init(n);

	//matrix_concat(wH, params->A, params->syndromT);
	mzd_t *tmp = matrix_concat(nullptr, params->A, params->syndromT);
	mzd_copy(wH, tmp);
	mzd_free(tmp);

	// GPU STUFF
	thrust::device_vector<T> de(config.M4RI_factor*params->A->width);
	checkCudaErrors(cudaPeekAtLastError());

	// Copy as much as possible on the cuda device. If it is not possible to pack everything on the device
	// it will be chunked. Also, every CPU thread needs its own `dCombination` vector. Otherwise, they could not concurrently work.
	thrust::device_vector<CombinationsType> dCombinations(p*params->tableSize);
	thrust::copy(params->combinations->begin(), params->combinations->begin()+(p*params->tableSize), dCombinations.begin());

	bool found_e        = false, set_e = false, *dFound_e = nullptr;
	uint32_t *dCompare  = nullptr;
	uint32_t zero       = 0;

	checkCudaErrors(cudaMalloc(&dCompare, sizeof(uint32_t)));
	checkCudaErrors(cudaMalloc(&dFound_e, sizeof(bool)));
	checkCudaErrors(cudaMemcpy(dCompare, &zero,    sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dFound_e, &found_e, sizeof(bool),     cudaMemcpyHostToDevice));

	// register memory for asynchronous execution of the cuda streams.
	//cudaHostRegister(HT->rows[0], HT->width * HT->nrows * sizeof(uint64_t), 0);
	//cudaHostRegister(&found_e, sizeof(bool), 0);

	// GPU stuff + Pointer abstraction
	T* pdHT;
	checkCudaErrors(cudaMalloc(&pdHT, config.M4RI_factor * config.matrix_rows * config.matrix_width * sizeof(T)));
	T* pdsT                 = &pdHT[config.M4RI_factor * (config.matrix_rows - 1) * config.matrix_width];
	T* pde                  = thrust::raw_pointer_cast(de.data());
	CombinationsType* pdC   = thrust::raw_pointer_cast(dCombinations.data());

	// Init the buckets for the radix sort.
	thrust::device_vector<T>          dS(params->tableSize*config.gpu_matrix_width);
	thrust::device_vector<BucketType> dT(params->tableSize);
	thrust::device_vector<BucketType> dTs(config.l_table_size);
	thrust::device_vector<BucketType> dTe(config.l_table_size);
	checkCudaErrors(cudaPeekAtLastError());

	T *pdS = thrust::raw_pointer_cast(dS.data());;
	BucketType *pdT = thrust::raw_pointer_cast(dT.data());
	BucketType *pdTs= thrust::raw_pointer_cast(dTs.data());
	BucketType *pdTe= thrust::raw_pointer_cast(dTe.data());


	// timing stuff
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float kernel_runtime = 0., total_kernel_runtime = 0.;

	double time     = (double)clock();
	uint64_t loops  = 0;
	uint32_t m4ri_k = matrix_opt_k(n - k, n);
	customMatrixData *matrix_data = init_matrix_data(wH->ncols);

	while (!found_e) {
		loops += 1;

		// create for each stream its own gaus elimination
		matrix_create_random_permutation(wH, wHT, permutation);
		matrix_echelonize_partial_plusfix(wH, m4ri_k, n-k-l, matrix_data, 0, n - k - l, 0, permutation);

		mzd_submatrix(H, wH, 0, n-k-l, n-k, n+1);
		matrix_transpose(HT, H);
		// matrix_transpose(wHT, wH);

		cudaEventRecord(start);

		// TODO opt memcopy in this functions
		MatrixToCuda<T>(pdHT, HT);
		checkCudaErrors(cudaPeekAtLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		kernel_runtime = 0.;
		//KernelBJMM<T>
		//	<<<params->blocks, params->tpb>>>
		//		();
		checkCudaErrors(cudaPeekAtLastError());
		checkCudaErrors(cudaDeviceSynchronize());


		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&kernel_runtime, start, stop);
		total_kernel_runtime += kernel_runtime;
		//checkCudaErrors(cudaStreamSynchronize(stream[cs]));
		//checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaMemcpy(&found_e, dFound_e, sizeof(bool), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaDeviceSynchronize());

		if (found_e && !set_e) {
			// this must be atomic in a multithreading environment
			set_e = true;

			// apply back perm
			mzd_t *tmpe = mzd_init(1, n);
			CudaToMatrix<T>(tmpe, de);
			for (uint32_t i = 0; i < n; i++) {
				mzd_write_bit(e, 0, permutation->values[i], mzd_read_bit(tmpe, 0, i));
			}

			mzd_print(tmpe);
			mzd_print(e);
			mzd_free(tmpe);
		}
		if ((loops%10000) == 0) {
			const uint64_t cloops = loops;
			const double time2 = ((double)clock() -  time)/CLOCKS_PER_SEC;
			const double ktime = total_kernel_runtime/1e3;
			std::cout << "loops: " << cloops << ", loops/s:" << cloops/time2 << ", kernel runtime: " << ktime << "s (" << ktime/time2*100. << "%) , runtime: " << time2 << "s" << "\n" << std::flush;
		}
	}

	time = ((double)clock() - time)/CLOCKS_PER_SEC;
	total_kernel_runtime /= 1e3;
	std::cout << "runtime: "  << time << "s, loops: " << loops << ", total kernel runtime: " << total_kernel_runtime << "s (" << total_kernel_runtime/time*100. << "%), loops/kernel s: " << loops/total_kernel_runtime << ", loops/s: " << loops/time << "\n";

	// check correctness
	mzd_transpose(eT, e);
	mzd_mul_naive(syndrom_tmp, params->A, eT);
	mzd_transpose(syndromT_tmp, syndrom_tmp);

	//mzd_print(A);
	std::cout << "s is:\t\t";
	mzd_print(syndromT_tmp);
	std::cout << "s should:\t";
	mzd_print(params->syndrom);


	for (int i = 0; i < params->syndrom->ncols; ++i) {
		if(mzd_read_bit(params->syndrom, 0, i) != mzd_read_bit(syndrom_tmp, i, 0)) {
			std::cout << "error not correct\n";
			break;
		}
	}

	// CPU
	mzd_free(wH);
	mzd_free(wHT);
	mzd_free(HT);
	mzd_free(H);
	mzp_free(permutation);
	mzd_free(syndrom_tmp);
	mzd_free(syndromT_tmp);
	mzd_free(e);
	mzd_free(eT);

	checkCudaErrors(cudaFree(dCompare));
	checkCudaErrors(cudaFree(dFound_e));
	checkCudaErrors(cudaFree(pdHT));
	//cudaStreamDestroy(stream[i]);
	//cudaHostUnregister(HT->rows[0]);
	//cudaHostUnregister(&found_e);

	return nullptr;
}


/// calculates the mem consumptions in bytes
/// \tparam param	parameter set of the instance
/// \tparam config	additional parameters needed to run the code
/// \param sizelist	size of the list to enumerate. Allows for Stern or Dumer
/// \return	approx bytes the algorithm needs.
template<typename T, const ConfigBJMM<T> &config>
uint64_t calc_bjmm_mem_consumption(const size_t sizelist) {
	// TODO generalize
	using BucketType            = LogTypeTemplate<MAX(config.l, 32)>;
	using CombinationsType      = uint16_t;
	const uint64_t mem = (sizelist*sizeof(CombinationsType)) + (sizelist*sizeof(T)) +   // T
						 (sizelist*config.gpu_matrix_width*sizeof(BucketType)) +        // S
						 ((uint32_t(1) << config.l)*sizeof(BucketType)*2);              // Ts+Te
	return mem;
}

template<typename T, const ConfigBJMM<T> &config>
int CudaBJMM() {
	// TODO generalize
	using BucketType            = LogTypeTemplate<MAX(config.l, 32)>;
	using CombinationsType      = uint32_t;

	srand(time(nullptr));
	config.print();

	// Try to allocate as much as possible in one thread and pass it to the other threads
	thrust::host_vector<CombinationsType> combinations;
	double time = (double)clock();
	auto numb_p_out_of_k_calc = calc_p_combinations<CombinationsType>(combinations, config.klpart_size, config.p);
	time = ((double)clock() - time)/CLOCKS_PER_SEC;
	if(config.numb_p_out_of_k != numb_p_out_of_k_calc) {
		std::cout << "ERROR: not correct combinations Dumer\n";
		std::cout << config.numb_p_out_of_k << " vs " << numb_p_out_of_k_calc << "\n";
		return -1;
	}

	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, 0);

	const uint64_t max_mem        = prop.totalGlobalMem;
	const uint64_t max_needed_mem = calc_bjmm_mem_consumption<T, config>(config.numb_p_out_of_k);
	const uint64_t tableReps      = ceil(double(max_needed_mem)/double(max_mem));
	const uint64_t tableSize      = config.numb_p_out_of_k/tableReps;   // size per repetition => Actually number of elements on the device
	const uint64_t needed_mem     = calc_bjmm_mem_consumption<T, config>(tableSize);
	assert(tableReps > 0);

	//const uint32_t number_sm    = prop.multiProcessorCount;
	constexpr uint64_t tpb        = MIN(256, config.numb_p_out_of_k);
	const uint64_t blocks         = MAX(1, tableSize/tpb); // take one extra block, s.t. we always have enough threads.
	const uint64_t nr_gpu_threads = blocks * tpb;
	const double rounding         = 1024.*1024.;

	// Some logging
	std::cout << "Precomputation took:  " << time << "s\n";
	std::cout << "max_mem_avail:        " << max_mem << "B = " << max_mem/rounding << "MB\n";
	std::cout << "max_needed_mem:       " << max_needed_mem << "B = " << max_needed_mem/rounding << "MB\n";
	std::cout << "needed mem:           " << needed_mem << "B = " << (needed_mem/rounding) << "MB\n";
	std::cout << "numb_p_out_of_k_calc: " << numb_p_out_of_k_calc << "\n";
	std::cout << "tableSize:            " << tableSize << "\n";
	std::cout << "tableReps:            " << tableReps << "\n";
	std::cout << "starting:             " << config.nr_threads << " threads with each: " << blocks << " blocks with each " << tpb << ", in total: : " << nr_gpu_threads << "\n";

	ParamsBJMM params (&combinations,blocks, tpb,tableSize, tableReps, mzd_from_str(n, n-k, h), mzd_from_str(1, n-k, s));
	params.A        = mzd_transpose(nullptr, params.AT);
	params.syndromT = mzd_transpose(nullptr, params.syndrom);

	BJMM_Thread<T, config>((void *)&params);
	cudaDeviceReset();


	// free everything we allocated
	mzd_free(params.syndrom);
	mzd_free(params.syndromT);
	mzd_free(params.A);
	mzd_free(params.AT);
	return 1;
}
