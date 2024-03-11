#ifndef DECODING_ISD_H
#define DECODING_ISD_H


#include "helper.h"
#include "combination/chase.h"
#include "simd/simd.h"
#include "popcount/popcount.h"
#include "math/bc.h"
#include "container/hashmap.h"
#include "nn/nn.h"
#include "matrix/matrix.h"
#include "element.h"

using namespace cryptanalysislib;

struct ConfigISD {
	// instance parameters
	const uint32_t n=0;   // code length
	const uint32_t k=0;   // code size
	const uint32_t q=0;   // field size
	const uint32_t w=0;   // code weight
	const uint32_t p=0;   // weight in the base lists. In this case the weight which is
	// mapped to by the collision functions
	const uint32_t l=0;   // bits to match on by the two collision functions

	// opt m4ri c
	const uint32_t c = 0;

	// inner threads
	const uint32_t threads = 0;

	// overlap
	const uint32_t epsilon = 0;

	/// if true only the `packed` versions of the data containers are used
	const bool packed = true;
	const bool simd = false; // TODO enable
	const bool parity_row = q == 2;
	const bool doom = false; // TODO

	const uint32_t q_bits = bits_log2(q);

	// if this value is set to a different value then -1 the program will end the
	// computation after `loops` permutation, regardless if it found the solution or not
#ifdef USE_LOOPS
	const uint64_t loops        = USE_LOOPS;
#else
	const uint64_t loops        = uint64_t(-1);
#endif

	// print every `print_loops` iterations a status update of the program
#ifdef PRINT_LOOPS
	const uint64_t print_loops  = PRINT_LOOPS;
#else
	const uint64_t print_loops  = 1ul << 12u;
#endif

	void print() const noexcept {
		std::cout << "{ \"n\": " << unsigned(n)
				  << ", \"k\": " << unsigned(k)
				  << ", \"q\": " << unsigned(q)
				  << ", \"w\": " << unsigned(w)
				  << ", \"l\": " << unsigned(l)
				  << ", \"n-k\":" << unsigned(n-k)
				  << ", \"p\": " << unsigned(p)
		          << ", \"c\": " << unsigned(c)
			   	  << ", \"threads\": " << unsigned(threads)
				  << ", \"epsilon\": " << unsigned(epsilon)
				  << ", \"packed\": " << unsigned(packed)
				  << ", \"simd\": " << unsigned(simd)
				  << ", \"parity_row\": " << unsigned(parity_row)
				  << ", \"doom\": " << unsigned(doom)
			   	  << ", \"loops\": " << unsigned (loops)
			   	  << ", \"print_loops\": " << unsigned (print_loops)
				  << "}" << std::endl;
	}
};

template<typename LimbType, const ConfigISD &config>
class ISDInstance {
public:
	constexpr static bool packed = config.packed;
	constexpr static bool simd = config.simd;
	constexpr static bool parity_row = config.parity_row;
	constexpr static uint32_t q_bits = config.q_bits;

	//  this is quite important.
	constexpr static uint32_t kp = config.k - 1u*parity_row;

	using limb_type = LimbType;
	using l_type = MinLogTypeTemplate<q_bits * config.l, 32>;

	// matrix types
	using PCMatrixOrg 	= FqMatrix<LimbType, config.n-config.k, config.n, config.q, packed>;
	using PCMatrixOrg_T = FqMatrix<LimbType, config.n, config.n-config.k, config.q, packed>;
	using PCMatrix 		= FqMatrix<LimbType, config.n-kp, config.n+1, config.q, packed>;
	using PCMatrix_T 	= FqMatrix<LimbType, config.n+1, config.n-kp, config.q, packed>;
	using PCSubMatrix 	= FqMatrix<LimbType, config.n-kp, config.k+config.l, config.q, packed>;
	using PCSubMatrix_T = FqMatrix<LimbType, config.k+config.l, config.n-kp, config.q, packed>;
	using Error 		= FqMatrix<LimbType, 1, config.n, config.q, packed>;
	using Syndrome 		= FqMatrix<LimbType, config.n-config.k, 1, config.q, packed>; // maybe transpose?

	using Value = typename std::conditional<packed,
			kAryPackedContainer_T<LimbType, config.k+config.l, config.q>,
			kAryContainer_T<LimbType, config.k, config.q>
	>::type;

	using Label = typename std::conditional<packed,
			kAryPackedContainer_T<LimbType, config.n-config.k, config.q>,
			kAryContainer_T<LimbType, config.n-config.k, config.q>
	>::type;
	using Element       = Element_T<Value, Label, PCSubMatrix_T>;

	/// internal matrices
	PCMatrixOrg A;		// org input matrix. Is not touched
	PCMatrix wA; 		// working matrix
	PCMatrix_T wAT; 	// transposed working matrix
	PCSubMatrix H;		// sub ISD matrix
	PCSubMatrix_T HT;	// transposed sub ISD matrix
	Syndrome s; 		//
	Label ws;			//
	Error e;			//

	// this is dangerous for the syndrome
	LimbType syndrome;
	static_assert(config.l * bits_log2(config.q) <= sizeof(LimbType)*8);

	/// permutation
	Permutation P{config.n};

	/// changelist stuff
	chase<(config.k+config.l)/2 + config.epsilon, config.p, config.q> c{};
	using cle = std::pair<uint16_t, uint16_t>;
	const size_t list_size = c.list_size();
	std::vector<cle> cL;

	/// not constant values.
	const double ghz = std::max(osfreq(), 1.); // just to make sure we are not dividing by zero
	uint64_t loops = 0, expected_loops = 0;
	bool not_found = true;
	uint64_t gaus_cycles = 0, extract_cycles = 0, cycles;

	/// some asserts
	static_assert(config.epsilon <= ((config.l+config.k)/2));
	static_assert(config.l < (config.n-config.k));

	constexpr ISDInstance(bool use_changelist=false) noexcept : c(){
		/// warn the user if the internal datastructures are
		/// not using a optimized arithmetic
		if constexpr (!Label::optimized()) {
			std::cout << "WARNING: the arithmetic is not optimized for "
					  << config.q << ", a generic fallback implementation will be used."
					  << std::endl;
		}

		if (use_changelist) {
			compute_changelist();
		}
	}

	virtual ~ISDInstance() {
	     periodic_print();
	};

	virtual void info() const noexcept {};


	constexpr void reset() noexcept {
		not_found = true;
		loops = 0;
		cycles = cpucycles();
		gaus_cycles = 0;
	}

	/// important: dont rename it
	virtual constexpr void print() const noexcept {
		std::cout << "{ \"ghz\": " << ghz
				  << ", \"loops\": " << loops
		          << ", \"cycles\": " << cycles
		          << ", \"walltime\": " << (double)cycles/(double)ghz
				  << ", \"not_found\": " << not_found
				  << " }" << std::endl;
		// and finally print the solution
		std::cout << "{ \"solution\": \"";
		e.print();
		std::cout << "\" }";
	}

	/// print periodically some information
	constexpr void periodic_print() const noexcept {
		const uint64_t cyc     = cpucycles() - cycles;
		const double time      = double(cyc) / ghz;
		const double gaus_proc = 100 * double(gaus_cycles) / double(cyc);
		const double extract_proc = 100 * double(extract_cycles) / double(cyc);
		const double tot_proc  = 100 * double(loops) / double(expected_loops);
		const double lps       = double(loops) / time;

		std::cout << "{ \"loops\": " << loops
		          << ", \"sec\": " << time
				  << ", \"gaus\": " <<  gaus_proc
		          << ", \"extract\": " <<  extract_proc
				  << ", \"lps\": " << lps
	  			  << ", \"tot\": " << tot_proc
				  << " }" << std::endl;
	}

	constexpr bool correct() const noexcept {
		bool ret = true;
		if (e.row_popcnt(0) > config.w){
			ret = false;
		}

		Syndrome s2;
		PCMatrixOrg::mul_transposed(s2, A, e);
		ret &= Syndrome::is_equal(s, s2);

		if (!ret) {
			s.print("Syndrome", false, true, true);
			s2.print("Computed Syndrome", false, true, true);
			e.print();
			std::cout << std::endl;
			std::cout << e.row_popcnt(0) << std::endl;
		}

		return ret;
	}

	constexpr void from_string(const char *H, const char *S) noexcept {
		static_assert(config.l < config.n-config.k);
		PCMatrixOrg_T AT(H);
		PCMatrixOrg_T::transpose(A, AT);
		s.from_string(S);
		PCMatrix::augment(wA, A, s);

		if constexpr (config.parity_row) {
			compute_parity_row();
		}

		// needed for marcov
		if constexpr (config.c > 0) {
			const uint32_t rank = wA.fix_gaus(P, wA.gaus(config.n-config.k), config.n-config.k-config.l);
			ASSERT(rank >= config.n-config.k-config.l);
		}
	}

	// generate a random instance
	constexpr void random() noexcept {
		A.random();
		A.gaus();
		e.random_row_with_weight(0, config.w);
		PCMatrixOrg::mul_transposed(s, A, e);
		PCMatrix::augment(wA, A, s);
	}

	constexpr void compute_changelist() noexcept {
		cL.resize(list_size);
		size_t ctr = 0;
		c.enumerate([&, this](const uint16_t p1, const uint16_t p2){
			cL[ctr] = cle{p1, p2};
			ctr += 1;
		});
	}

	constexpr void compute_parity_row() {
		for (uint32_t i = 0; i < config.n; ++i) {
			wA.set(1, config.n-config.k, i);
		}
		wA.set(w%2u, config.n-config.k, config.n);
	}

	template<const bool transpose=true,
	         const bool swap=true,
	         const bool sub=true>
	constexpr void step() noexcept {
		loops += 1;
		gaus_cycles -= cpucycles();

		if constexpr (config.c == 0) {
			wA.permute_cols(wAT, P);
			uint32_t rank = wA.gaus(config.n-config.k-config.l);
			rank = wA.fix_gaus(P, rank, config.n-config.k-config.l);
			ASSERT(rank >= config.n-config.k-config.l);
		} else {
			uint32_t rank = wA.template markov_gaus<config.c, config.n-config.k-config.l>(P);
			ASSERT(rank >= config.n-config.k-config.l);
		}

		gaus_cycles += cpucycles();

		extract_cycles -= cpucycles();
		if constexpr (sub) {
			PCMatrix::sub_matrix(H, wA, 0, config.n - config.k - config.l, config.n - config.k, config.n);
		}

		if constexpr (swap) {
			// otherwise it doesnt make any sense
			static_assert(sub);
			swap_matrix();
		}

		if constexpr (transpose) {
			PCSubMatrix::transpose(HT, H);
		}

		extract_syndrome();
		extract_cycles += cpucycles();

		if (loops % config.print_loops == 0) {
			periodic_print();
		}

		// print one time information
		if (loops == 1) {
			info();
		}
	}

	/// Swaps all rows in H, e.g. H[0] <-> H[n-k], H[1] <-> H[n-k-1], ...
	constexpr void swap_matrix() noexcept {
		// NOTE: im swapping H, not HT
		for (uint32_t i = 0; i < (config.n-config.k)/2u - ((config.n-config.k) % 2u); i++) {
			H.swap_rows(i, config.n-config.k-1u-i);
		}
	}

	/// extracts the l part of the parity check matrix HT and write
	/// it into lHT.
	/// Because we first swapped all rows in H previously (in `swap_matrix`)
	/// the lpart is in the first l bits of each row in HT
	template<typename TT=limb_type, const uint32_t offset=0, const uint32_t lprime=config.l>
	constexpr void extract_lHT(l_type *lHT) const noexcept {
		static_assert(config.l <= 64);
		static_assert(packed);

		if constexpr(offset == 0) {
			constexpr TT mask = (1u << (q_bits*lprime)) - 1u;

			for (uint32_t i = 0; i < config.k + config.l; i++) {
				lHT[i] = HT[i][0] & mask;
			}
		} else {
			constexpr uint32_t low = offset;
			constexpr uint32_t high = low + lprime;
			for (uint32_t i = 0; i < config.k + config.l; i++) {
				lHT[i] = extract<TT, low, high>(HT[i]);
			}
		}
	}

	/// extracts HT into pHT
	/// write the complete HT matrix into a continuously memory block.
	/// This is sadly needed because of the usage of `_mm256_i32gather_epi32`
	/// and the fact the `M4RI` version I'm using is not using a continuously
	/// memory block.
	template<typename TT=limb_type>
	constexpr void extract_pHT(TT *pHT) const noexcept {
		const size_t bytes_length = HT.limbs_per_row() * sizeof(TT);

		#pragma unroll
		for (uint32_t i = 0; i < config.k+config.l; i++) {
			memcpy(pHT + i*HT.limbs_per_row(), HT[i], bytes_length);
		}
	}

	///
	/// \tparam offset
	/// \return
	template<const uint32_t offset = 0, const uint32_t lprime=config.l>
	constexpr void extract_syndrome_limb() noexcept {
		// first reset the syndrome
		syndrome = 0;

		if constexpr (config.q == 2) {
			/// loop fusion is maybe a little bit over kill,
			/// the loops have different endings.
			for (uint16_t i = 0; i < lprime; i++) {
				const auto bit = wA.get(config.n-config.k-offset-i-1, config.n);
				syndrome ^= bit << i;
			}
		} else {
			for (uint16_t i = 0; i < lprime; i++) {
				const auto bits = wA.get(config.n-config.k-offset-i-1, config.n);
				syndrome ^= bits << (i * q_bits);
			}
		}
	}
	
	/// reads the l part of the syndrome into `syndrome` and `ws`
	/// NOTE: this function is directly swapping the positions
	constexpr void extract_syndrome() noexcept {
		extract_syndrome_limb();

		/// Read all n-k bits. This is needed, because I dont want to do
		/// any bitshifts in the `extract_pHT` function
		for (uint16_t i = 0; i < config.n-config.k; i++) {
			const auto bit = wA.get(config.n-config.k-1-i, config.n);
			ws.set(bit, i);
		}
	}


	/// This functions returns the number of buckets needed if you
	/// want the hashmap to be able to match on `nr_coordinates`.
	/// Where `nr_coordinates` is the number of Fq elements.
	/// E.g. if you want to match on 3 F4 elements, you need 4**3 many buckets
	constexpr static uint64_t HashMasp_BucketSize(const uint32_t nr_coordinates) noexcept {
		uint64_t ret = 1;
		for (uint32_t i = 0; i < nr_coordinates; ++i) {
			ret *= config.q;
		}

		return ret;
	}
};
#endif//DECODING_ISD_H
