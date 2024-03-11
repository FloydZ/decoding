#ifndef DECODING_FQ_SIEVING_H
#define DECODING_FQ_SIEVING_H

#include "helper.h"
#include "container/fq_vector.h"
#include "matrix/fq_matrix.h"
#include "list/list.h"
#include "list/enumeration/fq.h"
#include "sort.h"
#include "popcount/popcount.h"
#include "mitm.h"
#include "isd.h"

using namespace cryptanalysislib;

struct ConfigFqSieving : public ConfigISD {
public:
	/// Hashmap stuff
	const uint32_t HM_bs; 	// elements per bucket

	/// number of sieving iterations
	const uint32_t sieving_steps;

	/// specify the max `q` to enumerate in the base list.
	/// NOTE: this must not be equal q. It can be greater or smaller
	const uint32_t enumeration_q;
};

template<const ConfigISD &isd, const ConfigFqSieving &config>
class FqSieving : public  ISDInstance<uint64_t, isd>{
private:
	constexpr static uint32_t n = isd.n;
	constexpr static uint32_t k = isd.k;
	constexpr static uint32_t w = isd.w;
	constexpr static uint32_t q = isd.q;
	constexpr static uint32_t qbits = bits_log2(q);

	constexpr static uint32_t l = isd.l;
	constexpr static uint32_t p = isd.p;

	constexpr static uint32_t threads = isd.threads;

	constexpr static uint32_t sieving_steps = config.sieving_steps;
	constexpr static uint32_t enumeration_q = config.enumeration_q;

	constexpr static uint32_t full_l = l*sieving_steps;

public:
	using ISD = ISDInstance<uint64_t, isd>;
	using PCMatrixOrg 	= ISD::PCMatrixOrg;
	using PCMatrixOrg_T = ISD::PCMatrixOrg_T;
	using PCMatrix 		= ISD::PCMatrix;
	using PCMatrix_T 	= ISD::PCMatrix_T;
	using PCSubMatrix 	= ISD::PCSubMatrix;
	using PCSubMatrix_T = ISD::PCSubMatrix_T;
	using Syndrome 		= ISD::Syndrome;
	using Error 		= ISD::Error;
	using limb_type 	= ISD::limb_type;
	using ISD::A,ISD::H,ISD::wA,ISD::wAT,ISD::HT,ISD::s,ISD::ws,ISD::e,ISD::syndrome,ISD::P,ISD::not_found,ISD::loops,ISD::ghz;
	using ISD::cycles, ISD::gaus_cycles, ISD::periodic_print, ISD::HashMasp_BucketSize;

	constexpr static bool packed = ISD::packed;
	constexpr static uint32_t HM_nrb = ISD::HashMasp_BucketSize(l);
	constexpr static uint32_t HM_bs = config.HM_bs;
	using Value = typename std::conditional<packed,
			kAryPackedContainer_T<limb_type , k+full_l, q>,
			kAryContainer_T<limb_type , k+full_l, q>
	>::type;

	using Label = typename std::conditional<packed,
			kAryPackedContainer_T<limb_type , n-k, q>,
			kAryContainer_T<limb_type , n-k, q>
	>::type;
	using Element       = Element_T<Value, Label, PCSubMatrix_T>;

	/// minimal datatype to keep all l fq elements
	using l_type 	= MinLogTypeTemplate<qbits*l, 32>;
	using IndexType = TypeTemplate<HM_nrb * HM_bs>;

	/// needed list stuff
	constexpr static size_t list_size = compute_combinations_fq_chase_list_size<k, q, p>();
	using List = Parallel_List_FullElement_T<Element>;
	using Generator = ListEnumerateMultiFullLength<List, k, q, p>;
	Generator G;
	List *L1, *L2;


	/// as the list sizes can reduce, we need to keep to track of it.
	size_t current_list_size = list_size;

	/// computes the lower bit position of the l window of the current iteration
	template<const uint32_t current_iteration>
	constexpr static inline uint32_t lower_limit() noexcept {
		return current_iteration*qbits*l;
	}

	using valueType = TypeTemplate<list_size>[1];
	// using value_type = CollisionType<l_type, 1>;
	constexpr static SimpleHashMapConfig simpleHashMapConfig{
			HM_bs, HM_nrb, isd.threads
	};
	using HM = SimpleHashMap<l_type, valueType , simpleHashMapConfig, Hash<l_type, 0, l, q>>;
	HM *hm1, *hm2;
	using HM_LoadType = HM::load_type;


	Value value_solution_to_recover;
	Label label_solution_to_recover;

	// TODO move
	/// compress in the case of non packed data containers
	/// e.g. removes all the zeros from the containers
	constexpr inline static __uint128_t Compress(const Label &label) noexcept {
		/// some security measurements:
		if constexpr (qbits*l > 128) {
			ASSERT(false && "not implemented");
		}

		/// easy case
		if constexpr (packed) {
			constexpr __uint128_t mask = qbits*l == 128 ? __uint128_t(-1ull) : (__uint128_t(1ull) << (qbits*l)) - __uint128_t(1ull);
			using TT = LogTypeTemplate<qbits*l>;

			/// NOTE: that we fetch the first 128bits (and not the last, where we would assume
			/// normally the l bit window), as we assume we swapped all rows within the parity
			/// check matrix. Therefor the last `l` bit of every syndrome are now the first.
			const TT a = *((TT *)label.ptr());
			const __uint128_t aa = a;
			return aa&mask;
		}

		// important to init with zero here.
		__uint128_t ret = 0;

		#pragma unroll
		for (uint32_t i = 0u; i < l; ++i) {
			ret ^= (label.get(i) << (qbits*i));
		}

		return ret;
	}

	/// this extractor is needed, to be able to search on the second
	/// list for elements, which are equal zero summed together
	/// \return the negative of label[lower, q*l*nr_window) shifted to zero
	constexpr inline static __uint128_t NegateCompress(const Label &label) noexcept {
		Label tmp = label;
		tmp.neg();
		return Compress(tmp);
	}

	/// base constructor
	FqSieving() noexcept : G(HT, 0, &ws) {
		/// init lists
		L1 = new List(list_size, threads);
		L2 = new List(list_size, threads);
		ASSERT(L1 != nullptr && L2 != nullptr);

		/// init hashmap
		hm1 = new HM();
		hm2 = new HM();
		ASSERT(hm1 != nullptr && hm2 != nullptr);
	}

	/// initialize the list L1. Additionally resets the hashmap `hm1`
	/// regardless of the current ordering
	void init_list(const uint32_t tid) {
		/// reset list counter
		current_list_size = list_size;
		hm1->clear();

		/// this call simply inits the the list
		G.template run <HM, decltype(Compress), std::nullptr_t>
				(L1, nullptr, 0, tid, hm1, &Compress);
	}


	/// \param tid  thread id
	/// \param citeration current iteration
	/// \return new list size
	size_t sieving_final_step(const uint32_t tid) {
		constexpr uint32_t citeration = sieving_steps-1;
		constexpr uint32_t lower_l = lower_limit<citeration>();

		/// reset stuff
		hm2->clear();

		Element tmp;
		const size_t s_pos = L1->start_pos(tid);
		const size_t e_pos = std::min(L1->end_pos(tid), current_list_size);

		/// for every element in the current list
		size_t ctr = tid*(list_size / threads); // counts the new elements
		for (size_t i = s_pos; i < e_pos; ++i) {
			HM_LoadType load1;

			/// NOTE: the compressor negates the data, needed to be able
			/// to search for exact matches.
			l_type data1 = this->NegateCompress
			        <lower_l>(L1->at(i).label);
			IndexType pos1 = hm1->find(data1, load1);

			while (pos1 < load1) {
				const IndexType index = hm1->ptr(pos1)[0];
				pos1 += 1;
				/// make sure that we do not create zeros or doubles
				if (i >= index){
					continue;
				}

				/// add the two elements and run a few filters
				Value::add(tmp.value, L1->at(i).value, L1->at(index).value);
				if (tmp.value.popcnt() != 2*p) {
					continue;
				}

				/// only compute the label, if its needed
				Label::add(tmp.label, L1->at(i).label, L1->at(index).label);

				// filtering functions: Activate as needed:
				//if (!Value::template filter
				//        <2*p, -1u, -1, -1u, -1u>
				//    	(tmp.value, L1->at(i).value, L2->at(index).value)) {
				//	continue;
				//}

				/// some debug information
				//L1->at(i).print();
				//L1->at(index).print();
				//tmp.print();
				//std::cout << i << " " << index << std::endl;
				//std::cout << unsigned(data1) << " " << unsigned (hm1->__buckets[pos1-1].first) << std::endl << std::endl;

				/// NOTE: Debug check
				for (uint32_t j = 0; j < full_l; ++j) {
					ASSERT(tmp.label.get(j) == 0u);
				}


				/// if we pass this check, we found a global solution
				if (tmp.label.popcnt() <= (w-2*p)) {
					not_found = false;
					label_solution_to_recover = tmp.label;
					value_solution_to_recover = tmp.value;
					goto finish;
				}

				ctr += 1;
			}
		}

		finish:
		///
		return ctr;
	}

	/// normal step
	/// \param tid  thread id
	/// \param citeration current iteration
	/// \return new list size
	template<const uint32_t citeration>
	size_t sieving_step(const uint32_t tid) {
		constexpr uint32_t lower_l_bit = lower_limit<citeration>();
		constexpr uint32_t upper_l = (citeration+1)*l;

		/// reset stuff
		hm2->clear();

		Element tmp;
		IndexType npos[1];
		const size_t s_pos = L1->start_pos(tid);
		const size_t e_pos = std::min(L1->end_pos(tid), current_list_size);

		/// for every element in the current list
		size_t ctr = tid*(list_size / threads); // counts the new elements
		for (size_t i = s_pos; i < e_pos; ++i) {
			HM_LoadType load1;

			/// NOTE: the compressor negates the data
			l_type data1 = this->NegateCompress
					<lower_l_bit>(L1->at(i).label);

			/// NOTE: maybe I can prepare data1 before hand?
			IndexType pos1 = hm1->find(data1, load1);
			for (uint64_t j = pos1; j < pos1 + load1; j++) {
				const IndexType index = hm1->ptr(j)[0];
				/// make sure that we do not create zeros or doubles
				if (i >= index){
					continue;
				}

				/// add the two elements and run a few filters
				Value::add(tmp.value, L1->at(i).value, L1->at(index).value);
				if (tmp.value.popcnt() != 2*p) {
					continue;
				}

				/// only compute the label, if its needed
				Label::add(tmp.label, L1->at(i).label, L1->at(index).label);

				// filtering functions: Activate as needed:
				//if (!Value::template filter
				//        <2*p, -1u, -1, -1u, -1u>
				//    	(tmp.value, L1->at(i).value, L2->at(index).value)) {
				//	continue;
				//}

				// SOME DEBUGGING
				//std::cout << "Level: " << citeration << std::endl;
				//L1->at(i).label.print();
				//L1->at(index).label.print();
				//tmp.print();
				//std::cout << i << " " << index << std::endl;
				//std::cout << unsigned(data1) << " " << unsigned (hm1->__buckets[pos1-1].first) << std::endl << std::endl;

				/// some assertions
				for (uint32_t j = 0; j < upper_l; ++j) {
					ASSERT(tmp.label[j] == 0);
				}
				ASSERT(tmp.value.popcnt() == 2*p);


				/// insert this into the other list and hashmap
				l_type data2 = Compress(tmp.label);
				data2 >>= (citeration + 1) * l * qbits;
				npos[0] = ctr;
				hm2->insert(data2, npos, tid);
				L2->at(ctr++) = tmp;
				if (unlikely(ctr >= e_pos)) {
					/// exit if the new list is full
					goto finish;
				}

			}
		}

		finish:
		/// swap the lists and hashmaps
		std::swap(L1, L2);
		std::swap(hm1, hm2);

		current_list_size = ctr;
		return ctr;
	}

	/// tries to rebuild the solution
	/// in debug mode, this function asserts correctness
	/// in release mode: hope for the best.
	void rebuild_solution() {
		Error tmpe;
		tmpe.zero();
		e.zero();

		//std::cout << std::endl << "found:" << std::endl;
		//label_solution_to_recover.print();
		//value_solution_to_recover.print();

		ASSERT(label_solution_to_recover.popcnt() <= (w-2*p));
		ASSERT(value_solution_to_recover.popcnt() == 2*p);

		for (uint32_t i = 0; i < n-k-full_l; ++i) {
			tmpe.set(label_solution_to_recover.get(full_l + i), 0, (n-k-full_l) - i - 1);
		}

		for (uint32_t i = 0; i < k+full_l; ++i) {
			tmpe.set(value_solution_to_recover.get(i), 0, (n-k-full_l)+i);
		}

		ASSERT(tmpe.get(0).popcnt() <= w);

		// apply back permutation
		for (uint32_t i = 0; i < n; ++i) {
			const auto data = tmpe.get(0, i);
			e.set(data, 0, P.values[i]);
		}
	}


	/// \return number of loops needed
	uint64_t __attribute__ ((noinline))
	run() noexcept {
		not_found = true;
		loops = 0;
		while (not_found && (loops < isd.loops)) {
			ISD::step();

			//#pragma omp parallel default(none) shared(std::cout,L1,L2,not_found,loops) num_threads(threads)
			{
				const uint32_t tid = 0;//Thread::get_tid();
				init_list(tid);

			    constexpr_for<0, sieving_steps - 1u, 1u>([this](auto i){
				  sieving_step<i>(tid);
				});

				if (current_list_size == 0) {
					std::cout << "empty list" << std::endl;
					continue;
				}

				/// NOTE: this step additional checks for solutions
				sieving_final_step(tid);
			}
		}

		rebuild_solution();
		return loops;
	}
};

#endif//DECODING_SIEVING_H
