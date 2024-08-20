#ifndef DECODING_FQ_STERN_H
#define DECODING_FQ_STERN_H

#include "helper.h"
#include "container/fq_vector.h"
#include "matrix/matrix.h"
#include "list/list.h"
#include "sort.h"
#include "popcount/popcount.h"
#include "mitm.h"
#include "isd.h"
#include "../stern.h"

using namespace cryptanalysislib;


template<const ConfigISD &isd, const ConfigStern &config>
class FqStern : public ISDInstance<uint64_t, isd> {
private:
	constexpr static uint32_t n = isd.n;
	constexpr static uint32_t k = isd.k;
	constexpr static uint32_t w = isd.w;
	constexpr static uint32_t q = isd.q;
	constexpr static uint32_t qbits = bits_log2(q);

	constexpr static uint32_t l = isd.l;
	constexpr static uint32_t p = isd.p;

	constexpr static uint32_t threads = config.threads;
public:
	using ISD = ISDInstance<uint64_t, isd>;
	using Error 		= ISD::Error;
	using Label 		= ISD::Label;
	using Value 		= ISD::Value;
	using Element 		= ISD::Element;
	using limb_type 	= ISD::limb_type;
	using ISD::A,ISD::H,ISD::wA,ISD::wAT,ISD::HT,ISD::s,ISD::ws,ISD::e,ISD::syndrome,ISD::P,ISD::not_found,ISD::loops,ISD::ghz;
	using ISD::cycles, ISD::gaus_cycles, ISD::periodic_print, ISD::HashMasp_BucketSize;

	constexpr static uint32_t HM_nrb = ISD::HashMasp_BucketSize(l);
	constexpr static uint32_t HM_bs = config.HM_bucketsize;

	constexpr static bool packed = ISD::packed;

	/// minimal datatype to keep all l fq elements
	using l_type 	= MinLogTypeTemplate<qbits*l, 32>;
	using IndexType = TypeTemplate<HM_nrb * HM_bs>;

	/// needed list stuff
	constexpr static uint32_t enum_length = (k + l) >> 1u;
	constexpr static uint32_t enum_offset = k + l - enum_length;
	constexpr static size_t list_size = compute_combinations_fq_chase_list_size<enum_length, q, p>();
	// using List = Parallel_List_FullElement_T<Element>;
	using List = List_T<Element>;
	using Generator = ListEnumerateMultiFullLength<List, enum_length, q, p>;
	Generator G;
	List *L1, *L2;

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


	using valueType = TypeTemplate<list_size>[1];
    // using value_type = CollisionType<l_type, 1>;
	constexpr static SimpleHashMapConfig simpleHashMapConfig{
			HM_bs, HM_nrb, config.threads
	};
	using HM = SimpleHashMap<l_type, valueType , simpleHashMapConfig, Hash<l_type, 0, l, q>>;
	HM *hm;
	using HM_LoadType = HM::load_type;

	Value value_solution_to_recover;
	Label label_solution_to_recover;

	/// base constructor
	FqStern() noexcept : G(HT, 0, nullptr) {
		/// init lists
		L1 = new List(list_size, threads);
		L2 = new List(list_size, threads);
		ASSERT((L1 != nullptr) && (L2 != nullptr));

		///// init hashmap
		hm = new HM;
		ASSERT(hm != nullptr);
	}

	/// free all the memory
	~FqStern() {
		delete L1;
		delete L2;
		delete hm;
	}

	///
	inline void init_list(const uint32_t tid) noexcept {
		hm->clear();

		/// this call simply inits the the list
		G.template run <HM, decltype(Compress), std::nullptr_t>
				(L1, L2, enum_offset, 0, tid, hm, &Compress);
	}

	void find_collisions(const uint32_t tid) {
		const size_t start = L2->start_pos(tid);
		const size_t end = L2->end_pos(tid);

		Label tmp;
		for (size_t i = start; i < end; ++i) {
			l_type data = NegateCompress(L2->at(i).label);
			data = Label::template add_T<l_type>(data, syndrome);

			/// search in HM
			HM_LoadType load;
			IndexType pos = hm->find(data, load);
			for (uint64_t j = pos; j < pos + load; j++) {
				const IndexType index = hm->ptr(j)[0];

				/// TODO not really nice
				Label::add(tmp, L1->at(index).label, L2->at(i).label);
				Label::sub(tmp, ws, tmp);

				// std::cout << std::endl;
				// L2->at(i).label.print();
				// L1->at(index).label.print();
				// tmp.print();
				// ws.print();
				// std::cout << i << " " << index << std::endl;

				/// some debug checks
				for (uint32_t s = 0; s < l; ++s) {
					ASSERT(tmp.get(s) == 0);
				}

				/// if this checks passes we found a solution
				if (unlikely(tmp.popcnt() <= (w - 2*p))) {
					not_found = false;
					label_solution_to_recover = tmp;
					Value::add(value_solution_to_recover, L2->at(i).value, L1->at(index).value);
					goto finished;
				}
			}
		}

		/// forward jumps = best jumps
		finished:
		return;
	}

	///
	void rebuild_solution() {
		Error tmpe;
		tmpe.zero();
		e.zero();

		for (uint32_t i = 0; i < l; ++i) {
			ASSERT(label_solution_to_recover.get(i) == 0);
		}
		ASSERT(label_solution_to_recover.popcnt() <= (w-2*p));
		ASSERT(value_solution_to_recover.popcnt() == 2*p);

		for (uint32_t i = 0; i < n-k-l; ++i) {
			tmpe.set(label_solution_to_recover.get(l + i), 0, (n-k-l) - i - 1);
		}

		for (uint32_t i = 0; i < k+l; ++i) {
			tmpe.set(value_solution_to_recover.get(i), 0, (n-k-l)+i);
		}

		ASSERT(tmpe.get(0).popcnt() <= w);

		// apply back permutation
		for (uint32_t i = 0; i < n; ++i) {
			const auto data = tmpe.get(0, i);
			e.set(data, 0, P.values[i]);
		}
	}


	/// \return
	uint64_t __attribute__ ((noinline))
	run() noexcept {
		ISD::reset();

		while (not_found && (loops < config.loops)) {
			ISD::step();

			// #pragma omp parallel default(none) shared(std::cout,L1,not_found,loops) num_threads(threads)
			{
				const uint32_t tid = 0; //Thread::get_tid();
				init_list(tid);
				Thread::sync();
				find_collisions(tid);
			}
		}

		rebuild_solution();
		return loops;
	}
};
#endif//DECODING_FQ_STERN_H
