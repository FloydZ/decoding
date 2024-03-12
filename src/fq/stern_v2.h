#ifndef DECODING_FQ_STERNV2_H
#define DECODING_FQ_STERNV2_H

#include "helper.h"
#include "container/fq_vector.h"
#include "matrix/fq_matrix.h"
#include "list/list.h"
#include "list/enumeration/fq.h"
#include "sort.h"
#include "popcount/popcount.h"
#include "mitm.h"
#include "isd.h"
#include "../stern.h"

using namespace cryptanalysislib;


template<const ConfigISD &isd, const ConfigStern &config>
class FqSternV2 : public ISDInstance<uint64_t, isd> {
private:
	constexpr static uint32_t n = isd.n;
	constexpr static uint32_t k = isd.k;
	constexpr static uint32_t w = isd.w;
	constexpr static uint32_t q = isd.q;
	constexpr static uint32_t qbits = bits_log2(q);

	constexpr static uint32_t l = isd.l;
	constexpr static uint32_t p = isd.p;

	constexpr static uint32_t threads = config.threads;
	constexpr static size_t final_list_size = 1024;

public:
	using ISD = ISDInstance<uint64_t, isd>;
	using Error 		= ISD::Error;
	using Label 		= ISD::Label;
	using Value 		= ISD::Value;
	using Element 		= ISD::Element;
	using limb_type 	= ISD::limb_type;
	using PCSubMatrix_T = ISD::PCSubMatrix_T;
	using ISD::A,ISD::H,ISD::wA,ISD::wAT,ISD::HT,ISD::s,ISD::ws,ISD::e,ISD::syndrome,ISD::P,ISD::not_found,ISD::loops,ISD::ghz;
	using ISD::cycles, ISD::gaus_cycles, ISD::periodic_print, ISD::HashMasp_BucketSize;

	constexpr static uint32_t HM_nrb = ISD::HashMasp_BucketSize(l);
	constexpr static uint32_t HM_bs = config.HM_bucketsize;
	constexpr static uint32_t NR_HT_LIMBS = PCSubMatrix_T::limbs_per_row();

	constexpr static bool packed = ISD::packed;

	/// minimal datatype to keep all l fq elements
	using l_type 	= MinLogTypeTemplate<qbits*l, 32>;
	using IndexType = TypeTemplate<HM_nrb * HM_bs>;

	l_type *lHT;

	/// needed list stuff
	constexpr static uint32_t enum_length = (k + l + isd.epsilon) >> 1u;
	constexpr static uint32_t enum_offset = k + l - enum_length;
	constexpr static size_t enum_size = compute_combinations_fq_chase_list_size<enum_length, 2, p>();

    using V1 = CollisionType<l_type, uint16_t, 1>;
	constexpr static SimpleHashMapConfig simpleHashMapConfig{
			HM_bs, HM_nrb, config.threads
	};
	using HM = SimpleHashMap<l_type, V1, simpleHashMapConfig, Hash<l_type, 0, l, q>>;
	HM *hm;
	using HM_LoadType = HM::load_type;
	using HM_DataType_IndexType = V1::index_type;
	using HM_DataType = V1;


	/// changelist stuff
	chase<(config.k+config.l)/2 + config.epsilon, config.p, config.q> c{};
	using cle = std::pair<uint16_t, uint16_t>;
	std::vector<cle> cL;

	///
	size_t cfls = 0;
	alignas(256) std::array<uint32_t, final_list_size> final_list_left;
	alignas(256) std::array<uint32_t, final_list_size> final_list_right;

	/// list entries of the current permutation if a solution was found.
	size_t solutions[2*p] = {0};

	Value value_solution_to_recover;
	Label label_solution_to_recover;

	/// base constructor
	FqSternV2() noexcept {
		constexpr size_t size_lHT = roundToAligned<1024>(sizeof(l_type) * (k+l));
		lHT =(l_type *)cryptanalysislib::aligned_alloc(1024, size_lHT);
		ASSERT(lHT);

		hm = new HM{};
		ASSERT(hm);
		hm->print();


		cL.resize(enum_size);
		size_t ctr = 0;
		c.enumerate([&, this](const uint16_t p1, const uint16_t p2){
			cL[ctr] = cle{p1, p2};
			ctr += 1;
		});
		ASSERT(ctr == (enum_size - 1));
	}

	/// free all the memory
	~FqSternV2() {
		delete hm;
	}

	bool compute_finale_list() noexcept {
		alignas(32) uint16_t left[p], right[p];
		for (size_t m = 0; m < cfls; ++m) {
			biject<enum_length, p>(final_list_left[m], left);
			biject<enum_length, p>(final_list_right[m], right);

			/// TODO the following things are wrong:
			/// - biject is not correct, we need to match the +q-1 offset
			/// - missing lpart = 0 checks
			auto climb = ws.ptr(0);
			for (uint16_t j = 0; j < p; j++) {
				const auto tmp = Label::add_T(HT.limb(left[j], 0), HT.limb(right[j], 0));
				climb = Label::add_T(climb, tmp);
			}

			uint32_t wt = Label::popcnt_T(climb);
			if (likely(wt > (w - (2*p)))) {
				continue;
			}


			for (uint32_t i = 1; i < NR_HT_LIMBS; i++) {
				climb = ws.ptr(i);
				for (uint16_t j = 0; j < p; j++) {
					const auto tmp = Label::add_T(HT.limb(left[j], i), HT.limb(right[j], i));
					climb = Label::add_T(climb, tmp);
				}

				wt += cryptanalysislib::popcount::popcount(climb);
			}

			if ((wt <= w - (2*p)) && not_found) {
				not_found = false;
				cycles = cpucycles() - cycles;
				for (uint16_t j = 0; j < p; ++j) {
					solutions[j*p + 0] = left[j];
					solutions[j*p + 1] = right[j];
				}

				cfls = 0;
				return true;
			}
		}

		cfls = 0;
		return false;
	}

	///
	inline void init_list(const uint32_t tid) {
		hm->clear();

		l_type tmp = 0;
		for (uint32_t i = 0; i < p; ++i) {
			tmp = Label::template add_T<l_type>(tmp, lHT[i]);
		}

		for (size_t i = 0; i < enum_size; ++i) {
			const uint16_t ci = cL[i].second;
			for (uint32_t j = 0; j < (q - 1); ++j) {
				hm->insert(tmp, HM_DataType::create(tmp, (HM_DataType_IndexType *)&i));
				tmp = Label::template add_T<l_type>(tmp, lHT[ci]);
			}

			tmp = Label::template add_T<l_type>(tmp, lHT[cL[i].first]);
			tmp = Label::template add_T<l_type>(tmp, lHT[cL[i].second]);
		}
	}

	void find_collisions(const uint32_t tid) {
		l_type tmp = syndrome;
		for (uint32_t i = 0; i < p; ++i) {
			tmp = Label::template add_T<l_type>(tmp, lHT[i]);
		}

		for (size_t i = 0; i < enum_size; ++i) {
			const uint16_t ci = cL[i].second;
			for (uint32_t m = 0; m < (q - 1); ++m) {
				HM_LoadType load;
				IndexType pos = hm->find(tmp, load);
				for (uint64_t j = pos; j < pos + load; j++) {
					const IndexType index = hm->ptr(j).index[0];

					final_list_left[cfls]  = index;
					final_list_right[cfls] = i;
					cfls += 1;

					if (cfls >= final_list_size) {
						compute_finale_list();
					}
				}

				// next element
				tmp = Label::template add_T<l_type>(tmp, lHT[ci]);
			}

			// next element in the chase sequence
			tmp = Label::template add_T<l_type>(tmp, lHT[cL[i].first]);
			tmp = Label::template add_T<l_type>(tmp, lHT[cL[i].second]);
		}
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

		tmpe.print();
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
			ISD::template extract_lHT<l_type>(lHT);

			// #pragma omp parallel default(none) shared(std::cout,L1,not_found,loops) num_threads(threads)
			{
				const uint32_t tid = 0; //Thread::get_tid();
				init_list(tid);

				Thread::sync();
				find_collisions(tid);
				compute_finale_list();
			}
		}

		rebuild_solution();
		return loops;
	}
};
#endif//DECODING_FQ_STERN_H
