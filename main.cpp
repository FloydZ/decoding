#include "main.h"

#include <time.h>
#include <sys/resource.h>
#include <random>


/// IDEA: write an abstract interface every algorithm has to fullfill. 
template<class Algorithm>
concept AlgorithmAble = requires(Algorithm A){
	typename Algorithm::ValueType;
	typename Algorithm::LabelType;
	typename Algorithm::MatrixType;
	typename Algorithm::ListType;

	typename Algorithm::ValueContainerType;
	typename Algorithm::LabelContainerType;

	typename Algorithm::ValueDataType;
	typename Algorithm::LabelDataType;


	A.run();    // execute the algorithm
	A.loops();  // Expected number of loops.
	A.print();  // Print information about the current state of the Algorithm
	// TODO Constructor
};

int main(int argc, char** argv) {
	inittime = (double)clock();
	std::default_random_engine generator;
	std::uniform_int_distribution<uint64_t> distribution(1, std::numeric_limits<uint64_t>::max());

	// check if the external thread id passed via commandline
	uint32_t given_tid = 0;
	if (argc == 2) {
		given_tid = atoi(argv[1]);
	}

	// randomly see the internal prng.
	random_seed(given_tid + distribution(generator) + time(NULL));

	bool already_printed = false;
	uint64_t loops_sum = 0;
	double SumTime = (double)clock();
	finished.store(false);
	constexpr uint32_t addweightthresh = SYNDROM ? (4*G_p) : (LOW_WEIGHT ? (4*G_p+1) : 0);

	id_t pid = getpid();
	setpriority(PRIO_PROCESS, pid, -20);
	if (setpriority(PRIO_PROCESS, pid, -20) != 0)
		std::cout << omp_get_thread_num() << ", tid: " << given_tid << ". could not set NICE value\n";

#if TERNARY != 0
	static constexpr ListIteration listIters = TERNARY_ENUMERATION_TYPE == 0 ? SinglePartialSingle : EnumSinglePartialSingle;
	static constexpr ConfigMMTTernary MMTconfig(G_n, G_k, G_w/2, G_p, G_l, G_l1, BJMM_HM1_NRB, BJMM_HM2_NRB, BJMM_HM1_SIZEB, BJMM_HM2_SIZEB, TERNARY_NR1, TERNARY_NR2, TERNARY_FILTER2, G_k+G_l-TERNARY_ALPHA, listIters);
	using MatrixType = TernaryMMT<MMTconfig>::Matrix;

	MatrixType AT(n-k, n, h);
	MatrixType A(n, n-k);
	MatrixType ee(n, 1);
	MatrixType ss(n-k, 1, s);
	MatrixType ss_tmp(n-k, 1, s);

	MatrixType ee_T(n, 1);
	MatrixType::transpose(A, AT);

	for (uint32_t i = 0; i < n; ++i) {
		ee_T.set(2, 0, i);
	}
#else

#if USE_MO
	static constexpr ConfigBJMM config(n, k, G_w, G_p, G_l1, G_l1, HM1_SIZEB, HM2_SIZEB, HM1_NRB, 1, G_w-addweightthresh, NUMBER_THREADS, USE_DOOM, BJMM_DOOM_SPECIAL_FORM, FULLLENGTH, CUTOFF, LOW_WEIGHT, MO_l2, MO_NRHM, false, HM1_USESTDBINARYSEARCH, HM2_USESTDBINARYSEARCH, HM1_USEINTERPOLATIONSEARCH, HM2_USEINTERPOLATIONSEARCH, HM1_USELINEARSEARCH, HM2_USELINEARSEARCH, HM1_USELOAD, HM2_USELOAD);
#else
	static constexpr ConfigBJMM config(n, k, G_w, G_p, G_l, G_l1, HM1_SIZEB, HM2_SIZEB, HM1_NRB, HM2_NRB, G_w-addweightthresh, NUMBER_THREADS, USE_DOOM, BJMM_DOOM_SPECIAL_FORM, FULLLENGTH, CUTOFF, LOW_WEIGHT, MO_l2, MO_NRHM, false, HM1_USESTDBINARYSEARCH, HM2_USESTDBINARYSEARCH, HM1_USEINTERPOLATIONSEARCH, HM2_USEINTERPOLATIONSEARCH, HM1_USELINEARSEARCH, HM2_USELINEARSEARCH, HM1_USELOAD, HM2_USELOAD);
#endif
	mzd_t *AT = mzd_from_str(n, n-k, h);
	mzd_t *A = mzd_transpose(NULL, AT);
	mzd_t *ss;
	if (LOW_WEIGHT) {
		ss = mzd_init(1, n-k);
	} else {
		ss = mzd_from_str(1, n-k, s);
	}

	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);

	mzd_t *ee = mzd_init(1, n);
	mzd_t *ee_T = mzd_init(n, 1);
#endif

#pragma omp parallel default(none) shared(std::cout, ee, ss, A, ss_tmp, ee_T, given_tid, loops_sum, finished, already_printed) num_threads(NUMBER_OUTER_THREADS) if(NUMBER_OUTER_THREADS != 1)
	{
		const uint32_t e_tid = NUMBER_OUTER_THREADS == 1 ? 0 : omp_get_thread_num();
		const double time = (double)clock();

		// loops
		uint64_t r;

#if TERNARY != 0
		TernaryMMT<MMTconfig> *obj = new TernaryMMT<MMTconfig>(&ee, A, ss, e_tid);
#elif USE_NN
		BJMMNN<config> *obj = new BJMMNN<config>(ee, ss, A, e_tid);
#elif USE_MO
		MO<config> *obj = new MO<config>(ee, ss, A, e_tid);
#else
		BJMM<config> *obj = new BJMM<config>(ee, ss, A, e_tid);
#endif

		// This is it.
		r = obj->run();

		// measure time
		const double time2 = ((double)clock() - time)/CLOCKS_PER_SEC;

		// measure needed loops. Race cond. but who cares.
		loops_sum += r;

#pragma omp critical
		{
			if (!already_printed) {
				already_printed = true;

				const double lph = NUMBER_OUTER_THREADS * double(3600.) * double(r) / time2;
				std::cout << "Found outer tid:" << omp_get_thread_num() << ", e_tid: " << e_tid << " found solution, time: " << time2 << ", round: " << r << " loops/h:" << lph << "\n";
				FILE *f = fopen("solution.txt","a");
#if TERNARY
				ee.print();
				MatrixType::InternalRowType::sub(ee.__data[0], ee.__data[0], ee_T.__data[0]);
				ee.print();
#else
				mzd_fprint(f, ee);
#endif

				fprintf(f, "%lf s\n", time2);
				fprintf(f, "%lu iters\n", r);
				fprintf(f, "%lf loops/h\n", lph);

#ifndef TERNARY
				if constexpr(LOW_WEIGHT) {
					fprintf(f, "%u\n", hamming_weight(ee));
				}
#endif

				fclose(f);
			}
		}

		// cleanup mem
		delete(obj);
	}

	// write the results into a file:
	const double ttime = ((double)clock() - SumTime)/CLOCKS_PER_SEC;
	std::cout << "Finished: SUMTime: " << ttime << ", SUMLoops: " << loops_sum << ":" << loops_sum/NUMBER_OUTER_THREADS << "\n" << std::flush;
	FILE *f = fopen("solution.txt","a");
	fprintf(f, "loops_sum: %lu\n", loops_sum);
	fclose(f);

	return 0;
}
