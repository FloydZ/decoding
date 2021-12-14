#ifndef SMALLSECRETLWE_TEST_H
#define SMALLSECRETLWE_TEST_H

#define TESTSIZE 100

#define FOO(...) __VA_ARGS__

#define CUSTOM_COMPARE(subname)                                                             \
TEST(COMPARE, subname) {                                                                    \
	mzd_t *AT = mzd_from_str(n, n-k, h);                                                    \
	mzd_t *A = mzd_transpose(nullptr, AT);                                                  \
	mzd_t *ss = mzd_from_str(1, n-k, s);                                                    \
	mzd_t *ee = mzd_init(1, n);                                                             \
	static constexpr ConfigMcEliceV2 mcconfig(G_n, G_k, G_w, G_p, G_t, G_Thresh);           \
	static constexpr ConfigPrange config(G_n, G_n-G_k, G_w, G_n,-1);                        \
	double pr_time = 0.0;                                                                   \
	for (int i = 0; i < TESTSIZE; ++i) {                                                    \
		mzd_clear(ee);                                                                      \
		double t0 = ((double)clock()/CLOCKS_PER_SEC);                                       \
		prange<config>(ee, ss, A);                                                          \
		pr_time += ((double)clock()/CLOCKS_PER_SEC) - t0;                                   \
		EXPECT_EQ(w, hamming_weight(ee));                                                   \
	}                                                                                       \
	double mc_time = 0.0;                                                                   \
	double mc2_time = 0.0;                                                                  \
	for (int i = 0; i < TESTSIZE; ++i) {                                                    \
        mzd_clear(ee);                                                                      \
		double t0 = ((double)clock()/CLOCKS_PER_SEC);                                       \
		Sparsity<mcconfig>(ee, ss, A);                                                      \
		mc2_time += ((double)clock()/CLOCKS_PER_SEC) - t0;                                  \
		EXPECT_EQ(w, hamming_weight(ee));                                                   \
	}                                                                                       \
	std::cout << pr_time/TESTSIZE << " " <<                                                 \
	          mc_time/TESTSIZE << " " <<                                                    \
	          mc2_time/TESTSIZE << " " << "\n";                                             \
	mzd_free(A);                                                                            \
	mzd_free(AT);                                                                           \
	mzd_free(ss);                                                                           \
	mzd_free(ee);                                                                           \
}                                                                                           \



#define CUSTOM_TEST(name, subname, init, algo)                                              \
TEST(name, subname) {                                                                       \
	mzd_t *AT = mzd_from_str(n, n-k, h);                                                    \
	mzd_t *A = mzd_transpose(nullptr, AT);                                                  \
	mzd_t *ss = mzd_from_str(1, n-k, s);                                                    \
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);                                         \
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);                                      \
	mzd_t *ee = mzd_init(1, n);                                                             \
	mzd_t *ee_T = mzd_init(n, 1);                                                           \
	init;                                                                                   \
	auto aswd = 0;                                                                          \
	for (int j = 0; j < TESTSIZE; ++j) {                                                    \
        std::cout << j << "\n" << std::flush;                                               \
        mzd_clear(ee);                                                                      \
		aswd += algo;                                                                       \
		EXPECT_EQ(w, hamming_weight(ee));                                                   \
		mzd_transpose(ee_T, ee);                                                            \
		mzd_mul_naive(ss_tmp, A, ee_T);                                                     \
		mzd_transpose(ss_tmp_T, ss_tmp);                                                    \
		for (int i = 0; i < ss->ncols; ++i) {                                               \
			EXPECT_EQ(mzd_read_bit(ss, 0, i), mzd_read_bit(ss_tmp, i, 0));                  \
		}                                                                                   \
	}                                                                                       \
	print_matrix("FINAL e: ", ee);                                                          \
	print_matrix("FINAL should: ", ss);                                                     \
	print_matrix("FINAL is:", ss_tmp_T);                                                    \
	mzd_free(A);                                                                            \
	mzd_free(AT);                                                                           \
	mzd_free(ss);                                                                           \
	mzd_free(ss_tmp);                                                                       \
	mzd_free(ss_tmp_T);                                                                     \
	mzd_free(ee);                                                                           \
	mzd_free(ee_T);                                                                         \
}                                                                                           \

#define CUSTOM_PERF_TEST(name, subname, perff, init, algo)                                  \
TEST(name, subname) {                                                                       \
	mzd_t *AT = mzd_from_str(n, n-k, h);                                                    \
	mzd_t *A = mzd_transpose(nullptr, AT);                                                  \
	mzd_t *ss = mzd_from_str(1, n-k, s);                                                    \
	mzd_t *ss_tmp = mzd_init(ss->ncols, ss->nrows);                                         \
	mzd_t *ss_tmp_T = mzd_init( ss->nrows, ss->ncols);                                      \
	mzd_t *ee = mzd_init(1, n);                                                             \
	mzd_t *ee_T = mzd_init(n, 1);                                                           \
    perff;                                                                                  \
	init;                                                                                   \
	auto aswd = 0;                                                                          \
	for (int j = 0; j < TESTSIZE; ++j) {                                                    \
        mzd_clear(ee);                                                                      \
		aswd += algo;                                                                       \
		EXPECT_EQ(w, hamming_weight(ee));                                                   \
		mzd_transpose(ee_T, ee);                                                            \
		mzd_mul_naive(ss_tmp, A, ee_T);                                                     \
		mzd_transpose(ss_tmp_T, ss_tmp);                                                    \
		for (int i = 0; i < ss->ncols; ++i) {                                               \
			EXPECT_EQ(mzd_read_bit(ss, 0, i), mzd_read_bit(ss_tmp, i, 0));                  \
		}                                                                                   \
	}                                                                                       \
	print_matrix("FINAL e: ", ee);                                                          \
	print_matrix("FINAL s: ", ss);                                                          \
	print_matrix("FINAL ss:", ss_tmp_T);                                                    \
	perf.print();                                                                           \
   	perf.expected(config);                                                                  \
	mzd_free(A);                                                                            \
	mzd_free(AT);                                                                           \
	mzd_free(ss);                                                                           \
	mzd_free(ss_tmp);                                                                       \
	mzd_free(ss_tmp_T);                                                                     \
	mzd_free(ee);                                                                           \
	mzd_free(ee_T);                                                                         \
}                                                                                           \


#define Prange_TEST(subname)   CUSTOM_TEST(Prange, subname, FOO(static constexpr ConfigPrange config(G_n, G_n-G_k, G_w, G_n, -1);), prange<config>(ee, ss, A))
#define Sparsity_Test(subname) CUSTOM_TEST(McElieceV2, subname, FOO(static constexpr ConfigMcEliceV2 config(G_n, G_k, G_w, G_p, G_t, G_Thresh);), Sparsity<config>(ee, ss, A))
#define Decoding_TEST(subname) CUSTOM_TEST(Decoding, subname, ;, algorithm_only_m4ri_testing<G_n, G_k, G_l, G_w, G_p, G_d>(ee, ss, A))

#define Prange_Perf_TEST(subname)           CUSTOM_PERF_TEST(PerfPrange, subname, FOO(PerformancePrange perf;), FOO(static constexpr ConfigPrange config(G_n, G_n-G_k, G_w, G_n, -1);), prange<config>(ee, ss, A,  &perf))
//#define McEliece_v2_Perf_TEST(subname)    CUSTOM_PERF_TEST(PerfMcElieceV2, subname, FOO(static constexpr ConfigMcEliceV2 config(G_n, G_k, G_w, G_p, G_t, G_Thresh);), mc_eliece_d2_v2<config>(ee, ss, A))
//#define McEliece_v22_Perf_TEST(subname)   CUSTOM_PERF_TEST(PerfMcElieceV22, subname, FOO(static constexpr ConfigMcEliceV2 config(G_n, G_k, G_w, G_p, G_t, G_Thresh);), mc_eliece_d2_v22<config>(ee, ss, A))
//#define Decoding_Perf_TEST(subname)       CUSTOM_PERF_TEST(PerfDecoding, subname, ;, algorithm_only_m4ri_testing<G_n, G_k, G_l, G_w, G_p, G_d>(ee, ss, A))

#endif //SMALLSECRETLWE_TEST_H
