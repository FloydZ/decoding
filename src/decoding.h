#ifndef SMALLSECRETLWE_DECODING_H
#define SMALLSECRETLWE_DECODING_H

#include <sys/mman.h>
#include <cstdlib>
#include <strings.h>
#include <sys/random.h>
#include <random>

#include "m4ri/m4ri.h"
#include "m4ri/brilliantrussian.h"

#include "helper.h"
#include "glue_m4ri.h"
#include "custom_matrix.h"
#include "combinations.h"
#include "tree.h"

using DecodingValue     = Value_T<BinaryContainer<G_k + G_l>>;
using DecodingLabel     = Label_T<BinaryContainer<G_n - G_k>>;
using DecodingMatrix    = mzd_t *;
using DecodingElement   = Element_T<DecodingValue, DecodingLabel, DecodingMatrix>;
using DecodingList      = List_T<DecodingElement>;
using DecodingTree      = Tree_T<DecodingList>;

#include "list_generator.h"
#include "list_fill.h"
#include "list_check.h"

/// Output:
///				n
/// _________________________
///	|			|			|
///	|	I_n-k	|	H_2		|
/// |___________|___________|
/// |			|			|	n-k
/// |	0		|	H_1		|
/// |___________|___________|
/// Assumes: A is a (n-k)x(n+1) Matrix
/// Assumes: A has one additional column which should be excluded in all operations.
/// Output: a cxc identity matrix in the upper left corner and a zero matrix (n-k-c)x(c) in the lower left corner.
// permute the given matrix ans the solves it on the upper left n-k-l bits
// return -1 on error, dimension of created identity  matrix else.
inline int permute(mzd_t *A,  mzp_t *P_C) {
	ASSERT(A != nullptr && P_C != nullptr && A->ncols > 0);
	matrix_create_random_permutation(A, P_C);
	return mzd_echelonize(A, true);
}

struct ConfigLeeBrickell {
	///
	/// \tparam n #columns
	/// \tparam k n-k=#rows
	/// \tparam l addtioninal zero window
	/// \tparam w weight of the solution
	/// \tparam p weight on hte k-l part
public:
	const uint32_t n, k, w;     // instance Parameters
	const uint32_t l, p;        // opt parameters
	const int m4ri_k;           // opt parameter for faster gaus elimination
	constexpr ConfigLeeBrickell(uint32_t n, uint32_t k, uint32_t w, uint32_t p, uint32_t l) :
	n(n), k(k), w(w), p(p), l(l),
	m4ri_k(matrix_opt_k(k, n)) {}
};


/// \param e
/// \param s
/// \param A
/// \return
template<const uint64_t n, const uint64_t k, const uint64_t l, const uint64_t w, const uint64_t p, const uint64_t d>
int LeeBrickell(mzd_t *e, const mzd_t *const s, const mzd_t *const A) {
    ASSERT(n == A->ncols && n-k == A->nrows && n >= k-l && w > p);
    ASSERT(n-k == s->ncols && 1 == s->nrows);
    ASSERT(e->nrows == 1 && s->nrows == 1);

    // helper functions to better access H_1, H_2
    mzd_t *H1 = mzd_init(l, k+l);
    mzd_t *H1T = mzd_init(k+l, l);
    mzd_t *H2 = mzd_init(n-k-l, k+l);

    // helper to access s1/s2;
    mzd_t *s1 = mzd_init(n-k-l, 1);
    mzd_t *s1T = mzd_init(1, n-k-l);
    mzd_t *s2 = mzd_init(l, 1);
	mzd_t *s2T = mzd_init(1, l);

    // for the inner loop
    mzd_t *e1 = mzd_init(1, n-k-l);
    mzd_t *e2 = mzd_init(1, k+l);
    mzd_t *e2_old = mzd_init(1, k+l);

    mzd_t *le2 = mzd_init(1, l);    // IMPORTANT already transposed

    // needed for correctness
    mzd_t *e_T = mzd_init(n, 1);
    mzd_t *ss = mzd_init(1, n-k);
    mzd_t *ss_T = mzd_init(n-k, 1);
    mzd_t *e_prime = mzd_init(1, n);

    uint64_t outer_loops = 0, inner_loops = 0;
    mzp_t *P_C = mzp_init(n);

    mzd_t *work_matrix_H = mzd_init(A->nrows, A->ncols+1);
    mzd_t *columnTransposed = mzd_transpose(NULL, s);
    mzd_concat(work_matrix_H, A, columnTransposed);

	// init permutation
    for(rci_t i = 0; i < n; i++) P_C->values[i] = i;

    // init e2 table:
	using DecodingContainer  = BinaryContainer<k + l>;
	std::vector<DecodingContainer> table;
	std::vector<std::pair<uint64_t, uint64_t>> diff;

	Combinations_Chase2<DecodingContainer> cm{k+l, p, 0};
	cm.table(table, diff);

	// this will increase massively the access speed.
	MADIVE(table.data(), table.size()*sizeof(DecodingContainer)/8, POSIX_MADV_WILLNEED | POSIX_MADV_SEQUENTIAL);

    int ra = 1;
    while (ra != 0) {
        ASSERT(work_matrix_H->ncols == n+1 && work_matrix_H->nrows == n-k);
        ra = permute(work_matrix_H, P_C);
        if (ra < n-k-l) {
            std::cout << "permute error\n";
            continue;
        }

        // Extract the sub matrices
        copy_submatrix(H1, work_matrix_H, n-k-l, n-k-l, n-k, n);
        copy_submatrix(H2, work_matrix_H, 0, n-k-l, n-k-l, n);

        // because we permute s we need to extract it each turn
        copy_submatrix(s1, work_matrix_H, 0, n, n-k-l, n+1);
        copy_submatrix(s2, work_matrix_H, n-k-l, n, n-k, n+1);

        // Do as much transposing as possible in the outer loop. Because the inner loop is much more often looped
	    mzd_transpose(H1T, H1);
	    mzd_transpose(s1T, s1);
	    mzd_transpose(s2T, s2);

	    // approach 3;
	    uint64_t table_i = 1;
	    e2->rows[0] = table[1].data().data();
	    e2_old->rows[0] = table[0].data().data();
	    _mzd_mul_naive(le2, e2_old, H1, 1);

	    do {
		    if (m4ri_cmp_row_testing<l / 64 + 1>(s2T, le2) == 0) {
			    // calc e1 = s1 + H2*e2
			    //_mzd_mul_m4rm(e1, e2_old, H2, 256,  TRUE);
			    _mzd_mul_naive(e1, e2, H2, true);
			    mzd_add(e1, e1, s1T);
			    unsigned int wte1 = hamming_weight(e1);

			    if (wte1 == w - p) {
				    //speed is not that important here.
				    // first copy e1
				    for (int i = 0; i < e1->ncols; ++i) {
					    mzd_write_bit(e, 0, i, mzd_read_bit(e1, 0, i));
				    }

				    // now copy e2
				    for (int i = 0; i < e2_old->ncols; ++i) {
					    mzd_write_bit(e, 0, i + e1->ncols, mzd_read_bit(e2, 0, i));
				    }

				    // apply inverse permutation on columns
				    for (rci_t i = 0; i < n; ++i) {
					    mzd_write_bit(e_prime, 0, P_C->values[i], mzd_read_bit(e, 0, i));
				    }

				    mzd_transpose(e_T, e_prime);
				    mzd_mul_naive(ss_T, A, e_T);
				    mzd_transpose(ss, ss_T);

				    if (mzd_cmp(ss, s) == 0) {
					    goto finish;
				    }
			    }
			    inner_loops += 1;
		    }

		    e2_old->rows[0] = e2->rows[0];
		    e2->rows[0] = table[table_i].data().data();

		    // possible the fastest way.
		    ASSERT(diff[table_i].first < H1T->nrows && diff[table_i].second < H1T->nrows);
		    mzd_combine_even_in_place(le2, 0, 0, H1T, diff[table_i].first, 0);
		    mzd_combine_even_in_place(le2, 0, 0, H1T, diff[table_i].second, 0);

		    table_i += 1;
	    } while (table_i <= table.size());

        outer_loops += 1;
    }

finish:
#ifdef DEBUG
	std::cout << "number of outer loops: " << outer_loops << "\n";
	std::cout << "number of inner loops per outer loop: " << ((outer_loops > 0) ? inner_loops/outer_loops : inner_loops) << "\n";
#endif

	mzd_copy(e, e_prime);

	mzd_free(work_matrix_H);
    mzp_free(P_C);
    mzd_free(e1);
    mzd_free(e_T);
    mzd_free(e_prime);
    mzd_free(e2);
    mzd_free(le2);
    mzd_free(H1);
    mzd_free(H1T);
    mzd_free(H2);
    mzd_free(s1);
    mzd_free(s2);
    mzd_free(ss);
    mzd_free(ss_T);

    return 0;
}


#endif //SMALLSECRETLWE_DECODING_H
