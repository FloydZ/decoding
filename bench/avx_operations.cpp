#include "b63.h"
#include "counters/perf_events.h"
#include <helper.h>
#include <custom_matrix.h>

#include "m4ri/m4ri.h"
#include "../test/quasicyclic/challenges/qc46.h"
//#include "../test/quasicyclic/challenges/qc10.h"

constexpr uint64_t size = 10000;
constexpr uint64_t limbs = size;
B63_BASELINE(AVX2_Float_Aligned, nn) {
	uint64_t *v1, *v2, *v3;

	B63_SUSPEND {
		v1 = (uint64_t *)aligned_alloc(4096, sizeof(uint64_t) * size);
		v2 = (uint64_t *)aligned_alloc(4096, sizeof(uint64_t) * size);
		v3 = (uint64_t *)aligned_alloc(4096, sizeof(uint64_t) * size);
	}

	for (int j = 0; j < nn; ++j) {
		for (uint64_t i = 0; i+4 < limbs; i+=4) {
			__m256 x_avx = _mm256_load_ps((float*)v1 + 2*i);
			__m256 y_avx = _mm256_load_ps((float*)v2 + 2*i);
			__m256 z_avx = _mm256_xor_ps(x_avx, y_avx);
			_mm256_store_ps((float*)v3 + 2*i, z_avx);
		}
	}

	B63_SUSPEND {
		free(v1);
		free(v2);
		free(v3);
	}
	B63_KEEP(v3[0]);
}

B63_BENCHMARK(AVX2_Fload_Unaligned, nn) {
	uint64_t *v1, *v2, *v3;

	B63_SUSPEND {
		v1 = (uint64_t *)malloc(sizeof(uint64_t) * size);
		v2 = (uint64_t *)malloc(sizeof(uint64_t) * size);
		v3 = (uint64_t *)malloc(sizeof(uint64_t) * size);
	}

	for (int j = 0; j < nn; ++j) {
		for (uint64_t i = 0; i+4 < limbs; i+=4) {
			__m256 x_avx = _mm256_loadu_ps((float*)v1 + 2*i);
			__m256 y_avx = _mm256_loadu_ps((float*)v2 + 2*i);
			__m256 z_avx = _mm256_xor_ps(x_avx, y_avx);
			_mm256_storeu_ps((float*)v3 + 2*i, z_avx);
		}
	}

	B63_SUSPEND {
		free(v1);
		free(v2);
		free(v3);
	}
	B63_KEEP(v3[0]);

}

B63_BENCHMARK(AVX2_m256i_Aligned, nn) {
	uint64_t *v1, *v2, *v3;

	B63_SUSPEND {
		v1 = (uint64_t *)aligned_alloc(4096, sizeof(uint64_t) * size);
		v2 = (uint64_t *)aligned_alloc(4096, sizeof(uint64_t) * size);
		v3 = (uint64_t *)aligned_alloc(4096, sizeof(uint64_t) * size);
	}

	for (int j = 0; j < nn; ++j) {
		for (uint64_t i = 0; i < limbs/4; i++) {
			__m256 x_avx = _mm256_load_si256((__m256i *)v1 + i);
			__m256 y_avx = _mm256_load_si256((__m256i *)v2 + i);
			__m256 z_avx = _mm256_xor_si256(x_avx, y_avx);
			_mm256_store_si256((__m256i *)v3 + i, z_avx);
		}
	}

	B63_SUSPEND {
		free(v1);
		free(v2);
		free(v3);
	}
	B63_KEEP(v3[0]);

}

B63_BENCHMARK(AVX2_m256i_Unligned, nn) {
	uint64_t *v1, *v2, *v3;

	B63_SUSPEND {
		v1 = (uint64_t *)malloc(sizeof(uint64_t) * size);
		v2 = (uint64_t *)malloc(sizeof(uint64_t) * size);
		v3 = (uint64_t *)malloc(sizeof(uint64_t) * size);
	}

	for (int j = 0; j < nn; ++j) {
		for (uint64_t i = 0; i < limbs/4; i++) {
			__m256 x_avx = _mm256_loadu_si256((__m256i *)v1 + i);
			__m256 y_avx = _mm256_loadu_si256((__m256i *)v2 + i);
			__m256 z_avx = _mm256_xor_si256(x_avx, y_avx);
			_mm256_storeu_si256((__m256i *)v3 + i, z_avx);
		}
	}

	B63_SUSPEND {
		free(v1);
		free(v2);
		free(v3);
	}
	B63_KEEP(v3[0]);

}
int main(int argc, char **argv) {
	B63_RUN_WITH("time,lpe:branches,lpe:branch-misses,lpe:cache-misses,lpe:cycles,lpe:instructions,lpe:ref-cycles", argc, argv);
	return 0;
}