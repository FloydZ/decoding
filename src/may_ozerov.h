#ifndef DECODING_MAY_OZEROV_H
#define DECODING_MAY_OZEROV_H

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "isd.h"

#if defined(USE_CUDA) && defined(__CUDACC__)
// threads per block and elements per tile of the scan
constexpr uint32_t MO_SCAN_THREADS = 256, MO_SCAN_ITEMS = 8;
constexpr uint32_t MO_SCAN_TILE = MO_SCAN_THREADS * MO_SCAN_ITEMS;

/// exclusive scan of `in[0, n)` plus `carry` into `out` (may alias `in`), `n <= MO_SCAN_TILE`, returns the sum
__device__ uint32_t mo_tile_scan(uint32_t *out, const uint32_t *in, const uint32_t n, const uint32_t carry) {
	__shared__ uint32_t s[MO_SCAN_TILE], w[MO_SCAN_THREADS / 32u], total;
	// NOTE: all threads must be done with the previous tile
	__syncthreads();
	for (uint32_t i = threadIdx.x; i < MO_SCAN_TILE; i += MO_SCAN_THREADS) {
		s[i] = i < n ? in[i] : 0u;
	}
	__syncthreads();

	// each thread scans `MO_SCAN_ITEMS` consecutive values
	uint32_t *t = s + threadIdx.x * MO_SCAN_ITEMS, sum = 0;
	#pragma unroll
	for (uint32_t i = 0; i < MO_SCAN_ITEMS; i++) {
		const uint32_t x = t[i];
		t[i] = sum;
		sum += x;
	}

	// thread sums within the warp, then the warp sums by the first warp
	const uint32_t lane = threadIdx.x % 32u, warp = threadIdx.x / 32u;
	const uint32_t exc = warp_scan_exc_up_add_shuffled(sum);
	if (lane == 31u) { w[warp] = exc + sum; }
	__syncthreads();
	if (warp == 0) {
		const uint32_t v = lane < (MO_SCAN_THREADS / 32u) ? w[lane] : 0u;
		const uint32_t e = warp_scan_exc_up_add_shuffled(v);
		if (lane < (MO_SCAN_THREADS / 32u)) { w[lane] = e; }
		if (lane == 31u) { total = e + v; }
	}
	__syncthreads();

	const uint32_t o = carry + w[warp] + exc;
	#pragma unroll
	for (uint32_t i = 0; i < MO_SCAN_ITEMS; i++) { t[i] += o; }
	__syncthreads();

	for (uint32_t i = threadIdx.x; i < n; i += MO_SCAN_THREADS) { out[i] = s[i]; }
	return total;
}

/// scan 1/3: exclusive scan of each tile of `in[0, n)` into `out`, the tile sums into `sums`
__global__ void __launch_bounds__(MO_SCAN_THREADS)
mo_scan_tiles(uint32_t *out, uint32_t *sums, const uint32_t *in, const uint32_t n) {
	const uint32_t base = blockIdx.x * MO_SCAN_TILE;
	const uint32_t m = (n - base) < MO_SCAN_TILE ? (n - base) : MO_SCAN_TILE;
	const uint32_t s = mo_tile_scan(out + base, in + base, m, 0u);
	if (threadIdx.x == 0) { sums[blockIdx.x] = s; }
}

/// scan 2/3: exclusive scan of the `n` tile sums in place, by a single block
__global__ void __launch_bounds__(MO_SCAN_THREADS)
mo_scan_sums(uint32_t *sums, const uint32_t n) {
	uint32_t carry = 0;
	for (uint32_t base = 0; base < n; base += MO_SCAN_TILE) {
		const uint32_t m = (n - base) < MO_SCAN_TILE ? (n - base) : MO_SCAN_TILE;
		carry += mo_tile_scan(sums + base, sums + base, m, carry);
	}
}

/// scan 3/3: adds the scanned tile sums to `out[0, n)`, and copies `out[0, n-1)` into `cur`
__global__ void mo_scan_add(uint32_t *out, uint32_t *cur, const uint32_t *sums, const uint32_t n) {
	const uint32_t stride = gridDim.x * blockDim.x;
	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
		const uint32_t x = out[i] + sums[i / MO_SCAN_TILE];
		out[i] = x;
		if (i + 1u < n) { cur[i] = x; }
	}
}
/// writes the `p` rows of the subset with rank `x` into `rows`, same as `biject`
template<const uint32_t p>
__device__ __forceinline__ void mo_unrank(uint32_t x, uint32_t rows[p]) {
	#pragma unroll
	for (uint32_t j = 0; j < p; j++) {
		const uint32_t pj = p - j;
		if (pj == 1u) {
			rows[j] = x;
			break;
		}

		// max. c with bc(c, pj) <= x, estimated in `float`
		uint32_t c;
		if (pj == 2u) {
			c = (uint32_t(sqrtf(8.f * float(x) + 1.f)) + 1u) / 2u;
		} else if (pj == 3u) {
			c = uint32_t(cbrtf(6.f * float(x))) + 1u;
		} else {
			c = (uint32_t(sqrtf(4.f * floorf(sqrtf(24.f * float(x) + 1.f)) + 5.f)) + 3u) / 2u;
		}

		// NOTE: fixes rounding errors of the estimate
		uint64_t b;
		while (true) {
			b = 1;
			for (uint32_t t = 0; t < pj; t++) { b = b * (c - t) / (t + 1u); }
			if (b <= x) { break; }
			c--;
		}
		while (true) {
			uint64_t b1 = 1;
			for (uint32_t t = 0; t < pj; t++) { b1 = b1 * (c + 1u - t) / (t + 1u); }
			if (b1 > x) { break; }
			b = b1;
			c++;
		}
		rows[j] = c;
		x -= uint32_t(b);
	}
}

/// base lists 1/2: per subset `i` the left key `KL[i]` (+ histogram `cnt`), the right key `KR[i]` and element `R[i]`
template<typename T, const uint32_t limbs, const uint32_t half, const uint32_t p>
__global__ void mo_base_keys(uint32_t *KL, uint32_t *cnt, uint32_t *KR, T *R,
                             const uint32_t *RK, const T *RS, const uint32_t size) {
	const uint32_t stride = gridDim.x * blockDim.x;
	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += stride) {
		uint32_t rows[p];
		mo_unrank<p>(i, rows);

		uint32_t kl = 0, kr = 0;
		T r[limbs] = {0};
		for (uint32_t a = 0; a < p; a++) {
			kl ^= RK[rows[a]];
			kr ^= RK[rows[a] + half];
			for (uint32_t j = 0; j < limbs; j++) { r[j] ^= RS[(rows[a] + half) * limbs + j]; }
		}

		KL[i] = kl;
		atomicAdd(cnt + kl, 1u);
		KR[i] = kr;
		for (uint32_t j = 0; j < limbs; j++) { R[size_t(i) * limbs + j] = r[j]; }
	}
}

/// base lists 2/2: writes the left element of subset `i` to `L[cur[KL[i]]++]`, i.e. sorted by key
template<typename T, const uint32_t limbs, const uint32_t p>
__global__ void mo_base_scatter(T *L, uint32_t *cur, const uint32_t *KL,
                                const T *RS, const uint32_t size) {
	const uint32_t stride = gridDim.x * blockDim.x;
	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += stride) {
		uint32_t rows[p];
		mo_unrank<p>(i, rows);

		T x[limbs] = {0};
		for (uint32_t a = 0; a < p; a++) {
			for (uint32_t j = 0; j < limbs; j++) { x[j] ^= RS[rows[a] * limbs + j]; }
		}

		const uint32_t pos = atomicAdd(cur + KL[i], 1u);
		for (uint32_t j = 0; j < limbs; j++) { L[size_t(pos) * limbs + j] = x[j]; }
	}
}

/// level-1 lists: `L1` += L[bucket KR[j]^t1] ^ R[j], `L2` += L[bucket KR[j]^t2] ^ R[j] ^ wss; sizes in `E` (zeroed), max. `cap` written
template<typename T, const uint32_t limbs>
__global__ void mo_merge(T *L1, T *L2, uint32_t *E,
                         const T *L, const uint32_t *off,
                         const T *R, const uint32_t *KR, const T *wss,
                         const uint32_t t1, const uint32_t t2,
                         const uint32_t size, const uint32_t cap) {
	const uint32_t lane = threadIdx.x % 32u;
	const uint32_t stride = gridDim.x * blockDim.x;
	// NOTE: all threads of a warp run the same number of iterations
	for (uint32_t base = blockIdx.x * blockDim.x; base < size; base += stride) {
		const uint32_t j = base + threadIdx.x;
		T r[limbs] = {0};
		uint32_t s[2] = {0, 0}, c[2] = {0, 0};
		if (j < size) {
			const uint32_t key = KR[j];
			for (uint32_t i = 0; i < limbs; i++) { r[i] = R[size_t(j) * limbs + i]; }
			s[0] = off[key ^ t1]; c[0] = off[(key ^ t1) + 1u] - s[0];
			s[1] = off[key ^ t2]; c[1] = off[(key ^ t2) + 1u] - s[1];
		}

		for (uint32_t list = 0; list < 2; list++) {
			// reserve the space of the whole warp at once
			const uint32_t exc = warp_scan_exc_up_add_shuffled(c[list]);
			uint32_t o = 0;
			if (lane == 31u) { o = atomicAdd(E + list, exc + c[list]); }
			o = __shfl_sync(0xffffffff, o, 31) + exc;

			T *O = list ? L2 : L1;
			for (uint32_t a = 0; (a < c[list]) && (o + a < cap); a++) {
				for (uint32_t i = 0; i < limbs; i++) {
					O[size_t(o + a) * limbs + i] = L[size_t(s[list] + a) * limbs + i] ^ r[i] ^ (list ? wss[i] : 0);
				}
			}
		}
	}
}

/// copies the NN solution `L1[*f1]`, `L2[*f2]` into `S`, if there is one
template<typename T, const uint32_t limbs>
__global__ void mo_solution(T *S, const T *L1, const T *L2,
                            const size_t *f1, const size_t *f2) {
	const uint32_t i = threadIdx.x;
	if ((*f1 != size_t(-1)) && (i < limbs)) {
		S[i] = L1[*f1 * limbs + i];
		S[limbs + i] = L2[*f2 * limbs + i];
	}
}
#endif

/// May-Ozerov: depth 2 tree on the `l` window, NN on the other `n-k-l` bits; weight `p` per base list, i.e. `4p` on `k+l`
struct ConfigMayOzerov : public ConfigISD {
public:
	const uint32_t r = 2;      // NN levels
	const uint32_t N = 50;     // NN leaves per level
	const uint32_t dk = 12;    // NN max. weight per limb
	const uint32_t nnk = 32;   // NN bits per limb
	const uint32_t bf = 1024;  // NN bruteforce threshold

	// number of intermediate targets per permutation
	const uint32_t intermediate_loops = 1;

	// debug mode, enforces the bruteforce instead of the NN
	const bool enforce_bf = false;

	// runs the tree and the NN on the GPU, needs a CUDA compiler
	const bool cuda = false;

	/// expected number of permutations (ignores the representations)
	[[nodiscard]] consteval uint64_t compute_loops() const noexcept {
#ifdef EXPECTED_PERMUTATIONS
		return EXPECTED_PERMUTATIONS;
#else
		return bc(n, w)/(bc(n-k-l, w-4*p) * bc((k+l)/2, 2*p) * bc((k+l)/2, 2*p));
#endif
	}

	/// prints the ISD and the MO parameters
	void print() const noexcept {
		ConfigISD::print();
		std::cout << "{ \"r\": " << r
		          << ", \"N\": " << N
		          << ", \"dk\": " << dk
		          << ", \"nnk\": " << nnk
		          << ", \"bf\": " << bf
		          << ", \"intermediate_loops\": " << intermediate_loops
		          << ", \"cuda\": " << cuda
		          << " }" << std::endl;
	}
};

template<const ConfigISD &isd, const ConfigMayOzerov &config>
class MayOzerov : public ISDInstance<uint64_t, isd> {
public:
	constexpr static uint32_t n = isd.n;
	constexpr static uint32_t k = isd.k;
	constexpr static uint32_t w = isd.w;
	constexpr static uint32_t p = isd.p;
	constexpr static uint32_t l = isd.l;

	using ISD = ISDInstance<uint64_t, isd>;
	using Error     = ISD::Error;
	using Label     = ISD::Label;
	using limb_type = ISD::limb_type;
	using ISD::HT,ISD::ws,ISD::e,ISD::syndrome,ISD::P,ISD::not_found,ISD::loops,ISD::expected_loops;

	static_assert(isd.q == 2);
	static_assert((p >= 1) && (p <= 4));
	static_assert(4*p <= w);
	// NOTE: 2^l buckets, and the label shift needs 0 < l < 64
	static_assert((l > 0) && (l <= 28));
	static_assert((n-k-l > 32) && (n-k-l <= 256), "NN restriction");

	constexpr static uint32_t kl_half = (k+l)/2;
	// size of each base list
	constexpr static size_t B = bc(kl_half, p);
	static_assert(B < (1ull << 32u));
	constexpr static uint32_t nr_buckets = 1u << l;
	constexpr static limb_type lmask = (limb_type(1) << l) - 1u;
	// limbs of a label (n-k bits) and of a NN element (n-k-l bits)
	constexpr static uint32_t LABEL_LIMBS = (n-k+63)/64;
	constexpr static uint32_t NR_LIMBS = (n-k-l+63)/64;

	// expected size of a level-1 list, and its capacity
	constexpr static size_t E = (B * B) >> l;
	constexpr static size_t LIST_SIZE = E + E/4 + 1024;

	constexpr static NN_Config nn_config{
		n-k-l, config.r, config.N, config.nnk,
		LIST_SIZE, config.dk, w-(4*p), 0, config.bf};
	NN<nn_config> algo {};
	using NNElement = typename NN<nn_config>::Element;
	static_assert(NR_LIMBS == NN<nn_config>::ELEMENT_NR_LIMBS);
	static_assert(config.r <= NR_LIMBS);

	// simd type of an element: `NR_LIMBS` limbs, 3 are padded to 4
	constexpr static uint32_t VL = NR_LIMBS == 3 ? 4 : NR_LIMBS;
	using V = std::conditional_t<VL == 1, uint64_t,
	          std::conditional_t<VL == 2, cryptanalysislib::_uint64x2_t, uint64x4_t>>;
	using VElement = uint64_t[VL];

	// split rows of `HT`: l window and the rest
	uint32_t *RK = nullptr;
	VElement *RS = nullptr;
	// base lists sorted by key: elements, base indices, bucket offsets
	VElement *BL = nullptr, *BR = nullptr;
	uint32_t *BLi = nullptr, *BRi = nullptr, *offL = nullptr, *offR = nullptr;
	// keys of the current base list, padded by `prefetch_dist` zeros
	constexpr static uint32_t prefetch_dist = 16;
	uint32_t *tK = nullptr;
	// `ws` without the l window
	alignas(32) VElement wss;

	// current intermediate target, and the 4p rows of the solution
	uint32_t target = 0;
	uint32_t solutions[4*p] = {0};
	// number of merges which were cut, because a list was full
	size_t dropped = 0;

#if defined(USE_CUDA) && defined(__CUDACC__)
	// GPU: per slot (2) pinned split rows (`RS`, `wss`, `RK`), copies of `HT`, `ws`, `P`, `syndrome`
	constexpr static uint32_t threads = 256;
	constexpr static uint32_t blocks = std::min<size_t>((B + threads - 1) / threads, 8192);
	constexpr static size_t size_rows = (k+l+1)*NR_LIMBS + (k+l+1)/2;
	constexpr static size_t size_HT = (k+l+1)*LABEL_LIMBS;
	limb_type *hRows[2] = {nullptr, nullptr}, *hHT[2] = {nullptr, nullptr};
	uint32_t *hP[2] = {nullptr, nullptr};
	limb_type hSyn[2] = {0, 0};
	// pinned: sizes of the level-1 lists and the solution elements
	uint32_t *hE = nullptr;
	limb_type *hSol = nullptr;
	// device: split rows, base lists, keys, bucket counts/offsets/cursors, sizes, solution, scan tile sums
	constexpr static uint32_t scan_tiles = (nr_buckets + 1u + MO_SCAN_TILE - 1u) / MO_SCAN_TILE;
	limb_type *dRows = nullptr, *dBL = nullptr, *dBR = nullptr, *dSol = nullptr;
	uint32_t *dKL = nullptr, *dKR = nullptr, *dCnt = nullptr, *dOff = nullptr, *dCur = nullptr, *dE = nullptr;
	uint32_t *dSums = nullptr;
	cudaStream_t stream = nullptr;
	cudaEvent_t ev_merge = nullptr, ev_nn = nullptr;
#endif

	/// allocates the base and level-1 lists
	MayOzerov() noexcept {
		nn_config.print();
		expected_loops = config.compute_loops();

		constexpr size_t sB = roundToAligned<64>(B * sizeof(uint32_t));
		constexpr size_t sO = roundToAligned<64>((nr_buckets + 1) * sizeof(uint32_t));
		RK   = (uint32_t *)cryptanalysislib::aligned_alloc(64, roundToAligned<64>((k+l) * sizeof(uint32_t)));
		RS   = (VElement *)cryptanalysislib::aligned_alloc(64, roundToAligned<64>((k+l) * sizeof(VElement)));
		BL   = (VElement *)cryptanalysislib::aligned_alloc(64, roundToAligned<64>(B * sizeof(VElement)));
		BR   = (VElement *)cryptanalysislib::aligned_alloc(64, roundToAligned<64>(B * sizeof(VElement)));
		BLi  = (uint32_t *)cryptanalysislib::aligned_alloc(64, sB);
		BRi  = (uint32_t *)cryptanalysislib::aligned_alloc(64, sB);
		tK   = (uint32_t *)cryptanalysislib::aligned_alloc(64, roundToAligned<64>((B + prefetch_dist) * sizeof(uint32_t)));
		std::memset(tK, 0, (B + prefetch_dist) * sizeof(uint32_t));
		offL = (uint32_t *)cryptanalysislib::aligned_alloc(64, sO);
		offR = (uint32_t *)cryptanalysislib::aligned_alloc(64, sO);

		// NOTE: freed by `~NN`
		algo.L1 = (NNElement *)cryptanalysislib::aligned_alloc(64, roundToAligned<64>(LIST_SIZE * sizeof(NNElement)));
		algo.L2 = (NNElement *)cryptanalysislib::aligned_alloc(64, roundToAligned<64>(LIST_SIZE * sizeof(NNElement)));
	}

	/// frees the base lists
	~MayOzerov() noexcept {
		cryptanalysislib::aligned_free(RK);
		cryptanalysislib::aligned_free(RS);
		cryptanalysislib::aligned_free(BL);
		cryptanalysislib::aligned_free(BR);
		cryptanalysislib::aligned_free(BLi);
		cryptanalysislib::aligned_free(BRi);
		cryptanalysislib::aligned_free(tK);
		cryptanalysislib::aligned_free(offL);
		cryptanalysislib::aligned_free(offR);
#if defined(USE_CUDA) && defined(__CUDACC__)
		// NOTE: no `checkCudaErrors`, it calls `exit`
		for (uint32_t i = 0; i < 2; i++) {
			if (hRows[i]) { cudaFreeHost(hRows[i]); }
			free(hHT[i]);
			free(hP[i]);
		}
		if (hE) { cudaFreeHost(hE); }
		if (hSol) { cudaFreeHost(hSol); }
		for (void *ptr : {(void *)dRows, (void *)dBL, (void *)dBR, (void *)dSol, (void *)dKL, (void *)dKR,
		                  (void *)dCnt, (void *)dOff, (void *)dCur, (void *)dE, (void *)dSums}) {
			if (ptr) { cudaFree(ptr); }
		}
		if (ev_merge) { cudaEventDestroy(ev_merge); }
		if (ev_nn) { cudaEventDestroy(ev_nn); }
		if (stream) { cudaStreamDestroy(stream); }
#endif
	}

	/// loads an element
	static inline V vload(const uint64_t *ptr) noexcept {
		if constexpr (VL == 1) { return *ptr; }
		else { return V::aligned_load(ptr); }
	}

	/// xor of two elements
	static inline V vxor(const V a, const V b) noexcept {
		if constexpr (VL == 1) { return a ^ b; }
		else { return V::xor_(a, b); }
	}

	/// stores the first `NR_LIMBS` limbs of `v`
	static inline void vstore(uint64_t *ptr, const V v) noexcept {
		if constexpr (VL == 1) {
			*ptr = v;
		} else if constexpr (VL == NR_LIMBS) {
			V::unaligned_store(ptr, v);
		} else {
			alignas(32) uint64_t t[VL];
			V::aligned_store(t, v);
			std::memcpy(ptr, t, NR_LIMBS * sizeof(uint64_t));
		}
	}

	/// splits the label `in` into its l window `key` and the rest `out`
	static inline void split(VElement &out, uint32_t &key, const limb_type *in) noexcept {
		key = in[0] & lmask;
		#pragma unroll
		for (uint32_t j = 0; j < VL; j++) {
			const limb_type lo = (j < LABEL_LIMBS) ? (in[j] >> l) : 0;
			const limb_type hi = (j + 1u < LABEL_LIMBS) ? (in[j + 1u] << (64u - l)) : 0;
			out[j] = lo | hi;
		}
	}

	/// calls `f` with the xor (`get(x, row)`) of every `d`-subset of [0, end), in `biject` order
	template<const uint32_t d, typename T, typename G, typename F>
	static inline void enumerate(const G &get, const T x, const uint32_t end, F &&f) noexcept {
		for (uint32_t c = d - 1u; c < end; c++) {
			const T y = get(x, c);
			if constexpr (d == 1) { f(y); }
			else { enumerate<d - 1u>(get, y, c, f); }
		}
	}

	/// builds the base list of the rows [offset, offset + kl_half), sorted by key
	void build(VElement *L, uint32_t *Li, uint32_t *off, const uint32_t offset) noexcept {
		// keys and their histogram
		std::memset(off, 0, (nr_buckets + 1) * sizeof(uint32_t));
		uint32_t i = 0;
		const uint32_t *rk = RK + offset;
		enumerate<p, uint32_t>([rk](const uint32_t x, const uint32_t c) { return x ^ rk[c]; }, 0u, kl_half,
			[&](const uint32_t key) { tK[i++] = key; off[key + 1u] += 1; });
		for (uint32_t b = 0; b < nr_buckets; b++) { off[b + 1u] += off[b]; }

		// elements, written to their sorted position
		i = 0;
		const VElement *rs = RS + offset;
		const V zero = vxor(vload(rs[0]), vload(rs[0]));
		enumerate<p, V>([rs](const V x, const uint32_t c) { return vxor(x, vload(rs[c])); }, zero, kl_half,
			[&](const V x) {
				// NOTE: the scatter misses the cache, prefetch the next targets
				__builtin_prefetch(L + off[tK[i + prefetch_dist]], 1);
				__builtin_prefetch(Li + off[tK[i + prefetch_dist]], 1);
				const uint32_t pos = off[tK[i]]++;
				if constexpr (VL == 1) { L[pos][0] = x; }
				else { V::aligned_store(L[pos], x); }
				Li[pos] = i++;
			});

		// NOTE: `off[b]` is now the end of bucket `b`, shift it back
		for (uint32_t b = nr_buckets; b > 0; b--) { off[b] = off[b - 1u]; }
		off[0] = 0;
	}

	/// builds both base lists, each sorted by its l window
	void construct_base_lists() noexcept {
		for (uint32_t i = 0; i < k+l; i++) {
			split(RS[i], RK[i], HT[i]);
		}
		uint32_t key;
		split(wss, key, ws.ptr());

		build(BL, BLi, offL, 0);
		build(BR, BRi, offR, kl_half);
	}

	/// writes all `BL[a] ^ BR[j] (^ wss)` with key(a) ^ key(j) = `t` into `L`, returns their number
	template<const bool add_ws>
	size_t merge(NNElement *L, const uint32_t t) noexcept {
		const V W = vload(wss);
		size_t ctr = 0;
		for (uint32_t b = 0; b < nr_buckets; b++) {
			const uint32_t ls = offL[b ^ t], le = offL[(b ^ t) + 1u];
			const uint32_t rs = offR[b], re = offR[b + 1u];
			if (ctr + size_t(le - ls) * (re - rs) > LIST_SIZE) {
				dropped += 1;
				break;
			}

			for (uint32_t j = rs; j < re; j++) {
				V r = vload(BR[j]);
				if constexpr (add_ws) { r = vxor(r, W); }
				for (uint32_t a = ls; a < le; a++, ctr++) {
					vstore(L[ctr], vxor(vload(BL[a]), r));
				}
			}
		}
		return ctr;
	}

	/// finds the `2p` rows (left, then right) of `sol` in `merge<add_ws>(t)`
	template<const bool add_ws>
	bool find_pair(uint32_t *rows, const NNElement &sol, const uint32_t t) const noexcept {
		uint16_t r[p];
		for (uint32_t b = 0; b < nr_buckets; b++) {
			for (uint32_t j = offR[b]; j < offR[b + 1u]; j++) {
				for (uint32_t a = offL[b ^ t]; a < offL[(b ^ t) + 1u]; a++) {
					bool eq = true;
					for (uint32_t i = 0; i < NR_LIMBS; i++) {
						eq &= (BL[a][i] ^ BR[j][i] ^ (add_ws ? wss[i] : 0)) == sol[i];
					}
					if (!eq) { continue; }

					biject<kl_half, p>(BLi[a], r);
					for (uint32_t s = 0; s < p; s++) { rows[s] = r[s]; }
					biject<kl_half, p>(BRi[j], r);
					for (uint32_t s = 0; s < p; s++) { rows[p + s] = r[s] + kl_half; }
					return true;
				}
			}
		}
		return false;
	}

	/// computes `e` from the NN solution `sol1`, `sol2` of the current permutation and `target`
	void __attribute__ ((noinline))
	reconstruct(const limb_type *sol1, const limb_type *sol2) noexcept {
		NNElement s1, s2;
		std::memcpy(s1, sol1, sizeof(NNElement));
		std::memcpy(s2, sol2, sizeof(NNElement));

		// NOTE: the base lists are rebuilt, the GPU path does not keep them
		construct_base_lists();
		const bool f1 = find_pair<false>(solutions, s1, target);
		const bool f2 = find_pair<true>(solutions + 2*p, s2, target ^ uint32_t(syndrome));
		assert(f1 && f2);
		(void)f1; (void)f2;

		Label tmp = ws;
		for (uint32_t i = 0; i < 4*p; ++i) {
			Label::add(tmp.ptr(), tmp.ptr(), HT.row(solutions[i]));
		}

		Error tmpe;
		for (uint32_t i = 0; i < n-k; ++i) {
			tmpe.set(tmp.get(i), 0, n-k-1-i);
		}

		// NOTE: toggle, the two level-1 vectors may share rows
		for (uint32_t i = 0; i < 4*p; ++i) {
			const uint32_t pos = n-k-l + solutions[i];
			tmpe.set(tmpe.get(0, pos) ^ 1u, 0, pos);
		}

		for (uint32_t i = 0; i < n; ++i) {
			e.set(tmpe.get(0, i), 0, P.values[i]);
		}
	}

	/// one permutation: base lists, then per target the level-1 lists and the NN
	void step() noexcept {
		ISD::step();
		construct_base_lists();

		uint32_t t = 0;
		for (uint32_t il = 0; il < config.intermediate_loops; il++, t++) {
			// NOTE: `t` and `t ^ syndrome` give the same pairs
			while ((t ^ uint32_t(syndrome)) < t) { t++; }
			assert(t < nr_buckets);

			const size_t e1 = merge<false>(algo.L1, t);
			const size_t e2 = merge<true>(algo.L2, t ^ uint32_t(syndrome));
			if ((e1 == 0) || (e2 == 0)) { continue; }

			// NOTE: below the threshold the NN filters would only lose solutions
			if (config.enforce_bf || ((e1 < config.bf) && (e2 < config.bf))) {
				algo.bruteforce(e1, e2);
			} else {
				algo.nn(e1, e2);
			}
			if (algo.solutions_nr > 0) {
				target = t;
				not_found = false;
				return;
			}
		}
	}

#if defined(USE_CUDA) && defined(__CUDACC__)
	/// allocates the GPU buffers, once
	void cuda_alloc() noexcept {
		algo.cuda_alloc();
		if (dRows != nullptr) { return; }

		checkCudaErrors(cudaStreamCreate(&stream));
		checkCudaErrors(cudaEventCreateWithFlags(&ev_merge, cudaEventDisableTiming));
		checkCudaErrors(cudaEventCreateWithFlags(&ev_nn, cudaEventDisableTiming));
		for (uint32_t i = 0; i < 2; i++) {
			checkCudaErrors(cudaHostAlloc(&hRows[i], size_rows * sizeof(limb_type), cudaHostAllocDefault));
			hHT[i] = (limb_type *)malloc(size_HT * sizeof(limb_type));
			hP[i] = (uint32_t *)malloc(n * sizeof(uint32_t));
		}
		checkCudaErrors(cudaHostAlloc(&hE, 2 * sizeof(uint32_t), cudaHostAllocDefault));
		checkCudaErrors(cudaHostAlloc(&hSol, 2 * NR_LIMBS * sizeof(limb_type), cudaHostAllocDefault));

		checkCudaErrors(cudaMalloc(&dRows, size_rows * sizeof(limb_type)));
		checkCudaErrors(cudaMalloc(&dBL, B * NR_LIMBS * sizeof(limb_type)));
		checkCudaErrors(cudaMalloc(&dBR, B * NR_LIMBS * sizeof(limb_type)));
		checkCudaErrors(cudaMalloc(&dSol, 2 * NR_LIMBS * sizeof(limb_type)));
		checkCudaErrors(cudaMalloc(&dKL, B * sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&dKR, B * sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&dCnt, (nr_buckets + 1) * sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&dOff, (nr_buckets + 1) * sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&dCur, nr_buckets * sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&dE, 2 * sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&dSums, scan_tiles * sizeof(uint32_t)));
	}

	/// one ISD iteration: split rows into slot `s`, saves `HT`, `ws`, `P`, `syndrome`
	void cuda_step(const uint32_t s) noexcept {
		ISD::step();
		limb_type *rs = hRows[s];
		uint32_t *rk = (uint32_t *)(hRows[s] + (k+l+1)*NR_LIMBS);
		alignas(32) VElement t;
		for (uint32_t i = 0; i < k+l; i++) {
			split(t, rk[i], HT[i]);
			std::memcpy(rs + i*NR_LIMBS, t, NR_LIMBS * sizeof(limb_type));
			std::memcpy(hHT[s] + i*LABEL_LIMBS, HT[i], LABEL_LIMBS * sizeof(limb_type));
		}

		uint32_t key;
		split(t, key, ws.ptr());
		std::memcpy(rs + (k+l)*NR_LIMBS, t, NR_LIMBS * sizeof(limb_type));
		std::memcpy(hHT[s] + (k+l)*LABEL_LIMBS, ws.ptr(), LABEL_LIMBS * sizeof(limb_type));
		std::memcpy(hP[s], P.values, n * sizeof(uint32_t));
		hSyn[s] = syndrome;
	}

	/// restores `HT`, `ws`, `P`, `syndrome` of slot `s`
	void cuda_restore(const uint32_t s) noexcept {
		for (uint32_t i = 0; i < k+l; i++) {
			std::memcpy(HT[i], hHT[s] + i*LABEL_LIMBS, LABEL_LIMBS * sizeof(limb_type));
		}
		std::memcpy(ws.ptr(), hHT[s] + (k+l)*LABEL_LIMBS, LABEL_LIMBS * sizeof(limb_type));
		std::memcpy(P.values, hP[s], n * sizeof(uint32_t));
		syndrome = hSyn[s];
	}

	/// builds the base lists of slot `s` on the GPU, async on `stream`
	void cuda_tree(const uint32_t s) noexcept {
		const limb_type *dRS = dRows;
		const uint32_t *dRK = (const uint32_t *)(dRows + (k+l+1)*NR_LIMBS);
		checkCudaErrors(cudaMemcpyAsync(dRows, hRows[s], size_rows * sizeof(limb_type), cudaMemcpyHostToDevice, stream));
		checkCudaErrors(cudaMemsetAsync(dCnt, 0, (nr_buckets + 1) * sizeof(uint32_t), stream));
		mo_base_keys<limb_type, NR_LIMBS, kl_half, p><<<blocks, threads, 0, stream>>>(dKL, dCnt, dKR, dBR, dRK, dRS, B);
		checkCudaErrors(cudaGetLastError());
		mo_scan_tiles<<<scan_tiles, MO_SCAN_THREADS, 0, stream>>>(dOff, dSums, dCnt, nr_buckets + 1u);
		mo_scan_sums<<<1, MO_SCAN_THREADS, 0, stream>>>(dSums, scan_tiles);
		mo_scan_add<<<std::min<uint32_t>((nr_buckets + threads) / threads, 8192), threads, 0, stream>>>(dOff, dCur, dSums, nr_buckets + 1u);
		checkCudaErrors(cudaGetLastError());
		mo_base_scatter<limb_type, NR_LIMBS, p><<<blocks, threads, 0, stream>>>(dBL, dCur, dKL, dRS, B);
		checkCudaErrors(cudaGetLastError());
	}

	/// level-1 lists of slot `s` and target `t` on the GPU, async; sizes in `hE` after `ev_merge`
	void cuda_merge(const uint32_t s, const uint32_t t) noexcept {
		checkCudaErrors(cudaMemsetAsync(dE, 0, 2 * sizeof(uint32_t), stream));
		mo_merge<limb_type, NR_LIMBS><<<blocks, threads, 0, stream>>>
			((limb_type *)algo.dL1, (limb_type *)algo.dL2, dE, dBL, dOff, dBR, dKR,
			 dRows + (k+l)*NR_LIMBS, t, t ^ uint32_t(hSyn[s]), B, LIST_SIZE);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaMemcpyAsync(hE, dE, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
		checkCudaErrors(cudaEventRecord(ev_merge, stream));
	}

	/// NN on the level-1 lists, async; solution in `algo.hf`, its elements in `hSol` after `ev_nn`
	void cuda_nn(const uint32_t e1, const uint32_t e2) noexcept {
		algo.cuda_launch(e1, e2, stream);
		mo_solution<limb_type, NR_LIMBS><<<1, 32, 0, stream>>>
			(dSol, (const limb_type *)algo.dL1, (const limb_type *)algo.dL2, algo.df, algo.df + 1);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaMemcpyAsync(hSol, dSol, 2 * NR_LIMBS * sizeof(limb_type), cudaMemcpyDeviceToHost, stream));
		checkCudaErrors(cudaEventRecord(ev_nn, stream));
	}

	/// GPU loop of `run`: the CPU computes the next permutation while the GPU runs the current one
	void run_cuda() noexcept {
		static_assert(sizeof(limb_type) == sizeof(typename NN<nn_config>::Element) / NR_LIMBS);
		cuda_alloc();

		// first target of slot `s` >= `t`, see `step`
		auto next_target = [this](uint32_t t, const uint32_t s) {
			while ((t ^ uint32_t(hSyn[s])) < t) { t++; }
			assert(t < nr_buckets);
			return t;
		};

		uint32_t s = 0, il = 0;
		cuda_step(s);
		cuda_tree(s);
		uint32_t t = next_target(0, s);
		cuda_merge(s, t);
		while (true) {
			checkCudaErrors(cudaEventSynchronize(ev_merge));
			const uint32_t e1 = std::min<uint32_t>(hE[0], LIST_SIZE);
			const uint32_t e2 = std::min<uint32_t>(hE[1], LIST_SIZE);
			dropped += (hE[0] > LIST_SIZE) + (hE[1] > LIST_SIZE);
			const bool nn = (e1 > 0) && (e2 > 0);
			if (nn) { cuda_nn(e1, e2); }

			// NOTE: queued behind the NN: the next target or permutation
			uint32_t ns = s, nt = 0, nil = il + 1u;
			bool next = true;
			if (nil < config.intermediate_loops) {
				nt = next_target(t + 1u, s);
				cuda_merge(s, nt);
			} else {
				next = loops < isd.loops;
				if (next) {
					ns = s ^ 1u;
					nil = 0;
					cuda_step(ns);
					cuda_tree(ns);
					nt = next_target(0, ns);
					cuda_merge(ns, nt);
				}
			}

			if (nn) {
				checkCudaErrors(cudaEventSynchronize(ev_nn));
				if (algo.hf[0] != size_t(-1)) {
					// restore the iteration of the solution
					cuda_restore(s);
					loops -= (ns != s);
					target = t;
					not_found = false;
					break;
				}
			}

			if (!next) { break; }
			s = ns; t = nt; il = nil;
		}

		checkCudaErrors(cudaStreamSynchronize(stream));
		if (!not_found) {
			reconstruct(hSol, hSol + NR_LIMBS);
		}
	}
#endif

	/// runs May-Ozerov until a solution is found or `isd.loops` is reached
	uint64_t __attribute__ ((noinline))
	run() noexcept {
		ISD::reset();
#if defined(USE_CUDA) && defined(__CUDACC__)
		// NOTE: discarded, s.t. nvcc does not compile the CPU NN
		if constexpr (config.cuda) {
			run_cuda();
		} else
#else
		static_assert(!config.cuda, "`cuda` needs `USE_CUDA` and nvcc");
#endif
		{
			while (not_found && loops < isd.loops) {
				step();
			}

			if (!not_found) {
				const auto &sol = algo.solutions[0];
				reconstruct(algo.L1[sol.first], algo.L2[sol.second]);
			}
		}
		return loops;
	}

	/// important. Dont rename it
	constexpr void info() const noexcept {
		std::cout << "{ \"B\": " << B
		          << ", \"E\": " << E
		          << ", \"LIST_SIZE\": " << LIST_SIZE
		          << ", \"dropped\": " << dropped
		          << " }" << std::endl;
	}
};
#endif
