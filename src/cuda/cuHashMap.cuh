#ifndef DECODING_CUHASHMAP_H
#define DECODING_CUHASHMAP_H

#include <stdint.h>
#include <stddef.h>

// Internal data type.
template<typename T>
class __attribute__ ((packed)) cuHashMapElement {
public:
	using data_type = T;

	__device__
	constexpr cuHashMapElement() : start(T(-1)), length(0) {}

	__device__
	constexpr cuHashMapElement(const T s, const T e) : start(s), length(e) {}

	__device__
	constexpr bool is_zero() const noexcept {
	    return start == T(-1u);
	}

	__device__
	constexpr void clear() noexcept {
		start = T(-1);
		length = 0;
	}

	__device__
	constexpr void print() const noexcept {
		printf("%d:%d*\n", start, length);
	}

	__device__
	constexpr operator uint64_t () const { return data; }

	__device__
	constexpr operator unsigned long long() const { return data; }

//private :
	/// this is a abstraction of an distinguished point chain
	/// to a `uint64_t`
	union {
		struct {
			T start;
			uint32_t length;
		};
		uint64_t data;
	};
};

///
/// \tparam T type of the internal values
/// \tparam size number of element the hashmap can hold in total
/// \tparam HashFkt hash function
template<typename T,
         const size_t size,
         size_t (* HashFkt)(const T &)>
struct cuHashMap {
	/// construction: cuda flat hashmap
	///	[	|	|	 ... 	| in]
public:
	using data_type          = T;
	using internal_data_type = typename T::data_type;
	using index_type         = size_t;
	constexpr static index_type SIZE = size;

	/// constructor. Zero initializing everything
	__device__
	constexpr cuHashMap() : __internal_hashmap_array() {
		// THIS IS IMPORTANT. DO NOT DELETE.
		// this initializes the hashmap with -1
		clear();
	}

	/// hashes down `e` (Element) to an index where to store
	/// the element.
	/// NOTE: Boundary checks are performed in debug mode.
	/// NOTE: The insert is atomic
	/// \param e element to insert
	/// \return nothing
	__device__ __forceinline__
	constexpr void insert(const T &e) noexcept {
		// hash down the element to the index
		const size_t index = HashFkt(e);
		insert(e, index);
	}

	/// specially `insert` function which is independent of the
	/// given hash function. It simply puts the element wherever
	/// you want.
	/// NOTE: Boundary checks are performed in DEBUG mode
	/// NOTE: The insert is atomic
	/// \param e the element to insert
	/// \param index position to insert the element
	/// \return nothing
	__device__ __forceinline__
	constexpr void insert(const T &e, const index_type index) noexcept {
		// make sure we do not have made any errors.
		assert(index < size);

		// insert the element.
		atomicExch((unsigned long long*)&__internal_hashmap_array[index], (unsigned long long)(e));
	}

	/// checks if the entry in the hashmap is empty or not
	/// for the given element.
	/// NOTE: not atomic
	/// \param e the element which is first hashed down to an
	///				index and then checked if there is already an element.
	/// \return true/false
	__device__ __forceinline__
	constexpr bool probe(const T &e) const noexcept {
		const index_type index = HashFkt(e);
		return probe_internal(index);
	}

	///
	/// \tparam Hash
	/// \param e
	/// \param h
	/// \return
	template<class Hash>
	__device__ __forceinline__
	constexpr bool probe_internal(const internal_data_type &e, Hash h) const noexcept {
		const index_type index = h(e);
		return probe_internal(index);
	}

	/// Special probing function to check if a element is at the
	/// given position or not.
	/// \param index check if something at this position or not.
	/// \return true if something is in it
	///			false else
	__device__ __forceinline__
	constexpr bool probe_internal(const index_type index) const noexcept {
		assert(index < size);
		return !__internal_hashmap_array[index].is_zero();
	}

	/// Quite same to `probe` but instead it will directly return
	/// the position of the element.
	/// \param e Element to hash down.
	/// \return the position within the internal array of `e`
	__device__ __forceinline__
	constexpr index_type find(const T &e) const noexcept {
		const index_type index = HashFkt(e);
		assert(index < size);
		// return the index instead of the actual element, to
		// reduce the size of the returned element.
		return index;
	}

	/// prints the content of each bucket
	/// NOTE: only single threaded printing. If possible only call
	/// it with one thread.
	/// \return nothing
	__device__
	constexpr void print() const noexcept {
		const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx == 0) {
			for (index_type i = 0; i < size; i++) {
				__internal_hashmap_array[i].print();
			}
		}
	}

	/// overwrites the internal datat array
	/// with zero initialized elements.
	__device__
	constexpr void clear() noexcept {
		for (index_type i = 0; i < size; i++) {
			__internal_hashmap_array[i].clear();
		}
	}

	/// NOTE: only single threaded. Most likely you want to call
	///			this function only by a single thread
	/// \return the load, the number of buckets which are not empty
	__device__
	constexpr index_type load() const noexcept {
		index_type ret = index_type(0);

		const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx == 0) {
			for (index_type i = 0; i < size; i++) {
				// this comparison is on 8bytes
				const bool r = !__internal_hashmap_array[i].is_zero();
				ret += r;
			}
		}

		return ret;
	}

//private:
	// internal array
	alignas(32) T __internal_hashmap_array[size];
};

/// ahhhhh cuda sucks. Kernel need to be free functions.
/// printing function for a hashmap
/// \tparam HM type of the hashmap
/// \param hm the hashmap.
template<class HM>
__global__
void print_hashmap(const HM *hm) {
	const typename HM::index_type load = hm->load();
	printf("load: %ld/%ld(%f %)*\n", load, HM::SIZE, double(load)/double(HM::SIZE) * 100.);
	hm->print();
}



///
/// \tparam T type of the internal values
/// \tparam size number of element the hashmap can hold in total
/// \tparam 
/// \tparam HashFkt hash function
template<typename T,
         const size_t size,
         const size_t buckets,
         size_t (* HashFkt)(const T &)>
struct cuHashMapWithLoad {
	/// construction: cuda flat hashmap
	///	[	|	|	 ... 	| in]
public:
	using data_type          = T;
	using internal_data_type = T;
	using index_type         = size_t;
	using load_type          = uint32_t;
	constexpr static index_type SIZE = size;

	/// constructor. Zero initializing everything
	__device__
	constexpr cuHashMapWithLoad() noexcept : 
		__internal_hashmap_array(),
		__internal_load_array() {}

	/// hashes down `e` (Element) to an index where to store
	/// the element.
	/// NOTE: Boundary checks are performed in debug mode.
	/// NOTE: The insert is atomic
	/// \param e element to insert
	/// \return nothing
	__device__ __forceinline__
	constexpr void insert(const T &e) noexcept {
		// hash down the element to the index
		const size_t index = HashFkt(e);
		size_t pos = atomicAdd((uint32_t*) &__internal_load_array[index], 1);
        pos = min(buckets, pos);
		insert(e, index + pos);
	}

	/// specially `insert` function which is independent of the
	/// given hash function. It simply puts the element wherever
	/// you want.
	/// NOTE: Boundary checks are performed in DEBUG mode
	/// NOTE: The insert is atomic
	/// \param e the element to insert
	/// \param index position to insert the element
	/// \return nothing
	__device__ __forceinline__
	constexpr void insert(const T &e, const index_type index) noexcept {
		// make sure we do not have made any errors.
		assert(index < size*buckets);

		// insert the element.
		atomicExch((unsigned long long*)&__internal_hashmap_array[index], (unsigned long long)(e));
	}

	/// Quite same to `probe` but instead it will directly return
	/// the position of the element.
	/// \param e Element to hash down.
	/// \return the position within the internal array of `e`
	__device__ __forceinline__
	constexpr index_type find(const T &e) const noexcept {
		const index_type index = HashFkt(e);
		assert(index < size);
		// return the index instead of the actual element, to
		// reduce the size of the returned element.
		return index*buckets;
	}

	/// prints the content of each bucket
	/// NOTE: only single threaded printing. If possible only call
	/// it with one thread.
	/// \return nothing
	__device__
	constexpr void print() const noexcept {
		const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx == 0) {
			for (index_type i = 0; i < size; i++) {
				printf("%d\n", __internal_hashmap_array[i]);
			}
		}
	}

	/// can be called with any amount of threads
	/// overwrites the internal datat array
	/// with zero initialized elements.
	__device__
	constexpr void clear() noexcept {
		load_type idx = blockIdx.x * blockDim.x + threadIdx.x;
		const size_t block = blockDim.x*gridDim.x;
		while (idx < size) {
			__internal_load_array[idx] = 0;
			idx += block;
		}
	}

	/// NOTE: only single threaded. Most likely you want to call
	///			this function only by a single thread
	/// \return the load, the number of buckets which are not empty
	__device__
	constexpr index_type load() const noexcept {
		index_type ret = index_type(0);

		const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx == 0) {
			for (index_type i = 0; i < size; i++) {
				ret +=__internal_load_array[i];
			}
		}

		return ret;
	}

//private:
	// internal array
	alignas(32) T __internal_hashmap_array[size*buckets];
	alignas(32) load_type __internal_load_array[size];
};

#endif//DECODING_CUHASHMAP_H
