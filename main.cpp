#include <cstdint>

#include "prange.h"
#include "stern.h"
#include "stern_im.h"
#include "stern_mo.h"
#include "bjmm.h"

// NOTE: written by `decoding/optimize.py` into the build directory. Defines the
// 	instance (`h`, `s`) and `algorithm_type`.
#include "main.h"

int main() {
	// NOTE: on the heap, the algorithms are too large for the stack
	auto *algo = new algorithm_type{};
	algo->from_string(h, s);
	algo->run();
	algo->print();

	const bool ok = algo->correct();
	delete algo;
	return ok ? 0 : 1;
}
