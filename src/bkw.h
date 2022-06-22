#ifndef SMALLSECRETLWE_BKW_H
#define SMALLSECRETLWE_BKW_H

#include "sort.h"
#include "list.h"

template<class List>
class BKW {
private:
	typedef typename List::ElementType ElementType;
	typedef typename List::ValueType ValueType;
	typedef typename List::LabelType LabelType;

	typedef typename List::ValueContainerType ValueContainerType;
	typedef typename List::LabelContainerType LabelContainerType;

	typedef typename List::ValueDataType ValueDataType;
	typedef typename List::ValueDataType LabelDataType;

	typedef typename LabelContainerType::ContainerLimbType LimbType;

	// disable standard constructor
	BKW() {}

	List &l;
	const uint64_t a;   // number of blocks
	const uint64_t b;   // block width
	const uint64_t d;   // block width of the final block
public:
	BKW(List &l, const uint64_t a, const uint64_t b, const uint64_t d) : l(l), a(a), b(b), d(d) {
		// NOTE: no Sanity Check for a, b, d is done.
		std::vector<uint64_t> limits(a);
		for (int i = 0; i < a+1; ++i) {
			limits[i] = i*b;
		}

		for (uint64_t i = 0; i < a; ++i) {
			const uint64_t b0 = limits[i];
			const uint64_t b2 = limits[i+1];
			l.sort_level(b0, b2);

			// precompute as much as possible
			const LimbType lmask = ValueContainerType::higher_mask(b0);
			const LimbType rmask = ValueContainerType::lower_mask2(b2);
			const int64_t lower_limb = b0 / ValueContainerType::limb_bits_width();
			const int64_t higher_limb = (b2 - 1) / ValueContainerType::limb_bits_width();

			uint64_t current_pos = 0, current_limit = l.get_size();
			ElementType current = l[current_pos];

			if (lower_limb == higher_limb) {
				const uint64_t limb = lower_limb;
				const uint64_t mask = lmask&rmask;
				for (int j = 1; j < current_limit; ++j) {
					if (!current.get_value_container().is_equal_simple2(l[j].get_value_container(), limb, mask)) {
						current = l[j];
						l.erase(j);
						j -= 1;
						current_limit -= 1;
						continue;
					}

					l[j].get_label_container().add(current.get_label_container());
					l[j].get_value_container().add(current.get_value_container());
				}
			} else {
				for (int j = 1; j < current_limit; ++j) {
					if (!current.get_value_container().is_equal_ext2(l[j].get_value_container(), lower_limb, higher_limb, lmask, rmask)) {
						current = l[j];
						l.erase(j);
						j -= 1;
						current_limit -= 1;
						continue;
					}

					l[j].get_label_container().add(current.get_label_container());
					l[j].get_value_container().add(current.get_value_container());
				}
			}
		}

		// TODO not implemented.

		//  now do the majority vote
		l.sort_level(limits[a-1], ValueContainerType::size());
		std::cout << l;
	}
};
#endif //SMALLSECRETLWE_BKW_H
