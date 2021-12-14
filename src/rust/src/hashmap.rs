use std::mem::size_of;

// TODO
type ListElement        = u64;
type LabelContainerType = u32;  //
type ArgumentLimbType   = u32;  //
type LoadIntType        = u32;  //
type IndexType          = u32;  //
type BucketIntType      = u32;  //
type BucketElement      = (BucketIntType, [IndexType; 4]);

struct ParallelHashMap {
    l: u32,             // l

    loffset: u32,       // n-k-l
    loffset64: u32,
    lshift: u32,
    size_label: usize,

    b0: u16,
    b1: u16,
    b2: u16,

    nrb: BucketIntType,
    nrt: u32,
    nri: u32,

    size_bucket: u32,
    size_thread: u32,

    lmask1:     ArgumentLimbType,
    rmask1:     ArgumentLimbType,
    mask1:      ArgumentLimbType,
    lmask2:     ArgumentLimbType,
    rmask2:     ArgumentLimbType,
    mask2:      ArgumentLimbType,
    sortrmask2: ArgumentLimbType,

    chunks:     u64,

    __buckets: Vec<BucketElement>,
    __buckets_load: Vec<LoadIntType>,
    __acc_buckets_load: Vec<LoadIntType>,

}

impl ParallelHashMap {
    pub fn new(b0: u16, b1: u16, b2: u16,
               nrb: BucketIntType, nrt: u32, nri: u32,
               size_bucket: u32,
               label_offset: u32, l: u32, lvl: u8
        ) -> Self {
        let mut ret = ParallelHashMap {
            l: l,

            loffset:    label_offset,
            loffset64:  label_offset/64,
            lshift:     (label_offset - ((label_offset/64) * 64)),
            size_label: std::mem::size_of::<LabelContainerType>(),

            b0: b0,
            b1: b1,
            b2: b2,

            nrb: nrb,
            nrt: nrt,
            nri: nri,

            size_bucket: size_bucket,
            size_thread: 0,

            lmask1: !(((1 as ArgumentLimbType) << b0) - 1),
            rmask1: ((1 as ArgumentLimbType) << b1) - 1,
            mask1: 0, //TODO lmask1 & rmask1,
            lmask2: !(((1 as ArgumentLimbType) << b1) - 1),
            rmask2: ((1 as ArgumentLimbType) << b2) - 1,
            mask2: 0, // TODO lmask2 & rmask2,

            sortrmask2: 0, // TODO
            chunks: 0,      //TODO

            __buckets: vec![(0, [0; 4]); (nrb* (size_bucket as u64)) as usize],
            __buckets_load: vec![0; (nrb*(nrt as u64)) as usize],
            __acc_buckets_load: vec![0; (nrb) as usize],
        };

        assert!(ret.size_thread <= ret.size_bucket);
        assert!(1 << (b1-b0) <= nrb && b0 < b1 && b1 <= b2 && b2 <= 64);
        return ret;
    }

    fn thread_offset(&self, tid: u32) -> u32 { return tid*self.nrt; }
    fn bucket_offset_(&self, bid: u32) -> u32 { return bid*self.size_bucket; }
    fn bucket_offset(&self, tid: u32, bid: BucketIntType) -> BucketIntType {
        return bid*(self.size_bucket as BucketIntType) +
             ((tid*self.size_thread) as BucketIntType);
    }

    fn acc_bucket_load(&mut self, bid: BucketIntType) {
        let mut load: LoadIntType = 0;
        for tid in 0..self.nrt {
            load += self.get_bucket_load(tid, bid);
        }
        self.__acc_buckets_load[bid as usize] = load;
    }

    fn inc_bucket_load(&mut self, tid: u32, bid: BucketIntType) {
        assert!(tid < self.nrt && bid < self.nrb);
        self.__buckets_load[(tid as BucketIntType)*(self.nrb as BucketIntType) + bid] +=1;
    }

    fn get_bucket_load(&self, tid: u32, bid: BucketIntType) -> LoadIntType {
        assert!(tid < self.nrt && bid < self.nrb);
        return self.__buckets_load[tid*self.nrb + bid];
    }

    fn get_bucket_load2(&self, bid: BucketIntType) -> LoadIntType {
        assert!(bid < self.nrb);
        if self.nrt == 1 {
            return self.__buckets_load[bid];
        } else {
            return self.__acc_buckets_load[bid];
        }
    }

    fn hash(&self, data: ArgumentLimbType) -> BucketIntType {
        let bid: BucketIntType = (data & self.mask1) >> self.b0;
        assert!(bid < self.nrb);
        return bid;
    }


    fn hash_list(&mut self, L: &Vec<ListElement>, tid: u32) {
        let b_tid: LoadIntType = (L.len() as LoadIntType)/self.nrt;
        let s_tid: LoadIntType = tid*b_tid;

        // TODO
    }

    fn insert(&mut self, data: ArgumentLimbType, npos : &Vec<IndexType>, tid: u32) -> bool {
        assert!(tid < self.nrt);
        let bid: BucketIntType = self.hash(data);
        let load: LoadIntType = self.get_bucket_load(tid, bid);

        if self.size_bucket - load == 0 {
            return false;
        }

        let bucketOffset: BucketIntType = self.bucket_offset(tid, bid) + load;
        self.__buckets[bucketOffset as usize][0] = data;
        self.__buckets[bucketOffset as usize][1] = npos;
        self.inc_bucket_load(tid, bid);
        return true;
    }

    // TODO maybe replace with radix sort?
    fn bucket_sort(&mut self, bid: BucketIntType) {
        assert!(bid < self.nrb);
        self.__buckets[(bid*size_b) as usize .. ((bid+1)*size_b) as usize].sort_by(
            |a, v|
                (a[0] & self.sortrmask2) < (b[0] & self.sortrmask2))
    }

    fn sort(&mut self, tid: u32) {
        if self.b2 == self.b1 {
            if self.nrt == 1 {
                return;
            }

            for bid in 0 as BucketIntType..self.nrb {
                self.__acc_buckets_load(nrb);
            }

            return;
        }

        for i in (tid * self.chunks)..((tid + 1) * chunks) {
            self.bucket_sort(i);
        }
    }

    fn find(&self, data: ArgumentLimbType, load: &mut LoadIntType) -> usize {
        let bid: BucketIntType = self.hash(data);
        let boffset: usize = bid*self.size_bucket;
        *load = self.get_bucket_load2(bid);

        if self.b2 == self.b1 {
            if (load != 0){
                *load += boffset;
                return boffset;
            }
            return -1;
        }

        if load == 0 {
            return -1;
        }

        *load += boffset;

        let dummy: BucketElement; // TODO
        let r = self.__buckets[boffset, load].binary_search_by(|a|
                (a[0] & self.rmask2) == (dummy[0] & self.rmask2));
        match r {
            Ok(i) => return i,
            _ => return -1,
        }
    }

    fn traverse() -> ArgumentLimbType {


        return 0;
    }
}
