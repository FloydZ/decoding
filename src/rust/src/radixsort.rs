extern crate accel;
use accel::*;


trait RadixSort<T> {
    fn counting_sort_impl(B: &mut Vec<T>, A: &Vec<T>);
}


//impl<u8> RadixSort<u8>  {
    fn counting_sort_impl(B: &mut Vec<u8>, A: &Vec<u8>) {
        let n: usize = A.len();
        let mut counts = vec![0 as u8; 256];
        for it in 0..n {
            counts[A[it as usize] as usize] += 1;
        }
        let mut total: u8 = 0;

        //for mut count in counts {
        for it in 0..n {
            let old_count: u8 = counts[it];
            counts[it] = total;
            total += old_count;
        }
        for it in 0..n {
            let key = A[it];
            B[counts[key as usize] as usize] = A[it] as u8;
            counts[key as usize] += 1
        }
    }
//}


/// single threaded radix sort. for one block
#[kernel]
pub unsafe fn counting_sort_impl_single_block_single_thread(B: *mut u8, A: *const u8, counts: *mut u16, n: usize) {
    type CountType = u16;
    let nn = n as isize;
    for it in 0 as isize..nn {
        *counts.offset((*A.offset(it)) as isize) += 1;
    }
    let mut total: CountType = 0;
    for it in 0 as isize ..nn {
        let old_count = *counts.offset(it);
        *counts.offset(it) = total;
        total += old_count;
    }
    for it in 0 as isize..nn {
        let key = *A.offset(it) as isize;
        *B.offset((*counts.offset(key)) as isize) = *A.offset(it);
        *counts.offset(key) += 1
    }
}

/// TODO not finished
/// multi threaded radix sort. for one block
#[kernel]
pub unsafe fn counting_sort_impl_single_block_multiple_thread(B: *mut u8, A: *const u8, counts: *mut u16, n: usize) {
    type CountType = u16;

    // how the `index is calculated`
    //let block_id = accel_core::block_idx().into_id(accel_core::grid_dim());
    //let thread_id = accel_core::thread_idx().into_id(accel_core::block_dim());
    //index = (block_id + thread_id) as isize;
    accel_core::assert_eq!(accel_core::block_idx().into_id(accel_core::grid_dim()), 1);
    let i = accel_core::index() as isize;
    let nn = n as isize;
    let we: isize = nn/accel_core::block_dim() as isize;
    let start: isize = we * i;
    let end: isize = start + i;

    for it in start as isize..end {
        *counts.offset((*A.offset(it)) as isize) += 1;
    }
    let mut total: CountType = 0;
    for it in 0 as isize ..nn {
        let old_count = *counts.offset(it);
        *counts.offset(it) = total;
        total += old_count;
    }
    for it in 0 as isize..nn {
        let key = *A.offset(it) as isize;
        *B.offset((*counts.offset(key)) as isize) = *A.offset(it);
        *counts.offset(key) += 1
    }
}