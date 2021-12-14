#![allow(non_snake_case)]
extern crate accel;

use mceliece::*;
use accel::*;

fn main() -> error::Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();

    // Allocate memories on GPU
    const n:usize = 33;
    let mut a = DeviceMemory::<u8>::zeros(&ctx, n);
    let mut b = DeviceMemory::<u8>::zeros(&ctx, n);
    let mut c = DeviceMemory::<u16>::zeros(&ctx, n);

    // Accessible from CPU as usual Rust slice (though this will be slow)
    for i in 0..n {
        b[i] = (n-i) as u8;
    }
    println!("a = {:?}", a.as_slice());
    println!("b = {:?}", b.as_slice());

    // Launch kernel synchronously
    unsafe {
        radixsort::counting_sort_impl_single_block_single_thread(&ctx,
            1 /* grid */,
            1 /* block */,
            (a.as_mut_ptr(), b.as_ptr(), c.as_mut_ptr(), n)
        ).expect("Kernel call failed");
    }

    println!("a = {:?}", a.as_slice());
    println!("b = {:?}", b.as_slice());

    Ok(())
}