#![allow(non_snake_case)]
extern crate accel;

use mceliece::*;
use accel::*;

fn main() -> error::Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();

    // Allocate memories on GPU
    const n:usize = 32;
    let mut a = DeviceMemory::<f32>::zeros(&ctx, n);
    let mut b = DeviceMemory::<f32>::zeros(&ctx, n);
    let mut c = DeviceMemory::<f32>::zeros(&ctx, n);

    // Accessible from CPU as usual Rust slice (though this will be slow)
    for i in 0..n {
        a[i] = i as f32;
        b[i] = 2.0 * i as f32;
    }
    println!("a = {:?}", a.as_slice());
    println!("b = {:?}", b.as_slice());

    // Launch kernel synchronously
    unsafe {
        add(&ctx,
            1 /* grid */,
            n /* block */,
            (a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n)
        ).expect("Kernel call failed");
    }

    println!("c  = {:?}", c.as_slice());
    Ok(())
}

// fn main() {
//     //let mut a: BinVector = BinVector::with_capacity(10);
//     //let mut H: BinMatrix = BinMatrix::random(n-k, n);
//     //let mut s: BinMatrix = BinMatrix::random(n-k, 1);
//
//     let (syndrom, H) =
//         read_instace(instance::s.to_string(), instance::h.to_string());
//
//     for _ in 0..100 {
//         let sol = unsafe {
//             prange(&H, &syndrom)
//         };
//         sol.print();
//     }
// }
