#![allow(non_snake_case)]

use mceliece::*;

fn main() {
    //let mut a: BinVector = BinVector::with_capacity(10);
    //let mut H: BinMatrix = BinMatrix::random(n-k, n);
    //let mut s: BinMatrix = BinMatrix::random(n-k, 1);

    let (syndrom, H) =
        read_instace(instance::s.to_string(), instance::h.to_string());

    for _ in 0..100 {
        let sol = unsafe {
            prange(&H, &syndrom)
        };
        sol.print();

    }

}
