#![allow(non_snake_case)]
use mceliece::*;

pub fn main() {
    let mut c = Chase::new(10, 2, 0);
    let mut P = vec![0 as u64, 1];
    c.left_step(&mut P, true);

    let mut ctr: u32 = 0;
    while c.left_step(&mut P, false) {
        println!("{}, {:?}", ctr, P);
        ctr += 1;
    }
}