#![allow(non_snake_case)]
use mceliece::*;

pub fn main() {
    let mut c = ChaseVV:new(10, 2, 0);
    let mut P = vec![0 as u16, 2];

    while c.next(&mut P) {
        print!("{:?}", P);
    }
}