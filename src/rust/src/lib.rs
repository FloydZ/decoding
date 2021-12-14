#![allow(non_upper_case_globals)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused_mut)]
extern crate accel;
use accel::*;

use m4ri_rust::friendly::*;
//use m4ri_sys::*;
//use std::ptr;
use rand::Rng;

pub mod instance;
pub mod hashmap;
pub mod chase2;
pub mod radixsort;
//pub mod chase;

#[cfg(feature = "serde")]
use vob::Vob;

extern crate typenum;
use bit_array::BitArray;
use typenum::U2048;
use typenum::uint::SetBit;

const n: usize = 240;   // 156;
const k: usize = 192;   // 125;
const w: usize = 6;     // 4;

const l: usize = 1;
const p: usize = 1;

#[kernel]
pub unsafe fn add(a: *const f32, b: *const f32, c: *mut f32, n: usize) {
    let i = accel_core::index();
    if (i as usize) < n {
        *c.offset(i) = *a.offset(i) + *b.offset(i);
    }
}


//type Element = vob::Vob;
type Element = BitArray::<usize, U2048>;

const fn bc(nn: usize, kk: usize) -> usize {
    if kk > nn { return 0; }
    if kk==0 || nn == kk { return 1; }
    if kk==1 || kk==nn-1 { return nn; }
    if kk+kk < nn {
        return bc((nn - 1) as usize, (kk - 1) as usize)*nn/kk
    } else {
        return bc((nn - 1) as usize, kk as usize)/(nn-kk)
    }
}




// copies: out[0,..., end-start] = in[start,...,end]
fn m4ri_copy_row(out: &mut BinMatrix, inm: &BinMatrix, start: usize, end: usize, row: usize) {
    for i in start..end {
        out.set(row, i-start, inm.bit(row, i));
    }
}

fn copy_submatrix(out: &mut BinMatrix, inm: &BinMatrix, start_r: usize, start_c: usize, end_r: usize, end_c: usize) {
    for i in start_r..end_r {
        m4ri_copy_row(out, inm, start_c, end_c, i);
    }
}

fn weigh_colum(A: &BinMatrix, col: usize) -> usize {
    let mut ret = 0;
    for nrow in 0..A.nrows() {
        ret += A.bit(nrow, col) as usize;
    }
    return ret;
}

unsafe fn mceliece(H: &BinMatrix, s: &BinMatrix)  {
    let mut rng = rand::thread_rng();

    println!("{}", bc(l+k, p));
    let mut table: Vec<Element> = vec![Element::from_elem(false); bc(l+k, p)];
    let mut change: Vec<usize> = vec![0; bc(l+k, p)];

    chase2::chase_seq2(&mut table, &mut change, l+k, p);
    println!("{:?}", table);

    // Needed variables.
    let mut working_H =  H.augmented(&s);
    let mut working_H_T =  working_H.transposed();
    let mut H_prime: BinMatrix = BinMatrix::zero(n-k, k+l);
    let mut H_prime_T: BinMatrix = BinMatrix::zero(k+l, n-k);

    let mut working_s: BinMatrix = BinMatrix::zero(n-k, 1);
    let mut working_s_T: BinMatrix = BinMatrix::zero(1, n-k);

    // Init the Permutation
    let mut P = [0 as usize; n];
    for i in 1..n {
        P[i] = i;
    }

    // permute step
    for i in 0..n - 1 {
        let pos: usize = rng.gen::<usize>()%(n-i);
        let tmp = P[i];
        P[i] = P[i+pos];
        P[i+pos] = tmp;

        working_H_T.swap_rows(i as i32, (i+pos) as i32);
    }

    // transpose back
    //working_H = working_H_T.transposed();
    working_H_T.transpose(&mut working_H);

    // echolonize
    working_H.echelonize_full();
    working_H.print();

    copy_submatrix(&mut working_s, &working_H_T, 0, n, n-k, n+1);
    //working_s_T = working_s.transposed();
    working_s.transpose(&mut working_s_T);

    copy_submatrix(&mut H_prime, &working_H_T, 0, n-k-l, n-k, n);
    //H_prime_T = H_prime.transposed();
    H_prime.transpose(&mut H_prime_T);
}

pub unsafe fn prange(H: &BinMatrix, syndrom: &BinMatrix) -> BinMatrix {
    let mut rng = rand::thread_rng();

    // Needed variables.
    let mut working_H =  H.augmented(&syndrom);
    let mut working_H_T: BinMatrix = BinMatrix::zero(n+1, n-k);

    // Init the Permutation
    let mut P = [0 as usize; n];
    for i in 1..n {
        P[i] = i;
    }

    loop {
        working_H.transpose(&mut working_H_T);

        // permute step
        for i in 0..n - 1 {
            let pos: usize = rng.gen::<usize>()%(n-i);
            let tmp = P[i];
            P[i] = P[i+pos];
            P[i+pos] = tmp;

            working_H_T.swap_rows(i as i32, (i+pos) as i32);
        }

        // transpose back
        working_H_T.transpose(&mut working_H);

        // echolonize
        working_H._mzd_echelonize_m4ri();

        if(weigh_colum(&working_H, n)) == w {
            let mut e = BinMatrix::zero(1, n);
            for i in 0..n-k {
                e.set(0, P[i], working_H.bit(i, n));
            }
            return e;
        }
    }

}

pub fn read_instace(s: String, h: String) -> (BinMatrix, BinMatrix) {
    let ss = BinMatrix::from_string(1, n - k, s);
    let A = BinMatrix::from_string(n, n - k, h);
    return (ss.transposed(), A.transposed());
}