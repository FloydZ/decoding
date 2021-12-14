
use bit_array::BitArray;
use typenum::U2048;
type Element = BitArray::<usize, U2048>;


pub fn chase_seq2(table: &mut Vec<Element>, change: &mut Vec<usize>, nn: usize, kk: usize) -> u32{
    //debug_assert_eq!(table.len(), bc(nn, kk));
    debug_assert_eq!(table.len(), change.len());

    let mut aa:Vec<usize> = vec![0; kk+1];
    let mut ww:Vec<bool> = vec![true; kk+1];
    let mut r: usize = 0;
    let mut counter: usize = 0;

    for i in 0..aa.len() {
        aa[i] = nn - kk -i;
    }

    // init the first word
    for j in nn-kk..nn {
        table[0].set(j, true);
    }

    for v in table.iter_mut().skip(1) {
        //v.set(0, true);
        let mut found_r = false;
        let mut j: usize = r;
        while aa[j] != 0 {
            let mut b = aa[j]+1;
            let n_ = aa[j+1];
            let compare_val = match ww[j+1] {
                false => n_,
                true => n_ - (2 - (n_ & 1))
            };

            if b < compare_val {
                if (b&1)== 0 && b +1 < n_ { b += 1; }

                // clear a bit
                v.set(aa[j], false);
                v.set(b, true);
                change[counter] = b;
                aa[j] = b;

                if found_r == false {
                    r = match j > 1 {
                        true => j -1,
                        false => 0,
                    }
                }

                break;
                // return 1;
            }
            ww[j] = aa[j] - 1 >= j;
            if ww[j] && !found_r {
                r = j;
                found_r = true;
            }

            j += 1;
        }
        let mut b = aa[j];
        if (b&1 != 0) && b-1 >= j { b -=1; }

        v.set(aa[j], false);
        v.set(b, true);
        change[counter] = b;
        aa[j] = b;

        ww[j] = b - 1 >= j;
        if !found_r {r = j; }

        counter += 1;
    }

    return 0;
}

// does not work repeats itself
pub struct ChaseVV {
    offset: u16,
    n: u32,
    t: u32,
    N: u16,
    x: i32,
    r: usize,
    j: usize,
    c: Vec<u16>,
    z: Vec<u16>,
}

impl ChaseVV {
    pub fn new(nn: u32, t:u32, offset: u16) -> Self {
        let mut r = ChaseVV {
            n: nn,
            t: t,
            offset: offset,
            N: 0,
            r: 1,
            j: 0,
            x: 0,
            c: vec![0 as u16; (t+2) as usize],
            z: vec![0 as u16; (t+2) as usize],
        };

        for i in 1..t as usize + 2 {
            r.z[i] = 0;
        }

        for i in 1..t as usize + 2 {
            r.c[i] = (r.n - r.t - 1) as u16 + i as u16;
        }
        return r;
    }

    pub fn next(&mut self, P: &mut Vec<u16>) -> bool {
        for i in 1..self.t as usize + 1 {
            P[i-1] = self.c[i] + self.offset;
        }

        self.N += 1;
        self.j = self.r;
        loop {
            if self.z[self.j] > 0 {
                self.x = self.c[self.j as usize] as i32 + 2;
                if self.x < self.z[self.j] as i32 {
                    self.c[self.j] = self.x as u16;
                } else if self.x as u16 == self.z[self.j] && self.z[self.j + 1] == 1 {
                    self.c[self.j] = (self.x - (self.c[self.j + 1] % 2) as i32) as u16;
                } else {
                    self.z[self.j] = 0;
                    self.j += 1;
                    if self.j <= self.t as usize {
                        continue;
                    } else {
                        return false;
                    }
                }
                if self.c[1] > 0 {
                    self.r = 1;
                } else {
                    self.r = self.j - 1;
                }
            } else {
                self.x = self.c[self.j] as i32 + (self.c[self.j] % 2) as i32 - 2;
                if self.x as usize >= self.j {
                    self.c[self.j] = self.x as u16;
                    self.r = 1;
                } else if self.c[self.j] == self.j as u16 {
                    self.c[self.j] = self.j as u16 - 1;
                    self.z[self.j] = self.c[self.j + 1] - ((self.c[self.j + 1] + 1) % 2);
                    self.r = self.j;
                } else if self.c[self.j] < self.j as u16{
                    self.c[self.j] = self.j as u16;
                    self.z[self.j] = self.c[self.j + 1] - ((self.c[self.j + 1] + 1) % 2);

                    if self.j > 2 {
                        self.r = self.j - 1;
                    } else {
                        self.r = 1;
                    }
                } else {
                    self.c[self.j] = self.x as u16;
                    self.r = self.j;
                }
            }

            return true;
        }
    }
}
