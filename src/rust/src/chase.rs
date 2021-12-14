
// Das ist einfach richtig dumm. Keine Ahnung warum der VV Code nicht geht,
pub struct Chase {
    n: u32,
    k: u32,
    start: u32,
    to: Vec<u32>,
    td: Vec<i32>,
    tn: Vec<u32>,
    tp: Vec<u32>,
    tb: usize,
}

impl Chase {
    pub fn new(nn: u32, kk: u32, start: u32) -> Self {
        let mut ret = Chase {
            n: nn,
            k: kk,
            start: start,
            to: vec![0 as u32; nn as usize],
            td: vec![0 as i32; nn as usize],
            tn: vec![0 as u32; nn as usize],
            tp: vec![0 as u32; nn as usize],
            tb: 0
        };

        ret.td[0] = 1;
        ret.tn[0] = nn;

        return ret;
    }

    fn writespot(&self, ww: u64, spot: usize, bit: bool) -> u64 {
        return ww & (!((1 as u64) << spot)) | ((bit as u64) << spot);
    }
    fn readspot(&self, ww: u64, spot: usize) -> bool {
        return (ww >> spot) != 0;
    }

    fn writebit(&self, A: &mut Vec<u64>, pos: usize, bit: bool) {
        A[pos/64] = self.writespot(A[pos/64], pos%64, bit);
    }
    fn readbit(&self, A: &mut Vec<u64>, pos: usize) -> bool {
        return self.readspot(A[pos/64], pos%64);
    }
    fn left_write(&self, A: &mut Vec<u64>, b: usize, bit: bool) {
        let pos: u32 = self.start + (self.to[b] as i32 + (self.tp[b] as i32 * self.td[b])) as u32;
        self.writebit(A, pos as usize, bit);
    }
    fn left_round(&mut self, b: usize) {
        let s1:u32 = match (self.tp[b-1]&1) == 0 {
            true => self.tn[b-1]-1,
            false => self.tp[b],
        };

        self.to[b] = self.to[b-1]+((self.td[b-1] as u32)*s1);

        let s2:i32 = match (self.tp[b-1]&1) == 0 {
            true => -1,
            false => 1,
        };

        self.td[b] = self.td[b-1]*s2;
        self.tn[b] = self.tn[b-1] - self.tp[b-1] - 1;
        self.tp[b] = 0;
    }

    pub fn left_step(&mut self, A: &mut Vec<u64>, init: bool) -> bool {
        // cleanup of the prevoius round
        if !init {
            loop {
                self.left_write(A, self.tb, false);
                self.tp[self.tb] += 1;

                if (self.tp[self.tb] <= (self.tn[self.tb] + (self.tb as u32) - self.k)) || self.tb == 0 {
                    break;
                }

                self.tb -= 1;
            }
        }

        if self.tp[0] > (self.n-self.k) {
            return false;
        }

        self.left_write(A, self.tb, true);

        self.tb += 1;
        while (self.tb as u32) < self.k {
            self.left_round(self.tb);
            self.left_write(A, self.tb, true);
            self.tb += 1;
        }

        if self.tp[0] > (self.n-self.k) {
            return false;
        }

        self.tb = self.k as usize -1;
        return true;
    }

    fn left_write_element(&self, A: &mut Element, b: usize, bit: bool) {
        let pos: u32 = self.start + (self.to[b] as i32 + (self.tp[b] as i32 * self.td[b])) as u32;
        A.set(pos as usize, bit);
    }

    pub fn left_step_element(&mut self, A: &mut Element, init: bool) -> bool {
        // cleanup of the prevoius round
        if !init {
            loop {
                self.left_write_element(A, self.tb, false);
                self.tp[self.tb] += 1;

                if (self.tp[self.tb] <= (self.tn[self.tb] + (self.tb as u32) - self.k)) || self.tb == 0 {
                    break;
                }

                self.tb -= 1;
            }
        }

        if self.tp[0] > (self.n-self.k) {
            return false;
        }

        self.left_write_element(A, self.tb, true);

        self.tb += 1;
        while (self.tb as u32) < self.k {
            self.left_round(self.tb);
            self.left_write_element(A, self.tb, true);
            self.tb += 1;
        }

        if self.tp[0] > (self.n-self.k) {
            return false;
        }

        self.tb = self.k as usize -1;
        return true;
    }

}