#![allow(non_camel_case_types)]

extern "C" {
    pub fn to_pallas(ret: *mut pasta_fp, le_u64s: *const [u64; 4]);
    pub fn to_vesta(ret: *mut pasta_fq, le_u64s: *const [u64; 4]);

    pub fn add_pallas(ret: *mut pasta_fp, a: *const pasta_fp, b: *const pasta_fp);
    pub fn add_vesta(ret: *mut pasta_fq, a: *const pasta_fq, b: *const pasta_fq);

    pub fn sub_pallas(ret: *mut pasta_fp, a: *const pasta_fp, b: *const pasta_fp);
    pub fn sub_vesta(ret: *mut pasta_fq, a: *const pasta_fq, b: *const pasta_fq);

    pub fn mul_pallas(ret: *mut pasta_fp, a: *const pasta_fp, b: *const pasta_fp);
    pub fn mul_vesta(ret: *mut pasta_fq, a: *const pasta_fq, b: *const pasta_fq);

    pub fn sqr_pallas(ret: *mut pasta_fp, a: *const pasta_fp);
    pub fn sqr_vesta(ret: *mut pasta_fq, a: *const pasta_fq);
}

pub type limb_t = u64;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct pasta_fp {
    pub l: [limb_t; 4usize],
}

// Pallas base field (`Fp`) modulus (as a non-Montgomery form bigint):
// p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
pub const FP_MODULUS: [u64; 4] = [
    0x992d30ed00000001,
    0x224698fc094cf91b,
    0x0000000000000000,
    0x4000000000000000,
];

// Pallas base field (`Fp`) Montgomery multiplier `R = 2^256 (mod p)`
pub const FP_R: [u64; 4] = [
    0x34786d38fffffffd,
    0x992c350be41914ad,
    0xffffffffffffffff,
    0x3fffffffffffffff,
];

// Vesta base field (`Fq`) modulus (as a non-Montgomery form bigint):
// q = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
pub const FQ_MODULUS: [u64; 4] = [
    0x8c46eb2100000001,
    0x224698fc0994a8dd,
    0x0,
    0x4000000000000000,
];

// Vesta base field (`Fq`) Montgomery multiplier `R = 2^256 (mod q)`
pub const FQ_R: [u64; 4] = [
    0x5b2b3e9cfffffffd,
    0x992c350be3420567,
    0xffffffffffffffff,
    0x3fffffffffffffff,
];

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct Fp(pub pasta_fp);

impl Fp {
    pub const fn zero() -> Self {
        Fp(pasta_fp {
            l: [0, 0, 0, 0],
        })
    }

    pub const fn one() -> Self {
        Fp(pasta_fp {
            l: FP_R,
        })
    }

    // `-1 = q - 1 (mod p)`
    pub fn neg_one() -> Self {
        Self::zero().sub(&Self::one())
    }

    pub fn from_le_u64s_nonmont(le_u64s: [u64; 4]) -> Self {
        let mut ret = Self::zero();
        unsafe { to_pallas(&mut ret.0, &le_u64s); }
        ret
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut ret = Self::zero();
        unsafe { add_pallas(&mut ret.0, &self.0, &other.0); }
        ret
    }

    pub fn sub(&self, other: &Self) -> Self {
        let mut ret = Self::zero();
        unsafe { sub_pallas(&mut ret.0, &self.0, &other.0); }
        ret
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mut ret = Self::zero();
        unsafe { mul_pallas(&mut ret.0, &self.0, &other.0); }
        ret
    }

    pub fn square(&self) -> Self {
        let mut ret = Self::zero();
        unsafe { sqr_pallas(&mut ret.0, &self.0); }
        ret
    }

    pub fn modulus() -> Self {
        Fp(pasta_fp {
            l: FP_MODULUS,
        })
    }
}

impl From<pasta_fp> for Fp {
    fn from(fp: pasta_fp) -> Self {
        Fp(fp)
    }
}

impl From<Fp> for pasta_fp {
    fn from(fp: Fp) -> Self {
        fp.0
    }
}

impl From<u64> for Fp {
    fn from(limb: u64) -> Self {
        /*
        Fp(pasta_fp {
            l: [limb, 0, 0, 0],
        })
        */
        Self::from_le_u64s_nonmont([limb, 0, 0, 0])
    }
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct pasta_fq {
    pub l: [limb_t; 4usize],
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct Fq(pub pasta_fq);

impl Fq {
    pub fn zero() -> Self {
        Fq(pasta_fq::default())
    }

    pub fn one() -> Self {
        Fq(pasta_fq {
            l: FQ_R,
        })
    }

    // `-1 = q - 1 (mod q)`
    pub fn neg_one() -> Self {
        Self::zero().sub(&Self::one())
    }

    pub fn from_le_u64s_nonmont(le_u64s: [u64; 4]) -> Self {
        let mut ret = Self::zero();
        unsafe { to_vesta(&mut ret.0, &le_u64s); }
        ret
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut ret = Self::zero();
        unsafe { add_vesta(&mut ret.0, &self.0, &other.0); }
        ret
    }

    pub fn sub(&self, other: &Self) -> Self {
        let mut ret = Self::zero();
        unsafe { sub_vesta(&mut ret.0, &self.0, &other.0); }
        ret
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mut ret = Self::zero();
        unsafe { mul_vesta(&mut ret.0, &self.0, &other.0); }
        ret
    }

    pub fn square(&self) -> Self {
        let mut ret = Self::zero();
        unsafe { sqr_vesta(&mut ret.0, &self.0); }
        ret
    }

    pub fn modulus() -> Self {
        Fq(pasta_fq {
            l: FQ_MODULUS,
        })
    }
}

impl From<pasta_fq> for Fq {
    fn from(fq: pasta_fq) -> Self {
        Fq(fq)
    }
}

impl From<Fq> for pasta_fq {
    fn from(fq: Fq) -> Self {
        fq.0
    }
}

impl From<u64> for Fq {
    fn from(limb: u64) -> Self {
        Self::from_le_u64s_nonmont([limb, 0, 0, 0])
    }
}

#[cfg(test)]
macro_rules! offsetof {
    ($type:ty, $field:tt) => {
        unsafe {
            let v = std::mem::MaybeUninit::<$type>::uninit().assume_init();
            (&v.$field as *const _ as usize) - (&v as *const _ as usize)
        }
    };
}

#[test]
fn test_layout_pasta_fp() {
    assert_eq!(
        ::std::mem::size_of::<pasta_fp>(),
        32usize,
        concat!("Size of: ", stringify!(pasta_fp))
    );
    assert_eq!(
        ::std::mem::align_of::<pasta_fp>(),
        8usize,
        concat!("Alignment of ", stringify!(pasta_fp))
    );
    assert_eq!(
        offsetof!(pasta_fp, l),
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(pasta_fp),
            "::",
            stringify!(l)
        )
    );
}

#[test]
fn test_layout_pasta_fq() {
    assert_eq!(
        ::std::mem::size_of::<pasta_fq>(),
        32usize,
        concat!("Size of: ", stringify!(pasta_fq))
    );
    assert_eq!(
        ::std::mem::align_of::<pasta_fq>(),
        8usize,
        concat!("Alignment of ", stringify!(pasta_fq))
    );
    assert_eq!(
        offsetof!(pasta_fq, l),
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(pasta_fq),
            "::",
            stringify!(l)
        )
    );
}

#[test]
fn test_from_le_u64s_nonmont_pallas() {
    assert_eq!(Fp::from_le_u64s_nonmont(FP_MODULUS), Fp::zero());
    assert_eq!(Fp::from_le_u64s_nonmont(FP_MODULUS).add(&Fp::one()), Fp::one());
    assert_eq!(Fp::from_le_u64s_nonmont(FP_MODULUS).sub(&Fp::one()), Fp::neg_one());
}

#[test]
fn test_from_le_u64s_nonmont_vesta() {
    assert_eq!(Fq::from_le_u64s_nonmont(FQ_MODULUS), Fq::zero());
    assert_eq!(Fq::from_le_u64s_nonmont(FQ_MODULUS).add(&Fq::one()), Fq::one());
    assert_eq!(Fq::from_le_u64s_nonmont(FQ_MODULUS).sub(&Fq::one()), Fq::neg_one());
}

#[test]
fn test_add_pallas() {
    assert_eq!(Fp::one().add(&Fp::from(3)), Fp::from(4));
}

#[test]
fn test_add_vesta() {
    assert_eq!(Fq::one().add(&Fq::from(3)), Fq::from(4));
}

#[test]
fn test_sub_pallas() {
    assert_eq!(Fp::from(4).sub(&Fp::from(4)), Fp::zero());
    assert_eq!(Fp::from(4).sub(&Fp::one()), Fp::from(3));
}

#[test]
fn test_sub_vesta() {
    assert_eq!(Fq::from(4).sub(&Fq::from(4)), Fq::zero());
    assert_eq!(Fq::from(4).sub(&Fq::one()), Fq::from(3));
}

#[test]
fn test_mul_pallas() {
    assert_eq!(Fp::modulus().mul(&Fp::one()), Fp::zero());
    assert_eq!(Fp::one().mul(&Fp::neg_one()), Fp::neg_one());
    assert_eq!(Fp::from(3).mul(&Fp::neg_one()), Fp::modulus().sub(&Fp::from(3)));
    assert_eq!(Fp::from(4).mul(&Fp::from(3)), Fp::from(12));
}

#[test]
fn test_mul_vesta() {
    assert_eq!(Fq::modulus().mul(&Fq::one()), Fq::zero());
    assert_eq!(Fq::one().mul(&Fq::neg_one()), Fq::neg_one());
    assert_eq!(Fq::from(3).mul(&Fq::neg_one()), Fq::modulus().sub(&Fq::from(3)));
    assert_eq!(Fq::from(4).mul(&Fq::from(3)), Fq::from(12));
}

#[test]
fn test_square_pallas() {
    assert_eq!(Fp::one().square(), Fp::one());
    assert_eq!(Fp::neg_one().square(), Fp::one());
    assert_eq!(Fp::from(2).square(), Fp::from(4));
    assert_eq!(Fp::modulus().square(), Fp::zero());
}

#[test]
fn test_square_vesta() {
    assert_eq!(Fq::one().square(), Fq::one());
    assert_eq!(Fq::neg_one().square(), Fq::one());
    assert_eq!(Fq::from(2).square(), Fq::from(4));
    assert_eq!(Fq::modulus().square(), Fq::zero());
}
