// SPDX-License-Identifier: GPL-2.0

//! Atomic primitives.
//!
//! These primitives have the same semantics as their C counterparts: and the precise definitions of
//! semantics can be found at [`LKMM`]. Note that Linux Kernel Memory (Consistency) Model is the
//! only model for Rust code in kernel, and Rust's own atomics should be avoided.
//!
//! # Data races
//!
//! [`LKMM`] atomics have different rules regarding data races:
//!
//! - A normal write from C side is treated as an atomic write if
//!   CONFIG_KCSAN_ASSUME_PLAIN_WRITES_ATOMIC=y.
//! - Mixed-size atomic accesses don't cause data races.
//!
//! [`LKMM`]: srctree/tools/memory-model/

pub mod generic;
pub mod ops;
pub mod ordering;

pub use generic::Atomic;
pub use ordering::{Acquire, Full, Relaxed, Release};

// SAFETY: `i32` has the same size and alignment with itself, and is round-trip transmutable to
// itself.
unsafe impl generic::AllowAtomic for i32 {
    type Repr = i32;
}

// SAFETY: The wrapping add result of two `i32`s is a valid `i32`.
unsafe impl generic::AllowAtomicAdd<i32> for i32 {
    fn rhs_into_delta(rhs: i32) -> i32 {
        rhs
    }
}

// SAFETY: `i64` has the same size and alignment with itself, and is round-trip transmutable to
// itself.
unsafe impl generic::AllowAtomic for i64 {
    type Repr = i64;
}

// SAFETY: The wrapping add result of two `i64`s is a valid `i64`.
unsafe impl generic::AllowAtomicAdd<i64> for i64 {
    fn rhs_into_delta(rhs: i64) -> i64 {
        rhs
    }
}

// Defines an internal type that always maps to the integer type which has the same size alignment
// as `isize` and `usize`, and `isize` and `usize` are always bi-directional transmutable to
// `isize_atomic_repr`, which also always implements `AtomicImpl`.
#[allow(non_camel_case_types)]
#[cfg(not(CONFIG_64BIT))]
type isize_atomic_repr = i32;
#[allow(non_camel_case_types)]
#[cfg(CONFIG_64BIT)]
type isize_atomic_repr = i64;

// Ensure size and alignment requirements are checked.
crate::static_assert!(core::mem::size_of::<isize>() == core::mem::size_of::<isize_atomic_repr>());
crate::static_assert!(core::mem::align_of::<isize>() == core::mem::align_of::<isize_atomic_repr>());
crate::static_assert!(core::mem::size_of::<usize>() == core::mem::size_of::<isize_atomic_repr>());
crate::static_assert!(core::mem::align_of::<usize>() == core::mem::align_of::<isize_atomic_repr>());

// SAFETY: `isize` has the same size and alignment with `isize_atomic_repr`, and is round-trip
// transmutable to `isize_atomic_repr`.
unsafe impl generic::AllowAtomic for isize {
    type Repr = isize_atomic_repr;
}

// SAFETY: The wrapping add result of two `isize_atomic_repr`s is a valid `usize`.
unsafe impl generic::AllowAtomicAdd<isize> for isize {
    fn rhs_into_delta(rhs: isize) -> isize_atomic_repr {
        rhs as isize_atomic_repr
    }
}

// SAFETY: `u32` and `i32` has the same size and alignment, and `u32` is round-trip transmutable to
// `i32`.
unsafe impl generic::AllowAtomic for u32 {
    type Repr = i32;
}

// SAFETY: The wrapping add result of two `i32`s is a valid `u32`.
unsafe impl generic::AllowAtomicAdd<u32> for u32 {
    fn rhs_into_delta(rhs: u32) -> i32 {
        rhs as i32
    }
}

// SAFETY: `u64` and `i64` has the same size and alignment, and `u64` is round-trip transmutable to
// `i64`.
unsafe impl generic::AllowAtomic for u64 {
    type Repr = i64;
}

// SAFETY: The wrapping add result of two `i64`s is a valid `u64`.
unsafe impl generic::AllowAtomicAdd<u64> for u64 {
    fn rhs_into_delta(rhs: u64) -> i64 {
        rhs as i64
    }
}

// SAFETY: `usize` has the same size and alignment with `isize_atomic_repr`, and is round-trip
// transmutable to `isize_atomic_repr`.
unsafe impl generic::AllowAtomic for usize {
    type Repr = isize_atomic_repr;
}

// SAFETY: The wrapping add result of two `isize_atomic_repr`s is a valid `usize`.
unsafe impl generic::AllowAtomicAdd<usize> for usize {
    fn rhs_into_delta(rhs: usize) -> isize_atomic_repr {
        rhs as isize_atomic_repr
    }
}

// SAFETY: `*mut T` and `*mut ()` has the same size and alignment, and `*mut T` is round-trip
// transmutable to `*mut ()`.
unsafe impl<T> generic::AllowAtomic for *mut T {
    type Repr = *mut crate::ffi::c_void;
}

use crate::macros::kunit_tests;

#[kunit_tests(rust_atomics)]
mod tests {
    use super::*;

    // Call $fn($val) with each $type of $val.
    macro_rules! for_each_type {
        ($val:literal in [$($type:ty),*] $fn:expr) => {
            $({
                let v: $type = $val;

                $fn(v);
            })*
        }
    }

    #[test]
    fn atomic_basic_tests() {
        for_each_type!(42 in [i32, i64, u32, u64, isize, usize] |v| {
            let x = Atomic::new(v);

            assert_eq!(v, x.load(Relaxed));
        });

        let x = Atomic::new(core::ptr::null_mut::<i32>());
        assert!(x.load(Relaxed).is_null());
    }

    #[test]
    fn atomic_xchg_tests() {
        for_each_type!(42 in [i32, i64, u32, u64, isize, usize] |v| {
            let x = Atomic::new(v);

            let old = v;
            let new = v + 1;

            assert_eq!(old, x.xchg(new, Full));
            assert_eq!(new, x.load(Relaxed));
        });
    }

    #[test]
    fn atomic_cmpxchg_tests() {
        for_each_type!(42 in [i32, i64, u32, u64, isize, usize] |v| {
            let x = Atomic::new(v);

            let old = v;
            let new = v + 1;

            assert_eq!(Err(old), x.cmpxchg(new, new, Full));
            assert_eq!(old, x.load(Relaxed));
            assert_eq!(Ok(old), x.cmpxchg(old, new, Relaxed));
            assert_eq!(new, x.load(Relaxed));
        });
    }

    #[test]
    fn atomic_arithmetic_tests() {
        for_each_type!(42 in [i32, i64, u32, u64, isize, usize] |v| {
            let x = Atomic::new(v);

            assert_eq!(v, x.fetch_add(12, Full));
            assert_eq!(v + 12, x.load(Relaxed));

            x.add(13, Relaxed);

            assert_eq!(v + 25, x.load(Relaxed));
        });
    }

    #[test]
    fn atomic_ptr_tests() -> crate::error::Result {
        use crate::alloc::{flags::GFP_KERNEL, KBox};
        use core::ptr;

        let x = Atomic::new(ptr::null_mut::<i32>());

        assert!(x.load(Relaxed).is_null());

        let new = KBox::new(42, GFP_KERNEL)?;
        x.store(ptr::from_mut(KBox::leak(new)), Release);

        let ptr = x.load(Relaxed);
        assert!(!ptr.is_null());

        // SAFETY: `ptr` is a valid pointer from `KBox::leak()` and the address dependency
        // guarantees observation of the initialization of `KBox`.
        assert_eq!(42, unsafe { ptr.read_volatile() });

        x.xchg(ptr::null_mut(), Relaxed);
        assert!(x.load(Relaxed).is_null());

        // SAFETY: `ptr` is a valid pointer from `KBox::leak()` and no one is currently referencing
        // the pointer, so it's safety to convert the ownership back to a `KBox`.
        drop(unsafe { KBox::from_raw(ptr) });

        Ok(())
    }
}
