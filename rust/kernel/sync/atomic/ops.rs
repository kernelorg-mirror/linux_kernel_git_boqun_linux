// SPDX-License-Identifier: GPL-2.0

//! Atomic operation implementations.
//!
//! Provides 1:1 mapping to the C atomic operations.

use crate::bindings;
use crate::macros::paste;

mod private {
    /// Sealed trait marker to disable customized impls on atomic implementation traits.
    pub trait Sealed {}
}

// `i32` and `i64` are only supported atomic implementations.
impl private::Sealed for i32 {}
impl private::Sealed for i64 {}
impl private::Sealed for *mut crate::ffi::c_void {}

/// A marker trait for types that implement atomic operations with C side primitives.
///
/// This trait is sealed, and only types that have directly mapping to the C side atomics should
/// impl this:
///
/// - `i32` maps to `atomic_t`.
/// - `i64` maps to `atomic64_t`.
pub trait AtomicImpl: Sized + Copy + private::Sealed {
    /// The type of the delta in arithmetic or logical operations.
    ///
    /// For example, in `atomic_add(ptr, v)`, it's the type of `v`. Usually it's the same type of
    /// [`Self`], but it may be different for the atomic pointer type.
    type Delta;
}

// `atomic_t` implements atomic operations on `i32`.
impl AtomicImpl for i32 {
    type Delta = Self;
}

// `atomic64_t` implements atomic operations on `i64`.
impl AtomicImpl for i64 {
    type Delta = Self;
}

impl AtomicImpl for *mut crate::ffi::c_void {
    type Delta = isize;
}

/// Atomic representation.
pub(super) struct AtomicRepr<T: AtomicImpl>(core::cell::UnsafeCell<T>);

impl<T: AtomicImpl> AtomicRepr<T> {
    pub(super) const fn new(v: T) -> Self {
        Self(core::cell::UnsafeCell::new(v))
    }

    pub(super) const fn as_ptr(&self) -> *mut T {
        self.0.get()
    }
}

// This macro generates the function signature with given argument list and return type.
macro_rules! declare_atomic_method {
    (
        $(#[doc=$doc:expr])*
        $func:ident($($arg:ident : $arg_type:ty),*) $(-> $ret:ty)?
    ) => {
        paste!(
            $(#[doc = $doc])*
            fn [< atomic_ $func >]($($arg: $arg_type,)*) $(-> $ret)?;
        );
    };
    (
        $(#[doc=$doc:expr])*
        $func:ident [$variant:ident $($rest:ident)*]($($arg_sig:tt)*) $(-> $ret:ty)?
    ) => {
        paste!(
            declare_atomic_method!(
                $(#[doc = $doc])*
                [< $func _ $variant >]($($arg_sig)*) $(-> $ret)?
            );
        );

        declare_atomic_method!(
            $(#[doc = $doc])*
            $func [$($rest)*]($($arg_sig)*) $(-> $ret)?
        );
    };
    (
        $(#[doc=$doc:expr])*
        $func:ident []($($arg_sig:tt)*) $(-> $ret:ty)?
    ) => {
        declare_atomic_method!(
            $(#[doc = $doc])*
            $func($($arg_sig)*) $(-> $ret)?
        );
    }
}

// This macro generates the function implementation with given argument list and return type, and it
// will replace "call(...)" expression with "$ctype _ $func" to call the real C function.
macro_rules! impl_atomic_method {
    (
        ($ctype:ident) $func:ident($($arg:ident: $arg_type:ty),*) $(-> $ret:ty)? {
            call($($c_arg:expr),*)
        }
    ) => {
        paste!(
            #[inline(always)]
            fn [< atomic_ $func >]($($arg: $arg_type,)*) $(-> $ret)? {
                // SAFETY: Per function safety requirement, all pointers are aligned and valid, and
                // accesses won't cause data race per LKMM.
                unsafe { bindings::[< $ctype _ $func >]($($c_arg,)*) }
            }
        );
    };
    (
        ($ctype:ident) $func:ident[$variant:ident $($rest:ident)*]($($arg_sig:tt)*) $(-> $ret:ty)? {
            call($($arg:tt)*)
        }
    ) => {
        paste!(
            impl_atomic_method!(
                ($ctype) [< $func _ $variant >]($($arg_sig)*) $( -> $ret)? {
                    call($($arg)*)
            }
            );
        );
        impl_atomic_method!(
            ($ctype) $func [$($rest)*]($($arg_sig)*) $( -> $ret)? {
                call($($arg)*)
            }
        );
    };
    (
        ($ctype:ident) $func:ident[]($($arg_sig:tt)*) $( -> $ret:ty)? {
            call($($arg:tt)*)
        }
    ) => {
        impl_atomic_method!(
            ($ctype) $func($($arg_sig)*) $(-> $ret)? {
                call($($arg)*)
            }
        );
    }
}

// Delcares $ops trait with methods and implements the trait for `i32` and `i64`.
macro_rules! declare_and_impl_atomic_methods {
    ($(#[$attr:meta])* pub trait $ops:ident [$($rest:tt)*] {
        $(
            $(#[doc=$doc:expr])*
            fn $func:ident [$($variant:ident),*]($($arg_sig:tt)*) $( -> $ret:ty)? {
                bindings::#call($($arg:tt)*)
            }
        )*
    }) => {
        $(#[$attr])*
        pub trait $ops: AtomicImpl {
            $(
                declare_atomic_method!(
                    $(#[doc=$doc])*
                    $func[$($variant)*]($($arg_sig)*) $(-> $ret)?
                );
            )*
        }

        declare_and_impl_atomic_methods!(@impl
            pub trait $ops [$($rest)*] {
                $(
                    fn $func [$($variant),*]($($arg_sig)*) $( -> $ret)? {
                        bindings::#call($($arg)*)
                    }
                )*
            }
        );
    };
    (@impl pub trait $ops:ident [] {
        $(
            fn $func:ident [$($variant:ident),*]($($arg_sig:tt)*) $( -> $ret:ty)? {
                bindings::#call($($arg:tt)*)
            }
        )*
    }) => {
    };
    (@impl pub trait $ops:ident [$impl:ty => $c_type:ident $(, $rest_impl:ty => $rest_c_type:ident)*] {
        $(
            fn $func:ident [$($variant:ident),*]($($arg_sig:tt)*) $( -> $ret:ty)? {
                bindings::#call($($arg:tt)*)
            }
        )*
    }) => {
        impl $ops for $impl {
            $(
                impl_atomic_method!(
                    ($c_type) $func[$($variant)*]($($arg_sig)*) $(-> $ret)? {
                        call($($arg)*)
                    }
                );
            )*
        }

        declare_and_impl_atomic_methods!(@impl
            pub trait $ops [$($rest_impl => $rest_c_type),*] {
            $(
                fn $func [$($variant),*]($($arg_sig)*) $( -> $ret)? {
                    bindings::#call($($arg)*)
                }
            )*
            }
        );
    }
}

declare_and_impl_atomic_methods!(
    /// Basic atomic operations
    pub trait AtomicBasicOps
    [i32 => atomic, i64 => atomic64, *mut crate::ffi::c_void => atomic_ptr]
    {
        /// Atomic read (load).
        fn read[acquire](a: &AtomicRepr<Self>) -> Self {
            // SAFETY: `a.as_ptr()` is valid and properly aligned.
            bindings::#call(a.as_ptr().cast())
        }

        /// Atomic set (store).
        fn set[release](a: &AtomicRepr<Self>, v: Self) {
            // SAFETY: `a.as_ptr()` is valid and properly aligned.
            bindings::#call(a.as_ptr().cast(), v)
        }
    }
);

declare_and_impl_atomic_methods!(
    /// Exchange and compare-and-exchange atomic operations
    pub trait AtomicExchangeOps
    [i32 => atomic, i64 => atomic64, *mut crate::ffi::c_void => atomic_ptr]
    {
        /// Atomic exchange.
        ///
        /// Atomically updates `*a` to `v` and returns the old value.
        fn xchg[acquire, release, relaxed](a: &AtomicRepr<Self>, v: Self) -> Self {
            // SAFETY: `a.as_ptr()` is valid and properly aligned.
            bindings::#call(a.as_ptr().cast(), v)
        }

        /// Atomic compare and exchange.
        ///
        /// If `*a` == `*old`, atomically updates `*a` to `new`. Otherwise, `*a` is not
        /// modified, `*old` is updated to the current value of `*a`.
        ///
        /// Return `true` if the update of `*ptr` occured, `false` otherwise.
        fn try_cmpxchg[acquire, release, relaxed](
            a: &AtomicRepr<Self>, old: &mut Self, new: Self
        ) -> bool {
            // SAFETY: `a.as_ptr()` is valid and properly aligned. `core::ptr::from_mut(old)`
            // is valid and properly aligned.
            bindings::#call(a.as_ptr().cast(), core::ptr::from_mut(old), new)
        }
    }
);

declare_and_impl_atomic_methods!(
    /// Atomic arithmetic operations
    pub trait AtomicArithmeticOps
    [i32 => atomic, i64 => atomic64]
    {
        /// Atomic add (wrapping).
        ///
        /// Atomically updates `*a` to `(*a).wrapping_add(v)`.
        fn add[](a: &AtomicRepr<Self>, v: Self::Delta) {
            // SAFETY: `a.as_ptr()` is valid and properly aligned.
            bindings::#call(v, a.as_ptr().cast())
        }

        /// Atomic fetch and add (wrapping).
        ///
        /// Atomically updates `*a` to `(*a).wrapping_add(v)`, and returns the value of `*a`
        /// before the update.
        fn fetch_add[acquire, release, relaxed](a: &AtomicRepr<Self>, v: Self::Delta) -> Self {
            // SAFETY: `a.as_ptr()` is valid and properly aligned.
            bindings::#call(v, a.as_ptr().cast())
        }
    }
);
