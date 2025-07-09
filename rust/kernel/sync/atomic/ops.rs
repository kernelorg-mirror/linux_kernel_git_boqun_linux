// SPDX-License-Identifier: GPL-2.0

//! Atomic implementations.
//!
//! Provides 1:1 mapping of atomic implementations.

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

// `atomic_ptr` implements atomic operations on `*mut crate::ffi::c_void`
impl AtomicImpl for *mut crate::ffi::c_void {
    type Delta = isize;
}

// This macro generates the function signature with given argument list and return type.
macro_rules! declare_atomic_method {
    (
        $(#[doc=$doc:expr])*
        $func:ident($($arg:ident : $arg_type:ty),*) $(-> $ret:ty)?
    ) => {
        paste!(
            $(#[doc = $doc])*
            unsafe fn [< atomic_ $func >]($($arg: $arg_type,)*) $(-> $ret)?;
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
            unsafe fn [< atomic_ $func >]($($arg: $arg_type,)*) $(-> $ret)? {
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
            unsafe fn $func:ident [$($variant:ident),*]($($arg_sig:tt)*) $( -> $ret:ty)? {
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
                unsafe fn $func [$($variant),*]($($arg_sig)*) $( -> $ret)? {
                bindings::#call($($arg)*)
                }
            )*
            }
        );
    };
    (@impl pub trait $ops:ident [] {
        $(
            unsafe fn $func:ident [$($variant:ident),*]($($arg_sig:tt)*) $( -> $ret:ty)? {
                bindings::#call($($arg:tt)*)
            }
        )*
    }) => {
    };
    (@impl pub trait $ops:ident [$impl:ty => $c_type:ident $(, $rest_impl:ty => $rest_c_type:ident)*] {
        $(
            unsafe fn $func:ident [$($variant:ident),*]($($arg_sig:tt)*) $( -> $ret:ty)? {
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
                unsafe fn $func [$($variant),*]($($arg_sig)*) $( -> $ret)? {
                bindings::#call($($arg)*)
                }
            )*
            }
        );
    }
}

declare_and_impl_atomic_methods!(
    /// Basic atomic operations
    pub trait AtomicHasBasicOps
    [i32 => atomic, i64 => atomic64, *mut crate::ffi::c_void => atomic_ptr]
    {
        /// Atomic read (load).
        ///
        /// # Safety
        /// - `ptr` is aligned to [`align_of::<Self>()`].
        /// - `ptr` is valid for reads.
        ///
        /// [`align_of::<Self>()`]: core::mem::align_of
        unsafe fn read[acquire](ptr: *mut Self) -> Self {
            bindings::#call(ptr.cast())
        }

        /// Atomic set (store).
        ///
        /// # Safety
        /// - `ptr` is aligned to [`align_of::<Self>()`].
        /// - `ptr` is valid for writes.
        ///
        /// [`align_of::<Self>()`]: core::mem::align_of
        unsafe fn set[release](ptr: *mut Self, v: Self) {
            bindings::#call(ptr.cast(), v)
        }
    }
);

declare_and_impl_atomic_methods!(
    /// Exchange and compare-and-exchange atomic operations
    pub trait AtomicHasXchgOps
    [i32 => atomic, i64 => atomic64, *mut crate::ffi::c_void => atomic_ptr]
    {
        /// Atomic exchange.
        ///
        /// Atomically updates `*ptr` to `v` and returns the old value.
        ///
        /// # Safety
        /// - `ptr` is aligned to [`align_of::<Self>()`].
        /// - `ptr` is valid for reads and writes.
        ///
        /// [`align_of::<Self>()`]: core::mem::align_of
        unsafe fn xchg[acquire, release, relaxed](ptr: *mut Self, v: Self) -> Self {
            bindings::#call(ptr.cast(), v)
        }

        /// Atomic compare and exchange.
        ///
        /// If `*ptr` == `*old`, atomically updates `*ptr` to `new`. Otherwise, `*ptr` is not
        /// modified, `*old` is updated to the current value of `*ptr`.
        ///
        /// Return `true` if the update of `*ptr` occured, `false` otherwise.
        ///
        /// # Safety
        /// - `ptr` is aligned to [`align_of::<Self>()`].
        /// - `ptr` is valid for reads and writes.
        /// - `old` is aligned to [`align_of::<Self>()`].
        /// - `old` is valid for reads and writes.
        ///
        /// [`align_of::<Self>()`]: core::mem::align_of
        unsafe fn try_cmpxchg[acquire, release, relaxed](ptr: *mut Self, old: *mut Self, new: Self) -> bool {
            bindings::#call(ptr.cast(), old, new)
        }
    }
);

declare_and_impl_atomic_methods!(
    /// Atomic arithmetic operations
    pub trait AtomicHasArithmeticOps
    [i32 => atomic, i64 => atomic64]
    {
        /// Atomic add (wrapping).
        ///
        /// Atomically updates `*ptr` to `(*ptr).wrapping_add(v)`.
        ///
        /// # Safety
        /// - `ptr` is aligned to `align_of::<Self>()`.
        /// - `ptr` is valid for reads and writes.
        ///
        /// [`align_of::<Self>()`]: core::mem::align_of
        unsafe fn add[](ptr: *mut Self, v: Self::Delta) {
            bindings::#call(v, ptr.cast())
        }

        /// Atomic fetch and add (wrapping).
        ///
        /// Atomically updates `*ptr` to `(*ptr).wrapping_add(v)`, and returns the value of `*ptr`
        /// before the update.
        ///
        /// # Safety
        /// - `ptr` is aligned to `align_of::<Self>()`.
        /// - `ptr` is valid for reads and writes.
        ///
        /// [`align_of::<Self>()`]: core::mem::align_of
        unsafe fn fetch_add[acquire, release, relaxed](ptr: *mut Self, v: Self::Delta) -> Self {
            bindings::#call(v, ptr.cast())
        }
    }
);
