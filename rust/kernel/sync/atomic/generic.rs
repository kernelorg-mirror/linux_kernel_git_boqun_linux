// SPDX-License-Identifier: GPL-2.0

//! Generic atomic primitives.

use super::ops::*;
use super::ordering::*;
use crate::build_error;
use core::cell::UnsafeCell;

/// A generic atomic variable.
///
/// `T` must impl [`AllowAtomic`], that is, an [`AtomicImpl`] has to be chosen.
///
/// # Examples
///
/// Customized new types in [`Atomic`]:
///
/// ```rust
/// use kernel::sync::atomic::{generic::AllowAtomic, Atomic, Relaxed};
///
/// #[derive(Clone, Copy, PartialEq, Eq)]
/// #[repr(i32)]
/// enum State {
///     Uninit = 0,
///     Working = 1,
///     Done = 2,
/// };
///
/// // SAFETY: `State` and `i32` has the same size and alignment, and it's round-trip
/// // transmutable to `i32`.
/// unsafe impl AllowAtomic for State {
///     type Repr = i32;
/// }
///
/// let s = Atomic::new(State::Uninit);
///
/// assert_eq!(State::Uninit, s.load(Relaxed));
/// ```
/// # Invariants
///
/// Doing an atomic operation while holding a reference of [`Self`] won't cause a data race, this
/// is guaranteed by the safety requirement of [`Self::from_ptr`] and the extra safety requirement
/// of the usage on pointers returned by [`Self::as_ptr`].
#[repr(transparent)]
pub struct Atomic<T: AllowAtomic>(UnsafeCell<T>);

// SAFETY: `Atomic<T>` is safe to share among execution contexts because all accesses are atomic.
unsafe impl<T: AllowAtomic> Sync for Atomic<T> {}

/// Atomics that support basic atomic operations.
///
/// Implementers must guarantee that `into_repr()` and `from_repr()` provide the same results as
/// [`transmute()`] between [`Self`] and [`Self::Repr`].
///
/// # Safety
///
/// - [`Self`] must have the same size and alignment as [`Self::Repr`].
/// - [`Self`] and [`Self::Repr`] must have the round-trip transmutability:
///   - Any value of [`Self`] must be safe to [`transmute()`] to a [`Self::Repr`], this also means
///     that a valid pointer to [`Self`] is a valid pointer to [`Self::Repr`].
///   - If a value of [`Self::Repr`] is a result a [`transmute()`] from a [`Self`], it must be safe
///     to [`transmute()`] the value back to a [`Self`].
///
/// [`transmute()`]: core::mem::transmute
pub unsafe trait AllowAtomic: Sized + Send + Copy {
    /// The backing atomic implementation type.
    type Repr: AtomicImpl;

    /// Converts into a [`Self::Repr`].
    fn into_repr(self) -> Self::Repr {
        // SAFETY: Per the safety requirement of `AllowAtomic`, it's safe to `transmute()` from
        // [`Self`] to [`Self::Repr`].
        unsafe { core::mem::transmute_copy(&self) }
    }

    /// Converts from a [`Self::Repr`].
    ///
    /// # Safety
    ///
    /// Must guarantee `repr` is a result of a previous `into_repr()` or `transmute()` from
    /// [`Self`]
    unsafe fn from_repr(repr: Self::Repr) -> Self {
        // SAFETY: Per the safety requirement of `AllowAtomic`, it's safe to `transmute()` from
        // [`Self::Repr`] to [`Self`] if the [`Self::Repr`] is a result of a previous
        // `transmute()`.
        unsafe { core::mem::transmute_copy(&repr) }
    }
}

// An `AtomicImpl` is automatically an `AllowAtomic`.
//
// SAFETY: `T::Repr` is `Self` (i.e. `T`), so they have the same size and alignment, and it's
// round-trip transmutable to itself.
unsafe impl<T: AtomicImpl> AllowAtomic for T {
    type Repr = Self;
}

impl<T: AllowAtomic> Atomic<T> {
    /// Creates a new atomic.
    pub const fn new(v: T) -> Self {
        Self(UnsafeCell::new(v))
    }

    /// Creates a reference to [`Self`] from a pointer.
    ///
    /// # Safety
    ///
    /// - `ptr` has to be a valid pointer.
    /// - `ptr` has to be valid for both reads and writes for the whole lifetime `'a`.
    /// - For the duration of `'a`, other accesses to the object cannot cause data races (defined
    ///   by [`LKMM`]) against atomic operations on the returned reference. Note that if all other
    ///   accesses are atomic, then this safety requirement is trivially fulfilled.
    ///
    /// [`LKMM`]: srctree/tools/memory-model
    ///
    /// # Examples
    ///
    /// Using [`Atomic::from_ptr()`] combined with [`Atomic::load()`] or [`Atomic::store()`] can
    /// achieve the same functionality as `READ_ONCE()`/`smp_load_acquire()` or
    /// `WRITE_ONCE()`/`smp_store_release()` in C side:
    ///
    /// ```rust
    /// # use kernel::types::Opaque;
    /// use kernel::sync::atomic::{Atomic, Relaxed, Release};
    ///
    /// // Assume there is a C struct `Foo`.
    /// mod cbindings {
    ///     #[repr(C)]
    ///     pub(crate) struct foo { pub(crate) a: i32, pub(crate) b: i32 }
    /// }
    ///
    /// let tmp = Opaque::new(cbindings::foo { a: 1, b: 2});
    ///
    /// // struct foo *foo_ptr = ..;
    /// let foo_ptr = tmp.get();
    ///
    /// // SAFETY: `foo_ptr` is a valid pointer, and `.a` is in bounds.
    /// let foo_a_ptr = unsafe { &raw mut (*foo_ptr).a };
    ///
    /// // a = READ_ONCE(foo_ptr->a);
    /// //
    /// // SAFETY: `foo_a_ptr` is a valid pointer for read, and all accesses on it is atomic, so no
    /// // data race.
    /// let a = unsafe { Atomic::from_ptr(foo_a_ptr) }.load(Relaxed);
    /// # assert_eq!(a, 1);
    ///
    /// // smp_store_release(&foo_ptr->a, 2);
    /// //
    /// // SAFETY: `foo_a_ptr` is a valid pointer for write, and all accesses on it is atomic, so no
    /// // data race.
    /// unsafe { Atomic::from_ptr(foo_a_ptr) }.store(2, Release);
    /// ```
    ///
    /// However, this should be only used when communicating with C side or manipulating a C struct.
    pub unsafe fn from_ptr<'a>(ptr: *mut T) -> &'a Self
    where
        T: Sync,
    {
        // CAST: `T` is transparent to `Atomic<T>`.
        // SAFETY: Per function safety requirement, `ptr` is a valid pointer and the object will
        // live long enough. It's safe to return a `&Atomic<T>` because function safety requirement
        // guarantees other accesses won't cause data races.
        unsafe { &*ptr.cast::<Self>() }
    }

    /// Returns a pointer to the underlying atomic variable.
    ///
    /// Extra safety requirement on using the return pointer: the operations done via the pointer
    /// cannot cause data races defined by [`LKMM`].
    ///
    /// [`LKMM`]: srctree/tools/memory-model
    pub const fn as_ptr(&self) -> *mut T {
        self.0.get()
    }

    /// Returns a mutable reference to the underlying atomic variable.
    ///
    /// This is safe because the mutable reference of the atomic variable guarantees the exclusive
    /// access.
    pub fn get_mut(&mut self) -> &mut T {
        // SAFETY: `self.as_ptr()` is a valid pointer to `T`, and the object has already been
        // initialized. `&mut self` guarantees the exclusive access, so it's safe to reborrow
        // mutably.
        unsafe { &mut *self.as_ptr() }
    }
}

impl<T: AllowAtomic> Atomic<T>
where
    T::Repr: AtomicHasBasicOps,
{
    /// Loads the value from the atomic variable.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kernel::sync::atomic::{Atomic, Relaxed};
    ///
    /// let x = Atomic::new(42i32);
    ///
    /// assert_eq!(42, x.load(Relaxed));
    ///
    /// let x = Atomic::new(42i64);
    ///
    /// assert_eq!(42, x.load(Relaxed));
    /// ```
    #[doc(alias("atomic_read", "atomic64_read"))]
    #[inline(always)]
    pub fn load<Ordering: AcquireOrRelaxed>(&self, _: Ordering) -> T {
        let a = self.as_ptr().cast::<T::Repr>();

        // SAFETY:
        // - For calling the atomic_read*() function:
        //   - `self.as_ptr()` is a valid pointer, and per the safety requirement of `AllowAtomic`,
        //      a `*mut T` is a valid `*mut T::Repr`. Therefore `a` is a valid pointer,
        //   - per the type invariants, the following atomic operation won't cause data races.
        // - For extra safety requirement of usage on pointers returned by `self.as_ptr():
        //   - atomic operations are used here.
        let v = unsafe {
            match Ordering::TYPE {
                OrderingType::Relaxed => T::Repr::atomic_read(a),
                OrderingType::Acquire => T::Repr::atomic_read_acquire(a),
                _ => build_error!("Wrong ordering"),
            }
        };

        // SAFETY: Per the type invariants the value of the atomic variable is a valid `T`, so it's
        // safe to call `from_repr()`.
        unsafe { T::from_repr(v) }
    }

    /// Stores a value to the atomic variable.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kernel::sync::atomic::{Atomic, Relaxed};
    ///
    /// let x = Atomic::new(42i32);
    ///
    /// assert_eq!(42, x.load(Relaxed));
    ///
    /// x.store(43, Relaxed);
    ///
    /// assert_eq!(43, x.load(Relaxed));
    /// ```
    #[doc(alias("atomic_set", "atomic64_set"))]
    #[inline(always)]
    pub fn store<Ordering: ReleaseOrRelaxed>(&self, v: T, _: Ordering) {
        let v = T::into_repr(v);
        let a = self.as_ptr().cast::<T::Repr>();

        // SAFETY:
        // - For calling the atomic_set*() function:
        //   - `self.as_ptr()` is a valid pointer, and per the safety requirement of `AllowAtomic`,
        //      a `*mut T` is a valid `*mut T::Repr`. Therefore `a` is a valid pointer,
        //   - per the type invariants, the following atomic operation won't cause data races.
        // - For extra safety requirement of usage on pointers returned by `self.as_ptr():
        //   - atomic operations are used here.
        unsafe {
            match Ordering::TYPE {
                OrderingType::Relaxed => T::Repr::atomic_set(a, v),
                OrderingType::Release => T::Repr::atomic_set_release(a, v),
                _ => build_error!("Wrong ordering"),
            }
        };
    }
}

impl<T: AllowAtomic> Atomic<T>
where
    T::Repr: AtomicHasXchgOps,
{
    /// Atomic exchange.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kernel::sync::atomic::{Atomic, Acquire, Relaxed};
    ///
    /// let x = Atomic::new(42);
    ///
    /// assert_eq!(42, x.xchg(52, Acquire));
    /// assert_eq!(52, x.load(Relaxed));
    /// ```
    #[doc(alias("atomic_xchg", "atomic64_xchg", "swap"))]
    #[inline(always)]
    pub fn xchg<Ordering: Any>(&self, v: T, _: Ordering) -> T {
        let v = T::into_repr(v);
        let a = self.as_ptr().cast::<T::Repr>();

        // SAFETY:
        // - For calling the atomic_xchg*() function:
        //   - `self.as_ptr()` is a valid pointer, and per the safety requirement of `AllowAtomic`,
        //      a `*mut T` is a valid `*mut T::Repr`. Therefore `a` is a valid pointer,
        //   - per the type invariants, the following atomic operation won't cause data races.
        // - For extra safety requirement of usage on pointers returned by `self.as_ptr():
        //   - atomic operations are used here.
        let ret = unsafe {
            match Ordering::TYPE {
                OrderingType::Full => T::Repr::atomic_xchg(a, v),
                OrderingType::Acquire => T::Repr::atomic_xchg_acquire(a, v),
                OrderingType::Release => T::Repr::atomic_xchg_release(a, v),
                OrderingType::Relaxed => T::Repr::atomic_xchg_relaxed(a, v),
            }
        };

        // SAFETY: The value of the atomic variable is a valid `T`, so load from it is safe to call
        // `from_repr()`.
        unsafe { T::from_repr(ret) }
    }

    /// Atomic compare and exchange.
    ///
    /// Compare: The comparison is done via the byte level comparison between the atomic variables
    /// with the `old` value.
    ///
    /// Ordering: When succeeds, provides the corresponding ordering as the `Ordering` type
    /// parameter indicates, and a failed one doesn't provide any ordering, the read part of a
    /// failed cmpxchg should be treated as a relaxed read.
    ///
    /// Returns `Ok(value)` if cmpxchg succeeds, and `value` is guaranteed to be equal to `old`,
    /// otherwise returns `Err(value)`, and `value` is the value of the atomic variable when
    /// cmpxchg was happening.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kernel::sync::atomic::{Atomic, Full, Relaxed};
    ///
    /// let x = Atomic::new(42);
    ///
    /// // Checks whether cmpxchg succeeded.
    /// let success = x.cmpxchg(52, 64, Relaxed).is_ok();
    /// # assert!(!success);
    ///
    /// // Checks whether cmpxchg failed.
    /// let failure = x.cmpxchg(52, 64, Relaxed).is_err();
    /// # assert!(failure);
    ///
    /// // Uses the old value if failed, probably re-try cmpxchg.
    /// match x.cmpxchg(52, 64, Relaxed) {
    ///     Ok(_) => { },
    ///     Err(old) => {
    ///         // do something with `old`.
    ///         # assert_eq!(old, 42);
    ///     }
    /// }
    ///
    /// // Uses the latest value regardlessly, same as atomic_cmpxchg() in C.
    /// let latest = x.cmpxchg(42, 64, Full).unwrap_or_else(|old| old);
    /// # assert_eq!(42, latest);
    /// assert_eq!(64, x.load(Relaxed));
    /// ```
    #[doc(alias(
        "atomic_cmpxchg",
        "atomic64_cmpxchg",
        "atomic_try_cmpxchg",
        "atomic64_try_cmpxchg",
        "compare_exchange"
    ))]
    #[inline(always)]
    pub fn cmpxchg<Ordering: Any>(&self, mut old: T, new: T, o: Ordering) -> Result<T, T> {
        // Note on code generation:
        //
        // try_cmpxchg() is used to implement cmpxchg(), and if the helper functions are inlined,
        // the compiler is able to figure out that branch is not needed if the users don't care
        // about whether the operation succeeds or not. One exception is on x86, due to commit
        // 44fe84459faf ("locking/atomic: Fix atomic_try_cmpxchg() semantics"), the
        // atomic_try_cmpxchg() on x86 has a branch even if the caller doesn't care about the
        // success of cmpxchg and only wants to use the old value. For example, for code like:
        //
        //     let latest = x.cmpxchg(42, 64, Full).unwrap_or_else(|old| old);
        //
        // It will still generate code:
        //
        //     movl    $0x40, %ecx
        //     movl    $0x34, %eax
        //     lock
        //     cmpxchgl        %ecx, 0x4(%rsp)
        //     jne     1f
        //     2:
        //     ...
        //     1:  movl    %eax, %ecx
        //     jmp 2b
        //
        // This might be "fixed" by introducing a try_cmpxchg_exclusive() that knows the "*old"
        // location in the C function is always safe to write.
        if self.try_cmpxchg(&mut old, new, o) {
            Ok(old)
        } else {
            Err(old)
        }
    }

    /// Atomic compare and exchange and returns whether the operation succeeds.
    ///
    /// "Compare" and "Ordering" part are the same as [`Atomic::cmpxchg()`].
    ///
    /// Returns `true` means the cmpxchg succeeds otherwise returns `false` with `old` updated to
    /// the value of the atomic variable when cmpxchg was happening.
    #[inline(always)]
    fn try_cmpxchg<Ordering: Any>(&self, old: &mut T, new: T, _: Ordering) -> bool {
        let mut old_tmp = T::into_repr(*old);
        let old_p = &raw mut old_tmp;
        let new = T::into_repr(new);
        let a = self.0.get().cast::<T::Repr>();

        // SAFETY:
        // - For calling the atomic_try_cmpchg*() function:
        //   - `self.as_ptr()` is a valid pointer, and per the safety requirement of `AllowAtomic`,
        //      a `*mut T` is a valid `*mut T::Repr`. Therefore `a` is a valid pointer,
        //   - per the type invariants, the following atomic operation won't cause data races.
        //   - `old_p` is a valid pointer to write.
        // - For extra safety requirement of usage on pointers returned by `self.as_ptr():
        //   - atomic operations are used here.
        let ret = unsafe {
            match Ordering::TYPE {
                OrderingType::Full => T::Repr::atomic_try_cmpxchg(a, old_p, new),
                OrderingType::Acquire => T::Repr::atomic_try_cmpxchg_acquire(a, old_p, new),
                OrderingType::Release => T::Repr::atomic_try_cmpxchg_release(a, old_p, new),
                OrderingType::Relaxed => T::Repr::atomic_try_cmpxchg_relaxed(a, old_p, new),
            }
        };

        // SAFETY: The value of the atomic variable is a valid `T`, so load from it is safe to call
        // `from_repr()`.
        *old = unsafe { T::from_repr(old_tmp) };
        ret
    }
}
