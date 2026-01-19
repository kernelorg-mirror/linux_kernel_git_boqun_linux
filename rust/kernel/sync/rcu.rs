// SPDX-License-Identifier: GPL-2.0

//! RCU support.
//!
//! C header: [`include/linux/rcupdate.h`](srctree/include/linux/rcupdate.h)

use crate::{
    alloc::{Allocator, Box},
    bindings,
    field::{Field, HasField},
    macros::HasField,
    types::{ForeignOwnable, NotThreadSafe, Opaque},
};

use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ops::Deref;

use ffi::c_void;

/// Evidence that the RCU read side lock is held on the current thread/CPU.
///
/// The type is explicitly not `Send` because this property is per-thread/CPU.
///
/// # Invariants
///
/// The RCU read side lock is actually held while instances of this guard exist.
pub struct Guard(NotThreadSafe);

impl Guard {
    /// Acquires the RCU read side lock and returns a guard.
    #[inline]
    pub fn new() -> Self {
        // SAFETY: An FFI call with no additional requirements.
        unsafe { bindings::rcu_read_lock() };
        // INVARIANT: The RCU read side lock was just acquired above.
        Self(NotThreadSafe)
    }

    /// Explicitly releases the RCU read side lock.
    #[inline]
    pub fn unlock(self) {}
}

impl Default for Guard {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Guard {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: By the type invariants, the RCU read side is locked, so it is ok to unlock it.
        unsafe { bindings::rcu_read_unlock() };
    }
}

/// Acquires the RCU read side lock.
#[inline]
pub fn read_lock() -> Guard {
    Guard::new()
}

/// RCU head for call backs.
///
/// # Examples
///
/// Use `#[derive(HasField)]` macro to specify a struct has a RCU head.
///
/// ```
/// use kernel::sync::rcu::RcuHead;
///
/// #[derive(HasField)]
/// struct Foo {
///     a: i32,
///     #[field]
///     rcu_head: RcuHead,
///     b: i32,
/// }
///
/// const _: () = {
///     const fn assert_has_field<T: HasField<T, RcuHead>>() { }
///     assert_has_field::<Foo>();
/// };
/// ```
#[repr(transparent)]
pub struct RcuHead(Opaque<bindings::callback_head>);

impl<T> Field<T> for RcuHead {}

// SAFETY: `callback_head` doesn't hold anything local to the current execution context, so it's
// safe to transfer to another execution context.
unsafe impl Send for RcuHead {}
// SAFETY: `callback_head` should only be used when it's in the destructor, and accesses to it are
// already unsafe, hence make it `Sync`.
unsafe impl Sync for RcuHead {}

/// A wrapper that adds an `RcuHead` on `T`.
#[derive(HasField)]
pub struct WithRcuHead<T> {
    #[field]
    head: RcuHead,
    data: T,
}

impl<T> WithRcuHead<T> {
    /// Creates a new wrapper on `T` with `RcuHead`.
    pub fn new(data: T) -> Self {
        Self {
            head: RcuHead(Opaque::zeroed()),
            data,
        }
    }
}

impl<T> Deref for WithRcuHead<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// A allocation that supports dropping after one grace period.
///
/// [`Send`] is needed because RCU callback can execute on a different execution context.
pub trait DropRcu: ForeignOwnable + Send {
    /// Drops an object via a pointer after a grace period.
    ///
    /// # Safety
    ///
    /// `ptr` must be a return of a `into_foreign()`, and no corresponding `from_foreign()` has
    /// been called.
    unsafe fn drop_rcu_ptr(ptr: *mut c_void);

    /// Drops an object after a grace period.
    fn drop_rcu(self) {
        // SAFETY: `self.into_foreign()` is used immediately.
        unsafe { Self::drop_rcu_ptr(self.into_foreign()) };
    }

    /// TODO
    type RcuBorrowed<'a>;

    /// TODO
    ///
    /// # Safety
    ///
    /// TODO
    unsafe fn rcu_borrow<'a>(ptr: *mut c_void) -> Self::RcuBorrowed<'a>;
}

/// Drop and free the object in a rcu callback.
///
/// # Safety
///
/// `head` references the [`RcuHead`] field of a `T` that has no references to it. Ownership of the
/// [`Box<T, A>`] must be passed.
unsafe extern "C" fn box_drop_rcu_fn<T: HasField<T, RcuHead>, A: Allocator>(
    head: *mut bindings::callback_head,
) {
    // CAST: `RcuHead` is transparent to `callback_head`.
    let head = head.cast::<RcuHead>();

    // SAFETY: Per the function safety requirement, `head` points to the `RcuHead` field in a
    // `Box<T, A>`.
    let box_ptr = unsafe { T::field_container_of(head) };

    // SAFETY: Caller ensures exclusive access and passed ownership.
    drop(unsafe { Box::<T, A>::from_raw(box_ptr) });
}

/// [`Box<T, A>`] supports RCU async drop if `T` has [`RcuHead`] in it.
///
/// # Examples
/// ```
/// use kernel::sync::rcu::{DropRcu, RcuHead, WithRcuHead};
///
/// let kbox = KBox::new(WithRcuHead::<i32>::new(42), GFP_KERNEL)?;
///
/// kbox.drop_rcu(); // <- use kfree_rcu().
///
/// # Ok::<(), Error>(())
/// ```
impl<T: HasField<T, RcuHead> + Send + 'static, A: Allocator> DropRcu for Box<T, A> {
    #[inline]
    unsafe fn drop_rcu_ptr(ptr: *mut c_void) {
        // CAST: `Box::into_foreign()` returns the pointer value that points to `T`.
        let ptr = ptr.cast();

        // SAFETY: `ptr` is a valid pointer to `T` per function safety requirement.
        let head = unsafe { T::raw_get_field(ptr) };

        // CAST: `RcuHead` is transparent to `callback_head`.
        let head = head.cast();

        if core::mem::needs_drop::<T>() {
            // SAFETY: `head` is the `rcu_head` field of `T`. All users will be gone in an RCU
            // grace period. This is the destructor, so we may pass ownership of the allocation.
            unsafe {
                bindings::call_rcu(head, Some(box_drop_rcu_fn::<T, A>));
            }
        } else {
            // TODO: Maybe make it an Allocator API?
            // SAFETY: All users will be gone in an rcu grace period.
            unsafe {
                bindings::kvfree_call_rcu(head, ptr.cast());
            }
        }
    }

    type RcuBorrowed<'a> = &'a T;

    unsafe fn rcu_borrow<'a>(ptr: *mut c_void) -> Self::RcuBorrowed<'a> {
        // SAFETY: TODO
        unsafe { <Self as ForeignOwnable>::borrow(ptr) }
    }
}

/// A wrapper that uses the `drop_rcu()` instead of normal `drop()` of `T`.
///
/// # Examples
///
/// ```
/// use kernel::sync::rcu::{DropRcu, RcuDrop, RcuHead, read_lock, WithRcuHead};
/// use core::ops::Deref;
///
/// let rcu_drop = RcuDrop::new(KBox::new(WithRcuHead::<i32>::new(42), GFP_KERNEL)?);
///
/// let g = read_lock();
/// let w = rcu_drop.with_rcu(&g).deref();
///
/// drop(rcu_drop); // <- kfree_rcu()
///
/// assert_eq!(*w, 42);
///
/// # Ok::<(), Error>(())
/// ```
///
/// # Invariants
///
/// `self.0` is a return from `T::into_foreign()`.
pub struct RcuDrop<T: DropRcu>(*mut c_void, PhantomData<T>);

// SAFETY: `DropRcu` indicates `Send` hence it's safe to transfer `RcuDrop` to a different
// execution context.
unsafe impl<T: DropRcu> Send for RcuDrop<T> {}

impl<T: DropRcu> Drop for RcuDrop<T> {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a return of `T::into_foreign()`.
        unsafe {
            T::drop_rcu_ptr(self.0);
        }
    }
}

impl<T: DropRcu> RcuDrop<T> {
    /// Creates a new [`RcuDrop`] wrapper.
    pub fn new(t: T) -> Self {
        Self(t.into_foreign(), PhantomData)
    }

    /// Accesses the value while RCU read lock is held.
    pub fn with_rcu<'rcu>(&self, _guard: &'rcu Guard) -> <T as DropRcu>::RcuBorrowed<'rcu> {
        // SAFETY: The function signature guarantees that the object outlives the returned
        // reference since the `RcuDrop::drop()` waits for a grace period.
        unsafe { T::rcu_borrow(self.0) }
    }
}

// SAFETY: Per type invariants `self.0` is the return of `T::into_foreign()`, and it's guaranteed
// to be aligned to `T::FOREIGN_ALIGN` and not null.
unsafe impl<T: DropRcu> ForeignOwnable for RcuDrop<T> {
    const FOREIGN_ALIGN: usize = <T as ForeignOwnable>::FOREIGN_ALIGN;

    type Borrowed<'a> = <T as ForeignOwnable>::Borrowed<'a>;
    type BorrowedMut<'a> = <T as ForeignOwnable>::BorrowedMut<'a>;

    fn into_foreign(self) -> *mut c_void {
        ManuallyDrop::new(self).0
    }

    unsafe fn from_foreign(ptr: *mut c_void) -> Self {
        // INVARIANTS: `ptr` is a return of `T::into_foreign()`.
        Self(ptr, PhantomData)
    }

    unsafe fn borrow<'a>(ptr: *mut c_void) -> Self::Borrowed<'a> {
        // SAFETY: Per function safety requirement, `ptr` is `self.0` and per type invariants, it's
        // safe to call `T::borrow()`.
        unsafe { T::borrow(ptr) }
    }

    unsafe fn borrow_mut<'a>(ptr: *mut c_void) -> Self::BorrowedMut<'a> {
        // SAFETY: Per function safety requirement, `ptr` is `self.0` and per type invariants, it's
        // safe to call `T::borrow_mut()`.
        unsafe { T::borrow_mut(ptr) }
    }
}

/// `RcuDrop<T>` impl `Clone` if `T: Clone`.
///
/// # Examples
///
/// ```
/// use kernel::sync::{Arc, rcu::{DropRcu, RcuDrop, RcuHead, read_lock, WithRcuHead}};
/// use core::ops::Deref;
///
/// let rcu_drop = RcuDrop::new(Arc::new(WithRcuHead::<i32>::new(42), GFP_KERNEL)?);
/// let cloned = rcu_drop.clone();
///
/// let g = read_lock();
/// let w = cloned.with_rcu(&g).deref();
///
/// drop(cloned);
/// drop(rcu_drop); // <- kfree_rcu()
///
/// assert_eq!(*w, 42);
///
/// # Ok::<(), Error>(())
/// ```
impl<T: DropRcu + Clone> Clone for RcuDrop<T> {
    fn clone(&self) -> Self {
        let ptr = self.0;

        // SAFETY: Normally it's unsafe but here we keep it in a `ManuallyDrop` as if the
        // `from_foreign()` has not been called.
        let temp = ManuallyDrop::new(unsafe { <T as ForeignOwnable>::from_foreign(ptr) });

        let new = temp.deref().clone();

        // INVARIANTS: Trivial.
        Self(new.into_foreign(), PhantomData)
    }
}
