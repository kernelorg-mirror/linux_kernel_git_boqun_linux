// SPDX-License-Identifier: GPL-2.0

//! RCU support.
//!
//! C header: [`include/linux/rcupdate.h`](srctree/include/linux/rcupdate.h)

use crate::{
    bindings,
    field::{Field, HasField},
    macros::HasField,
    types::{NotThreadSafe, Opaque},
};

use core::ops::Deref;

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
