// SPDX-License-Identifier: GPL-2.0

//! A kernel thread (kthread).
//!
//! This module allows Rust code to create/wakeup/stop a kernel thread.

use core::ffi::CStr;
use core::ops::FnOnce;

use crate::{
    alloc::{flags::GFP_KERNEL, KBox},
    bindings,
    cpu::CpuId,
    error::{code::EINTR, from_err_ptr, to_result, Result},
    types::ARef,
};

use super::Task;

/// A running kernel thread.
pub struct Thread {
    /// The reference to the corresponding [Task].
    task: ARef<Task>,
}

/// A kernel thread that has not run yet.
pub struct ThreadNew<T> {
    /// The reference to the corresponding [Task].
    task: ARef<Task>,

    /// Pointer to the thread argument.
    arg_ptr: *mut T,
}

/// Bridge function of type `F` from Rust ABI to C.
extern "C" fn bridge<F>(data: *mut core::ffi::c_void) -> i32
where
    F: FnOnce() -> Result,
{
    // SAFETY: `data` is the result of `KBox::into_raw()`, therefore it's safe to
    // `KBox::from_raw()` here.
    let f = unsafe { KBox::from_raw(data as *mut F) };

    // TODO: Make KBox<FnOnce()> impl FnOnce<Args>.
    let f = KBox::into_inner(f);

    match f() {
        Ok(()) => 0,
        Err(e) => e.to_errno(),
    }
}

impl<T> ThreadNew<T> {
    /// Binds the kernel thread to a particular cpu.
    pub fn bind(&self, cpu: CpuId) {
        // SAFETY: `self.task.as_ptr()` is a valid pointer to a task, and `cpu` is a valid cpu number.
        unsafe { bindings::kthread_bind(self.task.as_ptr(), cpu.as_u32()) };
    }

    /// Starts a new thread.
    pub fn start(self) -> Thread {
        // Do not need to run `ThreadNew::drop()` since we are going start the thread. And we just
        // need pass the refcount from `ThreadNew` to `Thread.
        let task = &core::mem::ManuallyDrop::new(self).task;

        // CAST: `Task` is transparent to `bindings::task_struct`.
        let task = task.as_ptr().cast::<Task>();

        // SAFETY: `task` is not null.
        let task = unsafe { core::ptr::NonNull::new_unchecked(task) };

        // SAFETY: `ThreadNew` will never be dropped, so the refcount in it can be safely passed to
        // the `Thread`.
        let task = unsafe { ARef::from_raw(task) };

        // SAFETY: `task.as_ptr()` is a valid pointer, and `task` is newly created thread, so it's
        // safe to start.
        unsafe { bindings::kthread_start(task.as_ptr()) };

        Thread { task }
    }
}

impl<T> Drop for ThreadNew<T> {
    fn drop(&mut self) {
        // SAFETY: `self.task.as_ptr()` is a valid pointer to a task.
        let result = to_result(unsafe { bindings::kthread_stop(self.task.as_ptr()) });

        // If the thread has not run, the `result` from `kthread_stop` must be `EINTR`.
        assert_eq!(Err(EINTR), result);

        // SAFETY: `self.arg_ptr` came from a previous `KBox::into_raw()`, and the thread hasn't
        // run or won't run either, so it's safe to call `KBox::from_raw()`.
        drop(unsafe { KBox::from_raw(self.arg_ptr) });
    }
}

impl Thread {
    /// Creates a new thread.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use kernel::prelude::*;
    /// use kernel::{sync::{new_condvar, new_mutex, Arc, CondVar, Mutex}, task::Thread};
    ///
    /// let boxed = KBox::new(42, GFP_KERNEL)?;
    /// let lock = Arc::pin_init(new_mutex!(false), GFP_KERNEL)?;
    /// let cv = Arc::pin_init(new_condvar!(), GFP_KERNEL)?;
    ///
    /// let t = Thread::new(c"rust-thread", {
    ///     let lock = lock.clone();
    ///     let cv = cv.clone();
    ///     move || {
    ///         *(lock.lock()) = true;
    ///         cv.notify_all();
    ///         Ok(())
    ///     }
    /// })?;
    ///
    /// let t = t.start();
    ///
    /// let mut g = lock.lock();
    /// while !*g {
    ///     cv.wait(&mut g);
    /// }
    ///
    /// assert!(t.stop().is_ok());
    ///
    /// # Ok::<(), Error>(())
    /// ```
    ///
    /// The case where the thread has not started.
    ///
    /// ```rust
    /// # use kernel::prelude::*;
    /// use kernel::{sync::{new_condvar, new_mutex, Arc, CondVar, Mutex}, task::Thread};
    ///
    /// let boxed = KBox::new(42, GFP_KERNEL)?;
    /// let lock = Arc::pin_init(new_mutex!(false), GFP_KERNEL)?;
    ///
    /// let t = Thread::new(c"rust-thread", {
    ///     let lock = lock.clone();
    ///     move || {
    ///         *(lock.lock()) = true;
    ///         Ok(())
    ///     }
    /// })?;
    ///
    /// drop(t);
    /// assert!(!*lock.lock());
    ///
    /// # Ok::<(), Error>(())
    /// ```
    ///
    /// # Context
    ///
    /// This function might sleep in `kthread_create_on_node` due to the memory allocation and
    /// waiting for the completion, therefore do not call this in atomic contexts (i.e.
    /// preemption-off contexts).
    pub fn new<F>(name: &CStr, f: F) -> Result<ThreadNew<F>>
    where
        F: FnOnce() -> Result,
        F: Send + 'static,
    {
        let boxed_fn = KBox::new(f, GFP_KERNEL)?;
        let arg = KBox::into_raw(boxed_fn);

        // SAFETY:
        // - `bridge::<F>` is a valid pointer to a thread function.
        // - `arg` is a valid pointer.
        // - `kthread_create_on_node()` will copy the content of `name`, so `name` doesn't not need
        //   to live longer than this function call.
        let task_ptr = from_err_ptr(unsafe {
            bindings::kthread_create_on_node(
                Some(bridge::<F>),
                arg.cast(),
                bindings::NUMA_NO_NODE,
                c"%s".as_ptr(),
                name.as_ptr(),
            )
        })
        .map_err(|e| {
            // SAFETY: `kthread_create*()` failure means no thread taking the `arg`, and therefore
            // it is safe to convert back to a `KBox`.
            drop(unsafe { KBox::from_raw(arg) });

            e
        })?;

        // CAST: `Task` is transparent to `bindings::task_struct`.
        // SAFETY: `task_ptr` is a valid pointer for a `task_struct` because we've checked with
        // `from_err_ptr` above.
        let task_ref = unsafe { &*(task_ptr.cast::<Task>()) };

        // Increases the refcount of the task, so that it won't go away if it `do_exit`.
        Ok(ThreadNew {
            task: task_ref.into(),
            arg_ptr: arg,
        })
    }

    /// Creates a new thread and run it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use kernel::prelude::*;
    /// use kernel::{sync::{new_condvar, new_mutex, Arc, CondVar, Mutex}, task::Thread};
    ///
    /// let boxed = KBox::new(42, GFP_KERNEL)?;
    /// let lock = Arc::pin_init(new_mutex!(false), GFP_KERNEL)?;
    /// let cv = Arc::pin_init(new_condvar!(), GFP_KERNEL)?;
    ///
    /// let t = Thread::spawn(c"rust-thread", {
    ///     let lock = lock.clone();
    ///     let cv = cv.clone();
    ///     move || {
    ///         *(lock.lock()) = true;
    ///         cv.notify_all();
    ///         Ok(())
    ///     }
    /// })?;
    ///
    /// let mut g = lock.lock();
    /// while !*g {
    ///     cv.wait(&mut g);
    /// }
    ///
    /// assert!(t.stop().is_ok());
    ///
    /// # Ok::<(), Error>(())
    /// ```
    ///
    /// # Context
    ///
    /// This function might sleep in `kthread_create_on_node` due to the memory allocation and
    /// waiting for the completion, therefore do not call this in atomic contexts (i.e.
    /// preemption-off contexts).
    pub fn spawn<F>(name: &CStr, f: F) -> Result<Self>
    where
        F: FnOnce() -> Result,
        F: Send + 'static,
    {
        Ok(Self::new(name, f)?.start())
    }

    /// Creates a new thread and run it on a particular cpu.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use kernel::prelude::*;
    /// use kernel::{sync::{new_condvar, new_mutex, Arc, CondVar, Mutex}, task::Thread};
    /// use kernel::cpu::CpuId;
    ///
    /// let boxed = KBox::new(42, GFP_KERNEL)?;
    /// let lock = Arc::pin_init(new_mutex!(false), GFP_KERNEL)?;
    /// let cv = Arc::pin_init(new_condvar!(), GFP_KERNEL)?;
    ///
    /// let cpu = CpuId::from_i32(2).ok_or(kernel::error::code::EINVAL)?;
    ///
    /// let t = Thread::spawn_on(c"rust-thread", cpu, {
    ///     let lock = lock.clone();
    ///     let cv = cv.clone();
    ///     move || {
    ///         assert_eq!(CpuId::current().as_u32(), 2);
    ///         *(lock.lock()) = true;
    ///         cv.notify_all();
    ///         Ok(())
    ///     }
    /// })?;
    ///
    /// let mut g = lock.lock();
    /// while !*g {
    ///     cv.wait(&mut g);
    /// }
    ///
    /// assert!(t.stop().is_ok());
    ///
    /// # Ok::<(), Error>(())
    /// ```
    ///
    /// # Context
    ///
    /// This function might sleep in `kthread_create_on_node` due to the memory allocation and
    /// waiting for the completion, therefore do not call this in atomic contexts (i.e.
    /// preemption-off contexts).
    pub fn spawn_on<F>(name: &CStr, cpu: CpuId, f: F) -> Result<Self>
    where
        F: FnOnce() -> Result,
        F: Send + 'static,
    {
        let t = Self::new(name, f)?;

        t.bind(cpu);

        Ok(t.start())
    }

    /// Stops the thread.
    ///
    /// Waits for the closure to return or the thread `do_exit()` itself.
    ///
    /// Consumes the [`Thread`]. In case of error, returns the exit code of the thread.
    ///
    /// # Context
    ///
    /// This function might sleep, don't call it in atomic contexts.
    pub fn stop(self) -> Result {
        // SAFETY: `self.task.as_ptr()` is a valid pointer to a task.
        to_result(unsafe { bindings::kthread_stop(self.task.as_ptr()) })
    }
}
