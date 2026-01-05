// SPDX-License-Identifier: GPL-2.0

//! Field types to describe a field inside a struct.

/// A field.
///
/// The generic type `T` is usually the type that contains the field. For some field types, it
/// needs to be generic over the type containing it, because it needs to be initialized with
/// container-type-specific callbacks. For other types, simply implement [`Field<T>`] for all `T`
/// to indicate there is no restriction.
pub trait Field<T>: Sized {}

/// A struct `T` that has a field `F`.
///
/// # Safety
///
/// The methods [`raw_get_field()`] and [`field_container_of()`] must return valid pointers and
/// must be true inverses of each other; that is, they must satisfy the following invariants: -
/// `field_container_of(raw_get_field(ptr)) == ptr` for any `ptr: *mut Self`. -
/// `raw_get_field(field_container_of(ptr)) == ptr` for any `ptr: *mut Field<T>`.
///
/// Use [`macros::HasField`] to generate the impls automatically.
///
/// # Examples
///
/// ```
/// # use core::marker::PhantomData;
/// use kernel::{
///     macros::HasField,
///     field::{
///         Field,
///         HasField, //
///     }, //
/// };
///
/// struct Work<T, const ID: u64> {
///     _x: isize,
///     _inner: PhantomData<T>,
/// }
///
/// // Declare that `Work` is a `Field`.
/// impl<T, const ID: u64> Field<T> for Work<T, ID> {}
///
/// #[derive(HasField)]
/// struct B {
///     #[field]
///     w: Work<B, 2>,
///     a: i32,
/// }
///
/// const _: () = {
///     const fn assert_has_field<T: HasField<T, Work<T, 2>>>() { }
///     assert_has_field::<B>();
/// };
/// ```
///
/// [`raw_get_field()`]: HasField::raw_get_field
/// [`field_container_of()`]: HasField::field_container_of
pub unsafe trait HasField<T, F: Field<T>> {
    /// Returns a pointer to the [`Field<T>`] field.
    ///
    /// # Safety
    ///
    /// The provided pointer must point at a valid struct of type `Self`.
    unsafe fn raw_get_field(ptr: *mut Self) -> *mut F;

    /// Returns a pointer to the struct containing [`Field<T>`] field.
    ///
    /// # Safety
    ///
    /// The pointer must point at a [`Field<T>`] field in a struct of type `Self`.
    unsafe fn field_container_of(ptr: *mut F) -> *mut Self;
}
