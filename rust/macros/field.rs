// SPDX-License-Identifier: GPL-2.0

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{
    spanned::Spanned, Data, DataStruct, DeriveInput, Error, Fields, Generics, Ident, Result, Type,
};

fn impl_has_field(base: &Ident, field: &Ident, ty: &Type, generics: &Generics) -> TokenStream {
    let (impl_generics, type_generics, where_clause) = generics.split_for_impl();

    quote!(
        // SAFETY: The implementation of `raw_get_field()` only compiles if the field has the
        // right type.
        unsafe impl #impl_generics
        HasField<#base #type_generics, #ty>
        for #base #type_generics
        #where_clause {
            #[inline(always)]
            unsafe fn raw_get_field(ptr: *mut Self) -> *mut #ty {
                // SAFETY: Per function safety requirement, the pointer is valid.
                unsafe { &raw mut (*ptr).#field }
            }

            #[inline(always)]
            unsafe fn field_container_of(ptr: *mut #ty) -> *mut Self {
                // SAFETY: Per function safety requirement, the pointer is valid, and it points
                // to the right field of the struct.
                unsafe { kernel::container_of!(ptr, Self, #field) }
            }
        }
    )
}
fn handle_struct(
    ident: &Ident,
    generics: &Generics,
    st: &DataStruct,
    span: Span,
) -> Result<TokenStream> {
    let mut impls = vec![];

    if let Fields::Named(fields) = &st.fields {
        for field in &fields.named {
            let found = field
                .attrs
                .iter()
                .find(|attr| attr.path().is_ident("field"));

            if found.is_some() {
                if let Some(name) = &field.ident {
                    impls.push(impl_has_field(ident, name, &field.ty, generics));
                }
            }
        }

        Ok(quote!(
            #(#impls)*
        ))
    } else {
        Err(Error::new(
            span,
            "`#[derive(HasField)]` only supports structs with named fields",
        ))
    }
}

pub(crate) fn has_field(input: DeriveInput) -> Result<TokenStream> {
    let span = input.span();
    let data = &input.data;
    let ident = &input.ident;
    let generics = &input.generics;

    if let Data::Struct(st) = data {
        let impls = handle_struct(ident, generics, st, span)?;

        Ok(quote!(
            #impls
        ))
    } else {
        Err(Error::new_spanned(
            input,
            "`#[derive(HasField)]` only supports structs",
        ))
    }
}
