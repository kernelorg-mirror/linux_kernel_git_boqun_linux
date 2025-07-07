#include <asm/cmpxchg.h>
#include <linux/compiler.h>

__rust_helper void *rust_helper_atomic_ptr_read(void *const *v)
{
	return READ_ONCE(*v);
}

__rust_helper void *rust_helper_atomic_ptr_read_acquire(void *const *v)
{
	return smp_load_acquire(v);
}

__rust_helper void rust_helper_atomic_ptr_set(void **v, void *i)
{
	WRITE_ONCE(*v, i);
}

__rust_helper void rust_helper_atomic_ptr_set_release(void **v, void *i)
{
	smp_store_release(v, i);
}

#define GEN_XCHG_HELPER(sfx)						\
__rust_helper void *							\
rust_helper_atomic_ptr_xchg##sfx(void **v, void *new)			\
{									\
	return xchg##sfx(v, new);					\
}

GEN_XCHG_HELPER()
GEN_XCHG_HELPER(_relaxed)
GEN_XCHG_HELPER(_release)
GEN_XCHG_HELPER(_acquire)

#define GEN_TRY_CMPXCHG_HELPER(sfx)						\
__rust_helper bool								\
rust_helper_atomic_ptr_try_cmpxchg##sfx(void **v, void **oldp, void *new)	\
{										\
	return try_cmpxchg##sfx(v, oldp, new);					\
}

GEN_TRY_CMPXCHG_HELPER()
GEN_TRY_CMPXCHG_HELPER(_relaxed)
GEN_TRY_CMPXCHG_HELPER(_release)
GEN_TRY_CMPXCHG_HELPER(_acquire)
