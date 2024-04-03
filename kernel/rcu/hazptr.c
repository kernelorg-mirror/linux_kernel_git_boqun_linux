// SPDX-License-Identifier: GPL-2.0+

#include <linux/spinlock.h>
#include <linux/cleanup.h>
#include <linux/hazptr.h>

struct hazptr_context {
	struct list_head list;
	hazptr_t start_with_one;
};

struct hazptr_context_list {
	struct list_head list;
	spinlock_t lock;
};

DEFINE_PER_CPU(struct hazptr_context_list, hzc_list);

void init_hazptr_context(struct hazptr_context *hzcp)
{
	struct hazptr_context_list *this_hzc_list = this_cpu_ptr(&hzc_list);
	
	guard(spinlock)(&this_hzc_list->lock);
	list_add(&hzcp->list, &this_hzc_list->list);
}

hazptr_t *hazptr_alloc(struct hazptr_context *hzcp)
{
	if (((unsigned long)hzcp->start_with_one) != HAZPTR_UNUSED) {
		WRITE_ONCE(hzcp->start_with_one, NULL);
		return &hzcp->start_with_one;
	}

	return NULL;
}

void hazptr_free(struct hazptr_context *hzcp, hazptr_t *hzp)
{
	WARN_ON(((unsigned long)*hzp) == HAZPTR_UNUSED);
	WARN_ON(&hzcp->start_with_one != hzp);

	WRITE_ONCE(*hzp, (void *)HAZPTR_UNUSED);
}

void call_hazptr(struct hazptr_head *head, rcu_callback_t func)
{
	head->head.func = func;
	// TODO
}
