#include <linux/list.h>

typedef void* hazptr_t;

#define HAZPTR_UNUSED (1ul)

struct hazptr_context;

struct hazptr_head {
	struct rcu_head head;
};

void init_hazptr_context(struct hazptr_context *hzcp);
hazptr_t *hazptr_alloc(struct hazptr_context *hzcp);
void hazptr_free(struct hazptr_context *hzcp, hazptr_t *hzp);

#define hazptr_tryprotect(hzp, p, field)	__hazptr_tryprotect(hzp, p, offset_of(__typeof__(*p), field)

static inline bool __hazptr_tryprotect(hazptr_t *hzp, void **p, unsigned long head_offset)
{
	void *ptr;
	struct hazptr_head *head;

	ptr = READ_ONCE(*p);

	if (ptr == NULL)
		return false;
	
	head = (struct hazptr_head *)ptr + head_offset;
	
	WRITE_ONCE(*hzp, head);
	smp_mb();

	ptr = READ_ONCE(*p); // read again
	
	if (ptr + head_offset != *hzp) { // pointer changed
		WRITE_ONCE(*hzp, NULL);  // reset hazard pointer
		return false;
	} else
		return true;
}

static inline void hazptr_clear(hazptr_t *hzp)
{
	WRITE_ONCE(*hzp, NULL);
}

void call_hazptr(struct hazptr_head *head, rcu_callback_t func);
