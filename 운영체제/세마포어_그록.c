#include "types.h"
#include "defs.h"
#include "param.h"
#include "x86.h"
#include "memlayout.h"
#include "mmu.h"
#include "proc.h"
#include "spinlock.h"
#include "semaphore.h"

struct semaphore usema[NLOCK];

void
sem_init(struct semaphore *s, int init_value)
{
  s->value = init_value;
  initlock(&s->lock, "sem");
  s->queue_size = 0;
  for (int i = 0; i < NPROC; i++) {
    s->queue[i] = 0;
  }
}

void
sem_wait(struct semaphore *s)
{
  acquire(&s->lock);
  while (s->value <= 0) {
    if (s->queue_size < NPROC) {
      s->queue[s->queue_size++] = myproc();
      sleep(s, &s->lock);
    } else {
      panic("sem_wait: queue full");
    }
  }
  s->value--;
  release(&s->lock);
}

void
sem_post(struct semaphore *s)
{
  acquire(&s->lock);
  s->value++;
  if (s->queue_size > 0) {
    struct proc *p = s->queue[0];
    for (int i = 1; i < s->queue_size; i++) {
      s->queue[i-1] = s->queue[i];
    }
    s->queue[--s->queue_size] = 0;
    wakeup(p);
  }
  release(&s->lock);
}

void
sem_destroy(struct semaphore *s)
{
  acquire(&s->lock);
  s->value = 0;
  s->queue_size = 0;
  for (int i = 0; i < NPROC; i++) {
    s->queue[i] = 0;
  }
  release(&s->lock);
}