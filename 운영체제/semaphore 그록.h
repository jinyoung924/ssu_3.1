#define NLOCK 10

struct semaphore {
  int value;           // 세마포어 카운터
  struct spinlock lock; // 동기화를 위한 스핀락
  struct proc *queue[NPROC]; // 대기 프로세스 큐
  int queue_size;      // 큐에 있는 프로세스 수
};

extern struct semaphore usema[NLOCK];

void sem_init(struct semaphore *s, int init_value);
void sem_wait(struct semaphore *s);
void sem_post(struct semaphore *s);
void sem_destroy(struct semaphore *s);
