#include "types.h"
#include "defs.h"
#include "param.h"
#include "memlayout.h"
#include "mmu.h"
#include "spinlock.h"
#include "slab.h"

struct {
	struct spinlock lock;
	struct slab slab[NSLAB];
} stable;

void slabinit(){
    acquire(&stable.lock);  // 슬랩 초기화를 위한 락 획득

    // 슬랩 크기 목록 정의: 16B ~ 2048B 범위에서 2배씩 증가
    int sizes[] = {16, 32, 64, 128, 256, 512, 1024, 2048};

    for (int i = 0; i < NSLAB; i++) {
        stable.slab[i].size = sizes[i]; // 각 슬랩이 관리할 객체의 크기 설정

        // 비트맵 페이지 1개 할당: 각 객체의 사용 여부(0 또는 1)를 추적
        stable.slab[i].bitmap = kalloc();

        // 객체 저장용 슬랩 페이지 1개 할당 (초기에는 페이지 1개만 사용)
        stable.slab[i].memory_pages[0] = kalloc();

        // 현재 할당된 슬랩 페이지 수는 1개
        stable.slab[i].num_pages = 1;

        // 각 페이지에서 저장 가능한 객체 수 계산 (ex: size=64 → 4096/64 = 64개)
        int max_obj = PGSIZE / sizes[i];

        // 초기에는 모든 객체가 free 상태
        stable.slab[i].num_free_objects = max_obj;
        stable.slab[i].num_used_objects = 0;

        // 비트맵 전체를 0으로 초기화 (모든 객체가 비어 있음)
        memset(stable.slab[i].bitmap, 0, 4096);
    }

    release(&stable.lock); // 슬랩 초기화 완료 후 락 해제
}


char *kmalloc(int size) {
    acquire(&stable.lock); // 슬랩 접근을 위한 락 획득

    // 요청된 size보다 크거나 같은 슬랩 중 가장 작은 슬랩을 선택
    for (int i = 0; i < NSLAB; i++) {
        struct slab *s = &stable.slab[i];
        if (s->size >= size) {
            int max_objs_per_page = PGSIZE / s->size; // 한 페이지당 객체 수

            // 기존 슬랩 페이지들에서 비어있는 슬롯 탐색
            for (int p = 0; p < s->num_pages; p++) {
                for (int j = 0; j < max_objs_per_page; j++) {
                    int bit_index = p * max_objs_per_page + j;
                    int byte = bit_index / 8;
                    int bit = bit_index % 8;

                    // 해당 슬롯이 비어있다면 사용 처리 후 주소 반환
                    if (!(s->bitmap[byte] & (1 << bit))) {
                        s->bitmap[byte] |= (1 << bit); // 비트마크 설정
                        s->num_used_objects++;
                        s->num_free_objects--;

                        // 해당 슬롯의 실제 메모리 주소 계산
                        char *addr = s->memory_pages[p] + j * s->size;
                        release(&stable.lock);
                        return addr;
                    }
                }
            }

            // 기존 페이지에 여유 공간이 없을 경우: 새로운 페이지 추가 시도
            if (s->num_pages < 100) {
                char *new_page = kalloc();
                if (!new_page)
                    break; // 할당 실패 시 종료

                // 새 페이지를 memory_pages 배열에 추가
                s->memory_pages[s->num_pages] = new_page;

                // 새 페이지의 첫 번째 슬롯을 할당
                int base_bit_index = s->num_pages * max_objs_per_page;
                s->bitmap[base_bit_index / 8] |= (1 << (base_bit_index % 8));

                s->num_pages++;
                s->num_used_objects++;
                s->num_free_objects--;

                // 첫 번째 객체 주소 반환
                char *addr = new_page;
                release(&stable.lock);
                return addr;
            } else {
                break; // 최대 페이지 수 초과로 더 이상 확장 불가
            }
        }
    }

    release(&stable.lock);
    return 0x00; // 슬랩이 없거나 메모리 부족 시 실패
}

void kmfree(char *addr, int size){
    acquire(&stable.lock); // 락 획득

    for (int i = 0; i < NSLAB; i++) {
        struct slab *s = &stable.slab[i];

        if (s->size >= size) {
            int max_objs_per_page = PGSIZE / s->size;

            for (int p = 0; p < s->num_pages; p++) {
                char *base = s->memory_pages[p];
                if (addr >= base && addr < base + PGSIZE) {
                    int offset = addr - base;
                    int obj_index = offset / s->size;

                    int bit_index = p * max_objs_per_page + obj_index;
                    int byte = bit_index / 8;
                    int bit = bit_index % 8;

                    if (s->bitmap[byte] & (1 << bit)) {
                        s->bitmap[byte] &= ~(1 << bit);
                        s->num_used_objects--;
                        s->num_free_objects++;
                    }

                    // 현재 페이지가 완전히 비었는지 확인
                    int all_free = 1;
                    for (int j = 0; j < max_objs_per_page; j++) {
                        int index = p * max_objs_per_page + j;
                        int b = index / 8;
                        int k = index % 8;
                        if (s->bitmap[b] & (1 << k)) {
                            all_free = 0;
                            break;
                        }
                    }

                    // 페이지 완전히 비었으면 kfree()로 반납
                    if (all_free && s->num_pages > 1) {
                        kfree(s->memory_pages[p]);

                        // 페이지와 비트맵 shift
                        for (int q = p + 1; q < s->num_pages; q++) {
                            s->memory_pages[q - 1] = s->memory_pages[q];
                            for (int j = 0; j < max_objs_per_page; j++) {
                                int from = q * max_objs_per_page + j;
                                int to = (q - 1) * max_objs_per_page + j;

                                int from_byte = from / 8;
                                int from_bit = from % 8;
                                int to_byte = to / 8;
                                int to_bit = to % 8;

                                int val = (s->bitmap[from_byte] >> from_bit) & 1;
                                if (val)
                                    s->bitmap[to_byte] |= (1 << to_bit);
                                else
                                    s->bitmap[to_byte] &= ~(1 << to_bit);
                            }
                        }

                        // 마지막 페이지 비트마크 정리
                        int last_base = (s->num_pages - 1) * max_objs_per_page;
                        for (int j = 0; j < max_objs_per_page; j++) {
                            int idx = last_base + j;
                            int b = idx / 8;
                            int k = idx % 8;
                            s->bitmap[b] &= ~(1 << k);
                        }

                        s->num_pages--;
                        s->num_free_objects -= max_objs_per_page;
                    }

                    release(&stable.lock);
                    return;
                }
            }
        }
    }

    release(&stable.lock);
}

/* Helper functions */
void slabdump(){
	cprintf("__slabdump__\n");

	struct slab *s;

	cprintf("size\tnum_pages\tused_objects\tfree_objects\n");

	for(s = stable.slab; s < &stable.slab[NSLAB]; s++){
		cprintf("%d\t%d\t\t%d\t\t%d\n", 
			s->size, s->num_pages, s->num_used_objects, s->num_free_objects);
	}
}

int numobj_slab(int slabid)
{
	return stable.slab[slabid].num_used_objects;
}

int numpage_slab(int slabid)
{
	return stable.slab[slabid].num_pages;
}
