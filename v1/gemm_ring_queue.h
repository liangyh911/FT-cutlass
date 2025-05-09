#ifndef GEMM_RING_QUEUE_H
#define GEMM_RING_QUEUE_H

// #define QUEUE_SIZE 128

namespace cutlass{
    // struct RingQueue{
    //     int* buffer;
    //     volatile int head = 0;
    //     volatile int tail = 0;
    //     int capacity;

    //     __device__ void initial(int *buf, int size){
    //         buffer = buf;
    //         capacity = size;
    //     }

    //     __device__ bool isFull(){
    //         return ((tail+1) % capacity) == head;
    //     }

    //     __device__ bool isEmpty(){
    //         return head == tail;
    //     }

    //     __device__ bool enqueue(int value){
    //         if(isFull()){
    //             return false;
    //         }
    //         buffer[tail] = value;
    //         tail = (tail + 1) % capacity;
    //         return true;
    //     }

    //     __device__ bool dequeue(int *value){
    //         if(isEmpty()){
    //             return false;
    //         }
    //         *value = buffer[head];
    //         head = (head + 1) % capacity;
    //         return true;
    //     }
    // };

    struct RingQueue_v2{
        int *buffer;
        volatile int *head;
        volatile int *tail;
        int capacity;

        __device__ void initial(int size){
            capacity = size;
        }

        __device__ bool isFull(int offset){
            int cur_head = *(head + offset);
            int cur_tail = *(tail + offset);
            return ((cur_tail + 1) % capacity) == cur_head;
        }

        __device__ bool isEmpty(int offset){
            int cur_head = *(head + offset);
            int cur_tail = *(tail + offset);
            return cur_head == cur_tail;
        }

        __device__ bool enqueue(int offset, int value){
            int cur_tail = *(tail + offset);
            if(isFull(offset)){
                return false;
            }
            buffer[cur_tail + capacity*offset] = value;
            *(tail + offset) = (*(tail + offset) + 1) % capacity;
            return true;
        }

        __device__ bool dequeue(int offset, int *value){
            int cur_head = *(head + offset);
            if(isEmpty(offset)){
                return false;
            }
            *value = buffer[cur_head + capacity * offset];
            *(head + offset) = (*(head + offset) + 1) % capacity;
            return true;
        }
    };

}

#endif