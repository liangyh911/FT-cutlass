#ifndef GEMM_RING_QUEUE_H
#define GEMM_RING_QUEUE_H

// #define QUEUE_SIZE 128

namespace cutlass{
    struct RingQueue{
        int* buffer;
        int head = 0;
        int tail = 0;
        int QUEUE_SIZE;

        __device__ void initial(int *buf, int size){
            buffer = buf;
            QUEUE_SIZE = size;
        }

        __device__ bool isFull(){
            return ((tail+1) % QUEUE_SIZE) == head;
        }

        __device__ bool isEmpty(){
            return head == tail;
        }

        __device__ bool enqueue(int value){
            if(isFull()){
                return false;
            }
            buffer[tail] = value;
            tail = (tail + 1) % QUEUE_SIZE;
            return true;
        }

        __device__ bool dequeue(int *value){
            if(isEmpty()){
                return false;
            }
            *value = buffer[head];
            head = (head + 1) % QUEUE_SIZE;
            return true;
        }
    };
}

#endif