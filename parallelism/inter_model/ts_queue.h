//
// From lab4_integral_tasks_queue-prytula_korch_senyk
//

#ifndef DEEPDENDRO_TS_QUEUE_H
#define DEEPDENDRO_TS_QUEUE_H

#include <mutex>
#include <deque>
#include <condition_variable>

template<typename T>
class TSQueue {
    std::mutex mtx;
    std::deque<T> dq;
    std::condition_variable cv;
public:
    TSQueue() = default;
    ~ TSQueue() = default;
    TSQueue(const TSQueue&) = delete;
    TSQueue& operator=(const TSQueue&) = delete;

    void push(const T& el);
    T pop();
};

template<typename T>
void TSQueue<T>::push(const T &el) {
    {
        std::lock_guard<std::mutex> lock(mtx);
        dq.push_back(el);
    }
    cv.notify_one();
}

template<typename T>
T TSQueue<T>::pop() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this](){return !dq.empty();});
    T el = dq.front();
    dq.pop_front();
    return el;
}


#endif //DEEPDENDRO_TS_QUEUE_H
