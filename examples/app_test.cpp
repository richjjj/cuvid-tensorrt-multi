/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-02-02 10:12:18
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-13 19:45:48
 *************************************************************************************/
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

using namespace std;

class ProdCons {
public:
    ProdCons() {}

    // 生产者线程函数
    void producer() {
        int i = 0;
        while (true) {
            unique_lock<mutex> lock(mtx);
            while (queue_.size() == max_queue_size) {
                not_full.wait(lock);
            }
            queue_.push(i++);
            cout << "Producing: " << i - 1 << endl;
            not_empty.notify_one();
            lock.unlock();
        }
    }

    // 消费者线程函数
    void consumer() {
        while (true) {
            unique_lock<mutex> lock(mtx);
            while (queue_.empty()) {
                not_empty.wait(lock);
            }
            int item = queue_.front();
            queue_.pop();
            cout << "Consuming: " << item << endl;
            not_full.notify_one();
            lock.unlock();
        }
    }

private:
    mutex mtx;
    condition_variable not_empty;
    condition_variable not_full;
    queue<int> queue_;
    const int max_queue_size = 10;
};

int app_test() {
    ProdCons pc;
    thread producer_thread(&ProdCons::producer, &pc);
    thread consumer_thread(&ProdCons::consumer, &pc);
    producer_thread.join();
    consumer_thread.join();
    return 0;
}

// int app_test() {
//     return 0;
// }
