//
// Author:   Xuzzl
// Time:     2022/12/08
// Describe: 懒汉单例模式 - 两种
//
#ifndef LAZYSINGLETON_H
#define LAZYSINGLETON_H

#include <QObject>
#include <QDebug>
#include <QMutex>

// 检查锁模式
class LazySingleton
{
public:
    static LazySingleton *getInstance(){
        //第一个检查，如果只是读操作，就不许用加锁
        if (instance == nullptr) {
            //std::lock_guard<std::mutex> lck(m_mutex);
            m_mutex.lock();
            //第二个检查，保证只有一个
            if (instance == nullptr) {
                instance = new LazySingleton();
            }
            m_mutex.unlock();
        }
        return instance;
    };
    static void delInstance() {
        if(instance != nullptr) {
            delete instance;
            instance = nullptr;
        }
    };

private:
    LazySingleton(){
        qDebug() << "LazySingleton Hello";
    };
    ~LazySingleton(){
        // 私有化 可以避免 直接 delete s1 ，必须 使用 delInstance
        qDebug() << "LazySingleton Bye";
    };
    LazySingleton(const LazySingleton &other); //拷贝构造函数
    LazySingleton &operator = (const LazySingleton &other); //赋值运算操作符

    // static对象，可以保证对象只生成一次
    static LazySingleton *instance;
    static QMutex m_mutex;
};


// static变量模式
class LazySingleton2 {
public:
    static LazySingleton2 *getInstance(){
        static LazySingleton2 instance;
        return &instance;
    };

private:
    LazySingleton2(){
        qDebug() << "LazySingleton2 Hello";
    };
    ~LazySingleton2(){
        // 私有化 可以避免 直接 delete s1 ，必须 使用 delInstance
        qDebug() << "LazySingleton2 Bye";
    };
};

#endif // LAZYSINGLETON_H
