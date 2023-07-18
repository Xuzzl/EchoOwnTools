#include "lazysingleton.h"

LazySingleton* LazySingleton::instance = nullptr;
QMutex LazySingleton::m_mutex;
