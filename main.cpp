#include "EchoOwnTools.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    EchoOwnTools w;
    w.show();
    return a.exec();
}
