#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "lazysingleton.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    //void on_EagerSingleton_clicked();

    void on_LazySingleton_clicked();

    // void on_FactoryMethod_clicked();

    // void on_SimpleFactory_clicked();

    // void on_AbstractFactory_clicked();

    // void on_ProtoType_clicked();

    // void on_BuilderPattern_clicked();

    // void on_ProxyPattern_clicked();

    // void on_FacadePattern_clicked();

    // void on_ObserverPattern_clicked();

    // void on_FlyweightPattern_clicked();

    // void on_DecoratorPattern_clicked();

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
