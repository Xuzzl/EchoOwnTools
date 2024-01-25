#pragma once

#include <QtWidgets/QMainWindow>
#include <QSpinBox>
#include <QTextEdit>
#include <QComboBox>
#include "ui_EchoOwnTools.h"

//#include "lazysingleton.h"

class EchoOwnTools : public QMainWindow
{
    Q_OBJECT
public:
	Q_INVOKABLE EchoOwnTools(QWidget *parent = Q_NULLPTR);
	~EchoOwnTools();

public slots:
	void onPushButtonClicked();
	void onSpinboxValueChanged(int value);
	void onComboxIndexChanged(int index);
	void onTextEditValueChanged();

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
    Ui::EchoOwnToolsClass ui;
};
