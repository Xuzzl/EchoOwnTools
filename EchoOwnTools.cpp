#include "EchoOwnTools.h"

EchoOwnTools::EchoOwnTools(QWidget *parent) : QMainWindow(parent)
{
    ui.setupUi(this);

	for (auto _button : findChildren<QPushButton*>())
	{
		if (nullptr != _button)
		{
			connect(_button, &QPushButton::clicked, this, &EchoOwnTools::onPushButtonClicked);
		}
	}
	for (auto _spinbox : findChildren<QSpinBox*>())
	{
		if (nullptr != _spinbox)
		{
			connect(_spinbox, SIGNAL(valueChanged(int)), this, SLOT(onSpinboxValueChanged(int)));
		}
	}
	for (auto _text : findChildren<QTextEdit*>())
	{
		if (nullptr != _text)
		{
			connect(_text, SIGNAL(textChanged()), this, SLOT(onTextEditValueChanged()));
		}
	}
	for (auto _cmb : findChildren<QComboBox*>())
	{
		if (nullptr != _cmb)
		{
			connect(_cmb, SIGNAL(currentIndexChanged(int)), this, SLOT(onComboxIndexChanged(int)));
		}
	}

}

EchoOwnTools::~EchoOwnTools()
{


}

void EchoOwnTools::onPushButtonClicked()
{
	QPushButton* _cur_senser = static_cast<QPushButton*>(sender());
	if (nullptr == _cur_senser)
	{
		return;
	}

	if (ui.LazySingleton == _cur_senser)
	{
		on_LazySingleton_clicked();
	}


}

void EchoOwnTools::onSpinboxValueChanged(int value)
{

}

void EchoOwnTools::onComboxIndexChanged(int index)
{

}

void EchoOwnTools::onTextEditValueChanged()
{

}


void EchoOwnTools::on_LazySingleton_clicked()
{

}