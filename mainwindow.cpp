#include "mainwindow.h"
#include "./ui_mainwindow.h"
#pragma execution_character_set("utf-8")

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

// void MainWindow::on_EagerSingleton_clicked()
// {
//     qDebug() << "=== 饿汉单例模式示例 ===";
//     EagerSingleton *s1 = EagerSingleton::getInstance();
//     EagerSingleton *s2 = EagerSingleton::getInstance();

//     qDebug() << "s1 address = " << s1;
//     qDebug() << "s2 address = " << s2;

//     EagerSingleton::delInstance();
// }

void MainWindow::on_LazySingleton_clicked()
{
    qDebug() << "=== 懒汉单例模式示例 ===";
    LazySingleton *s1 = LazySingleton::getInstance();
    LazySingleton *s2 = LazySingleton::getInstance();

    qDebug() << "s1 address = " << s1;
    qDebug() << "s2 address = " << s2;

    LazySingleton::delInstance();

    LazySingleton2 *s3 = LazySingleton2::getInstance();
    LazySingleton2 *s4 = LazySingleton2::getInstance();

    qDebug() << "s3 address = " << s3;
    qDebug() << "s4 address = " << s4;
}

// void MainWindow::on_FactoryMethod_clicked()
// {
//     qDebug() << "=== 工厂方法模式示例 ===";
//     //定义工厂类对象和产品类对象
//     AbstractBallProduct *product = nullptr;
//     AbstractFactory *factory = nullptr;

//     factory = new BasketballFactory();
//     product = factory->createProduct();
//     product->productName();
//     delete factory;
//     delete product;


//     factory = new FootballFactory();
//     product = factory->createProduct();
//     product->productIntroduction();
//     delete factory;
//     delete product;

//     factory = new VolleyballFactory();
//     product = factory->createProduct();
//     product->productIntroduction();
//     delete factory;
//     delete product;
// }

// void MainWindow::on_SimpleFactory_clicked()
// {
//     qDebug() << "=== 简单工厂模式示例 ===";
//     //定义工厂类对象
//     AbstractSimpleBallProduct *product = nullptr;
//     product = SimpleFactory::getProduct("Basketball");
//     product->productName();
//     delete product;

//     product = SimpleFactory::getProduct("Football");
//     product->productIntroduction();
//     delete product;

//     product = SimpleFactory::getProduct("Volleyball");
//     product->productIntroduction();
//     delete product;
// }

// void MainWindow::on_AbstractFactory_clicked()
// {
//     qDebug() << "=== 抽象工厂模式示例 ===";
//     //定义工厂类对象和产品类对象
//     AbstractPhoneProduct *phone = nullptr;
//     AbstractTVProduct *tv = nullptr;
//     TheAbstractFactory *factory = nullptr;

//     factory = new HWFactory();
//     phone = factory->createPhone();
//     phone->productName();
//     tv = factory->createTV();
//     tv->productIntroduction();
//     delete factory;
//     delete phone;
//     delete tv;

//     factory = new MIFactory();
//     phone = factory->createPhone();
//     phone->productName();
//     tv = factory->createTV();
//     tv->productIntroduction();
//     delete factory;
//     delete phone;
//     delete tv;
// }

// void MainWindow::on_ProtoType_clicked()
// {
//     qDebug() << "=== 原型模式示例 ===";
// //    /// 用于复用的初始邮件创建
// //    auto *originalMail = new ConcretePrototypeMail("original_title","original_sender","original_rec","original_body","original_attachment");
// //    qDebug() << "originalMail address: "<< originalMail;
// //    originalMail->showMail();
// //    /// 浅拷贝
// //    qDebug() << "====浅拷贝====";
// //    auto *copyMail_A = originalMail;
// //    copyMail_A->changeTitle("copymail_title");
// //    copyMail_A->changeSender("copymail_sender");
// //    copyMail_A->changeRecipients("copymail_rec");
// //    copyMail_A->changeBody("copymail_body");
// //    copyMail_A->changeAtt("copymail_attachment");
// //    qDebug() << "====copyMail_A====";
// //    qDebug() << "copyMail_A address: "<< copyMail_A;
// //    copyMail_A->showMail();
// //    qDebug() << "====originalMail====";
// //    originalMail->showMail();
// //    delete originalMail;

//     /// 用于复用的初始邮件创建
//     auto *originalMail = new ConcretePrototypeMail("original_title","original_sender","original_rec","original_body");
//     originalMail->changeAtt("original_attachment");
//     qDebug() << "originalMail address: "<< originalMail;
//     originalMail->showMail();
//     /// 深拷贝
//     qDebug() << "====深拷贝====";
//     auto *copyMail_A = originalMail->clone();
//     copyMail_A->changeTitle("copymail_title");
//     copyMail_A->changeSender("copymail_sender");
//     copyMail_A->changeRecipients("copymail_rec");
//     copyMail_A->changeBody("copymail_body");
//     copyMail_A->changeAtt("copymail_attachment");
//     qDebug() << "====copyMail_A====";
//     qDebug() << "copyMail_A address: "<< copyMail_A;
//     copyMail_A->showMail();
//     qDebug() << "====originalMail====";
//     originalMail->showMail();
//     delete originalMail;
//     delete copyMail_A;
// }

// void MainWindow::on_BuilderPattern_clicked()
// {
//     qDebug() << "=== 建造者模式示例 ===";
//     //指挥者
//     Director director;
//     //抽象建造者
//     AbstractBuilder *builder;
//     //产品：套餐
//     ProductMeal *meal;

//     //指定具体建造者A
//     builder = new ConcreteBuilderMeal_A();
//     director.setBuilder(builder);
//     meal = director.construct();
//     meal->showMeal();
//     delete builder;
//     qDebug() << "======================";
//     //指定具体建造者C
//     builder = new ConcreteBuilderMeal_C();
//     director.setBuilder(builder);
//     meal = director.construct();
//     meal->showMeal();
//     delete builder;
//     qDebug() << "======================";
//     //指定具体建造者B
//     builder = new ConcreteBuilderMeal_B();
//     director.setBuilder(builder);
//     meal = director.construct();
//     meal->showMeal();
//     delete builder;

// }

// void MainWindow::on_ProxyPattern_clicked()
// {
//     qDebug() << "=== 代理模式示例 ===";
//     Subject *subject;
//     subject = new ProxySubject();
//     subject->business();
//     delete subject;
// }

// void MainWindow::on_FacadePattern_clicked()
// {
//     qDebug() << "=== 外观模式示例 ===";
//     auto *powerOnButton = new PowerOnButton();
//     powerOnButton->pressButton();
//     delete powerOnButton;
// }

// void MainWindow::on_ObserverPattern_clicked()
// {
//     qDebug() << "=== 外观模式示例 ===";
//     Blog *blog = new BlogCSDN("墨1024");
//     Observer *observer1 = new ObserverBlog("张三",blog);
//     Observer *observer2 = new ObserverBlog("李四",blog);
//     blog->attach(observer1);
//     blog->attach(observer2);
//     qDebug() << "==================";
//     blog->setStatus("发表了一篇Blog");
//     blog->notify();
//     qDebug() << "==================";
//     blog->remove(observer1);
//     blog->setStatus("更新了一条blink");
//     blog->notify();
//     qDebug() << "==================";
//     delete blog;
//     delete observer1;
//     delete observer2;
// }

// void MainWindow::on_FlyweightPattern_clicked()
// {
//     qDebug() << "=== 享元模式示例 ===";
//     ChessPiece *black1,*black2,*black3,*white1,*white2;
//     ChessPieceFactory *factory;

//     //获取享元工厂对象
//     factory = ChessPieceFactory::getInstance();

//     //通过享元工厂获取三颗黑子
//     black1 = factory->getChessPiece("b");
//     black2 = factory->getChessPiece("b");
//     black3 = factory->getChessPiece("b");
//     qDebug() << "两颗黑子是否相同：" << (black1==black2);

//     //通过享元工厂获取两颗白子
//     white1 = factory->getChessPiece("w");
//     white2 = factory->getChessPiece("w");
//     qDebug() << "两颗白子是否相同：" << (white1==white2);

//     std::vector<Coordinates *> coordinates;
//     //std::function<Coordinates *(Coordinates *)> func = [&coordinates](Coordinates *coord ) {
//     auto func = [&coordinates](Coordinates *coord) {
//         coordinates.push_back(coord);
//         return coord;
//     };
//     //显示棋子
//     black1->display(func(new Coordinates(1,3)));
//     black2->display(func(new Coordinates(2,6)));;
//     black3->display(func(new Coordinates(4,7)));;
//     white1->display(func(new Coordinates(5,8)));;
//     white2->display(func(new Coordinates(4,1)));;

//     for (auto & coordinate : coordinates) {
//         delete coordinate;
//     }
// }

// void MainWindow::on_DecoratorPattern_clicked()
// {
//     qDebug() << "=== 装饰器模式示例 ===";
//     //有个 张三 的顾客,他要了一个基础3元的手抓饼
//     auto *customerA = new Customer("张三");
//     customerA->buy(new ConcreteHandPancake(Base3));
//     delete customerA;
//     qDebug() << "================";
//     //有个 李四 的顾客,他要了一个基础5元的手抓饼,加鸡蛋
//     auto *customerB = new Customer("李四");
//     customerB->buy(new addEgg(new ConcreteHandPancake(Base5)));
//     delete customerB;
//     qDebug() << "================";
//     //有个 王五 的顾客,他要了一个基础5元的手抓饼,加鸡蛋,加火腿，加青菜。不愧是 钻石王老五
//     auto *customerC = new Customer("王五");
//     customerC->buy(new addVegetable(new addSausage(new addEgg(new ConcreteHandPancake(Base5)))));
//     delete customerC;
// }

