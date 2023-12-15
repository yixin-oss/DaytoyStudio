---

title: Python函数及变量的定义和使用
tags: python
categories: python学习笔记
---

## 1.可选参数传递

函数参数可有可无，但必须有括号.

<!--more-->

```
def <函数名>(非可选参数，可选参数):
    <函数体>
    return <返回值>
```

```
#e.g.1 这里m是可选参数，默认设为1，可以更改
def fact(n,m=1):
    s=1
    for i in range(1,n+1):
        s*=i
    return s//m
```

```
>>>fact(10)
>>>362800
>>>fact(10,3)
>>>1209600
```

<!--more-->

## 2.可变参数传递

不确定参数总数量

```
def <函数名>(参数，可变参数):
    <函数体>
    return <返回值>
```

```
#e.g.2  *b代表可变参数，可以有多个值
def fact(n,*b):
    s=1
    for i in range(1,n+1):
        s*=i
    for item in b:
        s=s//item
    return s
```

```
>>>fact(9,4)
>>>90720
>>>fact(9,4,7)
>>>12960
```

## 3.参数传递两种方式：位置传递&名称传递

```
def fact(n,m=1):
    s=1
    for i in range(1,n+1):
        s*=i
    return s//m
```

```
>>>fact(10,5)#位置传递
>>>725760
>>>fact(m=5,n=10)
>>>725760
```

## 4.函数返回值

函数可以返回0个或多个结果.

```
def fact(n,m=1):
    s=1
    for i in range(1,n+1):
        s*=i
    return s//m,n,m 
```

```
>>>fact(10,5)
>>>(725760, 10, 5) #元组形式
>>>a,b,c=fact(10,5)
>>>print(a,b,c)
>>>725760 10 5
```

## 5.全局变量和局部变量

- 局部变量是函数内部的占位符，可与全局变量重名，但不同
- 函数运算结束后，局部变量被释放
- 可以使用global关键字在函数内部使用全局变量
- 局部变量为组合数据类型且在函数内为真实创建，等同于全局变量

```
n,s=10,100 #s是全局变量
def fact(n)：
    s=1  #s是局部变量，与全局变量s不同
    for i in range(1,n+1):
        s*=i
    return s #s是局部变量
#函数运行结束后，局部变量会被释放
print(fact(n),s) #此处s是全局变量，s=100
```

```
>>> 3628800 100
```

```
n,s=10,100
def fact(n):
    global s  #使用global保留字声明s是全局变量s
    for i in range(1,n+1):
        s*=i
    return s  #此处s指全局变量s
print(fact(n),s)  #此处全局变量s被函数修改
```

```
>>>362880000 362880000
```

局部变量为组合数据类型且未创建，等同于全局变量

```
ls=['F','f'] #通过使用[]真实创建一个全局变量列表ls
def func(a):
    ls.append(a) #此处ls是列表类型，未真实创建，则等同于全局变量
    return
func('C')  #全局变量ls被修改
print(ls)
```

```
>>>['F', 'f', 'C']
```

```
ls=['F','f']
def func(a):
    ls=[]  #此处ls是列表类型，真实创建，ls是局部变量
    ls.append(a)
    return
print(func('C')) #修改了局部变量ls
print(ls) #输出全局变量ls
```

```
>>>['F', 'f']
```

## 6. lambda函数

lambda函数是一种匿名函数，即没有名字的函数.使用lambda保留字定义，函数名是返回结果；lambda函数用于定义简单的，能够在一行内定义的函数；lambda函数主要用作一些特定函数或方法的参数.

```
<函数名> = lambda<参数>:<表达式>
```

```
f=lambda x,y : x+y
f(6,8)
```

```
>>>14
```

```
f=lambda : '没有参数的lambda函数！'
print(f())
```

```
>>>没有参数的lambda函数！
```

