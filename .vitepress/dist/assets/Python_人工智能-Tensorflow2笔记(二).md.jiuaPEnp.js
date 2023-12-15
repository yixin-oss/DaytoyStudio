import{_ as s,o as n,c as a,R as p}from"./chunks/framework.bQUviISV.js";const d=JSON.parse('{"title":"","description":"","frontmatter":{"title:人工智能":"Tensorflow2笔记(二)","tags":["深度学习","Tensorflow","人工智能"],"categories":"Tensorflow学习笔记"},"headers":[],"relativePath":"Python/人工智能-Tensorflow2笔记(二).md","filePath":"Python/人工智能-Tensorflow2笔记(二).md","lastUpdated":null}'),l={name:"Python/人工智能-Tensorflow2笔记(二).md"},e=p(`<h2 id="神经网络实现鸢尾花分类" tabindex="-1">神经网络实现鸢尾花分类 <a class="header-anchor" href="#神经网络实现鸢尾花分类" aria-label="Permalink to &quot;神经网络实现鸢尾花分类&quot;">​</a></h2><p>准备数据</p><ul><li>数据集读入</li><li>数据集乱序</li><li>生成训练集和测试集,训练集，测试集不能有交集</li><li>配成（输入特征，标签）对，每次读入一部分(batch)</li></ul><p>搭建网络</p><ul><li>定义神经网络中所有可训练参数</li></ul><p>参数优化</p><ul><li>嵌套循环迭代，with结构更新参数，显示当前loss</li></ul><p>测试效果</p><ul><li>计算当前参数向后传播准确率，显示当前acc</li><li>准确率acc/损失函数loss可视化</li></ul><div class="language- vp-adaptive-theme line-numbers-mode"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>import tensorflow as tf</span></span>
<span class="line"><span>import numpy as np</span></span>
<span class="line"><span>from matplotlib import pyplot as plt</span></span>
<span class="line"><span>#从sklearn包datasets读入数据集：</span></span>
<span class="line"><span>from sklearn import datasets</span></span>
<span class="line"><span>x_data = datasets.load_iris().data  #返回iris数据集所有输入特征</span></span>
<span class="line"><span>y_data = datasets.load_iris().target #返回iris数据集中所有标签</span></span>
<span class="line"><span></span></span>
<span class="line"><span>#数据集乱序</span></span>
<span class="line"><span>np.random.seed(116) #使用相同的随机数种子，使输入特征/标签一一对应，即配对不会乱</span></span>
<span class="line"><span>np.random.shuffle(x_data)</span></span>
<span class="line"><span>np.random.seed(116)</span></span>
<span class="line"><span>np.random.shuffle(y_data)</span></span>
<span class="line"><span>tf.random.set_seed(116)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>#数据集分出训练集，测试集,不能有交集</span></span>
<span class="line"><span>#打乱数据集中前120个作为训练集</span></span>
<span class="line"><span>x_train = x_data[:-30]</span></span>
<span class="line"><span>y_train = y_data[:-30]</span></span>
<span class="line"><span>x_test = x_data[-30:]</span></span>
<span class="line"><span>y_test = y_data[-30:]</span></span>
<span class="line"><span></span></span>
<span class="line"><span># 转换x数据类型</span></span>
<span class="line"><span>x_train = tf.cast(x_train, tf.float32)</span></span>
<span class="line"><span>x_test = tf.cast(x_test, tf.float32)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>#from_tensor_slices配成【输入特征、标签】对，每次喂入神经网络一部分数据(batch)</span></span>
<span class="line"><span>#每32对打包为一个batch</span></span>
<span class="line"><span>train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)</span></span>
<span class="line"><span>test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)</span></span>
<span class="line"><span>#定义神经网络所有可训练参数</span></span>
<span class="line"><span>#输入特征是4，输入层为4个输入节点，只有一层网络，输出节点数=分类数，3分类</span></span>
<span class="line"><span>#参数w1是4行3列张量</span></span>
<span class="line"><span>w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1))</span></span>
<span class="line"><span>b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1))</span></span>
<span class="line"><span></span></span>
<span class="line"><span>lr = 0.1 #学习率为0.1</span></span>
<span class="line"><span>train_loss_results = []  #将每轮loss记录下来，为后面画loss曲线提供数据</span></span>
<span class="line"><span>test_acc = [] #记录acc</span></span>
<span class="line"><span>Epoch = 500 #循环500次</span></span>
<span class="line"><span>loss_all = 0 #每轮分4个step，loss_all记录四个step生成的4个loss的和</span></span>
<span class="line"><span></span></span>
<span class="line"><span>#两层循环迭代更新参数</span></span>
<span class="line"><span>#第一层for循环是针对整个数据集循环，用epoch表示</span></span>
<span class="line"><span>#第二层for循环是针对batch的，用step表示</span></span>
<span class="line"><span>for epoch in range(Epoch):</span></span>
<span class="line"><span>    for step, (x_train, y_train) in enumerate(train_db):</span></span>
<span class="line"><span>        with tf.GradientTape() as tape: #with结构记录梯度信息</span></span>
<span class="line"><span>            y = tf.matmul(x_train, w1) + b1 #神经网络乘加运算</span></span>
<span class="line"><span>            y = tf.nn.softmax(y) #使输出y符合概率分布</span></span>
<span class="line"><span>            y_ = tf.one_hot(y_train, depth=3) #将标签转换为独热码格式，方便计算loss</span></span>
<span class="line"><span>            loss = tf.reduce_mean(tf.square(y_ -y)) #采用均方误差损失函数MSE</span></span>
<span class="line"><span>            loss_all += loss.numpy() #将每个step计算出的loss累加，为后续求loss平均值提供数据</span></span>
<span class="line"><span>        grads = tape.gradient(loss, [w1, b1])</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        # 实现梯度更新</span></span>
<span class="line"><span>        w1.assign_sub(lr * grads[0]) #参数w1自更新</span></span>
<span class="line"><span>        b1.assign_sub(lr * grads[1]) #参数b1自更新</span></span>
<span class="line"><span>    #每个epoch 打印loss信息</span></span>
<span class="line"><span>    print(&#39;Epoch {}, loss: {}&#39;.format(epoch, loss_all/4)) #120组数据，需要batch级别循环4次，除以4求得每次step迭代平均loss</span></span>
<span class="line"><span>    train_loss_results.append(loss_all / 4) #将4个step的loss求平均记录在此变量中</span></span>
<span class="line"><span>    loss_all = 0 #loss_all归零，为记录下一个epoch的loss做准备</span></span>
<span class="line"><span></span></span>
<span class="line"><span></span></span>
<span class="line"><span>    #测试部分</span></span>
<span class="line"><span>    #计算当前参数前向传播后准确率，显示当前acc</span></span>
<span class="line"><span>    #total_corrrect为预测对的样本个数， total_number为测试的总样本数，初始化为0</span></span>
<span class="line"><span>    total_correct, total_number = 0, 0</span></span>
<span class="line"><span>    for x_test, y_test in test_db:</span></span>
<span class="line"><span>        y = tf.matmul(x_test, w1) + b1 #y为预测结果</span></span>
<span class="line"><span>        y = tf.nn.softmax(y) #y符合概率分布</span></span>
<span class="line"><span>        pred = tf.argmax(y, axis=1) #返回y中最大值索引，即预测分类</span></span>
<span class="line"><span>        #将pred转换为y_test数据类型</span></span>
<span class="line"><span>        pred = tf.cast(pred, dtype=y_test.dtype)</span></span>
<span class="line"><span>        # 若分类正确，correct=1，否则为0，将bool型转换为int型</span></span>
<span class="line"><span>        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)</span></span>
<span class="line"><span>        # 将每个batch的correct数加起来</span></span>
<span class="line"><span>        correct = tf.reduce_sum(correct)</span></span>
<span class="line"><span>        total_correct += int(correct) #将所有batch中correct数加起来</span></span>
<span class="line"><span>        #total_number为测试总样本数，即x_test行数，shape[0]</span></span>
<span class="line"><span>        total_number = x_test.shape[0]</span></span>
<span class="line"><span>    #总准确率为 total_correct / total_number</span></span>
<span class="line"><span>    acc = total_correct / total_number</span></span>
<span class="line"><span>    test_acc.append(acc)</span></span>
<span class="line"><span>    print(&quot;test_acc:&quot;, acc)</span></span>
<span class="line"><span>    print(&#39;----------------------&#39;)</span></span>
<span class="line"><span></span></span>
<span class="line"><span></span></span>
<span class="line"><span>#绘制loss曲线</span></span>
<span class="line"><span>plt.title(&#39;Loss Function Curve&#39;)</span></span>
<span class="line"><span>plt.xlabel(&#39;Epoch&#39;)</span></span>
<span class="line"><span>plt.ylabel(&quot;Loss&quot;)</span></span>
<span class="line"><span>plt.plot(train_loss_results, label=&#39;$loss$&#39;)</span></span>
<span class="line"><span>plt.legend()</span></span>
<span class="line"><span>plt.show()</span></span>
<span class="line"><span></span></span>
<span class="line"><span>#绘制Accuary曲线</span></span>
<span class="line"><span>plt.title(&#39;Acc Curve&#39;)</span></span>
<span class="line"><span>plt.xlabel(&#39;Epoch&#39;)</span></span>
<span class="line"><span>plt.ylabel(&#39;Acc&#39;)</span></span>
<span class="line"><span>plt.plot(test_acc, label=&#39;$Accuary$&#39;)</span></span>
<span class="line"><span>plt.legend()</span></span>
<span class="line"><span>plt.show()</span></span></code></pre><div class="line-numbers-wrapper" aria-hidden="true"><span class="line-number">1</span><br><span class="line-number">2</span><br><span class="line-number">3</span><br><span class="line-number">4</span><br><span class="line-number">5</span><br><span class="line-number">6</span><br><span class="line-number">7</span><br><span class="line-number">8</span><br><span class="line-number">9</span><br><span class="line-number">10</span><br><span class="line-number">11</span><br><span class="line-number">12</span><br><span class="line-number">13</span><br><span class="line-number">14</span><br><span class="line-number">15</span><br><span class="line-number">16</span><br><span class="line-number">17</span><br><span class="line-number">18</span><br><span class="line-number">19</span><br><span class="line-number">20</span><br><span class="line-number">21</span><br><span class="line-number">22</span><br><span class="line-number">23</span><br><span class="line-number">24</span><br><span class="line-number">25</span><br><span class="line-number">26</span><br><span class="line-number">27</span><br><span class="line-number">28</span><br><span class="line-number">29</span><br><span class="line-number">30</span><br><span class="line-number">31</span><br><span class="line-number">32</span><br><span class="line-number">33</span><br><span class="line-number">34</span><br><span class="line-number">35</span><br><span class="line-number">36</span><br><span class="line-number">37</span><br><span class="line-number">38</span><br><span class="line-number">39</span><br><span class="line-number">40</span><br><span class="line-number">41</span><br><span class="line-number">42</span><br><span class="line-number">43</span><br><span class="line-number">44</span><br><span class="line-number">45</span><br><span class="line-number">46</span><br><span class="line-number">47</span><br><span class="line-number">48</span><br><span class="line-number">49</span><br><span class="line-number">50</span><br><span class="line-number">51</span><br><span class="line-number">52</span><br><span class="line-number">53</span><br><span class="line-number">54</span><br><span class="line-number">55</span><br><span class="line-number">56</span><br><span class="line-number">57</span><br><span class="line-number">58</span><br><span class="line-number">59</span><br><span class="line-number">60</span><br><span class="line-number">61</span><br><span class="line-number">62</span><br><span class="line-number">63</span><br><span class="line-number">64</span><br><span class="line-number">65</span><br><span class="line-number">66</span><br><span class="line-number">67</span><br><span class="line-number">68</span><br><span class="line-number">69</span><br><span class="line-number">70</span><br><span class="line-number">71</span><br><span class="line-number">72</span><br><span class="line-number">73</span><br><span class="line-number">74</span><br><span class="line-number">75</span><br><span class="line-number">76</span><br><span class="line-number">77</span><br><span class="line-number">78</span><br><span class="line-number">79</span><br><span class="line-number">80</span><br><span class="line-number">81</span><br><span class="line-number">82</span><br><span class="line-number">83</span><br><span class="line-number">84</span><br><span class="line-number">85</span><br><span class="line-number">86</span><br><span class="line-number">87</span><br><span class="line-number">88</span><br><span class="line-number">89</span><br><span class="line-number">90</span><br><span class="line-number">91</span><br><span class="line-number">92</span><br><span class="line-number">93</span><br><span class="line-number">94</span><br><span class="line-number">95</span><br><span class="line-number">96</span><br><span class="line-number">97</span><br><span class="line-number">98</span><br><span class="line-number">99</span><br><span class="line-number">100</span><br><span class="line-number">101</span><br><span class="line-number">102</span><br><span class="line-number">103</span><br></div></div><p>程序在PyCharm中运行.</p><p><img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/Epoch.jpg" alt=""></p><p>从图中可见，随着迭代次数增加，损失函数值逐渐减小，对测试集的预测准确率逐渐增大直至达到100%正确.下面两张图给出了损失函数及预测准确率的变化图像.</p><p><img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/Loss_Function_Curve.png" alt=""></p><p><img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/Acc_Curve.png" alt=""></p>`,15),r=[e];function c(i,t,b,o,m,u){return n(),a("div",null,r)}const f=s(l,[["render",c]]);export{d as __pageData,f as default};
