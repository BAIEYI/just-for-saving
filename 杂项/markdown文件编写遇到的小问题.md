# 基础教程所在
网址：https://www.runoob.com/markdown/md-tutorial.html
runoob网站，可以学很多东西的菜鸟网站。

## 1.不识别图像插入
在插入图像时，要用到类似如下的语句  
```
#![alt text](0.png)
```
但是在一种情况下会失效：
```
…………
<br>
![alt text](0.png)<br>
```
因为上面已经有了一个《br》下面的可能不识别插入图片而是作为文字形式了。

## 2.终于知道咋加下标了
分割符合可以是~也可以是《sub》与《/sub》
但注意二者必须是成对出现的，正确示例：
H~2~O，O~2~，θ~max~，O<sub>2</sub>
错误示例：
H~2O
O~2
σ<sub>max,H~2~O


