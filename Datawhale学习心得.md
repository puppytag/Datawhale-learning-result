# Datawhale学习心得



### 首先夸一下代码写得挺好的，另外我有一些优化方向：

- 使用`sklearn.model_selection.GridSearchCV`类来对模型的参数进行网格搜索和交叉验证，这样可以找到最优的参数组合和评估模型的性能。例如，可以写成：

```python
from sklearn.model_selection import GridSearchCV
parameters = {
    'vectorizer__max_features': [1000, 2000, 3000],
    'model__C': [0.1, 1, 10]
}
grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring='accuracy')
grid_search.fit(train['text'], train['label'])
print(grid_search.best_params_)
print(grid_search.best_score_)
test['label'] = grid_search.predict(test['text'])
```

- 可以通过设置停用词来提高分数，可以在创建CountVectorizer对象时，传入一个stop_words参数，指定要移除的停用词列表。可以将代码中的第16行修改为：

```python
vector = CountVectorizer(stop_words='english').fit(train['text'])
```

- 我尝试自己用svm模型进行分类，但是效果差不太多，我也考虑过使用模型融合的方法，但是由于两个模型拟合程度相差不大，所以哪怕融合之后效果肯定也不会有很大改善，我就放弃了，第二阶段的时候我会试着优化一下代码。



### 另外，我对分词规则进行了学习，并自己敲代码加深理解：

```python
import re
import jieba

# 读取文本文件
text="我喜欢吃饭，但是我不喜欢喝饮料，"

# 使用正则表达式匹配非中文字符
pattern = re.compile(r'[^\u4e00-\u9fa5]')
text = re.sub(pattern, '', text)

# 对文本进行分词
words = jieba.cut(text)

# 将分词结果转换为列表
words_list = list(words)

# 使用空格将分词结果连接成字符串
text = ' '.join(words_list)

# 输出处理后的文本
print(text)
```

```
输出：我 喜欢 吃饭 但是 我 不 喜欢 喝 饮料
```

```python
pos={"喜欢","好看","好吃"}
neg={"讨厌","不","难吃"}

Pos_count=0
for item in words_list:
    for element in pos:
        if item == element:
            Pos_count += 1
print(Pos_count)
```

```
输出：2
```

```python
Neg_count=0
for item in words_list:
    for element in neg:
        if item == element:
            Neg_count += 1
print(Neg_count)
```

```
输出：1
```

```python
if Pos_count > Neg_count:
    print("正面的")
if Pos_count < Neg_count:
    print("负面的")
if Pos_count == Neg_count:
    print("中立的")
```

```
输出：正面的
```

```python
# 读取文本文件
text="小时候，看着满天的星斗，当流星飞过的时候，却总是来不及许愿，长大了，遇见了自己真正喜欢的人，却还是来不及。"

# 使用正则表达式匹配非中文字符
pattern = re.compile(r'[^\u4e00-\u9fa5]')
text = re.sub(pattern, '', text)

# 对文本进行分词
words = jieba.cut(text)

# 将分词结果转换为列表
words_list = list(words)

# 使用空格将分词结果连接成字符串
text = ' '.join(words_list)

# 输出处理后的文本
print(text)
```

```
输出：小时候 看着 满天 的 星斗 当 流星 飞过 的 时候 却 总是 来不及 许愿 长大 了 遇见 了 自己 真正 喜欢 的 人 却 还是 来不及
```

```python
pos={"喜欢","好看","好吃"}
neg={"讨厌","不","难吃"}

Pos_count=0
for item in words_list:
    for element in pos:
        if item == element:
            Pos_count += 1
print(Pos_count)

Neg_count=0
for item in words_list:
    for element in neg:
        if item == element:
            Neg_count += 1
print(Neg_count)

if Pos_count > Neg_count:
    print("正面的")
if Pos_count < Neg_count:
    print("负面的")
if Pos_count == Neg_count:
    print("中立的")
```

```
输出：1
0
正面的
```