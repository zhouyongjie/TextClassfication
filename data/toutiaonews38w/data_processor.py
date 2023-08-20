import pandas as pd
import random


## 数据源：https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset

label_string = """
100 民生 故事 news_story
101 文化 文化 news_culture
102 娱乐 娱乐 news_entertainment
103 体育 体育 news_sports
104 财经 财经 news_finance
106 房产 房产 news_house
107 汽车 汽车 news_car
108 教育 教育 news_edu 
109 科技 科技 news_tech
110 军事 军事 news_military
112 旅游 旅游 news_travel
113 国际 国际 news_world
114 证券 股票 stock
115 农业 三农 news_agriculture
116 电竞 游戏 news_game
"""

id2label = {}
for label in label_string.split('\n'):
    if label:
        tmp = label.split()
        id2label.update({tmp[0]: tmp[1]})
print(id2label)

with open("toutiao_cat_data.txt", encoding='utf-8') as f:
    data = f.readlines()

new_data_list = []
for text in data:
    tmp = text.split("_!_")
    new_data_list.append((id2label.get(tmp[1]), tmp[3]))

random.shuffle(new_data_list)

df = pd.DataFrame(new_data_list, columns=['Label', 'Text'])
df[:10000].to_csv("dev.csv", index=False)
df[10000:20000].to_csv("test.csv", index=False)
df[20000:].to_csv("train.csv", index=False)



