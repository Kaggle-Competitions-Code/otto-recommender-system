# otto-recommender-system

方案一（耗时过久，废弃）：
----
## data
* train data 与 test data中，session无重复
* `feature/candidate_comatrix_exploded_details.pkl` LB test集，每个session召回各100+
* `otto-validation/test_candidates/candidate_comatrix_exploded_details.pkl` CV 验证集，每个session召回100+
* `feature/carts_count.pkl` item feature， 每个aid被carts的次数
* `feature/orders_count.pkl` item feature, 每个aid被orders的次数
* `feature/clicks_count.pkl` item feature, 每个aid被clicks的次数
* `feature/session_carts_count.pkl` user feature, 每个session的carts次数
* `feature/session_clicks_count.pkl` user feature, 每个session的clicks次数
* `feature/session_orders_count.pkl` user feature, 每个session的orders次数

## feature engineering 
**+ 表示有提高，- 表示下降**

* 时间相对天数
* 时间相对天数余数
* 商品被clicks的数量
* 商品被orders的数量
* 商品被carts的数量
* 用户浏览商品中被carts的比例（-）
* 用户carts商品中被购买的比例（-）
* 滑窗行为分数

## experiment

Public Recall
* clicks recall = 0.5912166896226447
* carts recall = 0.4983971745865439
* orders recall = 0.6989974561367112

### orders
| 方法 | Summary | R@5E4 | R@5E5 | R@5E6 | LB |
| ---  |  ---   |  ---  |  ---  |  ---  | --- |
| XGBRanker+FE+Noisy| 5手工特征 | | 0.6665 |  |


### carts
| 方法 | Summary | R@5E4 | R@5E5 | R@5E6 | LB |
| ---  |  ---   |  ---  |  ---  |  ---  | --- |
| XGBRanker+FE+Noisy| 5手工特征 | | 0.48 | | 0.57|


----
## TODO:
* 滑窗特征
* 滑窗融合
* 训练集随机增加type为3的负样本
* 0.8 week + 0.2 week to split train&gt data (正样本过少，有人获得不错的效果，暂时不再尝试)
* Min-Max、Z-score、Log-based、L2-normalize、Gauss-Rank特征缩放
* downsample (-)

方案二（最终方案）
----
召回仅对session召回n个candidates，不携带其type和ts值，将type和ts信息以feature加入，具体参考2023.1.18 log  
该方案可独立制作特征并本地存储，训练阶段只需要遍历join即可，迭代非常高效  
复现可执行以下notebook
```
# candidates generate(require cuda>=11.x)
train_candidates.ipynb
test_candidates.ipynb

# feature generate
item_feature.ipynb
user_feature.ipynb
interaction_feature.ipynb

# rerank model train&valid&inference&submit
rerank-orders.ipynb
rerank-carts.ipynb

# ensemble
Ensemble.ipynb
```
----
# Log
## 2023.1.18
更换训练集、验证集、测试集及特征的生成方案，参考[Kaggle Link](https://www.kaggle.com/competitions/otto-recommender-system/discussion/370210)


## 2023.1.19
重新生成 Valid-A candidates, 原candidates postive:negative = 1:354

## 2023.1.25
不加入valid A数据集没有，但valid B中存在的正样本，LB 0.565->0.572

## 2023.1.28
生成top100 candidates，时序权重behavior score，是否会复购，点击后是否会购买，点击后是否会carts，carts后是否会购买

## 2023.1.29
生成top60, top100 candidates的效果不如top50, (是否会复购，点击后是否会购买，点击后是否会carts，carts后是否会购买) 没有做，时序权重采用10/(x+1)，x为距离当前数据集最新时间的天数距离，效果无显著提高

## 2023.1.30
将candidates的rank排序做为特征训练orders模型，LB 0.572->0.581，然后同样pipeline更换label训练carts模型LB 0.581->0.583，同样方式训练clicks模型无提高，原因可能是粗排效果很好，缺少其他差异性特征

## 2023.1.31
carts、orders、clicks以full data不同epoch融合，LB略有提高，但B榜无明显提高，不如未融合之前。

----
# Conclusion
* 不要嫌麻烦，尽早搭建一套完整正确的pipeline保证后半程能快速迭代
* 多花时间找到location cv方案和数据，不要以LB去验证自己的算法指标，很多discussion提供的数据集可能很可靠，尽量不要全部自己造轮子
* kaggle teamup ddl后大家分数会有很明显的涨幅，不追求solo golden的情况下，多尝试组队，小心被偷窃方案和答案
* 推荐比赛，召回或粗排的特征往往非常有用，多做特征
* 本次比赛最后方案包含44个特征，提升极为显著的特征分别为`粗排结果的排序order`和`interaction feature：历史行为分（无论加权与否）`