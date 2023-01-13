# otto-recommender-system

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

* 商品被clicks的数量
* 商品被orders的数量
* 商品被carts的数量
* 用户浏览商品中被carts的比例
* 用户carts商品中被购买的比例
* 滑窗行为分数

## experiment

Public Recall
<!-- * clicks recall = 0.5255597442145808
* carts recall = 0.4093328152483512
* orders recall = 0.6487936598117477 -->

* clicks recall = 0.5912166896226447
* carts recall = 0.4983971745865439
* orders recall = 0.6989974561367112


### carts
| 方法 | Summary | R@5E4 | R@5E5 | R@5E6 | LB |
| ---  |  ---   |  ---  |  ---  |  ---  | --- |
| XGBRanker+OpenFE | AutoFE生成44特征 + 11手工特征| | 0.1613 | |
| XGBRanker+FE | 11手工特征 | | 0.32 | 0.42 |
| XGBRanker+FE+Noisy| 11手工特征 | | 0.264 | |
| XGBRanker+FE+Noisy+BUG| 17特征 | | 0.509 | | 0.572 |


----
## TODO:
* 滑窗特征
* 滑窗融合
* 训练集随机增加type为3的负样本
* 0.8 week + 0.2 week to split train&gt data
* Min-Max、Z-score、Log-based、L2-normalize、Gauss-Rank特征缩放