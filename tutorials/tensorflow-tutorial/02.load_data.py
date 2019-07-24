import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 从csv文件中加载训练数据
training_data_df = pd.read_csv("./dataset/sales_data_training.csv", dtype=float)

# 拆分输入和输出
X_training = training_data_df.drop('销售总额', axis=1).values
Y_training = training_data_df[['销售总额']].values

# 加载测试数据
test_data_df = pd.read_csv("./dataset/sales_data_testing.csv", dtype=float)

# 拆分输入输出
X_testing = test_data_df.drop('销售总额', axis=1).values
Y_testing = test_data_df[['销售总额']].values

# 数据归一化：也可以像keras tutorial中那样统一用一个scaler
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

# 输入输出都归一化
X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

# 很重要：训练数据和测试数据必须采用相同的归一化
X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)

print(X_scaled_testing.shape)
print(Y_scaled_testing.shape)

print("Note: Y values were scaled by multiplying by {:.10f} and adding {:.4f}".format(Y_scaler.scale_[0], Y_scaler.min_[0]))
