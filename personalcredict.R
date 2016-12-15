# 数据
# 1）数据总体概述     
#  参赛者可用的训练数据包括用户的基本属性user_info.txt、银行流水记录bank_detail.txt、
#  用户浏览行为browse_history.txt、信用卡账单记录bill_detail.txt、放款时间loan_time.txt，
#  以及这些顾客是否发生逾期行为的记录overdue.txt。（注意：并非每一位用户都有非常完整的记录，
#  如有些用户并没有信用卡账单记录，有些用户却没有银行流水记录。）  

# 相应地，还有用于测试的用户的基本属性、银行流水、信用卡账单记录、浏览行为、放款时间等数据
# 信息，以及待预测用户的id列表脱敏处理：(a) 隐藏了用户的id信息；(b) 将用户属性信息全部数字化
# (c) 将时间戳和所有金额的值都做了函数变换。 

#（2）数据详细描述 （1）用户的基本属性user_info.txt。共6个字段，其中字段性别为0表示性别未知。     
# 用户id,性别,职业,教育程度,婚姻状态,户口类型     
#  6346,1,2,4,4,2     
#  2583,2,2,2,2,1
#  9530,1,2,4,4,2     
#  6707,0,2,3,3,2 
# 
#（2）银行流水记录bank_detail.txt。共5个字段，其中，第2个字段，时间戳为0表示时间未知；第3个字
# 段，交易类型有两个值，1表示支出、0表示收入；第5个字段，工资收入标记为1时，表示工资收入。
# 用户id,时间戳,交易类型,交易金额,工资收入标记     
#  6951,5894316387,0,13.756664,0     
#  6951,5894321388,1,13.756664,0     
#  18418,5896951231,1,11.978812,0     
#  18418,5897181971,1,12.751543,0     
#  18418,5897293906,0,14.456463,1 

#（3）用户浏览行为browse_history.txt。共4个字段。其中，第2个字段，时间戳为0表示时间未知。 
# 用户id,时间戳,浏览行为数据,浏览子行为编号	     
#  34724,5926003545,172,1     
#  34724,5926003545,163,4     
#  34724,5926003545,38,7     
#  67215,5932800403,163,4     
#  67215,5932800403,138,4     
#  67215,5932800403,109,7 
# 
#（4）信用卡账单记录bill_detail.txt。共15个字段，其中，第2个字段，时间戳为0表示时间未知。
# 为方便浏览，字段以表格的形式给出
 #  字段                   注释
 # 用户id                  整数
 # 账单时间戳              整数，0表示未知
 # 银行id                  枚举类型
 # 上期账单金额            浮点数
 # 上期还款金额            浮点数
 # 信用卡额度              浮点数
 # 本期账单余额            浮点数
 # 本期账单最低还款额      浮点数
 # 消费笔数                整点数
 # 本期账单金额            浮点数   
 # 调整金额                浮点数
 # 循环利息                浮点数
 # 可用余额                浮点数
 # 预借现金额度            浮点数
 # 还款状态                枚举值

#（5）放款时间信息loan_time.txt。共2个字段，用户id和放款时间。     
# 用户id,放款时间     
# 1,5914855887     
# 2,5914855887     
# 3,5914855887 

#（6）顾客是否发生逾期行为的记录overdue.txt。共2个字段。样本标签为1，表示逾期30天以上；
# 样本标签为0，表示逾期10天以内。注意：逾期10天~30天之内的用户，并不在此问题考虑的范围内。
# 用于测试的用户，只提供id列表，文件名为testUsers.csv。    
# 用户id,样本标签     
# 1,1     
# 2,0     
# 3,1

# 评分标准
# 采用kolmogorov-Smirnov(ks)统计量来衡量预测结果，KS是风险评分领域常用的评价标准，KS越高表明
# 模型对正负样本的区分能力越强。计算方法如下：
# 假设f(s|P)为正样本预测值的累积分布函数(cdf),f(s|N)为负样本预测值的累积分布函数，则
#    KS =  max{|f(s|P)-f(s|N)|}

library(data.table)
library(dplyr)
library(randomForest)
library(xgboost)
library(ggplot2)

library(bit64)

# 1.载入数据
# 用户的基本属性user_info_train.txt
user.info.train <- fread("./data/train/user_info_train.txt",header = F)
colnames(user.info.train) <- c("usrId","gender","profession","education","marital","nodetype")

# 银行流水记录bank_detail.txt
bank.detail <- fread("./data/train/bank_detail_train.txt",header = F)
colnames(bank.detail) <- c("usrId","timestamp","transactionType","transactionSum","salaryflag")

# 用户浏览行为browse_history.txt
browse.history <- fread("./data/train/browse_history_train.txt")
colnames(browse.history) <- c("usrId","timestamp","browseaction","browsesubactionId")


# 信用卡账单记录bill_detail.txt
bill.detail <- fread("./data/train/bill_detail_train.txt")
colnames(bill.detail) <- c("usrId","billtimestamp","bankId","lastbillsum",
                           "lastrepaysum","credictcardegree","billbalance","billrepayleastSum",
                           "consumcount","billsum","adjustsum","cycleinterest","usefullbalance",
                           "cashAdvanceLimit","repaystatus")

# 放款时间信息loan_time.txt
loan.time <- fread("./data/train/loan_time_train.txt")
colnames(loan.time) <- c("usrId","lendingtime")

# 顾客是否发生逾期行为的记录overdue.txt
overdue <- fread("./data/train/overdue_train.txt")
colnames(overdue) <- c("usrId","samplelabel")

# 2.数据清洗整合

# 对于银行流水记录数据一个用户对应多条记录，这里我们采用对每个用户每种交易类型
# 取均值进行聚合
bank.detail.gather <- bank.detail %>% group_by(usrId,transactionType) %>% 
  summarise(transactionSum = mean(transactionSum))
  
# 对于用户浏览数据，计算每个用户总浏览行为次数进行聚合
browse.history.gather <- browse.history  %>% group_by(usrId) %>% 
  summarise(browseaction=sum(browseaction))

# 对于信用卡账单数据按id取均值
bill.detail.gather <- bill.detail %>% group_by(usrId) %>% 
  summarise(lastbillsum = mean(lastbillsum),
            lastrepaysum= mean(lastrepaysum),
            )

#---------------------------------------------------------------------------------
loan_data <- fread("./output/loan_data.csv",header = T)

# # 构建模型
# 分开训练集、测试集
status <- overdue$samplelabel
train <- data.frame(loan_data[1:nrow(overdue),],status=status)
test <- loan_data[55597:nrow(loan_data),]

# 将训练集划分为7:3
library(caret)
set.seed(27)
val_index <- createDataPartition(train$status,p = 0.7, list=FALSE)
val_train_data <- train[val_index, ]
val_test_data  <- train[-val_index, ]
val_train_X <- val_train_data[,-1]
val_test_X <- val_test_data[,-1]


# 转换为matrix
matrix_train <- apply(val_train_X, 2, function(x) as.numeric(as.character(x)))
matrix_test <- apply(val_test_X, 2, function(x) as.numeric(as.character(x)))

xgb_train_matrix <- xgb.DMatrix(data = as.matrix(matrix_train), label = val_train_X$status)
xgb_test_matrix <- xgb.DMatrix(data = as.matrix(matrix_test), label = val_test_X$status)

watchlist <- list(train = xgb_train_matrix, test = xgb_test_matrix)
label <- getinfo(xgb_test_matrix, "label")

# using cross validation to evaluate the error rate.
param <- list("objective" = "binary:logistic")

# croos-validation 
bst <- xgb.cv(param = param, 
       data = xgb_train_matrix, 
       nfold = 3,
       label = getinfo(xgb_train_matrix, "label"),
       nrounds = 5)

# (1)Training with gbtree
bst_1 <- xgb.train(data = xgb_train_matrix, 
                   label = getinfo(xgb_train_matrix, "label"),
                   max.depth = 2, 
                   eta = 1, 
                   nthread = 4, 
                   nround = 50, # number of trees used for model building
                   watchlist = watchlist, 
                   objective = "binary:logistic")

# obtain important feature
features <- colnames(matrix_train)
importance_matrix_1 <- xgb.importance(features, model = bst_1)
print(importance_matrix_1)

xgb.plot.importance(importance_matrix_1) +
  theme_minimal()

# predict 
pred_1 <- predict(bst_1, xgb_test_matrix)
result_1 <- data.frame(usrId = rownames(val_test_data),
                       status = val_test_data$status, 
                       label = label, 
                       prediction_p_loan = round(pred_1, digits = 2),
                       prediction = as.integer(pred_1 > 0.5),
                       prediction_eval = ifelse(as.integer(pred_1 > 0.5) != label, "wrong", "correct"))
result_1