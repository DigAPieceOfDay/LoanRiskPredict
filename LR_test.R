library(data.table)
library(dplyr)
library(randomForest)
library(xgboost)
library(ggplot2)
library(bit64)

# 1.载入数据
# 训练集
# 用户的基本属性user_info_train.txt
setwd("H:/study/智慧杯大赛/客户违约风险分析/个人征信/train")
system.time(
  user_info <- fread(
    "user_info_train.txt",
    col.names = c("用户ID",
                  "性别",
                  "职业",
                  "教育层度",
                  "婚姻状态",
                  "户口类型")
  )
)

# 信用卡账单记录bill_detail.txt
system.time(
  bill_detail <- fread(
    "bill_detail_train.txt",
    col.names = c("用户ID",
                  "账单时间戳",
                  "银行ID",
                  "上期账单金额",
                  "上期还款金额",
                  "信用卡额度",
                  "本期账单余额",
                  "本期账单最低还款额",
                  "消费笔数",
                  "本期账单金额",
                  "调整金额",
                  "循环利息",
                  "可用余额",
                  "预借现金额度",
                  "还款状态")
  )
)
bill_detail$账单时间戳 <- as.integer(bill_detail$账单时间戳 %/% 86400)


# 放款时间信息loan_time.txt
loan_time <- fread("loan_time_train.txt", col.names = c("用户ID","放款时间"))
loan_time$放款时间 <- as.integer(loan_time$放款时间 %/% 86400)

# 顾客是否发生逾期行为的记录overdue.txt
overdue <- fread("overdue_train.txt",col.names = c("用户ID","样本标签"))

# join data
bill_detail <- merge(bill_detail,loan_time,by="用户ID")
data <- merge(overdue,user_info,by="用户ID")
data <- merge(data,loan_time,by="用户ID")

predict.data <- data
predict.data <- merge(predict.data,
                      bill_detail[, .(
                        userbilltime = length(unique(账单时间戳[账单时间戳  >  放款时间])),
                        userbilltimeone  = length(unique(账单时间戳[账单时间戳  >  放款时间 +1])),
                        userbilltimetwo  = length(unique(账单时间戳[账单时间戳  >  放款时间 +2]))
                      ), 用户ID],
                      by = "用户ID",
                      all.x = T)
predict.data [is.na(predict.data )] <- 0
predict.data$性别<-as.factor(predict.data$性别)
predict.data$职业<-as.factor(predict.data$职业)
predict.data$教育层度<-as.factor(predict.data$教育层度)
predict.data$婚姻状态<-as.factor(predict.data$婚姻状态)
predict.data$户口类型<-as.factor(predict.data$户口类型)
predict.data$样本标签<-as.factor(predict.data$样本标签)

library(MASS)
library(nortest)
library(car)
set.seed(123)
ids<-sample(1:dim(data)[1],10000)
train<-predict.data[-ids,]%>%filter(userbilltime>0)
test<-predict.data[ids,]%>%filter(userbilltime>0)
m<-glm(样本标签~.-用户ID-放款时间,data=train,family=binomial(link="logit"))
m1<-stepAIC(m,trace=F)
summary(m1)
m2<-glm(样本标签~.-用户ID-放款时间-教育层度-户口类型,data=train,family=binomial(link="logit"))
system.time(
m3<-stepAIC(m2,trace=F)
)
summary(m3)
prediction<-predict(m3,subset(test,select=c(性别,userbilltime,userbilltimeone,userbilltimetwo)),type="response")
prediction1<-data.frame(用户ID=test$用户ID,true=test$样本标签,pre=prediction)
ks.test(prediction1$true,prediction1$pre)



# -----------------------------------测试集--------------------------------------
# 用户的基本属性user_info_test.txt
rm(list=ls())
setwd("H:/study/智慧杯大赛/客户违约风险分析/个人征信/test")
system.time(
  user_info_test <- fread(
    "user_info_test.txt",
    col.names = c("用户ID",
                  "性别",
                  "职业",
                  "教育层度",
                  "婚姻状态",
                  "户口类型")
  )
)


# 信用卡账单记录bill_detail_test.txt
system.time(
  bill_detail_test <- fread(
    "bill_detail_test.txt",
    col.names = c("用户ID",
                  "账单时间戳",
                  "银行ID",
                  "上期账单金额",
                  "上期还款金额",
                  "信用卡额度",
                  "本期账单余额",
                  "本期账单最低还款额",
                  "消费笔数",
                  "本期账单金额",
                  "调整金额",
                  "循环利息",
                  "可用余额",
                  "预借现金额度",
                  "还款状态")
  )
)
bill_detail_test$账单时间戳 <- as.integer(bill_detail_test$账单时间戳 %/% 86400)
# 放款时间信息loan_time_test.txt
loantime_test <- fread("loan_time_test.txt", col.names = c("用户ID","放款时间"))
loantime_test$放款时间 <- as.integer(loantime_test$放款时间 %/% 86400)
# 顾客是否发生逾期行为的记录overdue.txt
test <- fread("usersID_test.txt",col.names = "用户ID")
# join data
bill_detail_test <- merge(bill_detail_test,loantime_test,by="用户ID")
test <- merge(test,user_info_test,by="用户ID")
test <- merge(test,loantime_test,by="用户ID")

predict.data <- test
predict.data <- merge(predict.data,
                      bill_detail_test[, .(
                        userbilltime = length(unique(账单时间戳[账单时间戳  >  放款时间])),
                        userbilltimeone  = length(unique(账单时间戳[账单时间戳  >  放款时间 +1])),
                        userbilltimetwo  = length(unique(账单时间戳[账单时间戳  >  放款时间 +2]))
                      ), 用户ID],
                      by = "用户ID",
                      all.x = T)
# 填补缺失值
predict.data [is.na(predict.data )] <- 0
predict.data1<-filter(predict.data,userbilltime!=0)
m4<-glm(样本标签~.-用户ID-放款时间-教育层度-户口类型,data=predict.data,family=binomial(link="logit"))
