### 载入包
if (!suppressWarnings(require(data.table))) {
  install.packages("data.table")
  require(data.table)
}
if (!suppressWarnings(require(dplyr))) {
  install.packages("dplyr")
  require(dplyr)
}
if (!suppressWarnings(require(data.table))) {
  install.packages("data.table")
  require(data.table)
}

##  读取数据
setwd("H:/study/智慧杯大赛/客户违约风险分析/个人征信/train")
bill_detail<-fread("bill_detail_train.txt")
names(bill_detail)<-c("用户ID","账单时间戳","银行ID","上期账单金额","上期还款金额","信用卡额度","本期账单余额","本期账单最低还款额","消费笔数","本期账单金额","调整金额","循环利息","可用余额","预借现金额度","还款状态")
overdue<-fread("overdue_train.txt")
names(overdue)<-c("用户ID","样本标签")
user_info<-fread("user_info_train.txt")
names(user_info)<-c("用户ID","性别","职业","教育层度","婚姻状态","户口类型")

## 数据初处理

### 这个部分按用户ID和还款状态对数据进行聚合
bill_detail<-bill_detail %>% group_by(用户ID,还款状态)
bill_new<-summarise(bill_detail,mean(上期账单金额),mean(上期还款金额),mean(信用卡额度),mean(本期账单余额),mean(本期账单最低还款额),mean(消费笔数),mean(本期账单金额),mean(调整金额),mean(循环利息),mean(可用余额),mean(预借现金额度))
names(bill_new)<-c("用户ID","还款状态","上期账单金额","上期还款金额","信用卡额度","本期账单余额","本期账单最低还款额","消费笔数","本期账单金额","调整金额","循环利息","可用余额","预借现金额度")

### 这个部分把用户信息，信用卡记录，和客户是否逾期按用户ID合并
user_info<-setorder(user_info,用户ID)
id<-user_info$用户ID  %in% bill_new$用户ID
user<-left_join(overdue[id,],user_info[id,])

bill_new<-left_join(user,bill_new)

## 现在开始撸随机森林算法了 

### 载入包
if (!suppressWarnings(require(plyr))) {
  install.packages("plyr")
  require(plyr)
}
if (!suppressWarnings(require(pROC))) {
  install.packages("pROC")
  require(pROC)
}
if (!suppressWarnings(require(randomForest))) {
  install.packages("randomForest")
  require(randomForest)
}

### 先把分类变量和目标变量换为因子
for(i in 1:8){
  bill_new[,i]<- as.factor(bill_new[,i])
}

### 构建交叉检验模型
CVgroup <- function(k, datasize, seed){
  cvlist <- list()
  set.seed(seed)
  n <- rep(1:k, ceiling(datasize/k))[1:datasize]
  temp <- sample(n, datasize)
  x <- 1:k
  dataseq <- 1:datasize 
  cvlist <- lapply(x, function(x) dataseq[temp==x])
  return(cvlist)
}
k <- 5
data<-bill_new[,-1]
datasize <- nrow(bill_new)
cvlist <- CVgroup(k = k, datasize = datasize, seed = 1024)

### 随机森林寻找最优树
auc_value<-data.frame(value=rep(0,90),kcross=rep(0,90),ntree=rep(0,90))
n <- seq(60,400,by=20)##如果数据量大尽量间隔大点，间隔过小没有实际意义
a<-0
set.seed(123)
system.time(for(j in n){
  for (i in 1:k) {
    train<- data[-cvlist[[i]],]
    test<- data[cvlist[[i]],]
    m<- randomForest(样本标签~., data=train, ntree = j)
    prediction <- predict(m, subset(test, select = - 样本标签))
    prediction<-data.frame(test$样本标签,prediction) 
    prediction1<-apply(prediction,2,as.numeric)
    r1<-roc(test.样本标签~prediction,data=prediction1)
    a<-a+1
    auc_value$value[a]<-r1$auc[1]
    auc_value$ntree[a]<-j
    auc_value$kcross[a]<-i
  }
})

auc_value1<-aggregate(data=auc_value,value~ntree,FUN=mean)
tree<-auc_value1$ntree[which.max(auc_value1$value)]#最优树数量
