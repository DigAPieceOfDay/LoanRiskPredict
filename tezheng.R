### 载入包
if (!suppressWarnings(require(data.table))) {
  install.packages("data.table")
  require(data.table)
}
if (!suppressWarnings(require(dplyr))) {
  install.packages("dplyr")
  require(dplyr)
}

# 数据探索&特征分析

## borwse_history数据部分

### 读取数据
setwd("H:/study/智慧杯大赛/客户违约风险分析/个人征信/train")
system.time(
  overdue<-fread("overdue_train.txt")
)
names(overdue)<-c("用户ID","样本标签")

system.time(
  browse_his<-fread("browse_history_train.txt")
)
names(browse_his)<-c("用户ID","时间戳","浏览行为数据","浏览行为子编号")

### 数据初步处理
browse_his$次数<-rep(1,22919547)
####这里先提取出每个用户的总浏览行为
bro_num<-browse_his%>%group_by(用户ID)%>%summarise(总次数=sum(次数))
browse_his<-browse_his%>%group_by(用户ID,浏览行为数据,浏览行为子编号)
browse_his_new<-browse_his%>%summarise(次数=sum(次数))
id<-overdue$用户ID%in%browse_his_new$用户ID
over_user<-overdue[id,]
browse_his_new<-left_join(over_user,browse_his_new)
rm(over_user,id) ###删掉过度变量，减小内存
browse_his_new<-left_join(browse_his_new,bro_num)
names(browse_his_new)
#### 这张表现在有这几个字段
#[1] "用户ID"         "样本标签"  
#[3] "浏览行为数据"   "浏览行为子编号"
#[5] "次数"           "总次数"

###提取违约客户中的高频行为，
browse_his_new<-browse_his_new%>%mutate(平均次数=次数/总次数)%>%subset(select=-c(次数,总次数))
####先剔除客户总行为次数的影响

id<-which(browse_his_new$样本标签==1)
browse_over<-browse_his_new[id,]
browse_over<-browse_over%>%group_by(浏览行为数据,浏览行为子编号)%>%summarise(平均次数=sum(平均次数))%>%setorder(-平均次数,浏览行为数据,浏览行为子编号)
####browse_over提取出了违约客户中每个行为（178条）的平均次数，并且按次数由高到底排列。
####我想的是接下来可以把它离散化，作为先验信息，就像那个天池的天猫移动推荐一样。我认为这个用户浏览历史行为实在不适合代入LR里