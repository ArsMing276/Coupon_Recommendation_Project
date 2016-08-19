library(data.table)
library(Matrix)

user_List = fread("user_list.csv")
coupon_area_train = fread("coupon_area_train.csv")
coupon_area_test = fread("coupon_area_test.csv")
coupon_list_train = fread("coupon_list_train.csv")
coupon_list_test = fread("coupon_list_test.csv")
coupon_detail_train = fread("coupon_detail_train.csv")
prefecture_location = fread("prefecture_locations.csv")
coupon_visit_train = fread("coupon_visit_train.csv")

# random sample function
feature_factor = function(df, feature, feature_ID_name,TF,range){
  class(df) = "data.frame"
  feature_ID = as.factor(df[,feature])
  #a = levels(feature_ID)
  if(TF)
    levels(feature_ID) = c(1:length(unique(feature_ID)))
  else
    levels(feature_ID) = range
  #feature_ID = as.character(feature_ID)
  #feature_ID = as.numeric(feature_ID)
  df[feature_ID_name] = feature_ID
  df
}

prefecture_location = fread("prefecture_locations.csv")
names(prefecture_location)[1] = "PREF_NAME"
user_List = fread("user_list.csv")
user_List = feature_factor(user_List, "USER_ID_hash", "USER_ID")
user_List = feature_factor(user_List, "SEX_ID","SEX_ID_integer")
user_List = merge(user_List, prefecture_location, by="PREF_NAME")
class(user_List) = "data.frame"
names(user_List)[10] = "user.LATITUDE"
names(user_List)[11] = "user.LONGITUDE"

# random select user_List
set.seed(171)
Idx_user_ID = sample(nrow(user_List), 1000, replace = FALSE)
user_List_r1 = user_List[Idx_user_ID, ]

# check the random sample of users
hist(as.numeric(user_List$SEX_ID_integer))
hist(as.numeric(user_List_r1$SEX_ID_integer))
hist(user_List_r1$AGE)
hist(user_List$AGE)


# coupon information

coupon_area_train = fread("coupon_area_train.csv")
coupon_list_train = fread("coupon_list_train.csv")
prefecture_location = fread("prefecture_locations.csv")
names(prefecture_location)[1] = "PREF_NAME"
coupon_info1 = merge(coupon_area_train, coupon_list_train, by = "COUPON_ID_hash")
coupon_info2 = merge(coupon_info1, prefecture_location, by="PREF_NAME")
coupon_info3 = feature_factor(coupon_info2, "COUPON_ID_hash", "COUPON_ID")


#coupon_detail_train = fread("coupon_detail_train.csv")
coupon_visit_train = fread("coupon_visit_train.csv")
class(coupon_visit_train) = "data.frame"
names(coupon_visit_train)[2] = "COUPON_ID_hash"
coupon_visit_train = coupon_visit_train[,c(1,5,6)]
coupon_visit1 = merge(coupon_visit_train, user_List_r1, by = "USER_ID_hash")
class(coupon_visit1) = "data.frame"
coupon_visit2 = merge(coupon_visit1, coupon_info3, by ="COUPON_ID_hash")
class(coupon_visit2) = "data.frame"
coupon_visit2$loc_distance = (coupon_visit2$user.LATITUDE - coupon_visit2$LATITUDE)^2 + (coupon_visit2$user.LONGITUDE - coupon_visit2$LONGITUDE)^2
coupon_visit2 = feature_factor(coupon_visit2, "GENRE_NAME", "GENRE_ID")
coupon_visit3 = coupon_visit2[,c("PURCHASE_FLG","loc_distance","AGE","SEX_ID_integer","user.LATITUDE","user.LONGITUDE","GENRE_ID","PRICE_RATE","CATALOG_PRICE","DISCOUNT_PRICE","DISPPERIOD","LATITUDE","LONGITUDE","COUPON_ID_hash", "USER_ID_hash","SEX_ID","COUPON_ID","USER_ID","GENRE_NAME")]
class(coupon_visit3) = "data.frame"
save(coupon_visit3, file = "coupon_visit3")


coupon_visit4 = feature_factor(coupon_visit3,"COUPON_ID_hash","COUPON_ID2",TRUE)
coupon_visit4 = feature_factor(coupon_visit4,"USER_ID_hash","USER_ID2",TRUE)

# coupon for tree model # 1 female 0 male
coupon_visit4$PURCHASE_FLG = as.factor(coupon_visit4$PURCHASE_FLG)
coupon_visit_final = coupon_visit4[,c(1:13)]
coupon_visit_final = coupon_visit_final[,c(1,4,7,2,3,5,6,8,9,10,11,12,13)]

Idx1 = which(coupon_visit_final$PURCHASE_FLG ==1)
Idx0 = which(coupon_visit_final$PURCHASE_FLG ==0)

set.seed(171)
Idx1_r = sample(Idx1, 30000, replace = FALSE)
Idx0_r = sample(Idx0, 30000, replace = FALSE)

coupon_visit_final2 = coupon_visit_final[c(Idx1_r, Idx0_r), ]
head(coupon_visit_final2)
coupon_visit_final3 = scale(coupon_visit_final2[,c(4:13)], center = TRUE, scale = TRUE)
attributes(coupon_visit_final3)
coupon_visit_final4 = as.data.frame(coupon_visit_final3)
coupon_visit_final5 = cbind(coupon_visit_final2[,c(1:3)], coupon_visit_final4)

coupon_visit_modelbuilding = coupon_visit_final5
save(coupon_visit_modelbuilding, file = "coupon_visit_modelbuilding")

coupon_visit_final5 = coupon_visit_final2
coupon_visit_final5$SEX_ID_integer = as.character(coupon_visit_final5$SEX_ID_integer)
coupon_visit_final5$SEX_ID_integer = as.numeric(coupon_visit_final5$SEX_ID_integer)
coupon_visit_final5$GENRE_ID = as.character(coupon_visit_final5$GENRE_ID)
coupon_visit_final5$GENRE_ID = as.numeric(coupon_visit_final5$GENRE_ID)

coupon_visit_final6 = scale(coupon_visit_final5[,c(2:13)], center = TRUE, scale = TRUE)
attributes(coupon_visit_final6)
coupon_visit_final6 = as.data.frame(coupon_visit_final6)
coupon_visit_final7 = cbind(coupon_visit_final2[,1], coupon_visit_final6)
names(coupon_visit_final7)[1]="PURCHASE_FLG"
coupon_modelbuilding_numeric = coupon_visit_final7
save(coupon_modelbuilding_numeric, file = "coupon_modelbuilding_numeric")

# data for matrix completion
coupon_visit4$COUPON_ID2 = as.character(coupon_visit4$COUPON_ID2)
coupon_visit4$COUPON_ID2 = as.numeric(coupon_visit4$COUPON_ID2)
coupon_visit4$USER_ID2 = as.character(coupon_visit4$USER_ID2)
coupon_visit4$USER_ID2 = as.numeric(coupon_visit4$USER_ID2)

coupon_visit5 = feature_factor(coupon_visit4,"PURCHASE_FLG","PURCHASE_FLG",c(-1,1))
coupon_visit5$PURCHASE_FLG = as.character(coupon_visit5$PURCHASE_FLG)
coupon_visit5$PURCHASE_FLG = as.numeric(coupon_visit5$PURCHASE_FLG)


# delete the cols
numUser = length(unique(coupon_visit3$USER_ID_hash))
numCOUPON = length(unique(coupon_visit3$COUPON_ID_hash))
spMatrix= Matrix(0, numUser, numCOUPON, sparse = TRUE)

for(i in 1:nrow(coupon_visit5)){
  spMatrix[coupon_visit5$USER_ID2[i], coupon_visit5$COUPON_ID2[i]] = coupon_visit5$PURCHASE_FLG[i]
}


col_zero = sapply(1:dim(spMatrix)[2], function(i) length(which(spMatrix[,i]==0)))
row_zero = sapply(1:dim(spMatrix)[1], function(i) length(which(spMatrix[,i]==0)))


Idx_coupon = which(col_zero<980)
coupon_visit_MatrixCompletion = coupon_visit5[which(!is.na(match(coupon_visit5$COUPON_ID2,Idx_coupon))),]
coupon_visit_MatrixCompletion2 = feature_factor(coupon_visit_MatrixCompletion,"PURCHASE_FLG","PURCHASE_FLG",c(0,1))
coupon_visit_MatrixCompletion3 = coupon_visit_MatrixCompletion2[,c(1,14,15)]
coupon_visit_MatrixCompletion4 = feature_factor(coupon_visit_MatrixCompletion3,"COUPON_ID_hash","COUPON_ID22",TRUE)
coupon_visit_MatrixCompletion5 = feature_factor(coupon_visit_MatrixCompletion4,"USER_ID_hash","USER_ID22",TRUE)
coupon_visit_MatrixCompletion5$COUPON_ID22 = as.character(coupon_visit_MatrixCompletion5$COUPON_ID22)
coupon_visit_MatrixCompletion5$COUPON_ID22 = as.numeric(coupon_visit_MatrixCompletion5$COUPON_ID22)
coupon_visit_MatrixCompletion5$USER_ID22 = as.character(coupon_visit_MatrixCompletion5$USER_ID22)
coupon_visit_MatrixCompletion5$USER_ID22 = as.numeric(coupon_visit_MatrixCompletion5$USER_ID22)
coupon_visit_MatrixCompletion5$PURCHASE_FLG = as.character(coupon_visit_MatrixCompletion5$PURCHASE_FLG)
coupon_visit_MatrixCompletion5$PURCHASE_FLG = as.numeric(coupon_visit_MatrixCompletion5$PURCHASE_FLG)

save(coupon_visit_MatrixCompletion5, file = "MatrixCompletion_user_coupon_hash")
coupon_visit_MatrixCompletion6 = coupon_visit_MatrixCompletion5[,c(5,4,1)]
save(coupon_visit_MatrixCompletion6, file = "coupon_visit_MatrixCompletion6")



head(coupon_visit5)



df = user_List
feature = "USER_ID_hash"
feature_ID_name = "USER_ID"


USER_ID = as.factor(user_List$USER_ID_hash)
a = levels(USER_ID)
levels(USER_ID) = c(1:length(a))
USER_ID = as.character(USER_ID)
USER_ID = as.numeric(USER_ID)
user_List$USER_ID = USER_ID

User_Num = length(unique(USER_ID_hash))

par(mfrow= c(1,2))
hist(user_List$AGE, main = "All Users", xlab = "AGE")
hist(user_List_r1$AGE, xlab = "AGE",main ="Random Selected Users")
length(which(user_List$SEX_ID=="m"))/length(which(user_List$SEX_ID=="f"))
length(which(user_List_r1$SEX_ID=="m"))/length(which(user_List_r1$SEX_ID=="f"))