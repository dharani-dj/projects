
library(MASS)
library(car)
library(e1071)
library(ggplot2)
library(cowplot)
library(caTools)
library(dummies)

emp_survey<-read.csv("employee_survey_data.csv")
gen_data<-read.csv("general_data.csv")
mang_survey<-read.csv("manager_survey_data.csv")

# Data collation in a single dataframe 
length(unique(tolower(emp_survey$EmployeeID)))   # 4410, confirming EmployeeID is key 
length(unique(tolower(gen_data$EmployeeID)))     # 4410, confirming EmployeeID is key 
length(unique(tolower(mang_survey$EmployeeID)))  # 4410, confirming EmployeeID is key 

setdiff(emp_survey$EmployeeID,gen_data$EmployeeID) # Identical EmployeeID across these datasets
setdiff(gen_data$EmployeeID,mang_survey$EmployeeID) # Identical EmployeeID across these datasets

emp_data<-merge(emp_survey,gen_data,by ="EmployeeID", all = F)
emp_data<-merge(emp_data,mang_survey,by ="EmployeeID", all = F)

#--------------------------------------------------------------------------------------------------

in_time<-read.csv("in_time.csv")
out_time<-read.csv("out_time.csv")

work_time<-in_time # This is just to initiate the dataframe - work_time 
work_time[,]<-0
work_time[,1]<-in_time[,1]
colnames(work_time)[1]<-"EmployeeID"

## Below step might take significant time for execution- (Approx- 15 mins)

for( i in 1:nrow(in_time)) # This step takes significant time as it has 4410x4410 iterations(Approx - 15 Mins) 
{
  for(j in 2:ncol(in_time))
  {
    tmp_in<-as.POSIXct(in_time[i,j],format="%Y-%m-%d %H:%M:%S") 
    tmp_out<-as.POSIXct(out_time[i,j],format="%Y-%m-%d %H:%M:%S") 
    in_time_h<-as.numeric(format(tmp_in,"%H"))
    in_time_m<-as.numeric(format(tmp_in,"%M"))
    out_time_h<-as.numeric(format(tmp_out,"%H"))
    out_time_m<-as.numeric(format(tmp_out,"%M"))
    time_difference<-as.numeric(((out_time_h*60+out_time_m)-(in_time_h*60+in_time_m)))
    work_time[i,j]<-time_difference/60
  }
}

avg_work_time<-work_time[,1:2] # This is just to initiate the dataframe - avg_work_time 
for(i in 1:nrow(work_time))
{
  avg_work_time[i,2]<-mean(as.numeric(work_time[i,-1],na.rm=TRUE),na.rm = TRUE)  
}
colnames(avg_work_time)[2]<-"Avg time spent"

emp_data<-merge(emp_data,avg_work_time,by ="EmployeeID", all = F)


##-------------- Derived metric--------
# We have computed 'Avg time spent', now this will be used to derive a new metric- time score(Avg time spent/Standard time)

emp_data$time_score<-emp_data$`Avg time spent`/emp_data$StandardHours

#Removing standard time variable
emp_data<-emp_data[,-21]

str(emp_data)

##-----------Outlier Treatment for continuous variables-------####

##Checking for outlier in Age####
quantile(emp_data$Age,probs=seq(0,1,0.005))
boxplot(emp_data$Age)
## No issues found

##Checking for outlier in Distancefromhome####
quantile(emp_data$DistanceFromHome,probs=seq(0,1,0.005))
boxplot(emp_data$DistanceFromHome)
## No issues found

##Checking for outlier in Monthlyincome####
quantile(emp_data$MonthlyIncome,probs=seq(0,1,0.005))
boxplot(emp_data$MonthlyIncome)
## No issues found

##Checking for outlier in Percentsalaryhike####
quantile(emp_data$PercentSalaryHike,probs=seq(0,1,0.005))
boxplot(emp_data$PercentSalaryHike)
## No issues found

##Checking for outlier in Totalworkingyears####
quantile(emp_data$TotalWorkingYears,probs=seq(0,1,0.005),na.rm = TRUE)
boxplot(emp_data$TotalWorkingYears)
## No issues found

##Checking for outlier in Trainingtimelastyear####
quantile(emp_data$TrainingTimesLastYear,probs=seq(0,1,0.005))
boxplot(emp_data$TrainingTimesLastYear)
## No issues found

##Checking for outlier in YearAtCompany####
quantile(emp_data$YearsAtCompany,probs=seq(0,1,0.005))
boxplot(emp_data$YearsAtCompany)
## There is a jump at 95%ile i.e from 20. so replacing values more than 20 with 20
emp_data$YearsAtCompany[which(emp_data$YearsAtCompany>20)]<-20

##Checking for outlier in Yearssincelastpromotion####
quantile(emp_data$YearsSinceLastPromotion,probs=seq(0,1,0.005))
boxplot(emp_data$YearsSinceLastPromotion)
## There is a jump at 95%ile i.e from 10. so replacing values more than 10 with 10
emp_data$YearsSinceLastPromotion[which(emp_data$YearsSinceLastPromotion>10)]<-10

##Checking for outlier in Yearswithcurrentmanager####
quantile(emp_data$YearsWithCurrManager,probs=seq(0,1,0.005))
boxplot(emp_data$YearsWithCurrManager)
## No issues found

##Checking for outlier in averagetimespent####
quantile(emp_data$`Avg time spent`,probs=seq(0,1,0.005))
boxplot(emp_data$`Avg time spent`)
## No issues found
##-----Outlier treatment is completed-------####

## CREATING DUMMY VARIABLES####
## By using package dummies we can create dummy variables for charector and factor type variables###
##Example for gender, the following command create dummy variables gender-male, 
##gender-female. We have to delete one dummy variable.

library(dummies)
emp_data<-dummy.data.frame(emp_data, sep = "_")
View(emp_data)


## Deleting extra dummy variables##
str(emp_data)

##Deleting Attrition_Yes,BusinessTravel_Non-Travel,Department_Human Resources,EducationField_Human Resources
##Gender_Female,JobRole_Research Director,  MaritalStatus_Divorced
emp_data <-emp_data[, -c(7,8,11,16,23,31,35)]
str(emp_data)

##Missing values

sapply(emp_data,function(x) sum(is.na(x))) # Nas are present in some fields 

tot_nas_recs<-nrow(subset(emp_data,is.na(emp_data$EnvironmentSatisfaction)|is.na(emp_data$JobSatisfaction)|is.na(emp_data$WorkLifeBalance)|is.na(emp_data$NumCompaniesWorked)|is.na(emp_data$TotalWorkingYears)))
tot_nas_recs/nrow(emp_data) # It is 2.49% of the entire dataset 

# Rows with Nas will be removed 

emp_data<-emp_data[-which(is.na(emp_data$EnvironmentSatisfaction)),]
emp_data<-emp_data[-which(is.na(emp_data$JobSatisfaction)),]
emp_data<-emp_data[-which(is.na(emp_data$WorkLifeBalance)),]
emp_data<-emp_data[-which(is.na(emp_data$NumCompaniesWorked)),]
emp_data<-emp_data[-which(is.na(emp_data$TotalWorkingYears)),]

##---------------------Duplicate Value-----------------##
duplicated(emp_data$EmployeeID)
sum(duplicated(emp_data$EmployeeID)) #No duplication in customer ID

##---------------------Standardization of Variable-----------------##
emp_data$EnvironmentSatisfaction<-scale(emp_data$EnvironmentSatisfaction)
emp_data$JobSatisfaction<-scale(emp_data$JobSatisfaction)
emp_data$WorkLifeBalance<-scale(emp_data$WorkLifeBalance)
emp_data$Age<-scale(emp_data$Age)
emp_data$DistanceFromHome<-scale(emp_data$DistanceFromHome)
emp_data$MonthlyIncome<-scale(emp_data$MonthlyIncome)
emp_data$Education<-scale(emp_data$Education)
emp_data$EmployeeCount<-scale(emp_data$EmployeeCount)
emp_data$JobLevel<-scale(emp_data$JobLevel)
emp_data$NumCompaniesWorked<-scale(emp_data$NumCompaniesWorked)
emp_data$PercentSalaryHike<-scale(emp_data$PercentSalaryHike)
emp_data$StockOptionLevel<-scale(emp_data$StockOptionLevel)
emp_data$TotalWorkingYears<-scale(emp_data$TotalWorkingYears)
emp_data$TrainingTimesLastYear<-scale(emp_data$TrainingTimesLastYear)
emp_data$YearsAtCompany<-scale(emp_data$YearsAtCompany)
emp_data$YearsSinceLastPromotion<-scale(emp_data$YearsSinceLastPromotion)
emp_data$YearsWithCurrManager<-scale(emp_data$YearsWithCurrManager)
emp_data$JobInvolvement<-scale(emp_data$JobInvolvement)
emp_data$PerformanceRating<-scale(emp_data$PerformanceRating)
emp_data$`Avg time spent`<-scale(emp_data$`Avg time spent`)
emp_data$time_score<-scale(emp_data$time_score)

##---------------------Format change for Target variable Attrition_No-----------------##
emp_data$Attrition_No<-factor(emp_data$Attrition_No)
summary(emp_data)
View(emp_data)
str(emp_data)

##---------------Removing EmployeeID---------------------------------------------##

emp_data$EmployeeID<-NULL
emp_data$EmployeeCount<-NULL

##----------------Putting Target variable at the start-----------------------------##

library(dplyr)
emp_data <- emp_data %>% select(Attrition_No, everything())

##------------------Data Splitting----------------------------------------------##
set.seed(100)

indices = sample.split(emp_data$Attrition_No, SplitRatio = 0.7)

train = emp_data[indices,]

test = emp_data[!(indices),]

##---------------------Model Building:Logistic Regression-----------------------##
#Initial model
model_1 = glm(Attrition_No ~ ., data = train, family = "binomial")
summary(model_1)

# Stepwise selection
library("MASS")
model_2<- stepAIC(model_1, direction="both")


summary(model_2)

# Removing multicollinearity through VIF check
library(car)
vif(model_2)

# EducationField_Life Sciences and EducationField_Medical have high VIFs but are highly significant variables, hence it cannot be removed 
# Other insignificant variables will be removed 

# Removing JobRole_Research Scientist
model_3<- glm(formula = Attrition_No ~ EnvironmentSatisfaction + JobSatisfaction + 
          WorkLifeBalance + Age + BusinessTravel_Travel_Frequently + 
      BusinessTravel_Travel_Rarely + `EducationField_Life Sciences` + 
      EducationField_Marketing + EducationField_Medical + EducationField_Other + 
      `EducationField_Technical Degree` + JobLevel + `JobRole_Healthcare Representative` + 
      `JobRole_Human Resources` + `JobRole_Laboratory Technician` + 
      JobRole_Manager + `JobRole_Manufacturing Director` + 
      `JobRole_Sales Representative` + MaritalStatus_Married + 
      MaritalStatus_Single + NumCompaniesWorked + StockOptionLevel + 
      TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
      YearsWithCurrManager + `Avg time spent`, family = "binomial", 
    data = train)
summary(model_3)

# Removing JobRole_Sales Representative
model_4<- glm(formula = Attrition_No ~ EnvironmentSatisfaction + JobSatisfaction + 
                WorkLifeBalance + Age + BusinessTravel_Travel_Frequently + 
                BusinessTravel_Travel_Rarely + `EducationField_Life Sciences` + 
                EducationField_Marketing + EducationField_Medical + EducationField_Other + 
                `EducationField_Technical Degree` + JobLevel + `JobRole_Healthcare Representative` + 
                `JobRole_Human Resources` + `JobRole_Laboratory Technician` + 
                JobRole_Manager + `JobRole_Manufacturing Director` + 
                + MaritalStatus_Married + 
                MaritalStatus_Single + NumCompaniesWorked + StockOptionLevel + 
                TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                YearsWithCurrManager + `Avg time spent`, family = "binomial", 
              data = train)
summary(model_4)

# Removing JobRole_Laboratory Technician
model_5<- glm(formula = Attrition_No ~ EnvironmentSatisfaction + JobSatisfaction + 
                WorkLifeBalance + Age + BusinessTravel_Travel_Frequently + 
                BusinessTravel_Travel_Rarely + `EducationField_Life Sciences` + 
                EducationField_Marketing + EducationField_Medical + EducationField_Other + 
                `EducationField_Technical Degree` + JobLevel + `JobRole_Healthcare Representative` + 
                `JobRole_Human Resources` +  
                JobRole_Manager + `JobRole_Manufacturing Director` + 
                + MaritalStatus_Married + 
                MaritalStatus_Single + NumCompaniesWorked + StockOptionLevel + 
                TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                YearsWithCurrManager + `Avg time spent`, family = "binomial", 
              data = train)
summary(model_5)

# Removing StockOptionLevel
model_6<- glm(formula = Attrition_No ~ EnvironmentSatisfaction + JobSatisfaction + 
                WorkLifeBalance + Age + BusinessTravel_Travel_Frequently + 
                BusinessTravel_Travel_Rarely + `EducationField_Life Sciences` + 
                EducationField_Marketing + EducationField_Medical + EducationField_Other + 
                `EducationField_Technical Degree` + JobLevel + `JobRole_Healthcare Representative` + 
                `JobRole_Human Resources` +  
                JobRole_Manager + `JobRole_Manufacturing Director` + 
                + MaritalStatus_Married + 
                MaritalStatus_Single + NumCompaniesWorked + 
                TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                YearsWithCurrManager + `Avg time spent`, family = "binomial", 
              data = train)
summary(model_6)

# Removing JobLevel 
model_7<- glm(formula = Attrition_No ~ EnvironmentSatisfaction + JobSatisfaction + 
                WorkLifeBalance + Age + BusinessTravel_Travel_Frequently + 
                BusinessTravel_Travel_Rarely + `EducationField_Life Sciences` + 
                EducationField_Marketing + EducationField_Medical + EducationField_Other + 
                `EducationField_Technical Degree` + `JobRole_Healthcare Representative` + 
                `JobRole_Human Resources` +  
                JobRole_Manager + `JobRole_Manufacturing Director` + 
                + MaritalStatus_Married + 
                MaritalStatus_Single + NumCompaniesWorked + 
                TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                YearsWithCurrManager + `Avg time spent`, family = "binomial", 
              data = train)
summary(model_7)

# Removing JobRole_Healthcare Representative 
model_8<- glm(formula = Attrition_No ~ EnvironmentSatisfaction + JobSatisfaction + 
                WorkLifeBalance + Age + BusinessTravel_Travel_Frequently + 
                BusinessTravel_Travel_Rarely + `EducationField_Life Sciences` + 
                EducationField_Marketing + EducationField_Medical + EducationField_Other + 
                `EducationField_Technical Degree` + 
                `JobRole_Human Resources` +  
                JobRole_Manager + `JobRole_Manufacturing Director` + 
                + MaritalStatus_Married + 
                MaritalStatus_Single + NumCompaniesWorked + 
                TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                YearsWithCurrManager + `Avg time spent`, family = "binomial", 
              data = train)
summary(model_8)

# Removing MaritalStatus_Married  
model_9<- glm(formula = Attrition_No ~ EnvironmentSatisfaction + JobSatisfaction + 
                WorkLifeBalance + Age + BusinessTravel_Travel_Frequently + 
                BusinessTravel_Travel_Rarely + `EducationField_Life Sciences` + 
                EducationField_Marketing + EducationField_Medical + EducationField_Other + 
                `EducationField_Technical Degree` + 
                `JobRole_Human Resources` +  
                JobRole_Manager + `JobRole_Manufacturing Director` + 
                + MaritalStatus_Single + NumCompaniesWorked + 
                TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                YearsWithCurrManager + `Avg time spent`, family = "binomial", 
              data = train)
summary(model_9)

# Removing JobRole_Human Resources   
model_10<- glm(formula = Attrition_No ~ EnvironmentSatisfaction + JobSatisfaction + 
                WorkLifeBalance + Age + BusinessTravel_Travel_Frequently + 
                BusinessTravel_Travel_Rarely + `EducationField_Life Sciences` + 
                EducationField_Marketing + EducationField_Medical + EducationField_Other + 
                `EducationField_Technical Degree` + 
                  JobRole_Manager + `JobRole_Manufacturing Director` + 
                + MaritalStatus_Single + NumCompaniesWorked + 
                TotalWorkingYears + TrainingTimesLastYear + YearsSinceLastPromotion + 
                YearsWithCurrManager + `Avg time spent`, family = "binomial", 
              data = train)
summary(model_10)

vif(model_10)

#####################################################

### Model Evaluation

### Test Data ####

#predicted probabilities of continuation(Non-attrition) for test data

test_pred = predict(model_10, type = "response", 
                    newdata = test[,-1])

summary(test_pred)

test$prob <- test_pred

test_pred_cont <- factor(ifelse(test_pred >= 0.50, "Yes", "No"))
test_actual_cont <- factor(ifelse(test$Attrition_No==1,"Yes","No"))

a<-table(test_actual_cont,test_pred_cont)
accuracy<-(a[1,1]+a[2,2])*100/(a[1,1]+a[1,2]+a[2,1]+a[2,2]) ## 85%
sensitivity<-a[2,2]/(a[2,1]+a[2,2])     ## 96%
specificity<-a[1,1]/(a[1,1]+a[1,2])     ## 29%
#######################################################

test_pred_cont <- factor(ifelse(test_pred >= 0.60, "Yes", "No"))

a<-table(test_actual_cont,test_pred_cont)
accuracy<-(a[1,1]+a[2,2])*100/(a[1,1]+a[1,2]+a[2,1]+a[2,2]) ## 85%
sensitivity<-a[2,2]/(a[2,1]+a[2,2])                         ## 94%
specificity<-a[1,1]/(a[1,1]+a[1,2])                         ## 42%

#######################################################

test_pred_cont <- factor(ifelse(test_pred >= 0.70, "Yes", "No"))

a<-table(test_actual_cont,test_pred_cont)
accuracy<-(a[1,1]+a[2,2])*100/(a[1,1]+a[1,2]+a[2,1]+a[2,2]) ## 83%
sensitivity<-a[2,2]/(a[2,1]+a[2,2])                         ## 88%
specificity<-a[1,1]/(a[1,1]+a[1,2])                         ## 55%

#######################################################

test_pred_cont <- factor(ifelse(test_pred >= 0.82, "Yes", "No"))

a<-table(test_actual_cont,test_pred_cont)
accuracy<-(a[1,1]+a[2,2])*100/(a[1,1]+a[1,2]+a[2,1]+a[2,2]) ## 76%
sensitivity<-a[2,2]/(a[2,1]+a[2,2])                         ## 75%
specificity<-a[1,1]/(a[1,1]+a[1,2])                         ## 77%

# Cut-off P value is 0.82 since it gives the best values of accuracy, sensitivity and specificity 
#######################################################

### KS -statistic - Test Data ######

test_cutoff_cont <- ifelse(test_pred_cont=="Yes",1,0)
test_actual_cont <- ifelse(test_actual_cont=="Yes",1,0)

install.packages('ROCR')
library(ROCR)
#on testing  data
pred_object_test<- prediction(test_cutoff_cont, test_actual_cont)

performance_measures_test<- performance(pred_object_test, "tpr", "fpr")

ks_table_test <- attr(performance_measures_test, "y.values")[[1]] - 
  (attr(performance_measures_test, "x.values")[[1]])

max(ks_table_test) # KS-statistics 52%



