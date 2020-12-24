Uber<- read.csv("uber Request Data.csv")
View(Uber)

#strsplit to separate date and time from the time stamps of request and drop
Uber$ReqDate <- sapply(strsplit(as.character(Uber$Request.timestamp), " "), "[", 1)
Uber$ReqTime <- sapply(strsplit(as.character(Uber$Request.timestamp), " "), "[", 2)
Uber$DropDate <- sapply(strsplit(as.character(Uber$Drop.timestamp), " NA"), "[", 1)
Uber$DropTime <- sapply(strsplit(as.character(Uber$Drop.timestamp), " "), "[", 2)

#as the date is in different formats in the columns to 
#bring all of them into a single syntax the code below is used
a <- as.Date(Uber$ReqDate,format="%d/%m/%Y") # Produces NA when format is not "%d-%m-%Y"
b <- as.Date(Uber$ReqDate,format="%d-%m-%Y") # Produces NA when format is not "%d/%m/%Y"
a[is.na(a)] <- b[!is.na(b)] # Combine both while keeping their ranks
Uber$ReqDate<- a # Put it back in the dataframe

#the same code for drop Time
c <- as.Date(Uber$DropDate,format="%d/%m/%Y") 
d <- as.Date(Uber$DropDate,format="%d-%m-%Y") 
c[is.na(c)] <- d[!is.na(d)] 
Uber$DropDate <- c

Uber$dayOfWeek<-weekdays((as.Date(Uber$ReqDate)))#extracting day of the week

#a graph to see if the day of the week is havng any impact on the problem
ggplot(Uber,aes(x=Uber$dayOfWeek,fill=Uber$Status))+geom_bar()

library(ggplot2)

ggplot(Uber,aes(x=Uber$Status))+geom_bar()#bar graph to see the count of statuses

ggplot(Uber,aes(x=Uber$Pickup.point,size=Uber$ReqHour,fill=Uber$Status))+geom_bar()
#a bar graph to see the ride status at airport and city


#A new column to extract the hour of the time for requestTime
Uber$ReqHour<-as.POSIXlt(Uber$ReqTime, format="%H:%M")$hour

#a bar chart to see hour wise distribution of the status of rides
ggplot(Uber,aes(x=Uber$ReqHour,fill=Uber$Status))+geom_bar() 




