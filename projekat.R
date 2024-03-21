
#Ucitavamo potrebne pakete

library(readr)
install.packages("tokenizers")
library(tokenizers)
install.packages("tidyverse")
library(tidyverse)
install.packages("tidytext")
library(tidytext)
install.packages("hcandersenr")
library(hcandersenr)
install.packages("stopwords")
library(stopwords)
install.packages("textstem")
library(textstem)
install.packages("stringr")
library(stringr)
install.packages("text2vec")
library(text2vec)
install.packages("tm")
library(tm)        #za 'text mining'
install.packages("SnowballC")
library(SnowballC) #za 'text steming'
install.packages("dplyr")
library(dplyr)     #za maniulaciju podataka
install.packages("ggplot2")
library(ggplot2)   #za vizualizaciju
install.packages("caret")
library(caret)     #za masinsko ucenje



#Ucitavamo bazu podataka

imdb_data <- read_csv("C:/Users/Aleksa/Documents/IMDB Dataset.csv")
head(imdb_data)

#kodiramo sentiment sa 0 i 1

imdb_data$sentiment<-ifelse(imdb_data$sentiment =="positive", 1, 0)
head(imdb_data)

#Pretprocesiranje

imdb_data <- imdb_data %>%
  mutate(review = gsub("[^[:alpha:][:space:]]", "", review), #uklanjamo interpukciju
         review = tolower(review),                           #pretvaramo u mala slova
         review = removeWords(review, stopwords("english")), #uklanjamo 'stopwords'
         review = wordStem(review)) 




#Delimo na test i trening skup
set.seed(1234)
split <- createDataPartition(imdb_data$sentiment, p = 0.8, list = FALSE)
train_data <- imdb_data[split, ]
test_data <- imdb_data[-split, ]


#cuvamo odvojeno kritike i sentiment
text <- imdb_data$review
sent <- imdb_data$sentiment

# kreiramo document term matrix, da prebroji frekvencije scakog termina u svakom dokumentu
dtm1 <- DocumentTermMatrix(text)
#kreiramo tf-idf matricu
tfidf1 <- weightTfIdf(dtm1)
#dodajemo kolonu sentimenata
tfidf1 <- cbind(tidy(tfidf1), sent)
matr <- tidy(tfidf1)




corpus <- Corpus(VectorSource(imdb_data$review))  #kreiramo korpus
dtm2 <- DocumentTermMatrix(corpus)                # kreiramo dtm matricu
tfidf2 <- weightTfIdf(dtm2)                       
sparse <- removeSparseTerms(tfidf2, 0.99)         #izdvajamo cesto ponavljane reci
finalWords <- tidy(notSparse)





#Pretvaramo tekstualne podatke u matricu
dtm <- TermDocumentMatrix(train_data, control=list(wordLengths = c(1, Inf)))
#Kreiramo sparse matricu od document term matrice
sparse_dtm <- removeSparseTerms(dtm, 0.995)
data1 <- tidy(sparse_dtm)
data1
head(data1)


#Treniramo model masinskog ucenja koristeci trening skup
model <- train(sentiment ~., data = data1, method = "glm", family = "binomial")

# Pravimo predvidjanje na test skupu
predictions <- predict(model, newdata = as.data.frame(TermDocumentMatrix(test_data, control = list(distionary = Terms(dtm)))))

#Evaluiramo performanse modela
confusionMatrix(predictions, test_data$sentiment)

