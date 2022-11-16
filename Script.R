
library(readxl)
library(dplyr)
library(ggplot2)
library(tidyr)
library(corrplot)
library(stringr)
library(randomForest)
library(caret)

df <- read.table('https://raw.githubusercontent.com/rosakavalco/Machine-Learning---Consumo-de-Carros-Eletricos/main/FEV-data-Excel.csv',
              header=TRUE, sep=";", na.strings = NA, encoding = "UTF-8", dec=",")

names(df) <- make.names(names(df), unique=TRUE)
View(df)
str(df)

# Vamos verificar quais colunas contém valores NA

Contem_NA <- names(which(colSums(is.na(df))>0))
Contem_NA

# Vamos retirar da base os itens com NA no Target.

df = df[!is.na(df$mean...Energy.consumption..kWh.100.km.),]

# Vamos criar uma coluna `cm3` para resumir Width, Heigth e Length, e então
# fazer uma matriz de correlação para verificar se podemos retirar alguma coluna do dataframe

df$cm3 = df$Width..cm. * df$Height..cm. * df$Length..cm.
df$Width..cm. = NULL
df$Height..cm. = NULL
df$Length..cm. = NULL

numeric_df = select_if(df, is.numeric)
numeric_df = na.omit(numeric_df)
corr_numeric = cor(numeric_df)
corrplot(corr_numeric,type="upper", tl.srt=45, tl.cex = .8, method="color",addCoef.col = "grey", number.cex=0.6)


# Baseado no corrplot, vamos fazer o primeiro modelo utilizando apenas a variável 
# numérica Minimal Empty Weight, já que ela apresenta alta correlação com o 
# consumo médio e não possui valores NA.
#
# Vamos verificar agora as variáveis categóricas `Type of brakes` e `Drive type`

ggplot(df, 
       aes(x=Type.of.brakes, y=mean...Energy.consumption..kWh.100.km.)) +
  geom_boxplot()

ggplot(df, 
       aes(x=Drive.type, y=mean...Energy.consumption..kWh.100.km.)) +
  geom_boxplot()

# Ambas as variáveis parecem importantes. Vamos alterar o Drive Type para mostrar
# Somente 2WD e 4WD, excluindo o "(front)" e o "(rear)".
# Além disso, Type of brakes possui valores NA. Vamos preencher com a moda ("disc (front + rear)").

### Início do pré-processamento ###

# Além das alterações mencionadas nos passos anteriores, vamos retirar do dataframe
# as linhas onde o Target está com NA, já que não é útil para treinar e testar o modelo.
# Vamos retirar também as colunas referentes ao modelo do carro.

df = df %>% 
  mutate(Drive.type = ifelse(Drive.type == "2WD (front)", "2WD", Drive.type)) %>%
  mutate(Drive.type = ifelse(Drive.type == "2WD (rear)", "2WD", Drive.type)) %>%
  mutate(Type.of.brakes = ifelse(is.na(Type.of.brakes), "disc (front + rear)", Type.of.brakes)) %>%
  select(-names(which(colSums(is.na(df))>0)), -X.U.FEFF.Car.full.name, -Make, -Model)

# Vamos dividir o dataset em treino e teste (80/20)

tamanho_treino = floor(0.8 * nrow(df))
set.seed(124)
train_ind = sample(seq_len(nrow(df)), size = tamanho_treino)

df_treino = df[train_ind,]
df_teste = df[-train_ind,]

# Agora vamos padronizar o dataset de treino e, com os mesmos parâmetros, o de teste

dummy = dummyVars(" ~ Drive.type", data=df_treino)
Normalizer = preProcess(df_treino, method = "range")

df_treino = cbind(predict(dummy,df_treino), df_treino)
df_treino = subset(df_treino, select=-Drive.type)
df_treino = predict(Normalizer, df_treino)

df_teste = cbind(predict(dummy,df_teste), df_teste)
df_teste = subset(df_teste, select=-Drive.type)
df_teste = predict(Normalizer, df_teste)

x_treino = df_treino %>%
  select(-mean...Energy.consumption..kWh.100.km.)
y_treino = df_treino$mean...Energy.consumption..kWh.100.km.

x_teste = df_teste %>%
  select(-mean...Energy.consumption..kWh.100.km.)
y_teste = df_teste$mean...Energy.consumption..kWh.100.km.

norm_df = rbind(df_teste,df_treino)
train_control <- trainControl(method="repeatedcv",
                              number=10, repeats = 3)

### Treinamento dos modelos ###
#
# Como forma de contornar o tamanho limitado de nosso dataset, vamos utilizar o
# Repeated K-fold cross-validation.
#
# Primeiro modelo --- Regressão linear utilizando o `Minimal empty weight`

set.seed(124)
model1 = train(mean...Energy.consumption..kWh.100.km. ~ Minimal.empty.weight..kg., 
               method="lm",
               data = norm_df,
               trControl=train_control)
print(model1)

# Segundo e terceiro modelos - Vamos utilizar Random Forest para selecionar as 4 principais variáveis explicativas.
# Com essas variáveis vamos fazer um modelo com Regressão Linear, e um com Random Forest. 

set.seed(124)
rf <- randomForest(as.formula("mean...Energy.consumption..kWh.100.km. ~ ."), data = norm_df)
rf_importance <- data.frame(importance(rf))
rf_importance$feature = unlist(row.names(rf_importance))
rf_importance <- rf_importance[order(rf_importance$IncNodePurity),]

top4features = select(tail(rf_importance, 4), feature)

string_model_2_3 = paste(unlist(top4features), collapse = " + ")
string_model_2_3 = paste("mean...Energy.consumption..kWh.100.km. ~ ", string_model_2_3)

# Segundo Modelo --- Vamos testar essa seleção de variáveis com regressão linear e comparar o desempenho com o Benchmark

set.seed(124)
model2 = train(as.formula(string_model_2_3), data = norm_df,
               method="lm",
               trControl=train_control)
print(model2)

# Terceiro Modelo --- Vamos testar a mesma seleção de variáveis com Random Forest

set.seed(124)
model3 = train(as.formula(string_model_2_3), data = norm_df,
               method="rf",
               trControl=train_control)
print(model3)

##### Conclusões #####################
#
# Apesar do conjunto de dados ser muito rico em variáveis, temos muitas variáveis correlacionadas 
# entre si. Além disso, temos poucos itens no dataset, o que resulta em uma grande variação
# no desempenho do modelo conforme se cria os conjuntos de treino e teste. 
#
# Como forma de contornar esses problemas, opta-se por utilizar Cross Validation e Feature Selection.
# 
# É possível observar que o benchmark já possui um desempenho razoável, com Rsquared = 0,77. Entretanto,
# utilizando mais variáveis explicativas, nosso indicador aumenta.
#
# O segundo modelo apresenta Rsquared = 0,81 e o terceiro 0,86.
# 
# Com base nas informações acima, o melhor modelo a ser utilizado é o 3.