setwd('C:\\Users\\leoka\\Documents\\DataScienceAcademy\\BigDataRAzure\\Projetos')

library(readxl)
library(dplyr)
library(ggplot2)
library(tidyr)
library(corrplot)
library(stringr)
library(caret)

df <- read_excel('FEV-data-Excel.xlsx')
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
  select(-names(which(colSums(is.na(df))>0)), -Car.full.name, -Make, -Model)

# Vamos dividir o dataset em treino e teste (80/20)

tamanho_treino = floor(0.8 * nrow(df))
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


str(df_treino)
View(df_treino)

x_treino = df_treino %>%
  select(-mean...Energy.consumption..kWh.100.km.)
y_treino = df_treino$mean...Energy.consumption..kWh.100.km.

x_teste = df_teste %>%
  select(-mean...Energy.consumption..kWh.100.km.)
y_teste = df_teste$mean...Energy.consumption..kWh.100.km.


### Treinamento dos modelos ###

# Primeiro modelo - regressão linear utilizando o `Drive type` e o `Minimal empty weight`

model1 = lm("mean...Energy.consumption..kWh.100.km. ~ Drive.type2WD + Minimal.empty.weight..kg.", data = df_treino)
summary(model1)
postResample(pred=predict(model1, x_teste), y_teste)

# Segundo modelo - Vamos utilizar RFE para selecionar as variáveis explicativas da regressão linear

control <- rfeControl(functions = rfFuncs, # random forest
                      method = "repeatedcv", # repeated cv
                      repeats = 5, # number of repeats
                      number = 10) # number of folds

result_rfe <- rfe(x=x_treino,
                  y=y_treino,
                  sizes=c(1:ncol(x_treino)),
                  rfeControl = control)

?rfeControl

selected_cols = predictors(result_rfe)
selected_cols

# Vamos testar essa seleção com regressão linear e comparar o desempenho com o Benchmark

string_lm = paste(selected_cols, collapse = " + ")
string_lm = paste("mean...Energy.consumption..kWh.100.km. ~ ", string_lm)

model2 = lm(string_lm, data = df_treino)
summary(model2)
postResample(pred=predict(model2, x_teste), y_teste)

##### Conclusões #####
# É possível observar que o benchmark tem indicadores piores no summary, porém quando utilizamos os
# dados de teste o desempenho tende a ser melhor.
# O segundo modelo, por outro lado, apresenta bons indicadores no summary, mas seu poder de generalização
# é menor. Como consequência, o desempenho com os dados de teste não superam o benchmark.
#
# O melhor modelo a ser utilizado é o 1.