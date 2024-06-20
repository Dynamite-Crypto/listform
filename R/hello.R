formule <- "# Vettore medie: colMeans(dati)
# n=nrow(dati)
# medie=t(dati)%*% rep(1,n)/n
# matrice di correlazione: → cor(dati)
# uno=matrix(rep(1,n),ncol=1)
# X=as.matrix(dati)
# S=(1/n)*t(X-uno%*%t(medie))%*%(X-uno%*%t(medie)) matrice di covarianza =var(dati)
# Cor=diag(1/sqrt(diag(S)))
# R=cor%*% S %*% cor
# Varianza: sum(diag(S))
# Oppure con autovalori:
#   eigs=eigen(S)
# eigs$values
# eigs&vectors
# sum(eigs$values)
# Varianza generalizzata: det(S)
# Oppure(con comandi sopra:) prod(eigs$values)
# Trasformare i dati in dataset. Definire matrice a e vettore b per trasformaz lineare
# A=matrix(c(……rep(0,7)…), byrow=T, nrow=5))
# Datin=as.matrix(dati)%*%t(A)
# b=matrix(c(-9,-6),ncol=1)
# Y=as.matrix(dati)%*%t(A)+rep(1,n)%*%t(b)
# Valore della funzione densit‡ della normale descritta in consegna nel generico punto x0=(2,4,6)
# sigma=matrix(c(1,1/2,-3/4,1/2,1,-1/3,-3/4,-1/3,1),nrow=3)
# mu=c(3,2,5)
# library(mvtnorm)
# dmvnorm(c(2,4,6),mean=mu,sigma=sigma)
# probabilit‡ P(x<=x0)
# pmvnorm(upper=c(2,4,6),mean=mu, sigma=sigma)
# sotto ipotesi di NORMALE BIVARIATA:
#   stimare media:
#   n=nnrow(dati) opuure dim(dati)[1]
# mu=colMeans(dati)
# mu=(1/n)*t(dati)%*%rep(1/n)
# stimare varianza (campionaria):
#   sigmacapp=(1/n)*(t(dati)-as.numeric(mu))%*%t(t(dati)-as.numeric(mu))
# var=var(dati)*(n-1)/n
# testare ipotesi H0:µ=µ0 con µ0=(175,15) con quantit‡ T2 HOTELLING
# n=dim(dati)[1]
# mu=apply(dati[,7:9],2,mean)
# m=c(175,15)
# S=var(dati[,7:9])
# P=ncol(dati[,7:9]
#        T2=n*t(mu-m)%*%solve(S)%*%(mu-m)
#        T2
#        (n-1)*p/(n-p)*qf(0.95,p,n-p)
#        Pval=a-pf(T2*(n-p)/(n-1)/p,0.99,p,n-p)
#        Rifiuto se ..
#        ““
#        CALCOLARE IC (NO ASSUNZIONE DI DISTRIBUZIONE)
#        Alfa=0.01
#        Tit=dati[,2:7]
#        Medie=apply(tit,2,mean)
#        S=var(dati[,2:7])
#        N=nrow(dati)
#        P=ncol(tit)
#        BONFERRONI
#        Bonf(i)=medie[i]+c(-1,1)*qt(1-(alfa/2)/p,n-1)*sqrt(S[i,i]/n)[1]
#        IC SIM:
#          sim(i)=medie[i]+c(-1,1)*sqrt(p*(n-1)/(n-p)*qf(1-alfa,p,n-p)*S[i,i]/n)[1]
#        singoli t: t.test(dati)
#        T2 HOTELLING:
#          hot=medie[i]+c(-1,1)*sqrt(p*(n-1)/(n-p)*qf(1-alfa,p,n-p)*S[i,i]/n)[1]
#        PCA: identifica combinazioni lineare ortogonali di variab origin,da usare a fin descrit o x sostiuire le variab origin con
#        un numero pi˘ piccolo
#        str(dati) quante variab
#        Pairs(dati) grafico di corr
#        pca=prcomp(dati)
#        summary(pca)
#        pca$rotation
#        plot(pca, type=“l”)
#        var(eco)
#        pcas=prcomp(dati,scal.=T)
#        summary(pcas)
#        plot(pcas, type=“l”)
#        cor(dati, pcas$x)
#        biplot(pcas)
#        n° max di PC che pox be estratte da un dataset Ë uguale al minimo tra il numero di righe ed il numero di colonne del
#        dataset. il criterio per una buona proiezione nella PCA Ë una elevata varianza per quella proiezione, tengo quindi
#        quelle PC con varianze elevate. Problema coinvolge i valori degli autovalori di matrice di varianza/cov campionaria.
#        Scelgo in base a criterio di:varianza spiegata: si tiene un n° di comp suff a riprodurre una grande % specificata della
#        variabilit‡ complessiva di variab originarie. Sn suggeriti valori compresi tra il 70% e il 90%. scree plot:Ë un grafico k
#        rappr autovalori ordinati in senso decresc verso il loro n° d’ordinE(assex). Se i pi˘ grandi autovalori campi dominano in
#        dimensione, ed i rimanenti autovalori campionari sono molto piccoli, allora lo scree plot mostrer‡ un “gomito” nel
#        grafico in corrispondenza alla divisione tra “grandi” e “piccoli” valori degli autovalori campionari. Posizione in cui si
#        presenta gomito, puÚ be usata come n° di PC da tenere.
#        Da pairs vedo variabili originariamente correlate e qnd ha senso elaborare dati con PCA. Standariz per dare peso
#        uguale alle variabili
#        Dall’output possiamo vedere che le prime 3 componenti spiegano più dell’84%
# Dal summary(pca), in particolare in proportion of variance,si nota come la 1a comp princ da sola spieghi quasi il 73%
# della var tot, mentre prime due bastano a spiegare pi˘ del 98% di var tot. La matrice di rotazione mostra che la 1comp
# prin riflette essenzi la variabile (quella con valore + alto in 1° colonna), mentre la 2° comp princ riflette... tuttavia,
# controllando la var si nota che ce molta differenza in scala di variab analizzate, x questo si procede con analisi di dati
# standard prima di applicare pca. Ora la 1° comp princ spiega il%, la 2° spiega il..% della σ tot pertanto le prime due
# comp princ spiegano il 74% della σ tot. Da screeplot si può ritenere che le prime 3 cp, che raccolgono l’86% della σ tot
# possano bastare per riassum i dati. Attrav il biplot si vede che .. e .. sono le variaibli pi˘ corr, mentre .. e .. formano un
# angolo da 90° e quindi hanno correlazione nulla. Si vede anche che la prima cp Ë formata da (variab a sx), mentre la
# seconda da..
# Lo scree plot sembra avere 2 gomiti. La propor di σ cumula porterebbe a scegliere 3 cp, mentre la σ prop farebbe
# tenere le prime 2, pertanto 2 comp potrebbe essere suff. Biplot nota rendimenti corr negativam con la 1 cp
# FA: ipotizza che le variab origin possano essere modellate come combinaz lineari di un inx ridotto di variab causali non
# osservab, aka fattori comuni. I coeff di combinaz linea sono laoding fattoriali. Richiede livello minimo di correlaz tra le
# variab. Factanal usa max ver
# analisi a 1 fact:
#   se è accettabile l’ipotesi di normalità multivariata, per determinare numero ragionevole di fattori da estrarr posso
# usare il test di rapporto di ver fornito dalla funzione factanal()
# fa1= factanal(dati, factors=1, rotation=“none”), faccio con 2
# guardo p value e gradi di libert‡, tendenzialmente mi aspetto che 2 fatt spieghino pi˘ variabilit‡ comune. Noto che
# uniquenesses cambiano al variare del n° di fattori. Guardo PROPORT varia (0.704): in questo caso il 70.4% della
# variabilit‡ dipende dai 2 fattori comuni
# RUOTO I FATTORI PER INTERPRETARE I DATI (VARIMAX di default)
# faR=factanal(dati, factors=2)
# pval= sapply(1:7, function(nf) factanal(dati, factors = nf)$PVAL)
# names(pval)=sapply(1:7, function(nf) paste0(“nf=“,nf))
# questi valori suggeriscono che la soluzione 4 fattori fornisce una adeguata descrizione. Scrivo >fa4=factanal(dati,
#                                                                                                               factors=4)
# le variab che pesano di pi˘ sul primo fattorie sono ..elenco di loadings primo fact con %>50. Sul secondo fattore
# pesano di pi˘.. i fattori rimanenti pesano principalmente su una sola variabile
# scegliendo 2 fatt sono in grado di spiegare il 42% (cumu varia sotto factor2) della varianza/cov complessiva delle
# variabi origin, ma cmq guardando loading fattoriali e unicit‡ segnalano che gran parte della variabilit‡ per molte variab
# resta non spiegata. Quindi questa prima soluzione non Ë appropriata
# errori:
#   round(R<<-cor(dati),4) 4 sono n° variab da 7 a 10
# round(R1<<-fa1$loadings %*% t(fa1&loadings)+diag(fa1$ uniquenesses),4)
# sum((R-R1)^2)
# specificit‡:
#   fa2$uniquenesses
# faR$uniquenesses (vedo che rimangono uguali)
# fa2$PVAL
# faR$PVAL
# grafico:
#   fa2=factanal (dati, factors=2, scores=“regression o bartlett”)
# plot(fa2$scores)
# pca con analisi dei fattori:
# eis=cor(dati)
# ee=eigen(eis)
# loads=(ee$vectors%*%diag(sqrt(ee$values)))[,1:4]
# comm=diag(loads%*%t(loads))
# spec=diag(eis)-comm
# quanti fattori sono necessari?
#   ee$values/5
# cumsum(ee$values)/5
# [1] 0.7030034 0.8114055 0.8898527 0.9567353 1.0000000
# NE SCELGO DUE
# spec1=1-diag(loads[,1]%*%t(loads[,1]))
# CON 2 FATTORI:
#   spec2=1-diag(loads[,1:2]%*%t(loads[,1:2]))
# spec2
# errori:
#   R=cor(dati[,7:11])
# sum((R-loads[,1]%*%t(loads[,1])-diag(spec1))^2)
# sum((R-loads[,1:2]%*%t(loads[,1:2])-diag(spec2))^2)
# CCA: unire due vettori
# Library(cca)
# Library(corrplot)
# Se ho 2 vettori: colonne
# X=as.matrix(,dati[,c(4,3)])
# Y=as.matrix(,dati[-c(3,4)])
# Se devo aggregare colonne per avere due vettori:
#   X=dati[,c(2,3,4,5,)]
# Y=dati[,c(10,12,16)]
# rho=matcor(X,Y)
# img.matcor(rho, type=2)
# cca=cc(X,Y) summary(cca)
# cca$cor cca$xcoef cca$ycoef
# cca$scores$corr.X.xscores →corr(X,u)
# cca$scores$corr.Y.xscores →corr(Y,u)
# cca$scores$corr.Y.yscores →corr(X,v)
# cca$scores$corr.X.yscores →corr(X,v)
# plot.cc(cca, σ.label=T)
# guardo cca$xcoef e se ha basse % dico che prima correlaz can non risulta particolarm elevata
# guardo X.x e vedo se c’è correlaz tra prima colonna (variab can X) e le due righe(variabili) e dico che Ë alta (Ë pi˘ alta
#                                                                                                                 in X.x che con le variabili di Y.x)
# guardo Y.x se hanno risultati elevati tutte le righe dico che sono il gruppo di variabili correlate maggiorm anche se in
# direz opposta con le due variab di prima in riga
# guardo anche le ultime due fx e se sono basse dico che seconda variab can bassa bassa correlaz con le variab di
# partenza/ non aggiunge molta info
# CLUSTER:analisi dei gruppi con metodo gerarch agglomerativo distanza euclidea e diversi tipi di legame.
# d=dist(dati)
# hcc=hclust(d, method=“complete”)
# hcs=hclust(d method=“single”)
# hca=hclust(d, method=“average”)
# hcw=hclust(d, method=“ward.D2”)
# par(mfrow=c(2,2))
# plot(hcc)
# plot(hcs)…
# il metodo del legame singolo tende a creare grandi gruppi aggiungendo in successione item a gruppi gi‡ creati. Gli altri
# 3 metodi ritornano simili tra loro con cluster pi˘ strutturati
# confrontare ward con risultati ottenuti con distanza euc calcolata su dati stand
# ds=dist(scale(dati))
# par(mfrow=c(1,1))
# new=hclust(ds, method=“ward.D2”)
# plot(new)
# dat=dati[,1:7]
# d=dist(dat)
# hcc=hclust(d, method=“complete”)
# hcs=hclust(d, method=“single”)
# hca=hclust(d, method=“average”)
# hcw=hclust(d, method=“ward.D2”)
# plot(hcc,labels=dati[,8])
# rect.hclust(hcc,k=3,border=“red”)
# rect.hclust(hcw, k=3, border=“red”)
# cutree(hcc, k=3)
# par(mfrow=c(2,2))
# plot(cutree(hcc, k=3), col=dati[,8])
# plot(cutree(hcs, k=3), col=dati[,8])
# plot(cutree(hcw, k=3), col=dati[,8])
# plot(cutree(hca, k=3), col=dati[,8])
# se scelgo numero gruppi: dendogramma, k= numero gruppo
# par(mfrow=c(2,2))
# plot(cutree(hcc,k=3, col=dati, main=“complete”))
# …
# silhouette:
#   library(cluster)
# par(mfrow=c(2,2))
# plot(silhouette(cutree(hcw, k=2), d))
# clusplot(dati,cutree(hca,k=3))
# plot(silhouette(cutree(hca,k=3),d))
# # distanza media dell osservazione 1 con le altre del suo cluster
# d0 <- sum(dm[which(cl==1),1])/80
# # distanza media dell oss 1 con quelle del cluster 2
# d1 <- sum(dm[which(cl==2),1])/64
# # distanza media dell
# oss 1 con quelle del cluster 3
# d2 <- sum(dm[which(cl==3),1])/65
# # silhouette per losservazione 1
# (d2-d0)/d2
# silhouette(cutree(hca, k=3), seed.dist)[1,]
# ….
# il clustering gerarchico aggregativo produce raggruppamenti di variab seguendo un processo gerarchico: parte da tutti
# gli item separati in gruppi individuali e procede iterativamente con l’aggregazione di coppie di gruppi che si somigliano
# di pi˘ arrivando al termine a produrre un unico gruppo contenente tutti gli item
# metodo di partizione:
#   dati=read.table()
# km=kmeans(dati,centers=2)
# km$centers →centroidi dei due cluster
# km$totss →info su devianza totale, su quella interna ai singoli cluster e su quella tra cluster
# plot(dati[,6:7], col=km2$cluster,pch=20)
# FARE LO STESSO CON 3 CLUSTER: vedo quanta % di dev viene spiegata da una diversa numerosit‡ dei cluster e
# cerchiamo di capire quando l’incremento marginale sia trascurabile: faccio center=6
# km1 <- kmeans(seed[,1:7], centers=1)
# km4 <- kmeans(seed[,1:7], centers=4)
# km5 <- kmeans(seed[,1:7], centers=5)
# km6 <- kmeans(seed[,1:7], centers=6)
# explained.var <- c(km1$betweenss/km1$totss, km2$betweenss/km2$totss,
#                    km3$betweenss/km3$totss, km4$betweenss/km4$totss,
#                    km5$betweenss/km5$totss, km6$betweenss/km6$totss)
# plot(c(1:6), explained.var, xlab=“N. clusters”,
#      ylab=“% varianza spiegata”, type=“l”)"
hello <- function() {
  cat(formule,"\n")
}
