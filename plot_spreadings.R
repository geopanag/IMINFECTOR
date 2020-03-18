  library(ggplot2)
library(reshape2)

#------------- Plot weibo
setwd("/Weibo/spreading")

df  = data.frame(matrix(ncol=10,nrow=0))
nam = c()
for(f in dir(pattern="*.txt")){
  dat = read.csv(f)
  dat = dat[order(dat[,2]),]
  df = rbind(df,dat[,1])
  nam = c(nam,f)  
  #}
}

x = 1000
step = 100
names(df) = seq(step,x,step)

nam = c("AVG Cascade Size","IMINFECTOR","IMM DB","K-cores")
dni_means = apply(df,1,mean)
df$Method = factor(nam,nam[order(dni_means,decreasing=T)])
reshaped = melt(df,id=c("Method"))

reshaped$variable = as.numeric(as.character(reshaped$variable))


#------------- Plot Digg
setwd("/Digg/spreading")

df  = data.frame(matrix(ncol=10,nrow=0))
nam = c()
for(f in dir(pattern="*.txt")){
  bool = F
  dat = read.csv(f)
  dat = dat[order(dat[,2]),]
  df = rbind(df,dat[,1])
  nam = c(nam,f)  
  #}
}

df
x = 50
step = 10
names(df) = seq(step,x,step)


nam = c("AVG Cascade Size","Credit Distribution","IMINFECTOR","IMM DB","IMM Inf2vec","K-cores","Simpath DB","Simpath Inf2vec")
dni_means = apply(df,1,mean) 
df$Method = factor(nam,nam[order(dni_means,decreasing=T)])

df[1,1:5] = as.numeric(df[1,4:8])
#df[6,5]=NaN
df = df[,-c(6:10)]

reshaped2 = melt(df,id=c("Method"))

reshaped2$variable = as.numeric(as.character(reshaped2$variable))


#------------ MAG
setwd("/MAG/spreading")

df  = data.frame(matrix(ncol=10,nrow=0))
nam = c()
for(f in dir(pattern="*.txt")){
  dat = read.csv(f)
  #dat=dat[,c(2,1)]
#  write.csv(dat,f)
  dat = dat[order(dat[,1]),]
  df = rbind(df,dat[,1])
  nam = c(nam,f)  
}


x = 10000
step = 1000
names(df) = seq(step,x,step)
nam = c("AVG Cascade Size","Credit Distribution","IMINFECTOR","IMM DB","IMM Inf2vec","K-cores","Simpath DB")
dni_means = apply(df,1,mean) 
df$Method = factor(nam,nam[order(dni_means,decreasing=T)])
reshaped3 = melt(df,id=c("Method"))

reshaped3$variable = as.numeric(as.character(reshaped3$variable))


options(scipen = 999)


#------------ Plot everything
reshaped3$dataset = "MAG"
reshaped2$dataset = "Digg"
reshaped$dataset = "Weibo"

res = rbind(reshaped3,reshaped2,reshaped)

res = res[order(res$Method, decreasing=TRUE),]

res$dataset = factor(res$dataset,levels=c("Digg","MAG","Weibo"))
res$Method = factor(res$Method, levels = c("IMINFECTOR","Credit Distribution","IMM DB","IMM Inf2vec","Simpath DB","Simpath Inf2vec","AVG Cascade Size","K-cores"))

ggplot(res,aes(x=variable,y=value,color=Method))+geom_line(size=1,position=position_dodge(width=0.3))+
  geom_point(size=3,aes(shape = Method))+xlab("Seed Set Size")+ylab("DNI")+
  facet_wrap(~dataset,scales = "free")+theme_bw()+
  theme(plot.margin=unit(c(0.7,0.7,0.7,1),"cm"),legend.position="top",text = element_text(size=18),legend.title=element_blank(),
        strip.text = element_text(size=15,face="bold"),legend.text= element_text(size=15),axis.text = element_text(size=14))+
  scale_shape_manual(values=c(17,16,15,3,6,10,12,11))+
  scale_color_manual(values=c("#E69F00","steelblue","darkgreen","brown","black","cyan","red","gray"))


ggsave("results_spreading.pdf")



