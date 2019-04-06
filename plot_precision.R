library(ggplot2)
library(reshape2)


#------------- Plot weibo
setwd("Path/To/Weibo/")

d = read.csv("accuracy.txt",header=FALSE)
feats = c(as.character(d[1,]),as.character(d[12,]),as.character(d[23,]),as.character(d[34,]))
feats = unlist(lapply(feats,function(x) gsub("_seeds.txt","",tail(unlist(strsplit(x,"/")), 1))))
vals = list(as.numeric(as.character(d[2:11,])),as.numeric(as.character(d[13:22,])),as.numeric(as.character(d[24:33,])),
            as.numeric(as.character(d[35:44,]))) #,as.numeric(as.character(d[46:55,])))

df = data.frame(vals)
names(df) = feats
reshaped1 = melt(df)
reshaped1$Method = as.character(reshaped1$variable)
step1=100
reshaped1$variable=rep(seq(step1,step1*10,step1),4)


#------------- Plot digg
setwd("Path/To/Digg/")

d = read.csv("accuracy.txt",header=FALSE)
feats = c(as.character(d[1,]),as.character(d[12,]),as.character(d[23,]),as.character(d[34,]))
feats = unlist(lapply(feats,function(x) gsub("_seeds.txt","",tail(unlist(strsplit(x,"/")), 1))))
vals = list(as.numeric(as.character(d[2:11,])),as.numeric(as.character(d[13:22,])),as.numeric(as.character(d[24:33,])),
            as.numeric(as.character(d[35:44,]))) #,as.numeric(as.character(d[46:55,])))

df = data.frame(vals)
names(df) = feats
reshaped2 = melt(df)
reshaped2$Method = as.character(reshaped2$variable)
step2=10
reshaped2$variable=rep(seq(step2,step2*10,step2),4)


#------------- Plot MAG
setwd("Path/To/MAG/")

d = read.csv("accuracy.txt",header=FALSE)
feats = c(as.character(d[1,]),as.character(d[12,]),as.character(d[23,]),as.character(d[34,]))
feats = unlist(lapply(feats,function(x) gsub("_seeds.txt","",tail(unlist(strsplit(x,"/")), 1))))
vals = list(as.numeric(as.character(d[2:11,])),as.numeric(as.character(d[13:22,])),as.numeric(as.character(d[24:33,])),
            as.numeric(as.character(d[35:44,]))) #,as.numeric(as.character(d[46:55,])))

df = data.frame(vals)
names(df) = feats
reshaped3 = melt(df)
reshaped3$Method = as.character(reshaped3$variable)
step3=1000
reshaped3$variable=rep(seq(step3,step3*10,step3),4)



#------------ Plot everything
reshaped3$dataset="MAG"
reshaped2$dataset="Digg"
reshaped1$dataset="Weibo"
res = rbind(reshaped3,reshaped2,reshaped1)

unique(res$Method)
res$Method[res$Method=="final"]="IMINFECTOR"
#res = res[-which(res$Method=="fun"),]
res$Method[res$Method=="imm"]="IMM"
res$Method[res$Method=="avg_cascades_size"]="AVG Cascade Size"
res$Method[res$Method=="kcores"]="K-cores"
res$dataset = factor(res$dataset,levels=c("Digg","Weibo","MAG"))
res$Method = factor(res$Method, levels = c("IMINFECTOR","IMM","AVG Cascade Size","K-cores"))

#d = res[res$dataset=="MAG",]
ggplot(res,aes(x=variable,y=value,color=Method))+geom_line(size=1)+
  geom_point(size=3,aes(shape = Method))+xlab("Seed Set Size")+ylab("Precision")+
  facet_wrap(~dataset,scales = "free")+theme_bw()+
  theme(plot.margin=unit(c(0.7,0.7,0.7,1),"cm"),legend.position="top",text = element_text(size=18),legend.title=element_blank(),
        strip.text = element_text(size=15,face="bold"),legend.text= element_text(size=15),axis.text = element_text(size=14))+
  scale_shape_manual(values=c(17,16,15,3))+
  scale_color_manual(values=c("#E69F00","steelblue","darkgreen","brown"))#,"black","cyan","yellow","gray"))

ggsave("../../Figures/results_precision.pdf")
