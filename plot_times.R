


#---------------------- Plot computational time
df_digg = data.frame(step = factor(rep(c("Preprocessing","Training","Algorithm"),each=6),levels=c("Preprocessing","Training","Algorithm")),
                     model = factor(rep(c("IMINFECTOR","Credit\nDistribution","IMM\nDB","SimPath\nDB","IMM\nInf2vec","SimPath\nInf2vec"),3),levels=c("IMINFECTOR","Credit\nDistribution","IMM\nDB","SimPath\nDB","IMM\nInf2vec","SimPath\nInf2vec")),
                     time = c(4.77, #iminfector preprcessing
                              2321, # cd  
                              1768, # db
                              1768, # db
                              2285, # inf2vec
                              2285, # inf2vec
                              436, #infector training
                              0, #cd
                              0,#db
                              0,#db
                              25926,#inf2vec
                              25926,#inf2vec
                              22, #iminfector algorithm
                              2460,  #cd
                              86,  #imm
                              5.6,  #simpath
                              320,#imm  (16482 with 0.01)
                              6184  #simpath  
                     ))


#df_mag = data.frame(step = factor(rep(c("Preprocessing","Training","Algorithm"),each=5),levels = c("Preprocessing","Training","Algorithm")),
#                    model = factor(rep(c("IMINFECTOR","IMM\nDB","IMM\nInf2vec","Credit\nDistribution","SimPath\nDB"),3),levels=c("IMINFECTOR","Credit\nDistribution","IMM\nDB","IMM\nInf2vec","SimPath\nDB")),
df_mag = data.frame(step = factor(rep(c("Preprocessing","Training","Algorithm"),each=6),levels=c("Preprocessing","Training","Algorithm")),
                    model = factor(rep(c("IMINFECTOR","Credit\nDistribution","IMM\nDB","SimPath\nDB","IMM\nInf2vec","SimPath\nInf2vec"),3),levels=c("IMINFECTOR","Credit\nDistribution","IMM\nDB","SimPath\nDB","IMM\nInf2vec","SimPath\nInf2vec")),
                    time = c(57, #iminfector preprocessing
                             7414, # cd  
                             21892, # imm db
                             21892, # simpath db
                             44952, # imm inf2vec 
                             0, # simpath inf2vec did not scale
                             919, #infector training
                             0, #cd
                             0,#db
                             0,#db
                             173915,#94602,inf2vec
                             0,#inf2vec
                             9768, #iminfector algorithm
                             1560,  #cd 
                             80, # imm db
                             61680,  #simpath db
                             879, # imm inf2vec
                             0 #simpath inf2vec
                             ))

df_weibo = data.frame(step = factor(rep(c("Preprocessing","Training","Algorithm"),each=6),levels=c("Preprocessing","Training","Algorithm")),
                    model = factor(rep(c("IMINFECTOR","Credit\nDistribution","IMM\nDB","SimPath\nDB","IMM\nInf2vec","SimPath\nInf2vec"),3),levels=c("IMINFECTOR","Credit\nDistribution","IMM\nDB","SimPath\nDB","IMM\nInf2vec","SimPath\nInf2vec")),
                    time = c(229, #iminfector preprocessing
                             0, # cd  
                             21999, # imm db
                             0, # simpath db
                             0, # imm inf2vec 
                             0, # simpath inf2vec did not scale
                             3287, #infector training
                             0, #cd
                             0,#db
                             0,
                             0,
                             0,
                             2865, #iminfector algorithm
                             0,  #cd 
                             300, # imm db
                             0,  #simpath db
                             0, # imm inf2vec
                             0 #simpath inf2vec
                             
                    ))



df_mag$dataset = "MAG"
df_digg$dataset = "Digg"
df_weibo$dataset = "Weibo"

df = rbind(df_mag,df_digg,df_weibo)
library(ggplot2)



anno = data.frame(model = c("SimPath\nInf2vec", "Credit\nDistribution","IMM\nInf2vec","SimPath\nDB","SimPath\nInf2vec"), 
                   time = c(50000, rep(5000,4)),
                   lab = rep("X",5),
                   dataset = c("MAG", rep("Weibo",4)))

ggplot(df,aes(y=time,x=model,fill=step))+geom_bar(stat="identity",width = 0.4)+ylab("Time [sec]")+xlab("")+theme_bw()+
  geom_text(data = anno, mapping = aes(y = time,x = model, label = lab,fill=NULL),size=15,colour="red",face="bold") +
  #geom_text(aes(x=3.3, y = 65,dataset = "MAG"), label = 'n = 123', size = 6)+
  theme(strip.text = element_text(size = 17,face="bold"),
        axis.text.y = element_text(size=15),
        axis.title.y = element_text(size=18),
        legend.text = element_text(size=20),legend.position="top",
        legend.title = element_blank(),
        axis.text.x = element_text(size=10))+
  scale_fill_manual(values=c("#E69F00","steelblue","darkgreen","brown"))+
  facet_wrap(~dataset,scales = "free")



ggsave("/Figures/time_results.pdf")


