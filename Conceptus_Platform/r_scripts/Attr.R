library(readr)
library(tidyverse)
library(Matrix)
library(reshape2)
library(SpiecEasi)
library(phyloseq)
library(igraph)
library(devtools)
library(SpiecEasi)
setwd(as.character("E:/Test/R Test/SSN"))
meta <- read_csv("meta.csv")
OTUcount <- read_csv("OTUcount_filter.csv")%>%merge(meta,by.x='Samples',by.y = 'Sampleno')

meta_fil = filter(meta, Subgroup=="SS")
slist = paste0(meta$Sampleno,"_OTU_0.65_cityedge4.csv_0.3SSNcitymatr15.gml")
slist = append(slist,paste0(c("CN","CS","SN","SS"),"_OTU_0.65_edge.gml"))

arpack_defaults$maxiter=99999999999999999
Rgs = 73616
flist = list()
flist=append(flist,as.numeric(Rgs))

for (folder in flist){
  mypath<-paste0(as.character(folder),'/')
  attrlist=list()
  for(file in slist){
    G1<-read_graph(paste0(mypath,file), format = c("gml"))
    tryCatch(
      {#I want
        E(G1)$weight<-abs(E(G1)$weight)
        G_frame<-as.data.frame(list(mean(degree(G1)),
                                    #average.path.length(G1),
                                    #transitivity(G1),
                                    #assortativity.degree(G1),
                                    #diameter(G1, unconnected=TRUE),#the linear size of a network.
                                    #spectrum(G1)$value,
                                    #mean(betweenness(G1)),#invalid in SE
                                    #mean(closeness(G1)),
                                    length(E(G1)),
                                    length(V(G1))))
        colnames(G_frame)<-c('mean_degree','edgenumber','nodenumber')
        #colnames(G_frame)<-c('mean_degree','average_path','transitivity','assortativity','diameter','mean_betweenness','edgenumber','nodenumber')
        rownames(G_frame)<-file
        attrlist[[file]]=G_frame
      },
      error=function(error_message) {
        message("Error is")
        print(file)
        return(NA)
      }
    )
  }
  attrtable<-bind_rows(attrlist,.id='gml')
  colnames(attrtable)[2:4]=paste0(colnames(attrtable)[2:4],as.character(folder))
  attrtable$gml = substr(attrtable$gml,1,3)
  write.csv(attrtable,paste0(mypath,'attrtable_SSN.csv'),row.names = F)
  if(folder == flist[1]){
    SSN_degree = attrtable[,c(1,2)]
    SSN_edge = attrtable[,c(1,3)]
    SSN_node = attrtable[,c(1,4)]
  }
  else{
    SSN_degree = merge(SSN_degree,attrtable[,c(1,2)],by="gml")
    SSN_edge = merge(SSN_edge,attrtable[,c(1,3)],by="gml")
    SSN_node = merge(SSN_node,attrtable[,c(1,4)],by="gml")
  }
  #SSN = append(SSN,c(folder,"Edge",substr(attrtable[order(attrtable$edgenumber,attrtable$gml),]$gml,1,3)))
  #SSN = append(SSN,c(folder,"Node",substr(attrtable[order(attrtable$nodenumber,attrtable$gml),]$gml,1,3)))
  #SSN = append(SSN,c(folder,"Degree",substr(attrtable[order(attrtable$mean_degree,attrtable$gml),]$gml,1,3)))
}

write.csv(melt(SSN_degree),"SSN_degree.csv",row.names = TRUE)
write.csv(melt(SSN_edge),"SSN_edge",row.names = TRUE)
write.csv(melt(SSN_node),"SSN_node",row.names = TRUE)
