library(batchelor)
library(Seurat)

read_data <- function(path) {
    # return matrix
    data = as.matrix(read.csv(path, row.names=1))
    return (data)
}

read_label <- function(path) {
    #return matrix
    label = as.matrix(read.csv(path))
    return (label)
}

select_feature <- function(data,label,nf=2000){
    M <- nrow(data); new.label <- label[,1]
    pv1 <- sapply(1:M, function(i){
        mydataframe <- data.frame(y=as.numeric(data[i,]), ig=new.label)
        fit <- aov(y ~ ig, data=mydataframe)
        summary(fit)[[1]][["Pr(>F)"]][1]})
    names(pv1) <- rownames(data)
    pv1.sig <- names(pv1)[order(pv1)[1:nf]]
    egen <- unique(pv1.sig)
    return (egen)
}

normalize_data <- function(data) {
#     data <- as.matrix(Seurat:::NormalizeData.default(data,verbose=F))
    data = t(data) # 先转置，变成cell * gene
    row_sum = apply(data, 1, sum)
    mean_t = mean(row_sum)
#     细胞表达量为0的地方不用管，设置为1，表示不影响
    row_sum[row_sum==0] = 1
#
#     row_sum是vector，会自动广播
    data = data/row_sum * mean_t
#     再次转置回来，变成 gene * cell
    data = t(data)
    return (data)

}

GenerateGraph <- function(Dat1,Dat2,Lab1,K,check.unknown){
    object1 <- CreateSeuratObject(counts=Dat1,project = "1",assay = "Data1",
                                  min.cells = 0,min.features = 0,
                                  names.field = 1,names.delim = "_")

    object2 <- CreateSeuratObject(counts=Dat2,project = "2",assay = "Data2",
                                  min.cells = 0,min.features =0,names.field = 1,
                                  names.delim = "_")

    objects <- list(object1,object2)
    objects1 <- lapply(objects,function(obj){
        obj <- NormalizeData(obj,verbose=F)

	obj <- FindVariableFeatures(obj,
                                selection.method = "vst",
                                nfeatures = 2000,verbose=F)
	obj <- ScaleData(obj,features=rownames(obj),verbose=FALSE)
# 	这句RunPCA可要可不要，反正不会用到
    obj <- RunPCA(obj, features=rownames(obj), verbose = FALSE)
        return(obj)})
    #'  Inter-data graph
    # 这个函数默认使用cca
    object.nn <- FindIntegrationAnchors(object.list = objects1,k.anchor=K,verbose=F)
    arc=object.nn@anchors
    d1.arc1=cbind(arc[arc[,4]==1,1],arc[arc[,4]==1,2],arc[arc[,4]==1,3])
    grp1=d1.arc1[d1.arc1[,3]>0,1:2]-1

    if (check.unknown){
        obj <- objects1[[2]]
        obj <- RunPCA(obj, features = VariableFeatures(object = obj),npcs=30,verbose=F)
        obj <- FindNeighbors(obj,verbose=F)
        obj <- FindClusters(obj, resolution = 0.5,verbose=F)
        hc <- Idents(obj); inter.graph=grp1+1
        scores <- metrics(lab1=Lab1,inter_graph=inter.graph,clusters=hc)
        saveRDS(scores,file='./input/statistical_scores.RDS')
    }
    #'  Intra-data graph
    d2.list <- list(objects1[[2]],objects1[[2]])
    d2.nn <- FindIntegrationAnchors(object.list =d2.list,k.anchor=K,verbose=F)
    d2.arc=d2.nn@anchors
    d2.arc1=cbind(d2.arc[d2.arc[,4]==1,1],d2.arc[d2.arc[,4]==1,2],d2.arc[d2.arc[,4]==1,3])
    d2.grp=d2.arc1[d2.arc1[,3]>0,1:2]-1
    final <- list(inteG=grp1,intraG=d2.grp)

    # 尝试构造一下reference内部的图
#     d3.list <- list(objects1[[1]],objects1[[1]])
#     d3.nn <- FindIntegrationAnchors(object.list =d3.list,k.anchor=K,verbose=F)
#     d3.arc=d3.nn@anchors
#     d3.arc1=cbind(d3.arc[d3.arc[,4]==1,1],d3.arc[d3.arc[,4]==1,2],d3.arc[d3.arc[,4]==1,3])
#     d3.grp=d3.arc1[d3.arc1[,3]>0,1:2]-1
#     final <- list(inteG=grp1,intraG=d2.grp, intraG2=d3.grp)

    return (final)
}


main <- function(ref_data_path, query_data_path, ref_label_path, ref_save_path, query_save_path){

    ref_data = t(read_data(ref_data_path)) # gene x cell
    ref_label = read_label(ref_label_path)
    query_data = t(read_data(query_data_path)) # gene x cell

    # gene selection
    print(dim(ref_data))
    print(dim(ref_label))
    sel.features <- select_feature(ref_data, ref_label)
    sel.ref_data = ref_data[sel.features, ]
    sel.query_data = query_data[sel.features, ]

    # Norm
    norm.ref_data = normalize_data(sel.ref_data)
    norm.query_data = normalize_data(sel.query_data)

    # Mnn correct
#     out = mnnCorrect(norm.ref_data, norm.query_data, cos.norm.in = FALSE, cos.norm.out=FALSE)
#     out = mnnCorrect(norm.ref_data, norm.query_data)

#     out = mnnCorrect(norm.ref_data, norm.query_data, sigma=0.3)

#     new_data = out@assays@data@listData$corrected

#     new.ref_data = t(out@assays@data$corrected[,out$batch==1])
#     new.query_data = t(out@assays@data$corrected[,out$batch==2])

    new.ref_data = t(norm.ref_data)
    new.query_data = t(norm.query_data)

    graphs <- suppressWarnings(GenerateGraph(Dat1=sel.ref_data,Dat2=sel.query_data,
                                                 Lab1=ref_label,K=10,
                                                 check.unknown=false))

    write.csv(graphs[[1]],file=paste(ref_save_path, 'inter_graph.csv'), quote=F,row.names=T)
    write.csv(graphs[[2]],file=paste(query_save_path, 'intra_graph.csv'),quote=F,row.names=T)
    write.csv(new.ref_data, file=ref_save_path, row.names=TRUE)
    write.csv(new.query_data, file=query_save_path, row.names=TRUE)

}


args = commandArgs(trailingOnly = TRUE)
base_path = args[[1]]
ref_data_path = paste(base_path, 'raw_data' ,'ref', 'data_1.csv', sep='/')
query_data_path = paste(base_path, 'raw_data', 'query', 'data_1.csv', sep='/')
ref_label_path = paste(base_path, 'raw_data', 'ref', 'label_1.csv', sep='/')

ref_save_path = paste(base_path, 'data', 'ref', 'data_1.csv',sep='/')
query_save_path = paste(base_path, 'data', 'query', 'data_1.csv', sep='/')

print("Path is")
print(base_path)
main(ref_data_path, query_data_path, ref_label_path, ref_save_path, query_save_path)
