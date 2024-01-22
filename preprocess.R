suppressWarnings(library(batchelor))
suppressWarnings(library(Seurat))
suppressWarnings(library(SeuratDisk))
suppressWarnings(library(SeuratData))

readH5AD <- function(file_path) {
    Covert(path, 'h5seurat', overwrite=True)
    file_name = strsplit(path, "/")
    file_name = file_name[length(file_name)]    
    data = LoadH5Seurat(paste(file_name, '.h5seurat'))
    return (data)
    
}
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
    # median normalization

    data = t(data) # 先转置，变成cell * gene
    row_sum = apply(data, 1, sum)
    mean_t = mean(row_sum)
    # 细胞表达量为0的地方不用管，设置为1，表示不影响
    row_sum[row_sum==0] = 1

    # row_sum是vector，会自动广播
    data = data/row_sum * mean_t
    # turn into gene * cell format
    data = t(data)
    return (data)

}

GenerateGraph <- function(Dat1,Dat2,Dat3,Lab1,K){
    object1 <- CreateSeuratObject(counts=Dat1,project = "1",assay = "Data1",
                                  min.cells = 0,min.features = 0,
                                  names.field = 1,names.delim = "_")

    object2 <- CreateSeuratObject(counts=Dat2,project = "2",assay = "Data2",
                                  min.cells = 0,min.features =0,names.field = 1,
                                  names.delim = "_")

    object3 <- CreateSeuratObject(counts=Dat3,project = "3",assay = "Data3",
                                  min.cells = 0,min.features =0,names.field = 1,
                                  names.delim = "_")
    
    objects <- list(object1,object2, object3)
    
    objects1 <- lapply(objects,function(obj){
        obj <- NormalizeData(obj,verbose=F)

        obj <- FindVariableFeatures(obj,
                                    selection.method = "vst",
                                    nfeatures = 2000,verbose=F)
        obj <- ScaleData(obj,features=rownames(obj),verbose=FALSE)        
        obj <- RunPCA(obj, features=rownames(obj), verbose = FALSE)
            return(obj)
    })
    
    # Inter-data graph
    # 这个函数默认使用cca
    d1.list <- list(objects1[[1]],objects1[[2]])   
    object.nn <- FindIntegrationAnchors(object.list = d1.list, k.anchor=K,verbose=F)
    arc=object.nn@anchors
    d1.arc1=cbind(arc[arc[,4]==1,1],arc[arc[,4]==1,2],arc[arc[,4]==1,3])
    grp1=d1.arc1[d1.arc1[,3]>0,1:2]-1
    
    # Intra-data graph
    d2.list <- list(objects1[[2]],objects1[[2]])    
    d2.nn <- FindIntegrationAnchors(object.list =d2.list,k.anchor=K,verbose=F)
    d2.arc=d2.nn@anchors
    d2.arc1=cbind(d2.arc[d2.arc[,4]==1,1],d2.arc[d2.arc[,4]==1,2],d2.arc[d2.arc[,4]==1,3])
    d2.grp=d2.arc1[d2.arc1[,3]>0,1:2]-1

    # auxilary data graph
    d3.list <- list(objects1[[3]],objects1[[3]])
    d3.nn <- FindIntegrationAnchors(object.list =d3.list,k.anchor=K,verbose=F)
    d3.arc=d3.nn@anchors
    d3.arc1=cbind(d3.arc[d3.arc[,4]==1,1],d3.arc[d3.arc[,4]==1,2],d3.arc[d3.arc[,4]==1,3])
    d3.grp=d3.arc1[d3.arc1[,3]>0,1:2]-1

    final <- list(inteG=grp1,intraG=d2.grp, auxilaryG=d3.grp)


    return (final)
}


main <- function(ref_data_path, 
                query_data_path,
                auxilary_data_path,                
                ref_save_path,                 
                query_save_path,
                auxilary_save_path){
    
    ref_data_h5 = readH5AD(ref_data_path)
    query_data_h5 = readH5AD(query_data_path)
    auxilary_data_h5 = readH5AD(auxilary_data_path)    

    # use as() function to get dense matrix
    ref_data = as(ref_data_h5@assays$RNA@counts, 'matrix')
    query_data = as(query_data_h5@assays$RNA@counts, 'matrix')
    auxilary_data = as(auxilary_data_h5@assays$RNA@counts, 'matrix')
    
    ref_label = ref_data_h5@meta.data$cell_type
                
    # gene intersection
    inter_genes = intersect(intersect(rownames(ref_data), rownames(query_data)), rownames(auxilary_data))
    ref_data = ref_data[inter_genes, ]
    query_data = query_data[inter_genes, ]
    auxilary_data = auxilary_data[inter_genes, ]
    
    
    # gene selection
    print(dim(ref_data))
    print(dim(ref_label))
    
    sel.features <- select_feature(ref_data, ref_label)
    sel.ref_data = ref_data[sel.features, ]
    sel.query_data = query_data[sel.features, ]
    sel.auxilary_data = auxilary_data[sel.features, ]
            
    # Norm: gene * cell
    norm.ref_data = normalize_data(sel.ref_data)
    norm.query_data = normalize_data(sel.query_data)
    norm.auxilary_data = normalize_data(sel.auxilary_data)
    

    graphs <- suppressWarnings(GenerateGraph(Dat1=norm.ref_data,Dat2=norm.query_data,Dat3=norm.auxilary_data,
                                                 Lab1=ref_label,K=5 #这里修改了K值 2024.1.18
                                                 ))

    ref_data_h5@assays$RNA@counts = Matrix(new.ref_data, sparse = TRUE)
    query_data_h5@assays$RNA@counts = Matrix(new.query_data, sparse = TRUE)
    auxilary_data_h5@assays$RNA@counts = Matrix(new.auxilary_data, sparse = TRUE)
    
                                          
    write.csv(graphs[[1]],file=paste(paste(base_path, 'data' , 'inter_graph.csv', sep='/')), quote=F,row.names=T)
    write.csv(graphs[[2]],file=paste(paste(base_path, 'data' , 'intra_graph.csv', sep='/')),quote=F,row.names=T)
    write.csv(graphs[[3]],file=paste(paste(base_path, 'data' , 'auxilary_graph.csv', sep='/')),quote=F,row.names=T)
    
    SaveH5Seurat(ref_data_h5, filename = ref_save_path)
    SaveH5Seurat(query_data_h5, filename = query_save_path)
    SaveH5Seurat(auxilary_data_h5, filename = auxilary_save_path)
    Convert(ref_save_path, dest = "h5ad")
    Convert(query_save_path, dest = "h5ad")
    Convert(auxilary_save_path, dest = "h5ad")

}


args = commandArgs(trailingOnly = TRUE)
base_path = args[[1]]
ref_data_path = paste(base_path, 'raw_data' , 'ref_data.h5ad', sep='/')
query_data_path = paste(base_path, 'raw_data', 'query_data.h5ad', sep='/')
auxilary_data_path = paste(base_path, 'raw_data','auxilary_data.h5ad', sep='/')


ref_save_path = paste(base_path, 'data', 'ref_data.h5Seurat',sep='/')
query_save_path = paste(base_path, 'data', 'query_data.h5Seurat', sep='/')
auxilary_save_path = paste(base_path, 'data', 'auxilary_data.h5Seurat', sep='/')

print("Path is")
print(base_path)
main(ref_data_path, query_data_path, auxilary_data_path, ref_save_path, query_save_path, auxilary_save_path)
