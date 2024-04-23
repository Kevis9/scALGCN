suppressPackageStartupMessages(library(Seurat))
# suppressPackageStartupMessages(library(CHETAH)) // CHETAH不支持 R4.1 直接删除
suppressPackageStartupMessages(library(SingleR))
suppressPackageStartupMessages(library(scmap))
# suppressPackageStartupMessages(library(conos))
suppressPackageStartupMessages(library(pagoda2))
suppressPackageStartupMessages(library(Matrix))

read_data <- function(path) {
    # return matrix    
    data = as(readMM(path), 'matrix') # cell * gene    
    return (data)
}

read_label <- function(path) {
    #return matrix
    label = as.matrix(read.csv(path))
    return (label)
}
read_gene <- function(path) {    
    gene = as.matrix(read.csv(path))
    return (gene)
}
read_name <- function(path) {
    name = as.matrix(read.csv(path))
    return (name)
}

read_ref_query_data_label <- function(path) {
    
    ref_data_path = paste(path, 'ref_data_middle.mtx', sep='/')            
    ref_label_path = paste(path, 'ref_label_middle.csv', sep='/')            
    ref_gene_path = paste(path, 'ref_gene_middle.csv', sep='/')
    ref_name_path = paste(path, 'ref_name_middle.csv', sep='/')
    
    query_data_path = paste(path, 'query_data_middle.mtx', sep='/')            
    query_label_path = paste(path, 'query_label_middle.csv', sep='/')            
    query_gene_path = paste(path, 'query_gene_middle.csv', sep='/')
    query_name_path = paste(path, 'query_name_middle.csv', sep='/')

    
    ref_data = t(read_data(ref_data_path)) # gene x cell
    ref_label = read_label(ref_label_path)
    ref_gene = read_gene(ref_gene_path)
    ref_name = read_name(ref_name_path)


    query_data = t(read_data(query_data_path)) # gene x cell
    query_label = read_label(query_label_path)
    query_gene = read_gene(query_gene_path)
    query_name = read_name(query_name_path)

    ref_label[ref_label == "Unassigned"] <- "unassigned"
    query_label[query_label == "Unassigned"] <- "unassigned"

    rownames(ref_data) = ref_gene
    rownames(query_data) = query_gene
    colnames(ref_data) = ref_name
    colnames(query_data) = query_name

    # gene intersection
    inter_genes = intersect(rownames(ref_data), rownames(query_data))
    ref_data = ref_data[inter_genes, ]
    query_data = query_data[inter_genes, ]    
        
    
    rownames(ref_label) = ref_name
    rownames(query_label) = query_name
    
    colnames(ref_label) = c('type')
    colnames(query_label) = c('type')

    return (list(ref_data, query_data, ref_label, query_label))

}

acc_score <- function(pred, true_label){
    prediction.match <- pred == true_label
    acc = sum(prediction.match)/length(prediction.match)
}


save_pred <- function(pred, proj_name, method_name) {
    save_path = paste("result", proj_name, sep='/')
    if(!file.exists(save_path)) {
        dir.create(save_path)
    }
    save_path = paste("result", proj_name, method_name, sep='/')
    if(!file.exists(save_path)) {
        dir.create(save_path)
    }
    save_file_path = paste(save_path, "query_pred.csv", sep="/")
    write.table(as.matrix(pred), file=save_file_path, sep=',', row.names=FALSE, col.names=TRUE,quote=FALSE)

}

save_labels <- function(labels, proj_name, method_name){
    save_path = paste("result", proj_name, sep='/')
    if(!file.exists(save_path)) {
        dir.create(save_path)
    }
    save_path = paste("result", proj_name, method_name, sep='/')
    if(!file.exists(save_path)) {
        dir.create(save_path)
    }
    save_file_path = paste(save_path, "query_true.csv", sep="/")
    write.table(as.matrix(labels), file=save_file_path, sep=',', row.names=FALSE, col.names=TRUE,quote=FALSE)

}

save_prob <- function(prob, proj_name, method_name){
    save_path = paste("result", proj_name, sep='/')
    if(!file.exists(save_path)) {
        dir.create(save_path)
    }
    save_path = paste("result", proj_name, method_name, sep='/')
    if(!file.exists(save_path)) {
        dir.create(save_path)
    }
    save_file_path = paste(save_path, "query_embeddings.csv", sep="/")
    write.table(as.matrix(prob), file=save_file_path, sep=',', row.names=FALSE, col.names=TRUE,quote=FALSE)

}

seurat_pca_pred <- function(ref_data, query_data, ref_label, query_label, proj_name) {

    # all input data must be matrix
    # reurn prediction (matrix)
    # ref_data = as.data.frame(ref_data)
    # query_data = as.data.frame(query_data)
    ref_label = as.data.frame(ref_label)
    query_label = as.data.frame(query_label)



    object1 <- CreateSeuratObject(
            counts=ref_data,
            project = "reference",
            assay = "RNA",
            meta.data = ref_label)

    object2 <- CreateSeuratObject(
        counts=query_data,
        project = "query",
        assay = "RNA",
        meta.data = query_label
        )


    objs <- list(object1,object2)
    objs1 <- lapply(objs,function(data){
        data <- NormalizeData(data)
        data <- FindVariableFeatures(data,
                                       selection.method = "vst",
                                       nfeature=500)
        return(data)})

    reference.object <- objs1[[1]];
    query.object <- objs1[[2]]
    reference.object <- ScaleData(reference.object, verbose = FALSE)
    reference.object <- RunPCA(reference.object, npcs = 30, verbose = FALSE)
    reference.anchors <- FindTransferAnchors(reference = reference.object, query = query.object, dims = 1:30, reduction='pcaproject', reference.reduction='pca')
    reference.object = RunUMAP(reference.object, dims = 1:30, return.model = TRUE)
#     predictions <- TransferData(anchorset = reference.anchors, refdata = as.factor(reference.object$type), dims = 1:30)    
    query <-  MapQuery(anchorset = reference.anchors, reference = reference.object, query = query.object,    
    refdata = list(celltype = "type"), reduction.model = "umap")
    
    pred = query$predicted.celltype

    prob = pmax(as.matrix(query$predicted.celltype.score))
    save_prob(prob, proj_name, "seurat")
    ref_umap_data = reference.object@reductions$umap@cell.embeddings
    query_umap_data = query@reductions$ref.umap@cell.embeddings
    print("proj name")
    print(proj_name)
    # 保存Seurat的embeddings
    save_path = paste("result", proj_name, sep='/')
    if(!file.exists(save_path)) {
        dir.create(save_path)
    }
    save_path = paste("result", proj_name, 'seurat', sep='/')
    if(!file.exists(save_path)) {
        dir.create(save_path)
    }
    embeddings = rbind(ref_umap_data, query_umap_data)
    pred = as.matrix(pred)
    ref_label = as.matrix(ref_label)
    colnames(pred) = c('type')
    colnames(ref_label)=c('type')

    all_preds = rbind(as.matrix(ref_label), as.matrix(pred))
    print("save path" )
    print(save_path)
    write.csv(embeddings, paste(save_path, "embeddings_2d.csv", sep='/'), row.names=F)
    write.csv(all_preds, paste(save_path, "all_preds.csv", sep='/'), row.names=F)
    write.csv(as.matrix(reference.anchors@anchors), paste(save_path, "anchors.csv", sep='/'), row.names=F)
    save_pred(pred, proj_name, "seurat")

    return (as.matrix(pred))
}

seurat_cca_pred <- function(ref_data, query_data, ref_label, query_label, proj_name) {
    # all input data must be matrix
    # reurn prediction (matrix)
    ref_data = as.data.frame(ref_data)
    query_data = as.data.frame(query_data)
    ref_label = as.data.frame(ref_label)
    query_label = as.data.frame(query_label)

    object1 <- CreateSeuratObject(
            counts=ref_data,
            project = "reference",
            assay = "RNA",
            meta.data = ref_label)

    object2 <- CreateSeuratObject(
        counts=query_data,
        project = "query",
        assay = "RNA",
        meta.data = query_label
        )


    objs <- list(object1,object2)
    objs1 <- lapply(objs,function(data){
        data <- NormalizeData(data)
        data <- FindVariableFeatures(data,
                                       selection.method = "vst",
                                       nfeature=2000)
        return(data)})

    reference.object <- objs1[[1]];
    query.object <- objs1[[2]]
    reference.object <- ScaleData(reference.object, verbose = FALSE)
    reference.object <- RunPCA(reference.object, npcs = 30, verbose = FALSE)
    reference.anchors <- FindTransferAnchors(reference = reference.object, query = query.object, dims = 1:30, reduction = 'cca')
    reference.object = RunUMAP(reference.object, dims = 1:30, return.model = TRUE)
#     predictions <- TransferData(anchorset = reference.anchors, refdata = as.factor(reference.object$type), dims = 1:30)
    # weight.reduction 会自动设置成cca
    query <-  MapQuery(anchorset = reference.anchors, reference = reference.object, query = query.object,
    refdata = list(celltype = "type"), reduction.model = "umap")

    pred = query$predicted.celltype
    prob = pmax(as.matrix(query$predicted.celltype.score))
    save_prob(prob, proj_name, "seurat_cca")
    ref_umap_data = reference.object@reductions$umap@cell.embeddings
    query_umap_data = query@reductions$ref.umap@cell.embeddings
    # 保存Seurat的embeddings
    save_path = paste("result", proj_name, sep='/')
    if(!file.exists(save_path)) {
        dir.create(save_path)
    }
    save_path = paste("result", proj_name, 'seurat_cca', sep='/')
    if(!file.exists(save_path)) {
        dir.create(save_path)
    }
    embeddings = rbind(ref_umap_data, query_umap_data)
    pred = as.matrix(pred)
    ref_label = as.matrix(ref_label)
    colnames(pred) = c('type')
    colnames(ref_label)=c('type')

    all_preds = rbind(as.matrix(ref_label), as.matrix(pred))

    write.csv(embeddings, paste(save_path, "embeddings_2d.csv", sep='/'), row.names=F)
    write.csv(all_preds, paste(save_path, "all_preds.csv", sep='/'), row.names=F)
    write.csv(as.matrix(reference.anchors@anchors), paste(save_path, "anchors.csv", sep='/'), row.names=F)
    save_pred(pred, proj_name, "seurat_cca")
    return (as.matrix(pred))
}

conos_pred <- function(ref_data, query_data, ref_label) {
    cellannot = data.frame(
        cell_name = as.vector(colnames(ref_data)),
        cell_type = ref_label[, 1]
    )
#     rownames(cellannot) = c(1:dim(cellannot)[1])
    cellannot <- setNames(cellannot[,2], cellannot[,1])
    panel <- list(ref=ref_data, query=query_data)
    panel.preprocessed <- lapply(panel, basicSeuratProc)
    con <- Conos$new(panel.preprocessed, n.cores=1)
    con$buildGraph(k=30, k.self=5, space='PCA', ncomps=30, n.odgenes=2000, matching.method='mNN', metric='angular', score.component.variance=TRUE, verbose=TRUE)
    pred <- con$propagateLabels(labels = cellannot)$labels



    query_pred = pred[colnames(query_data)]
    return (as.matrix(query_pred))
}


single_r_pred <- function(ref_data, query_data, ref_label, proj_name) {
    # all input must be matrix
    # Single R can take matrix as input, reference data must be log-normalized
    ref_sce = SingleCellExperiment(assays = list(normcounts = as.matrix(ref_data)))
    # Normalize reference data
#     ref_data <- as.matrix(log2(SingleCellExperiment::normcounts(ref_sce) + 1))

    pred <- SingleR(test = query_data, ref = ref_data, labels = ref_label)

    prob <- as.matrix(apply(as.matrix(pred$scores), 1, max, na.rm=TRUE))
    pred <- pred$labels

    save_prob(prob, proj_name, "singler")
    save_pred(pred, proj_name, "singler")
    return (pred)
}




scmap_pred <- function(ref_data, query_data, ref_label, proj_name) {
    # all input must be matrix
    colnames(ref_label) = c('cell_type1')
    ref_sce <- SingleCellExperiment(assays = list(normcounts = as.matrix(ref_data)), colData = as.data.frame(ref_label))
    logcounts(ref_sce) <- log2(SingleCellExperiment::normcounts(ref_sce) + 1)
    rowData(ref_sce)$feature_symbol <- rownames(ref_sce)
    ref_sce <- ref_sce[!duplicated(rownames(ref_sce)), ]

    query_sce <- SingleCellExperiment(assays=list(normcounts = as.matrix(query_data)))
    logcounts(query_sce) <- log2(normcounts(query_sce) + 1)
    rowData(query_sce)$feature_symbol <- rownames(query_sce)


    # feature selection, default will be 500 genes being selected
    ref_sce <- selectFeatures(ref_sce, suppress_plot = TRUE)


    # scmap Cluster
    ref_sce <- scmap::indexCluster(ref_sce)


    scmapCluster_results <- scmapCluster(projection = query_sce,
                                         index_list = list(yan = metadata(ref_sce)$scmap_cluster_index),
                                         threshold=0.5)

    prob = scmapCluster_results$scmap_cluster_siml

    pred <- scmapCluster_results$scmap_cluster_labs

    save_prob(prob, proj_name, "scmap")

    save_pred(pred, proj_name, "scmap")
    return (as.matrix(pred))

}



chetah_pred <- function(ref_data, query_data, ref_label, proj_name) {
    # all input must be matrix
    ref_label = as.character(ref_label) #label 必须是字符串

    ref_ct = as.data.frame(ref_label)
    colnames(ref_ct) = c('celltypes')

    reference <- SingleCellExperiment(assays = list(counts = ref_data),
                                         colData = ref_ct)
    # Normalize reference data
#     assay(reference, "counts") <- apply(assay(reference, "counts"), 2, function(column) log2((column/sum(column) * 1e4) + 1))
    # aka query
    input <- SingleCellExperiment(assays = list(counts = query_data))
    # 正常，默认版
    # unknown_cell的检测就用默认版本
    input <- CHETAHclassifier(input = input, ref_cells = reference)
    # Unknwon Cell type，设置默认的thresh为0.5， 之前的0.1也太低了
#     input <- CHETAHclassifier(input = input, ref_cells = reference, thresh=0.05)

    ## Extract celltypes:
    pred <- input$celltype_CHETAH
    # Node0应该涵盖了所有类型的confidence score
    prob <- as.matrix(apply(input@int_colData$CHETAH$conf_scores$Node0, 1, max, na.rm=TRUE))
#     prob = pmax(input@int_colData$CHETAH$conf_scores$Node0)
    save_prob(prob, proj_name, "chetah")
    save_pred(pred, proj_name, "chetah")
    return (as.matrix(pred))
}

main <- function(data, method, proj_name){

    ref_data = data[[1]]
    query_data = data[[2]]
    ref_label = data[[3]]
    query_label = data[[4]]
    
    print(proj_name)
    if(method == 'seurat') {
        pred = seurat_pca_pred(ref_data, query_data, ref_label, query_label, proj_name)
        acc = acc_score(pred, query_label)
    }    
    if(method == 'singler') {
        pred = single_r_pred(ref_data, query_data, ref_label, proj_name)
        acc = acc_score(pred, query_label)
    }
    if(method == 'scmap') {
        pred = scmap_pred(ref_data, query_data, ref_label, proj_name)
        acc = acc_score(pred, query_label)
    }
    if(method == 'chetah') {
        pred = chetah_pred(ref_data, query_data, ref_label, proj_name)
        acc = acc_score(pred, query_label)
    }    

    print(acc)

#     write.csv(pred, paste("result/", paste(paste(save_path, method, sep='_'), ".csv", sep=''), sep=''))
    return (acc)
}

final_acc = list()


args = commandArgs(trailingOnly = TRUE)
base_path = args[[1]]
proj = args[[2]]



data_path = paste(base_path,'raw_data', sep='/')
data = read_ref_query_data_label(data_path)

query_label = data[[4]]
    
save_labels(query_label,proj, "seurat")
save_labels(query_label,proj, "scmap")    
# save_labels(query_label,proj, "chetah")
save_labels(query_label, proj, "singler")

print("数据读取完成")

acc = c(
    main(data, 'seurat', proj),
    main(data, 'singler', proj),
    main(data, 'scmap', proj)
    # main(data, 'chetah', proj)
)
acc = as.matrix(acc)
print(acc)