#' This functions takes raw counts and labels of reference/query set to generate scGCN training input
#' @param count.list list of reference data and query data; rows are genes and columns are cells
#' @param label.list list of reference label and query label (if any), both are data frames with rownames identical with colnames of data; the first column is cell type
#' @return This function returns files saved in folders "input" & "process_data"
#' @export: all files are saved in current path
#' @examples: load count.list and label.list from folder "example_data"
#' save_processed_data(count.list,label.list)

source('data_preprocess_utility.R')
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

args = commandArgs(trailingOnly = TRUE)
path = args[[1]]
args.rgraph = args[[2]]
if(args.rgraph == "True"){
    rgraph = TRUE
} else {
    rgraph = FALSE
}

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
colnames(ref_label) = c('type')
colnames(query_label) = c('type')

ref_label = as.data.frame(ref_label)
query_label = as.data.frame(query_label)

# count.list <- readRDS('example_data/count.list.RDS')
# label.list <- readRDS('example_data/label.list.RDS')

print(dim(ref_data))
print(dim(query_data))
count.list <- list(ref_data, query_data)
label.list <- list(ref_label, query_label)


save_processed_data(count.list,label.list, Rgraph=rgraph)

