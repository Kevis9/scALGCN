#' This functions takes raw counts and labels of reference/query set to generate scGCN training input
#' @param count.list list of reference data and query data; rows are genes and columns are cells
#' @param label.list list of reference label and query label (if any), both are data frames with rownames identical with colnames of data; the first column is cell type
#' @return This function returns files saved in folders "input" & "process_data"
#' @export: all files are saved in current path
#' @examples: load count.list and label.list from folder "example_data"
#' save_processed_data(count.list,label.list)

source('data_preprocess_utility.R')
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

args = commandArgs(trailingOnly = TRUE)
base_path = args[[1]]
args.rgraph = args[[2]]
if(args.rgraph == "True"){
    rgraph = TRUE
} else {
    rgraph = FALSE
}
ref_data_path = paste(base_path, 'data' ,'ref', 'data_1.csv', sep='/')
query_data_path = paste(base_path, 'data', 'query', 'data_1.csv', sep='/')
ref_label_path = paste(base_path, 'data', 'ref', 'label_1.csv', sep='/')
query_label_path = paste(base_path, 'data', 'query', 'label_1.csv', sep='/')

ref_data = t(read_data(ref_data_path))
query_data = t(read_data(query_data_path))
ref_label = read_label(ref_label_path)
query_label = read_label(query_label_path)

rownames(ref_label) = colnames(ref_data)
rownames(query_label) = colnames(query_data)
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

