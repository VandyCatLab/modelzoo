library(corrplot)
library(jsonlite)
library(tidyverse)

# Reading all the data
trial1_data <- read.csv("csvs/csv_data_maker_threeACF-noise.csv", stringsAsFactors = FALSE)
trial2_data <- read.csv("csvs/csv_data_maker_many_odd-noise.csv", stringsAsFactors = FALSE)
trial3_data <- read.csv("csvs/csv_data_maker_learn_exemp-noise.csv", stringsAsFactors = FALSE)

# Extracting unique SbjIDs
subjids_trial1 <- unique(trial1_data$SbjID)
subjids_trial2 <- unique(trial2_data$SbjID)
subjids_trial3 <- unique(trial3_data$SbjID)

# Identifying common SbjIDs
common_subjids <- Reduce(intersect, list(subjids_trial1, subjids_trial2, subjids_trial3))

# Filtering function
filter_data <- function(dataset, common_ids) {
  dataset %>% 
    filter(SbjID %in% common_ids)
}

# Filtering data
data1 <- filter_data(trial1_data, common_subjids)
data2 <- filter_data(trial2_data, common_subjids)
data3 <- filter_data(trial3_data, common_subjids)

# json file path
json_file_path <- "./hubModels_timm.json" 

# Read the JSON data
json_data <- fromJSON(json_file_path)

# Extract the relevant attributes (num_classes, num_params, num_layers)
json_extracted <- data.frame(
  SbjID = names(json_data),
  num_classes = sapply(json_data, function(x) x$num_classes),
  num_params = sapply(json_data, function(x) x$num_params),
  num_layers = sapply(json_data, function(x) x$num_layers)
)


model_data <- filter_data(json_extracted, common_subjids)
rownames(model_data) <- NULL

# Write attributes file to CSV
write.csv(model_data, "./csvs/model_data.csv", row.names = FALSE)

# Function to process each CSV file and calculate average score per subject
process_data <- function(data) {
  data %>%
    group_by(SbjID) %>%
    summarize(AvgScore = mean(CorrRes == response)) %>%
    ungroup()
}

# Read and process each CSV file
test1_data <- process_data(data1)
test2_data <- process_data(data2)
test3_data <- process_data(data3)

# Calculate overall average scores for each test
avg_score_test1 <- mean(test1_data$AvgScore)
avg_score_test2 <- mean(test2_data$AvgScore)
avg_score_test3 <- mean(test3_data$AvgScore)

# Calculate z-scores for each test
z_score <- function(test_data, avg_score) {
  (test_data$AvgScore - avg_score) / sd(test_data$AvgScore)
}

test1_data$ZScore <- z_score(test1_data, avg_score_test1)
test2_data$ZScore <- z_score(test2_data, avg_score_test2)
test3_data$ZScore <- z_score(test3_data, avg_score_test3)


# Merge the results based on subjects who participated in all three tests
merged_data <- reduce(list(test1_data, test2_data, test3_data), inner_join, by = "SbjID")

# Calculate the average of the three z-scores
merged_data$AvgZ <- rowMeans(merged_data[, c("ZScore.x", "ZScore.y", "ZScore")])

merged_data <- select(merged_data, -c(ZScore.x, ZScore.y, ZScore))

# Rename columns for clarity
colnames(merged_data) <- c("SbjID", "3ACF", "MOO", "LE", "OScore")

merged_data

# Save the final data to a CSV file
write.csv(merged_data, "./csvs/means_oscore_data.csv", row.names = FALSE)

# Calculate the Pearson correlation coefficient
correlation_matrix <- cor(merged_data[c("3ACF", "MOO", "LE")], method = "pearson")

corrplot(correlation_matrix, method = "color", order = "hclust",
         addCoef.col = "white",
         col.axis="black",
         tl.col = "black")


merged_data$AvgTestScore <- rowMeans(merged_data[, c("3ACF", "MOO", "LE")], na.rm = TRUE)
merged_data$MaxTestScore <- apply(merged_data[, c("3ACF", "MOO", "LE")], 1, max, na.rm = TRUE)

# Merge the data frames on the 'SbjID' column
merged_df <- merge(merged_data, model_data, by = "SbjID")

# Calculate correlations for specific categories
categories <- c("num_classes", "num_params", "num_layers")
correlations_o <- sapply(merged_df[categories], function(x) cor(x, merged_df$OScore, use = "complete.obs"))
correlations_avg <- sapply(merged_df[categories], function(x) cor(x, merged_df$AvgTestScore, use = "complete.obs"))
correlations_max <- sapply(merged_df[categories], function(x) cor(x, merged_df$MaxTestScore, use = "complete.obs"))
# Create a data frame for the correlations
correlation_df_o <- data.frame(Category = categories, Correlation = correlations_o)
correlation_df_avg <- data.frame(Category = categories, Correlation = correlations_avg)
correlation_df_max <- data.frame(Category = categories, Correlation = correlations_max)

# View the correlations
print(correlation_df_o)
print(correlation_df_avg)
print(correlation_df_max)