# Load necessary libraries
library(tidyverse) 
library(dplyr)
library(GGally)
library(psych)
library(readr)


# Read the CSV file for one of them
csv_filename = "csvs/csv_data_maker_learn_exemp-noise.csv"
data <- read.csv(csv_filename)

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


# Display the first few rows of the data
head(data)

# Check the structure of the data
str(data)

# Summary of the data
summary(data)

# Function to reformat data to summary format
summary_format_data <- function(data) {
    summarized_data <- data %>%
        group_by(SbjID) %>%
        summarize(
            total_trials = max(TrialN),
            number_correct = sum(AnsCatagory == "correct"),
            proportion_correct = number_correct / total_trials
        )
    return(summarized_data)
}

# make summary format with all trials
# make them all, then do a join

summary1 <- summary_format_data(data1)
summary1 <- summary1[c('SbjID', 'proportion_correct')]
names(summary1) <- c('SbjID', 'Task1')
summary2 <- summary_format_data(data2)
summary2 <- summary2[c('SbjID', 'proportion_correct')]
names(summary2) <- c('SbjID', 'Task2')
summary3 <- summary_format_data(data3)
summary3 <- summary3[c('SbjID', 'proportion_correct')]
names(summary3) <- c('SbjID', 'Task3')

allSummary <- left_join(summary1, summary2, by = 'SbjID')
allSummary <- left_join(allSummary, summary3, by = 'SbjID')



transform_to_wide <- function(data) {

  # Create a binary column for the answer category
  data$AnsBinary <- if_else(data$AnsCatagory == "correct", 1, 0)
  
  # Check if AnsBinary was created successfully
  if (!"AnsBinary" %in% colnames(data)) {
    stop("AnsBinary column could not be created.")
  }
  
  # Create a wide data frame with trials as columns and binary responses as values
  wide_data <- data %>%
    mutate(Trial = paste("Trial", TrialN, sep="")) %>%
    select(SbjID, Trial, AnsBinary) %>%
    pivot_wider(names_from = Trial, values_from = AnsBinary) 
  
  # Calculate the percentage of correct responses for each subject
  percentage_correct <- data %>%
    group_by(SbjID) %>%
    summarize(PercentageCorrect = 100 * mean(AnsBinary, na.rm = TRUE), .groups = 'drop')
  
  # Join the percentage with the wide data frame
  final_data <- left_join(percentage_correct, wide_data, by = "SbjID")
  
  # Ensure that PercentageCorrect is the second column
  final_data <- final_data %>%
    select(SbjID, PercentageCorrect, everything())
  
  return(final_data)
}


# Function to compute overall descriptive statistics for accuracy
accuracy_descriptive_stats <- function(data) {
  # Get the summarized data by calling the function
  summarized_data <- summary_format_data(data)
  
  # Now access the proportion_correct column from the summarized data
  accuracy <- summarized_data$proportion_correct
  
  # Calculate statistics
  stats <- list(
    mean_accuracy = mean(accuracy, na.rm = TRUE),
    median_accuracy = median(accuracy, na.rm = TRUE),
    variance_accuracy = var(accuracy, na.rm = TRUE),
    sd_accuracy = sd(accuracy, na.rm = TRUE),
    range_accuracy = range(accuracy, na.rm = TRUE),
    Q1 = quantile(accuracy, 0.25, na.rm = TRUE),
    Q3 = quantile(accuracy, 0.75, na.rm = TRUE),
    IQR = IQR(accuracy, na.rm = TRUE)
  )
  
  return(stats)
}



summary_data <- summary_format_data(data)
head(summary_data)
wide_data <- transform_to_wide(data)
head(wide_data)
accuracy_descriptives <- accuracy_descriptive_stats(data)
stats_dataframe <- as.data.frame(accuracy_descriptives)
print(stats_dataframe)

# do correlation matrix / plotting correlation scatterplots
# (3 plots not 3 dimentions)
# afterword psycometrics/test statistics


# Function to create scatterplots for the three csv files
create_correlation_matrix <- function(data1, data2, data3) {
  # Read and transform each CSV file to wide format
  df1 <- summary_format_data(data1)
  df2 <- summary_format_data(data2)
  df3 <- summary_format_data(data3)
  
  # Assuming that 'data1', 'data2', and 'data3' have the same subjects in the same order
  # and the column 'accuracy' represents the accuracy score for each subject:
  accuracies <- data.frame(
    accuracy1 = df1$proportion_correct,
    accuracy2 = df2$proportion_correct,
    accuracy3 = df3$proportion_correct
  )
  
  # Calculate the correlation matrix
  corr_matrix <- cor(accuracies, use = "complete.obs")  # 'complete.obs' handles missing data
  
  return(corr_matrix)
}

# Function to calculate item-rest correlations
compute_item_rest_correlations <- function(data) {
  item_rest_correlations <- apply(data, 2, function(item) {
    # Calculate the total score for all other items
    rest_total <- rowSums(data) - item
    # Compute correlation between this item and the rest total
    cor(item, rest_total)
  })
  
  return(item_rest_correlations)
}


correlation_matrix <- create_correlation_matrix(data1, data2, data3)

# To print the correlation matrix
print(correlation_matrix)

allSummary <- allSummary[allSummary$Task3 > 0,]
ggpairs(allSummary[c('Task1', 'Task2', 'Task3')])
allSummary
wide_data
#splitHalf()allSummary

subjects_with_zero_score <- summary3$SbjID[summary3$Task3 == 0]


trial_data <- trial_data[!trial_data$SbjID %in% subjects_with_zero_score, ]
trial_data <- wide_data[-c(1,2)]

# Ensure that all data is numeric
trial_data <- data.frame(lapply(trial_data, as.numeric))

out <- colMeans(trial_data)
out

# Calculate item-rest correlations
item_rest_correlations <- compute_item_rest_correlations(trial_data)

# Print the item-rest correlations
print(item_rest_correlations)


# Set the threshold for the correlation
correlation_threshold <- 0.15

# Find trials with correlations below the threshold
trials_below_threshold <- names(item_rest_correlations)[item_rest_correlations < correlation_threshold]
trials_below_zero <- names(item_rest_correlations)[item_rest_correlations < 0]


# Output the trials below the threshold
print(trials_below_threshold)
print(trials_below_zero)

data_cats <- read_csv("csvs/csv_data_maker_threeACF-noise.csv")

# Compute accuracy for each subject under each condition
accuracy_data <- data_cats %>%
  group_by(SbjID, View, Noise) %>%
  summarize(
    total_trials = n(),
    number_correct = sum(AnsCatagory == "correct"),
    accuracy = number_correct / total_trials,
    .groups = 'drop'  # This ensures the group_by is dropped after summarization
  )

# Perform statistical tests (e.g., ANOVA) to compare accuracy across conditions
# Here we use ANOVA as an example; you may need a different test depending on your data structure
ANOVA_results <- aov(accuracy ~ View * Noise, data = accuracy_data)
summary(ANOVA_results)

# Output the results
print(accuracy_data)
print(summary(ANOVA_results))


# Convert AnsCatagory to a binary variable (1 for 'correct', 0 for 'incorrect')
data_cats$AnsCatagory <- ifelse(data_cats$AnsCatagory == "correct", 1, 0)

# Set up the logistic regression model
logistic_model <- glm(AnsCatagory ~ View + Noise, data = data_cats, family = binomial)

# Summarize the model
summary(logistic_model)

# To get the odds ratios instead of the log(odds) you can exponentiate the coefficients
exp(coef(logistic_model))


splitHalf(wide_data[,3:50], check.keys=FALSE)