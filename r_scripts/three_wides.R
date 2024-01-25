# Load necessary libraries
library(tidyverse) 
library(dplyr)
library(GGally)
library(psych)
library(readr)

# function to get wides
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


wide_3ACF = transform_to_wide(data1)
wide_MOO = transform_to_wide(data2)
wide_LE = transform_to_wide(data3)

# write wides to csvs
write.csv(wide_3ACF, "./csvs/wides/wide_3ACF.csv", row.names = FALSE)
write.csv(wide_MOO, "./csvs/wides/wide_MOO.csv", row.names = FALSE)
write.csv(wide_LE, "./csvs/wides/wide_LE.csv", row.names = FALSE)
