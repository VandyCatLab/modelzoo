# Load necessary libraries
library(tidyverse) 
library(dplyr)
library(GGally)
library(psych)
library(readr)
library(ggplot2)
library(gridExtra)
library(broom)

# Read the CSV file for one of them
csv_filename = "csvs/csv_data_maker_threeACF-noise.csv"
data <- read.csv(csv_filename)


# --- Plot 1: Histogram with Normal Distribution Overlay ---

# Calculate the average score for each subject
average_scores <- data %>%
  group_by(SbjID) %>%
  summarize(Avg_Score = mean(AnsCatagory == 'correct'))

# Calculate mean and standard deviation for the normal distribution
mean_score <- mean(average_scores$Avg_Score)
sd_score <- sd(average_scores$Avg_Score)

# Create the first plot
plot1 <- ggplot(average_scores, aes(x = Avg_Score)) +
  geom_histogram(aes(y = ..density..), binwidth = 0.05, fill = "blue", alpha = 0.5) +
  stat_function(fun = dnorm, args = list(mean = mean_score, sd = sd_score), color = "red", size = 1) +
  xlab("Average Score") +
  ylab("Density") +
  ggtitle(sprintf("Average Scores (%s)", round(mean_score,2)))

# --- Plot 2: Comparison of First Half vs. Second Half ---

# Convert AnsCatagory to numeric values (1 for 'correct', 0 for 'incorrect')
data$AnsCatagory <- ifelse(data$AnsCatagory == 'correct', 1, 0)

# Determine the halfway point of the trials
half_trial <- max(data$TrialN) / 2

# Calculate average scores for first and second half
avg_scores_first <- data %>%
  filter(TrialN <= half_trial) %>%
  group_by(SbjID) %>%
  summarize(Avg_Score = mean(AnsCatagory))

avg_scores_second <- data %>%
  filter(TrialN > half_trial) %>%
  group_by(SbjID) %>%
  summarize(Avg_Score = mean(AnsCatagory))

# Calculate means and standard deviations for each half
mean_first <- mean(avg_scores_first$Avg_Score)
sd_first <- sd(avg_scores_first$Avg_Score)
mean_second <- mean(avg_scores_second$Avg_Score)
sd_second <- sd(avg_scores_second$Avg_Score)

# Combine data and create a factor for the legend
avg_scores_combined <- rbind(avg_scores_first %>% mutate(Half = sprintf("First Half (%s)", round(mean_first, 2))),
                             avg_scores_second %>% mutate(Half = sprintf("Second Half (%s)", round(mean_second,2))))

# Plot histograms with a legend
plot2 <- ggplot(avg_scores_combined, aes(x = Avg_Score, fill = Half)) +
  geom_histogram(position = "identity", alpha = 0.5, binwidth = 0.05, aes(y = ..density..)) +
  stat_function(fun = dnorm, args = list(mean = mean_first, sd = sd_first), color = "darkblue", size = 1) +
  stat_function(fun = dnorm, args = list(mean = mean_second, sd = sd_second), color = "darkred", size = 1) +
  scale_fill_manual(values = c("blue", "red")) +
  xlab("Average Score") +
  ylab("Density") +
  ggtitle("Average Scores for First and Second Half") +
  theme_minimal() +
  theme(legend.position = "top", legend.title = element_blank())


# Combine the plots side by side
grid.arrange(plot1, plot2, ncol = 2)

# Determine the halfway point of the trials for each subject
half_trial <- ceiling(max(data$TrialN) / 2)

# Define the threshold for significant improvement
improvement_threshold <- 1.1  # Change this value to your desired threshold

# Calculate average scores for each half for each subject
avg_scores <- data %>%
  group_by(SbjID) %>%
  summarize(First_Half_Avg = mean(AnsCatagory[TrialN <= half_trial]), 
            Second_Half_Avg = mean(AnsCatagory[TrialN > half_trial]))

# Identify subjects with higher scores in the second half by at least the threshold
improved_subjects <- filter(avg_scores, Second_Half_Avg / First_Half_Avg > improvement_threshold)

# Print subjects who improved in the second half by at least the threshold
print(improved_subjects)


# get wide datas to Jason, get attibutes