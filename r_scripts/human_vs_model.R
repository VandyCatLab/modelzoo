# Loading packages ----
library(tidyverse)
library(psych)
library(ggplot2)
library(GGally)
library(ggpp)
library(BayesFactor)
library(rstanarm)
library(lavaan)
library(semPlot)

# Helper functions ----
corMatrixPlot <- function(
    data,
    reliability,
    nullInterval = NULL,
    rscale = NULL,
    relSymbol = "r",
    draw_dist = FALSE,
    showN = FALSE) {
    # Setup test labels and reliability symbols
    testLabels <- names(data)
    nTests <- length(testLabels)
    if (length(relSymbol) == 1) {
        relSymbol <- rep(relSymbol, times = nTests)
    }

    # Create base plot matrix
    plots <- list()
    for (i in 1:(nTests * nTests)) {
        plots[[i]] <- ggally_text(paste("Plot #", i, sep = ""))
    }
    corMatPlot <- ggmatrix(
        plots = plots,
        nrow = nTests,
        ncol = nTests,
        xAxisLabels = testLabels,
        yAxisLabels = testLabels,
    ) +
        theme(
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            panel.background = element_rect(color = "black"),
            strip.background = element_rect(
                fill = "white"
            )
        )

    # Edit each cell
    for (row in 1:corMatPlot$nrow) {
        for (col in 1:corMatPlot$ncol) {
            if (row == col) { # Diagonal reliability
                rel <- round(reliability[row], digits = 2)
                rel <- str_replace(rel, "0.", ".")
                if (str_length(rel) == 2) {
                    rel <- paste0(rel, "0")
                }

                corMatPlot[row, row] <- ggplot(
                    data,
                    aes(x = .data[[testLabels[row]]])
                ) +
                    geom_density(
                        color = ifelse(draw_dist, "gray", NA)
                    ) +
                    annotate(
                        "text_npc",
                        npcx = "center",
                        npcy = "middle",
                        label = paste0(
                            relSymbol[row], " = ", rel,
                            "\n\n", testLabels[row]
                        )
                    ) +
                    theme(
                        panel.grid.major = element_blank(),
                        panel.grid.minor = element_blank(),
                        axis.text.x = element_blank(),
                        axis.text.y = element_blank(),
                        axis.ticks = element_blank()
                    )
            } else if (row < col) { # Off diagonal correlations
                x <- pull(data[, col])
                y <- pull(data[, row])

                # Calculate BF
                corBF <- correlationBF(
                    x, y,
                    nullInterval = nullInterval,
                    rscale = rscale
                )

                # Correlation with BFs
                if (length(corBF) == 2) {
                    bfSamples <- posterior(
                        corBF[2],
                        iterations = 10000,
                        progress = FALSE
                    )
                    rho <- summary(bfSamples)$statistics[1, 1]
                    bf <- extractBF(corBF[2])$bf
                } else {
                    bfSamples <- posterior(
                        corBF,
                        iterations = 10000,
                        progress = FALSE
                    )
                    rho <- summary(bfSamples)$statistics[1, 1]
                    bf <- extractBF(corBF)$bf
                }

                # Make the text
                rhoText <- str_replace(
                    as.character(round(rho, digits = 2)), "0.", "."
                )
                BFText <- as.character(round(bf, digits = 2))

                if (showN) {
                    # Count number of full observations
                    nObs <- sum(complete.cases(data[, c(row, col)]))
                    corText <- paste0(
                        "atop(", paste0("r(", nObs, ') == "', rhoText), '",',
                        paste0("\nBF[+0] == ", BFText), ")"
                    )
                } else {
                    corText <- paste0(
                        "atop(", paste0('r == "', rhoText), '",',
                        paste0("\nBF[+0] == ", BFText), ")"
                    )
                }

                # Place text in the cell
                pieData <- data.frame(
                    Value = c(bf, 1),
                    Hypo = c("H1", "H0")
                )
                pieData$Fraction <- pieData$Value / sum(pieData$Value)
                pieData$YMax <- cumsum(pieData$Fraction)
                pieData$YMin <- c(0, head(pieData$YMax, n = -1))

                corMatPlot[row, col] <- ggplot(pieData) +
                    aes(
                        ymax = YMax, # nolint
                        ymin = YMin, xmax = 4, xmin = 1, fill = Hypo # nolint
                    ) +
                    geom_rect(color = "white", alpha = .5) +
                    coord_polar("y", start = (1 / (bf + 1)) * pi + pi) +
                    scale_fill_manual(values = c("coral", "aquamarine3")) +
                    xlim(c(1, 4)) +
                    annotate(
                        "text",
                        x = 1,
                        y = 1,
                        label = corText,
                        parse = TRUE
                    ) +
                    theme(
                        panel.background = element_rect(color = "white"),
                        axis.text = element_blank(),
                        axis.ticks = element_blank()
                    )


                # Create scatter plot for the lower diagonal cell
                corMatPlot[col, row] <- tryCatch(
                    {
                        # Calculate line of best fit
                        plotLM <- stan_glm(
                            as.formula(
                                paste0(testLabels[col], "~", testLabels[row])
                            ),
                            data = data,
                            refresh = 0,
                        )
                        drawsLines <- as.data.frame(as.matrix(plotLM))
                        colnames(drawsLines) <- c("intercept", "slope", "sigma")
                        ggplot(data, aes(
                            x = .data[[testLabels[row]]],
                            y = .data[[testLabels[col]]]
                        )) +
                            geom_abline(
                                data = drawsLines,
                                aes(intercept = intercept, slope = slope), # nolint
                                color = "gray",
                                linewidth = 0.1,
                                alpha = 0.1
                            ) +
                            geom_abline(
                                intercept = coef(plotLM)[1],
                                slope = coef(plotLM)[2]
                            ) +
                            geom_point(alpha = 0.3)
                        # Aspect ratio issue https://github.com/ggobi/ggally/issues/415
                    },
                    error = function(e) {
                        ggplot(data, aes(
                            x = .data[[testLabels[row]]],
                            y = .data[[testLabels[col]]]
                        )) +
                            geom_point(alpha = 0.3)
                    }
                )
            }
        }
    }

    return(corMatPlot)
}

long2Matrix <- function(data, idCol, corrCol, trialNCol, standardize = FALSE) {
    # Get subjects and trials
    sbjs <- data[idCol] |>
        unique() |>
        pull() |>
        sort()
    trials <- unique(data[trialNCol]) |>
        pull() |>
        as.character()

    # Create a tibble with subject id row
    taskMatrix <- tibble(!!idCol := sbjs)

    for (trial in trials) {
        trialData <- data[data[trialNCol] == trial, c(idCol, corrCol)]
        # Rename corr to the trial
        names(trialData)[2] <- trial

        taskMatrix <- suppressMessages(taskMatrix |>
            left_join(
                trialData
            ))
    }

    # Set the SbjID column as row names
    taskMatrix <- column_to_rownames(taskMatrix, var = idCol)

    # Save raw matrix for later
    rawInfo <- taskMatrix

    # Add mean row
    taskMatrix <- rbind(taskMatrix, mean = apply(taskMatrix, 2, mean))

    # Add participant mean row
    taskMatrix <- cbind(taskMatrix, Sum = apply(taskMatrix, 1, sum))

    # Add correlation row
    totals <- taskMatrix[seq_len(nrow(taskMatrix) - 1), "Sum"]
    tmp <- c()
    for (col in seq_len(ncol(rawInfo))) {
        tmp <- c(tmp, cor(rawInfo[col], totals - rawInfo[col]))
    }
    taskMatrix["cor", ] <- c(tmp, 1)

    # Add variance row
    sbjs <- as.character(t(unique(data[idCol])))
    taskMatrix <- rbind(taskMatrix, var = apply(taskMatrix[sbjs, ], 2, var))

    return(taskMatrix)
}

disattCor <- function(rxy, rxx, ryy) {
    return(rxy / sqrt(rxx * ryy))
}

matrix2Rel <- function(mat) {
    tmp <- mat[, !is.na(mat["cor", ])]
    tmp <- tmp[seq_len(nrow(tmp) - 3), seq_len(ncol(tmp) - 1)]

    return(splitHalf(tmp, check.keys = FALSE))
}

aggRel <- function(data, rel) {
    # Calculate correlation matrix
    corMat <- cor(data, use = "complete.obs")
    diag(corMat) <- NA

    numer <- sum(rel) + sum(corMat, na.rm = TRUE)
    denom <- length(rel) + sum(corMat, na.rm = TRUE)

    return(numer / denom)
}

# Loading data ----
# Load human data
humanLE <- read_csv("./data_storage/humanData/le1.csv") |>
    column_to_rownames("SbjID")
humanMatch <- read_csv("./data_storage/humanData/match1.csv") |>
    column_to_rownames("SbjID")
humanMOO <- read_csv("./data_storage/humanData/moo1.csv") |>
    column_to_rownames("SbjID")

# Change the column names to just be the trial number (removing the X)
colnames(humanLE) <- gsub("X", "", colnames(humanLE))
colnames(humanMatch) <- gsub("X", "", colnames(humanMatch))
colnames(humanMOO) <- gsub("X", "", colnames(humanMOO))

# Load human data 2
humanLE2 <- read_csv("./data_storage/humanData/le2.csv") |>
    column_to_rownames("SbjID")
humanMatch2 <- read_csv("./data_storage/humanData/match2.csv") |>
    column_to_rownames("SbjID")
humanMOO2 <- read_csv("./data_storage/humanData/moo2.csv") |>
    column_to_rownames("SbjID")

# Change the column names to just be the trial number (removing the Corr.)
colnames(humanLE2) <- gsub("Corr.", "", colnames(humanLE2))
colnames(humanMatch2) <- gsub("Corr.", "", colnames(humanMatch2))
colnames(humanMOO2) <- gsub("Corr.", "", colnames(humanMOO2))

# Load model data
modelLE <- read_csv("./data_storage/results/results_learn_exemp_noise-1.0.csv")
modelLE <- long2Matrix(modelLE, "ModelName", "Corr", "Trial")
modelLE <- modelLE[!rownames(modelLE) %in% c("mean", "cor", "var"), !colnames(modelLE) %in% c("Sum")]

modelMatch <- read_csv("./data_storage/results/results_threeAFC_noise-1.0_encNoise-1.0.csv")
modelMatch <- long2Matrix(modelMatch, "ModelName", "Corr", "Trial")
modelMatch <- modelMatch[!rownames(modelMatch) %in% c("mean", "cor", "var"), !colnames(modelMatch) %in% c("Sum")]

modelMOO <- read_csv("./data_storage/results/results_many_odd_noise-1.0_encNoise-1.0.csv")
modelMOO <- long2Matrix(modelMOO, "ModelName", "Corr", "Trial")
modelMOO <- modelMOO[!rownames(modelMOO) %in% c("mean", "cor", "var"), !colnames(modelMOO) %in% c("Sum")]

# Get the rownames of all models
modelNames <- unique(c(rownames(modelLE), rownames(modelMatch), rownames(modelMOO)))
modelNames <- modelNames[modelNames %in% rownames(modelLE)]
modelNames <- modelNames[modelNames %in% rownames(modelMatch)]
modelNames <- modelNames[modelNames %in% rownames(modelMOO)]

# Filter the models
modelLE <- modelLE[modelNames, ]
modelMatch <- modelMatch[modelNames, ]
modelMOO <- modelMOO[modelNames, ]

# Calculate subject wise performance on each task
humanSummary <- tibble(
    SbjID = rownames(humanLE),
    LE = rowMeans(humanLE, na.rm = TRUE),
    Match = rowMeans(humanMatch, na.rm = TRUE),
    MOO = rowMeans(humanMOO, na.rm = TRUE)
) |>
    mutate(
        o = c((scale(LE) + scale(Match) + scale(MOO))) / 3
    )
humanSummary2 <- tibble(
    SbjID = rownames(humanLE2),
    LE = rowMeans(humanLE2, na.rm = TRUE),
    Match = rowMeans(humanMatch2, na.rm = TRUE),
    MOO = rowMeans(humanMOO2, na.rm = TRUE)
) |>
    mutate(
        o = c((scale(LE) + scale(Match) + scale(MOO))) / 3
    )

modelSummary <- tibble(
    SbjID = rownames(modelLE),
    LE = rowMeans(modelLE, na.rm = TRUE),
    Match = rowMeans(modelMatch, na.rm = TRUE),
    MOO = rowMeans(modelMOO, na.rm = TRUE)
) |>
    mutate(
        o = c((scale(LE) + scale(Match) + scale(MOO))) / 3
    )

# Load model attribute summary
modelInfo <- read_csv("./data_storage/results/models_summary_timm.csv")

# Test specific measures ----
# Reliability
humanLERel <- splitHalf(humanLE, check.keys = FALSE)$lambda2
humanMatchRel <- splitHalf(humanMatch, check.keys = FALSE)$lambda2
humanMOORel <- splitHalf(humanMOO, check.keys = FALSE)$lambda2
humanLE2Rel <- splitHalf(humanLE2, check.keys = FALSE)$lambda2
humanMatch2Rel <- splitHalf(humanMatch2, check.keys = FALSE)$lambda2
humanMOO2Rel <- splitHalf(humanMOO2, check.keys = FALSE, covar = TRUE)$lambda2

modelLERel <- splitHalf(modelLE, check.keys = FALSE)$lambda2
modelMatchRel <- splitHalf(modelMatch, check.keys = FALSE)$lambda2
modelMOORel <- splitHalf(modelMOO, check.keys = FALSE, covar = TRUE)$lambda2

# Get trial difficulty
humanLEDiff <- colMeans(humanLE)
humanMatchDiff <- colMeans(humanMatch)
humanMOODiff <- colMeans(humanMOO)
humanLE2Diff <- colMeans(humanLE2)
humanMatch2Diff <- colMeans(humanMatch2)
humanMOO2Diff <- colMeans(humanMOO2)

modelLEDiff <- colMeans(modelLE)
modelMatchDiff <- colMeans(modelMatch)
modelMOODiff <- colMeans(modelMOO)

# Correlate difficulty between models and humans
LEDiffCor <- cor(humanLEDiff, modelLEDiff, use = "complete.obs")
matchDiffCor <- cor(humanMatchDiff, modelMatchDiff)
MOODiffCor <- cor(humanMOODiff, modelMOODiff)

# Correlate difficulty between humans and humans
LEHumanDiffCor <- cor(humanLEDiff, humanLE2Diff)
matchHumanDiffCor <- cor(humanMatchDiff, humanMatch2Diff)
MOOHumanDiffCor <- cor(humanMOODiff, humanMOO2Diff)

# Plot difficulty correlations where points are trial numbers
LEDiffPlot <- ggplot(
    data = tibble(
        trial = seq_len(length(humanLEDiff)),
        human = humanLEDiff,
        model = modelLEDiff
    ),
    aes(x = human, y = model)
) +
    geom_text(aes(label = trial)) +
    geom_abline(intercept = 0, slope = 1) +
    annotate(
        "text",
        x = 0.9,
        y = 0,
        label = paste0("r = ", round(LEDiffCor, digits = 2))
    ) +
    labs(x = "Human", y = "Model", title = "LE Difficulty") +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
    theme_bw() +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
    )
LEDiffPlot

matchDiffPlot <- ggplot(
    data = tibble(
        trial = seq_len(length(humanMatchDiff)),
        human = humanMatchDiff,
        model = modelMatchDiff
    ),
    aes(x = human, y = model)
) +
    geom_text(aes(label = trial)) +
    geom_abline(intercept = 0, slope = 1) +
    annotate(
        "text",
        x = 0.9,
        y = 0,
        label = paste0("r = ", round(matchDiffCor, digits = 2))
    ) +
    labs(x = "Human", y = "Model", title = "Match Difficulty") +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
    theme_bw() +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
    )
matchDiffPlot

MOODiffPlot <- ggplot(
    data = tibble(
        trial = seq_len(length(humanMOODiff)),
        human = humanMOODiff,
        model = modelMOODiff
    ),
    aes(x = human, y = model)
) +
    geom_text(aes(label = trial)) +
    geom_abline(intercept = 0, slope = 1) +
    annotate(
        "text",
        x = 0.9,
        y = 0,
        label = paste0("r = ", round(MOODiffCor, digits = 2))
    ) +
    labs(x = "Human", y = "Model", title = "MOO Difficulty") +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
    theme_bw() +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
    )
MOODiffPlot

LEHumanDiffPlot <- ggplot(
    data = tibble(
        trial = seq_len(length(humanLEDiff)),
        human1 = humanLEDiff,
        human2 = humanLE2Diff
    ),
    aes(x = human1, y = human2)
) +
    geom_text(aes(label = trial)) +
    geom_abline(intercept = 0, slope = 1) +
    annotate(
        "text",
        x = 0.9,
        y = 0,
        label = paste0("r = ", round(LEHumanDiffCor, digits = 2))
    ) +
    labs(x = "Human 1", y = "Human 2", title = "LE Difficulty") +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
    theme_bw() +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
    )
LEHumanDiffPlot

matchHumanDiffPlot <- ggplot(
    data = tibble(
        trial = seq_len(length(humanMatchDiff)),
        human1 = humanMatchDiff,
        human2 = humanMatch2Diff
    ),
    aes(x = human1, y = human2)
) +
    geom_text(aes(label = trial)) +
    geom_abline(intercept = 0, slope = 1) +
    annotate(
        "text",
        x = 0.9,
        y = 0,
        label = paste0("r = ", round(matchHumanDiffCor, digits = 2))
    ) +
    labs(x = "Human 1", y = "Human 2", title = "Match Difficulty") +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
    theme_bw() +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
    )

matchHumanDiffPlot

MOOHumanDiffPlot <- ggplot(
    data = tibble(
        trial = seq_len(length(humanMOODiff)),
        human1 = humanMOODiff,
        human2 = humanMOO2Diff
    ),
    aes(x = human1, y = human2)
) +
    geom_text(aes(label = trial)) +
    geom_abline(intercept = 0, slope = 1) +
    annotate(
        "text",
        x = 0.9,
        y = 0,
        label = paste0("r = ", round(MOOHumanDiffCor, digits = 2))
    ) +
    labs(x = "Human 1", y = "Human 2", title = "MOO Difficulty") +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
    theme_bw() +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
    )

MOOHumanDiffPlot

# Split Model Analyses ----
# Split the model data into two groups randomly
splitModel <- sample(1:2, nrow(modelSummary), replace = TRUE)

# Calculate trial difficulty for each group
modelLE1 <- modelLE[splitModel == 1, ]
modelMatch1 <- modelMatch[splitModel == 1, ]
modelMOO1 <- modelMOO[splitModel == 1, ]
modelLE2 <- modelLE[splitModel == 2, ]
modelMatch2 <- modelMatch[splitModel == 2, ]
modelMOO2 <- modelMOO[splitModel == 2, ]

# Calculate difficulty correlations between the two groups
LEModelDiffCor <- cor(colMeans(modelLE1), colMeans(modelLE2))
matchModelDiffCor <- cor(colMeans(modelMatch1), colMeans(modelMatch2))
MOOModelDiffCor <- cor(colMeans(modelMOO1), colMeans(modelMOO2))

# Plot
LEModelDiffPlot <- ggplot(
    data = tibble(
        trial = seq_len(length(colMeans(modelLE1))),
        model1 = colMeans(modelLE1),
        model2 = colMeans(modelLE2)
    ),
    aes(x = model1, y = model2)
) +
    geom_text(aes(label = trial)) +
    geom_abline(intercept = 0, slope = 1) +
    annotate(
        "text",
        x = 0.9,
        y = 0,
        label = paste0("r = ", round(LEModelDiffCor, digits = 2))
    ) +
    labs(x = "Model 1", y = "Model 2", title = "LE Difficulty") +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
    theme_bw() +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
    )

LEModelDiffPlot

matchModelDiffPlot <- ggplot(
    data = tibble(
        trial = seq_len(length(colMeans(modelMatch1))),
        model1 = colMeans(modelMatch1),
        model2 = colMeans(modelMatch2)
    ),
    aes(x = model1, y = model2)
) +
    geom_text(aes(label = trial)) +
    geom_abline(intercept = 0, slope = 1) +
    annotate(
        "text",
        x = 0.9,
        y = 0,
        label = paste0("r = ", round(matchModelDiffCor, digits = 2))
    ) +
    labs(x = "Model 1", y = "Model 2", title = "Match Difficulty") +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
    theme_bw() +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
    )

matchModelDiffPlot

MOOModelDiffPlot <- ggplot(
    data = tibble(
        trial = seq_len(length(colMeans(modelMOO1))),
        model1 = colMeans(modelMOO1),
        model2 = colMeans(modelMOO2)
    ),
    aes(x = model1, y = model2)
) +
    geom_text(aes(label = trial)) +
    geom_abline(intercept = 0, slope = 1) +
    annotate(
        "text",
        x = 0.9,
        y = 0,
        label = paste0("r = ", round(MOOModelDiffCor, digits = 2))
    ) +
    labs(x = "Model 1", y = "Model 2", title = "MOO Difficulty") +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
    theme_bw() +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
    )

MOOModelDiffPlot

# Create model/human correlation matrix plot ----
lambda2 <- "\U03BB\U2082"
humanCorMatrix <- corMatrixPlot(
    data = humanSummary |> select(LE, Match, MOO),
    reliability = c(humanLERel, humanMatchRel, humanMOORel),
    nullInterval = c(-1, 0),
    rscale = 1 / 3,
    relSymbol = lambda2,
    draw_dist = TRUE,
    showN = TRUE
)
humanCorMatrix

human2CorMatrix <- corMatrixPlot(
    data = humanSummary2 |> select(LE, Match, MOO),
    reliability = c(humanLERel, humanMatchRel, humanMOORel),
    nullInterval = c(-1, 0),
    rscale = 1 / 3,
    relSymbol = lambda2,
    draw_dist = TRUE,
    showN = TRUE
)
human2CorMatrix

modelCorMatrix <- corMatrixPlot(
    data = modelSummary |> select(LE, Match, MOO),
    reliability = c(modelLERel, modelMatchRel, modelMOORel),
    nullInterval = c(-1, 0),
    rscale = 1 / 3,
    relSymbol = lambda2,
    draw_dist = TRUE,
    showN = TRUE
)
modelCorMatrix

# Measurement invariance testing between human and model ----
# Concatenate the model and the human data
allSummary <- rbind(
    humanSummary |> mutate(Group = "Human"),
    modelSummary |> mutate(Group = "Model")
)

oModel <- "
    oLat =~ LE + Match + MOO
"
confInvarFit <- cfa(
    oModel,
    data = allSummary,
    group = "Group"
)
metricInvarFit <- cfa(
    model = oModel,
    data = allSummary,
    group = "Group",
    group.equal = c("loadings")
)
scalarInvarFit <- cfa(
    model = oModel,
    data = allSummary,
    group = "Group",
    group.equal = c("loadings", "intercepts")
)
residInvarFit <- cfa(
    model = oModel,
    data = allSummary,
    group = "Group",
    group.equal = c("loadings", "intercepts", "residuals")
)

lavTestLRT(confInvarFit, metricInvarFit, scalarInvarFit, residInvarFit)
summary(confInvarFit, standardized = TRUE, fit.measures = TRUE)

# Manual partial invariance model
partialMetricInvarModel <- "
    # Remember that LE is the first and is always set to 1 anyways
    oLat =~ c(l1, l1)*LE + c(l2, l2)*Match + MOO
"
partialMetricInvarFit <- cfa(
    model = partialMetricInvarModel,
    data = allSummary,
    group = "Group"
)

partialScalarInvar <- "
    LE ~ c(i1, i1)*1
    Match ~ c(i2, i2)*1
"
partialScalarInvarFit <- cfa(
    model = c(partialMetricInvarModel, partialScalarInvar),
    data = allSummary,
    group = "Group"
)
lavTestLRT(confInvarFit, partialMetricInvarFit, partialScalarInvarFit)

# Plot confInvarFit model
semPaths(
    confInvarFit,
    "std",
    intercepts = FALSE,
    edge.color = "black",
    edge.label.cex = 2,
    ask = FALSE,
    panelGroups = TRUE
)

semPaths(
    partialMetricInvarFit,
    "std",
    intercepts = FALSE,
    edge.color = "black",
    edge.label.cex = 2,
    ask = FALSE,
    panelGroups = TRUE
)

# Measurement invariance between human and human ----
# Bind together human data only
allHumanSummary <- rbind(
    humanSummary |> mutate(Group = "Human"),
    humanSummary2 |> mutate(Group = "Human2")
)

# We can use the exact same models as before, just fits
confInvarHumanFit <- cfa(
    oModel,
    data = allHumanSummary,
    group = "Group"
)

metricInvarHumanFit <- cfa(
    model = oModel,
    data = allHumanSummary,
    group = "Group",
    group.equal = c("loadings")
)

scalarInvarHumanFit <- cfa(
    model = oModel,
    data = allHumanSummary,
    group = "Group",
    group.equal = c("loadings", "intercepts")
)

residInvarHumanFit <- cfa(
    model = oModel,
    data = allHumanSummary,
    group = "Group",
    group.equal = c("loadings", "intercepts", "residuals")
)

lavTestLRT(
    confInvarHumanFit,
    metricInvarHumanFit,
    scalarInvarHumanFit,
    residInvarHumanFit
)

semPaths(
    metricInvarHumanFit,
    "std",
    intercepts = FALSE,
    edge.color = "black",
    edge.label.cex = 2,
    ask = FALSE,
    panelGroups = TRUE
)

# Model attribute analysis ----
# First figure out which models are missing from the modelInfo file
missingModels <- modelSummary$SbjID[!modelNames %in% modelInfo$Model]

# Combine model info with modelSummary
modelSummary <- modelSummary |>
    left_join(modelInfo, by = c("SbjID" = "Model"))

# Change any column names with spaces to use underscore
colnames(modelSummary) <- gsub(" ", "_", colnames(modelSummary))

regModel <- lm(
    o ~ Layers + Parameters + as.factor(Training_Dataset) + as.factor(Family),
    data = modelSummary,
    na.action = na.omit
)
summary(regModel)
