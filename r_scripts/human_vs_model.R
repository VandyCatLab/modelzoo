library(tidyverse)
library(lavaan)
library(psych)
library(ggplot2)
library(GGally)
library(ggpp)
library(BayesFactor)
library(rstanarm)

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
                    aes_string(x = testLabels[row])
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
                            as.formula(paste0(testLabels[col], "~", testLabels[row])),
                            data = data,
                            refresh = 0,
                        )
                        drawsLines <- as.data.frame(as.matrix(plotLM))
                        colnames(drawsLines) <- c("intercept", "slope", "sigma")
                        ggplot(data, aes_string(
                            x = testLabels[row],
                            y = testLabels[col]
                        )) +
                            geom_abline(
                                data = drawsLines,
                                aes(intercept = intercept, slope = slope), # nolint
                                color = "gray",
                                size = 0.1,
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
                        ggplot(data, aes_string(
                            x = testLabels[row],
                            y = testLabels[col]
                        )) +
                            geom_point(alpha = 0.3)
                    }
                )
            }
        }
    }

    return(corMatPlot)
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

# Load model data
modelLE <- read_csv("./data_storage/results/r_results/wides/wide_LE.csv") |>
    column_to_rownames("SbjID") |>
    select(-PercentageCorrect)
modelMatch <- read_csv("./data_storage/results/r_results/wides/wide_3ACF.csv") |>
    column_to_rownames("SbjID") |>
    select(-PercentageCorrect)
modelMOO <- read_csv("./data_storage/results/r_results/wides/wide_MOO.csv") |>
    column_to_rownames("SbjID") |>
    select(-PercentageCorrect)

# Change column name to just be the trial number (remove the X and +1 index)
colnames(modelLE) <- as.character(
    as.numeric(gsub("Trial", "", colnames(modelLE))) + 1
)
colnames(modelMatch) <- as.character(
    as.numeric(gsub("Trial", "", colnames(modelMatch))) + 1
)
colnames(modelMOO) <- as.character(
    as.numeric(gsub("Trial", "", colnames(modelMOO))) + 1
)

# Find the colums where every model gets the LE trials correct
LECorrect <- apply(modelLE, 2, function(x) all(x == 1))

# Set the LE trials to NA for the models that got them all correct
modelLE[, LECorrect] <- NA

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
modelSummary <- tibble(
    SbjID = rownames(modelLE),
    LE = rowMeans(modelLE, na.rm = TRUE),
    Match = rowMeans(modelMatch, na.rm = TRUE),
    MOO = rowMeans(modelMOO, na.rm = TRUE)
) |>
    mutate(
        o = c((scale(LE) + scale(Match) + scale(MOO))) / 3
    )


# Test specific measures ----
# Reliability
humanLERel <- splitHalf(humanLE, check.keys = FALSE)$lambda2
humanMatchRel <- splitHalf(humanMatch, check.keys = FALSE)$lambda2
humanMOORel <- splitHalf(humanMOO, check.keys = FALSE)$lambda2

# Only use LE trials that have no NA
tmp <- modelLE[, !is.na(modelLE[1, ])]
modelLERel <- splitHalf(tmp, check.keys = FALSE)$lambda2
modelMatchRel <- splitHalf(modelMatch, check.keys = FALSE)$lambda2
modelMOORel <- splitHalf(modelMOO, check.keys = FALSE)$lambda2

# Get trial difficulty
humanLEDiff <- colMeans(humanLE)
humanMatchDiff <- colMeans(humanMatch)
humanMOODiff <- colMeans(humanMOO)

modelLEDiff <- colMeans(modelLE)
modelMatchDiff <- colMeans(modelMatch)
modelMOODiff <- colMeans(modelMOO)

# Correlate difficulty between models and humans
LEDiffCor <- cor(humanLEDiff, modelLEDiff, use = "complete.obs")
matchDiffCor <- cor(humanMatchDiff, modelMatchDiff)
MOODiffCor <- cor(humanMOODiff, modelMOODiff)

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
