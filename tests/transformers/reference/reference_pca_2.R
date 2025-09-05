library(onlinePCA)

# --- 1. Define Expanded Parameters ---
n <- 10  # number of sample paths
d <- 8   # number of observation points
q <- 2   # number of PCs to compute
n0 <- 5  # number of sample paths used for initialization

# --- 2. Hardcoded Artificial Data ---
# This data is identical to the data provided for the Python implementation
data <- c(5.74, 3.28, 5.25, 6.12, 5.34, 4.33, 7.33, 4.49,
          6.13, 5.11, 4.42, 5.08, 5.82, 5.08, 6.49, 4.38,
          3.16, 4.29, 5.49, 4.03, 3.48, 5.48, 2.82, 6.11,
          5.91, 5.76, 5.45, 6.88, 5.61, 5.03, 5.61, 4.22,
          5.94, 5.27, 4.16, 5.54, 4.21, 6.13, 2.44, 4.88,
          2.28, 5.68, 5.82, 5.56, 5.99, 4.41, 5.12, 4.84,
          7.02, 3.82, 3.32, 5.11, 6.51, 5.17, 4.79, 4.26,
          3.88, 4.43, 6.55, 4.58, 4.38, 4.22, 5.75, 4.41,
          2.89, 4.49, 4.59, 4.43, 3.42, 6.11, 4.29, 2.67,
          4.25, 4.78, 6.78, 4.22, 6.15, 4.43, 5.73, 5.08)
mat <- t(matrix(data, nrow = d, ncol = n))


# --- 3. Initialization Step ---
# Perform initial PCA on the first n0 samples
pca_initial <- stats::prcomp(mat[1:n0,], center=FALSE)
# Store the results in the list format required by incRpca
pca <- list(values = pca_initial$sdev[1:q]^2, vectors = pca_initial$rotation[,1:q])

print('--- Initial PCA state after n0=5 samples ---')
print(pca)


# --- 4. Incremental Learning Steps ---
# Loop through the remaining data points from n0+1 to n
for (i in (n0 + 1):n) {
  # Get the new data vector for the current step
  new_vector <- mat[i,]

  # The count of samples seen *before* this new one is (i - 1)
  pca <- onlinePCA::incRpca(pca$values, pca$vectors, new_vector, i - 1, q = q)

  # Print the state after learning the new vector
  print(paste('--- PCA state after learning vector', i, '---'))
  print(pca)
}