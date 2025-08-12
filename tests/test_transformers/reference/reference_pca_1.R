# Load the necessary library for online PCA
library(onlinePCA)

# --- 1. Define Parameters ---
# These parameters control the dimensions and initialization of our test data.
n <- 3   # Total number of initial sample paths
d <- 5   # Dimensionality (number of features or observation points)
q <- 2   # The number of principal components we want to compute
n0 <- 3  # The number of samples to use for the initial batch PCA

# --- 2. Define Hardcoded Data ---
# The initial dataset used to start the PCA process.
initial_data <- t(c(1,  2,  2.5,  5,  5,  # Sample 1
                    10, 10.5, 11, 8,  4,  # Sample 2
                    3,  3.5,  7, 10,  9)) # Sample 3

# Reshape the vector into a matrix with samples as rows and features as columns.
# The `t()` function transposes the matrix to the correct orientation.
mat <- t(matrix(initial_data, nrow = d, ncol = n))

# Define the new data points that will be learned incrementally.
x <- c(2, 3, 3.5, 11, 5) # This will be the 4th sample
y <- c(4, 3.4, 9.5, 1, 1) # This will be the 5th sample

# --- 3. Incremental PCA (IPCA, uncentered) ---

## Step 3.1: Initialization
# We start by performing a standard PCA on the first `n0` samples.
# `center=FALSE` ensures the data is not mean-centered, matching the online method.
pca_initial <- stats::prcomp(mat[1:n0, ], center = FALSE)

# Extract the eigenvalues (squared standard deviations) and eigenvectors (rotation).
# This list structure is what the `incRpca` function expects as input.
pca_state <- list(
  values = pca_initial$sdev[1:q]^2,
  vectors = pca_initial$rotation[, 1:q]
)

# Print the initial state of the PCA model
print("pca n0 uncentered (after initialization)")
print(pca_state)


## Step 3.2: Learn the first new vector 'x'
# The current number of samples processed is 3. The new sample 'x' will be the 4th.
# The `incRpca` function takes the old PCA state and updates it with the new vector.
i <- 4
pca_state <- onlinePCA::incRpca(
  pca_state$values,
  pca_state$vectors,
  x,
  i - 1, # The total number of samples processed before this new one
  q = q
)

# Print the state after learning 'x'
print("pca online uncentered, x learned")
print(pca_state)


## Step 3.3: Learn the second new vector 'y'
# Now, we update the model with vector 'y', which will be the 5th sample.
i <- 5
pca_state <- onlinePCA::incRpca(
  pca_state$values,
  pca_state$vectors,
  y,
  i - 1, # The total number of samples processed before this new one
  q = q
)

# Print the final state after learning 'y'
print("pca online uncentered, y learned")
print(pca_state)
