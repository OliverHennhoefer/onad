library(onlinePCA)

n <- 3 # number of sample paths
d <- 5	 # number of observation points
q <- 2	 # number of PCs to compute
n0 <- 3 # number of sample paths used for initialization

data <- t(c(1, 2, 2.5, 5, 5,
            10, 10.5 , 11, 8, 4,
            3, 3.5, 7, 10, 9))
mat <- t(matrix(data, nrow = 5, ncol = 3))
x <- c(2, 3, 3.5, 11, 5)
y <- c(4, 3.4, 9.5, 1, 1)

## Incremental PCA (IPCA, uncentered)
pca <- stats::prcomp(mat[1:n0,],center=FALSE) # initialization
pca <- list(values = pca$sdev[1:q]^2, vectors = pca$rotation[,1:q])
print('pca n0 uncentered list')
print(pca)

i <- 4
pca <- onlinePCA::incRpca(pca$values, pca$vectors, x, i-1, q = q)
print('pca online uncentered, x learned')
print(pca)

i <- 5
pca <- onlinePCA::incRpca(pca$values, pca$vectors, y, i-1, q = q)
print('pca online uncentered, y learned')
print(pca)


# Expected output: 

#[1] "pca n0 uncentered (after initialization)"
#$values
#[1] 325.52805  35.94132
#
#$vectors
#            PC1        PC2
#[1,] -0.3792350 -0.4771219
#[2,] -0.4164001 -0.4290828
#[3,] -0.5165665 -0.1729495
#[4,] -0.5218317  0.4036324
#[5,] -0.3790019  0.6288179

#[1] "pca online uncentered, x learned"
#$values
#[1] 263.20439  31.11496
#
#$vectors
#           [,1]       [,2]
#[1,] -0.3380927  0.5038719
#[2,] -0.3841677  0.4439442
#[3,] -0.4758005  0.2912170
#[4,] -0.5986203 -0.4938852
#[5,] -0.3916327 -0.4693579

#[1] "pca online uncentered, y learned"
#$values
#[1] 215.25829  31.20006
#
#$vectors
#           [,1]       [,2]
#[1,] -0.3528824 -0.3990229
#[2,] -0.3885382 -0.3074019
#[3,] -0.5330556 -0.4358383
#[4,] -0.5541488  0.5795662
#[5,] -0.3650793  0.4695027