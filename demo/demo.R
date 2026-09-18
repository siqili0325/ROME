script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
parent_dir <- dirname(script_dir)  
source(file.path(parent_dir, "R", "rome.R"))
source(file.path(parent_dir, "R", "sim.R"))

n <- 8000 
p <- 20   
p_S <- 5        
p_A <- p-p_S
G <- 4      
prev=-1. 
lm=T
index_OS=c(1,2,3,4,5)
max_iter = 100
tol1 = 1e-3
tol2 = 5e-3
Sig.X <- diag(p)
gamma <- matrix(c(
  2,2,2,2,2,
  -3,-2,-5,0.1,0.1, 
  0.1,-10,0.1,0.1,0.1,
  -2,-2,-2,-2,-2
), nrow = G, ncol = p_S, byrow = TRUE)
epsilon <- epsilon.gen(delta=0.8, G=G, p=p, uni_group=F)
n_b=2
family <- c(rep("gaussian", p-p_S),rep("gaussian", p_S-n_b), rep("binomial",n_b))
ones_rate <- c(rep(NA, p-n_b),0.1,0.2)
beta=readRDS(file.path(parent_dir, "R", "beta.rds"))
beta=cbind(runif(G,-1,1),beta) 

sim_result <- simulateDataSoftmax(
  n = n,
  p = p,
  p_S = p_S,
  index_OS =index_OS,
  G = G,
  misspecify = T,
  gamma = gamma,
  beta = beta,
  Sig.X = Sig.X,
  family = family,      
  ones_rate = ones_rate,
  rand_fake=0.5,
  myseed = 1,
  lm=T
)
gammma_truth=sim_result$gamma 
beta_truth=sim_result$beta
table(sim_result$dat$G0)/n
table(sim_result$dat$G0_fake)/n  
mydata=sim_result$dat
v0=ratio.tr=table(sim_result$group_label)/nrow(mydata)%>%as.vector
ratio.tr
Y=mydata$Y
A=as.matrix(mydata[,grepl("A",colnames(mydata))])
S=as.matrix(mydata[,grepl("S",colnames(mydata))])
X=cbind(A,S)%>%as.matrix()
n0=n
G0=mydata$G0_fake
fit.pool.1 = lm(Y~X)

res.EM.1=EM.iter.lm(Y=mydata$Y, A=as.matrix(mydata[,grepl("A",colnames(mydata))]), S=as.matrix(mydata[,grepl("S",colnames(mydata))]), index_OS=index_OS, initial_labels = mydata$G0_fake,max_iter=max_iter, tol1 = tol1,tol2 =tol2, G=G,myseed=1)

beta_truth
beta.EM = res.EM.1$omega
beta.pool = coef(fit.pool.1)


pred0.mat.1= matrix(NA, nrow=n, ncol=G)
for(g in 1:G) pred0.mat.1[,g] = as.matrix(cbind(1,X)) %*% res.EM.1$omega[g,]
omega1 = res.EM.1$omega
consts.set = c(1, 0.6, seq(0.5, 0.01, by=-0.02))
hOmega1=compute_maximin.2(X,Y,pred0.mat.1,omega1,n,G0,v0=v0,consts.set)
v.mag.mat.1= matrix(NA, nrow=length(consts.set), ncol=G)
for(i.const in 1:length(consts.set)){
  out1 = compute_v(hOmega1, L=G, const=consts.set[i.const], as.numeric(v0))
  v.mag.mat.1[i.const,] = out1$v
}

sim_result_test <- simulateDataSoftmax(
  n = 3000,
  p = p,
  p_S = p_S,
  index_OS =index_OS,
  G = G,
  misspecify = T,
  gamma = gamma,
  beta = beta,
  Sig.X = Sig.X,
  family = family,    
  ones_rate = ones_rate, 
  rand_fake=0.5,
  myseed = 101,
  lm=T
)
dat_test=sim_result_test$dat
Y.t=dat_test$Y
A.t=as.matrix(dat_test[,grepl("A",colnames(dat_test))])
S.t=as.matrix(dat_test[,grepl("S",colnames(dat_test))])
X.t=cbind(A.t,S.t)%>%as.matrix()
G0.t=dat_test$G0

mse_mat=matrix(0, nrow=G, ncol=1+length(consts.set))
colnames(mse_mat)=c('pool.1',consts.set)
rownames(mse_mat)=paste0('latentgroup_',c(1:G))

for(g in 1:G){
  sampled_indices=which(G0.t == g)
  Y.g=Y.t[sampled_indices]
  A.g=A.t[sampled_indices,]
  X.g=X.t[sampled_indices,]
  pred_baseline <- cbind(1, X.g) %*% coef(fit.pool.1)
  mse_mat[g, 1] <- Metrics::mse(Y.g, pred_baseline)
  for(i.const in 1:length(consts.set)){
    pred_mm <- as.vector(pred0.mat.1 %*% v.mag.mat.1[i.const, ])
    mse_mat[g, i.const+1] <- Metrics::mse(Y.g, pred_mm[sampled_indices])
  }
}

print("MSE by group:")
print(mse_mat)
print("Worst-group MSE (maximum):")
print(apply(mse_mat, 2, max))
