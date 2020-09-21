#include <math.h>
#include <stdio.h>
#include <string.h>
#define SQUARE(x) ((x)*(x))


/* -------------------- */
/* covariance functions */
/* -------------------- */

double C_covScalingFactor(const char *type) {
	if (strcmp(type, "gauss") == 0) return(sqrt(2.)/2.);
	else if (strcmp(type, "matern3_2") == 0) return(sqrt(3.)); 
	else if (strcmp(type, "matern5_2") == 0) return(sqrt(5.));
	else return(1.);
}


double C_covWhiteNoise(const double *x1, const int *n1, const double *x2, const int *n2, const int *d, const int *i1, const int *i2, const double *param, const double *scaling_factor, const double *var) {
  double s = 0.;
  for (int k = 0; k < *d; k++) {
    s += fabs((x1[*i1 + *n1 * k] - x2[*i2 + *n2 * k]));
  }
  if (s < 0.000000000000001) return(*var);
  else return(0.);
}

double C_covGauss(const double *x1, const int *n1, const double *x2, const int *n2, const int *d, const int *i1, const int *i2, const double *param, const double *scaling_factor, const double *var) {
  double s = 0.;
  for (int k = 0; k < *d; k++) {
    s += SQUARE((x1[*i1 + *n1 * k] - x2[*i2 + *n2 * k])/ (param[k] / *scaling_factor));
  }
  return(exp(-s) * *var);
}


double C_covExp(const double *x1, const int *n1, const double *x2, const int *n2, const int *d, const int *i1, const int *i2, const double *param, const double *scaling_factor, const double *var) {
  double s = 0.;
  for (int k = 0; k < *d; k++) {
    s += fabs(x1[*i1 + *n1 * k] - x2[*i2 + *n2 * k]) / param[k];
  }
  return(exp(-s) * *var);
}

double C_covMatern3_2(const double *x1, const int *n1, const double *x2, const int *n2, const int *d, const int *i1, const int *i2, const double *param, const double *scaling_factor, const double *var) {
  double s = 0.;
  double ecart = 0.;
  for (int k = 0; k < *d; k++) {
  	 ecart = fabs(x1[*i1 + *n1 * k] - x2[*i2 + *n2 * k]) / (param[k] / *scaling_factor);
    s += ecart - log(1+ecart);
  }
  return(exp(-s) * *var);
}

void C_covMatern3_2_hao(const double *dist, const int n,  const int d, const double *param, double* ans) {
  
  double scaling_factor = sqrt(3.);
  double s;
  for (int i = 0; i < n; i++) {
	  s = 0;
	  for (int k = 0; k < d; k++) {
		  s += SQUARE(dist[k + i * d]) * param[k];
  	  }
  	  s = sqrt(s) * scaling_factor;
	  ans[i] = exp(-s) * (1.0 + s);
  }
}

double C_covMatern5_2(const double *x1, const int *n1, const double *x2, const int *n2, const int *d, const int *i1, const int *i2, const double *param, const double *scaling_factor, const double *var) {
  double s = 0.;
  double ecart = 0.;
  for (int k = 0; k < *d; k++) {
  	 ecart = fabs(x1[*i1 + *n1 * k] - x2[*i2 + *n2 * k]) / (param[k] / *scaling_factor);
    s += ecart - log(1+ecart+SQUARE(ecart)/3);
  }
  return(exp(-s) * *var);
}

double C_covPowExp(const double *x1, const int *n1, const double *x2, const int *n2, const int *d, const int *i1, const int *i2, const double *param, const double *scaling_factor, const double *var) {
  double s = 0.;
  for (int k = 0; k < *d; k++) {
    s += pow(fabs(x1[*i1 + *n1 * k] - x2[*i2 + *n2 * k]) / param[k], param[k+*d]);
  }
  return(exp(-s) * *var);
}



/* ------------------- */
/* covariance matrices */
/* ------------------- */
void C_covMatrix(const double *x, const int n, const int d, const double *param, const double var, const char *type, double *ans) {
		
	double (*C_covFunction)(const double *, const int *, const double *, const int *, const int *, const int *, const int *, const double *, const double *, const double *);
	double scf = C_covScalingFactor(type);   /* scaling factor */ 

	if (strcmp(type, "gauss") == 0) C_covFunction = C_covGauss;
	else if (strcmp(type, "exp") == 0) C_covFunction = C_covExp;
	else if (strcmp(type, "matern3_2") == 0) C_covFunction = C_covMatern3_2;
	else if (strcmp(type, "matern5_2") == 0) C_covFunction = C_covMatern5_2;
	else if (strcmp(type, "powexp") == 0) C_covFunction = C_covPowExp;
	
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < i; j++) {	
			ans[j + n * i] = ans[i + n * j] = (*C_covFunction)(x, &n, x, &n, &d, &i, &j, param, &scf, &var);
		}
		ans[i + n *i] = var;
	}
} 

void C_covMat1Mat2(const double *x1, const int *n1, const double *x2, const int *n2, const int *d, const double *param, const double *var, const char **type, double *ans) {
	
	double (*C_covFunction)(const double *, const int *, const double *, const int *, const int *, const int *, const int *, const double *, const double *, const double *);
	
	double scf = C_covScalingFactor(*type);   /* scaling factor */ 
	
	if (strcmp(*type, "gauss") == 0) C_covFunction = C_covGauss;
	else if (strcmp(*type, "exp") == 0) C_covFunction = C_covExp;
	else if (strcmp(*type, "matern3_2") == 0) C_covFunction = C_covMatern3_2;
	else if (strcmp(*type, "matern5_2") == 0) C_covFunction = C_covMatern5_2;
	else if (strcmp(*type, "powexp") == 0) C_covFunction = C_covPowExp;
	else if (strcmp(*type, "whitenoise") == 0) C_covFunction = C_covWhiteNoise;
		
	for (int i1 = 0; i1 < *n1; i1++) {
		for (int i2 = 0; i2 < *n2; i2++) {				ans[i1 + *n1 * i2] = (*C_covFunction)(x1, n1, x2, n2, d, &i1, &i2, param, &scf, var);
		}
	}
	
} 



/* ----------------------------------------------------------- */
/* covariance functions derivatives with respect to parameters */
/* ----------------------------------------------------------- */

double C_covGaussDerivative(const double *X, const int *n, const int *d, const int *i, const int *j, const double *param, const double *scaling_factor, const int *k, const double *C) {
  	/* derive C(X_i, X_j) with respect to param_k */
  double dlnC = 0.;
  double v = param[*k] / *scaling_factor;
  dlnC = SQUARE((X[*i + *n * *k] - X[*j + *n * *k]) / v) * 2 / v;
  return(dlnC * C[*i + *n * *j] / *scaling_factor);
}

double C_covExpDerivative(const double *X, const int *n, const int *d, const int *i, const int *j, const double *param, const double *scaling_factor, const int *k, const double *C) {
  	/* derive C(X_i, X_j) with respect to param_k */
  double dlnC = 0.;
  dlnC = fabs((X[*i + *n * *k] - X[*j + *n * *k]) / param[*k]) / param[*k];
  return(dlnC * C[*i + *n * *j]);
}

double C_covMatern3_2Derivative(const double* X, const int n, const int d, const int *i, const int *j, const double *param, const double *scaling_factor, const int *k, const double *C) {
  	/* derive C(X_i, X_j) with respect to param_k */
	  for (int ) {
		diff = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2
	  }
  double dlnC = 0.;
  double v = param[*k] / *scaling_factor;
  double ecart = fabs((X[*i + *n * *k] - X[*j + *n * *k])) / v;
  dlnC = (SQUARE(ecart) / (1+ecart)) / v;
  return(dlnC * C[*i + *n * *j] / *scaling_factor);


   diff = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2.

        if self.corr_type == 'squared_exponential':
            grad = -diff * R[..., np.newaxis]

        elif self.corr_type == 'matern':
            c = np.sqrt(3)
            D = np.sqrt(np.sum(theta * diff, axis=-1))

            if nu == 0.5:
                grad = - diff * theta / D * R
            elif nu == 1.5:
                grad = -3 * np.exp(-c * D)[..., np.newaxis] * diff / 2.
            elif nu == 2.5:
                pass
}

// double C_covMatern3_2Derivative(const double *X, const int *n, const int *d, const int *i, const int *j, const double *param, const double *scaling_factor, const int *k, const double *C) {
//   	/* derive C(X_i, X_j) with respect to param_k */
//   double dlnC = 0.;
//   double v = param[*k] / *scaling_factor;
//   double ecart = fabs((X[*i + *n * *k] - X[*j + *n * *k])) / v;
//   dlnC = (SQUARE(ecart) / (1+ecart)) / v;
//   return(dlnC * C[*i + *n * *j] / *scaling_factor);
// }

double C_covMatern5_2Derivative(const double *X, const int *n, const int *d, const int *i, const int *j, const double *param, const double *scaling_factor, const int *k, const double *C) {
  	/* derive C(X_i, X_j) with respect to param_k */
  double dlnC = 0.;
  double v = param[*k] / *scaling_factor;
  double ecart = fabs((X[*i + *n * *k] - X[*j + *n * *k])) / v;
  double a = 1+ecart;
  double b = SQUARE(ecart)/3;
  dlnC = a*b / (a+b) / v;
  return(dlnC * C[*i + *n * *j] / *scaling_factor);
}


double C_covPowExpDerivative(const double *X, const int *n, const int *d, const int *i, const int *j, const double *param, const double *scaling_factor, const int *k, const double *C) {
  	/* derive C(X_i, X_j) with respect to param_k */
  
  int kmodd = *k % *d;
  double ecart = X[*i + *n * kmodd] - X[*j + *n * kmodd];
  
  if (ecart==0.) 
      return(0.);	
  else {
	  ecart = fabs(ecart) / param[kmodd];
	  double dlnC = 0.;	  
	  if (*k<=((*d)-1)) 
	      dlnC = pow(ecart, param[kmodd + *d]) * param[kmodd + *d] / param[kmodd];  
	  else 
	      dlnC = - pow(ecart, param[kmodd + *d]) * log(ecart);
	  
	  return(dlnC * C[*i + *n * *j]);
  }
}




/* --------------------------------------------------------- */
/* covariance matrice derivatives with respect to parameters */
/* --------------------------------------------------------- */

void C_covMatrixDerivative(const double *X, const int *n, const int *d, const double *param, const char **type, int *k, double *C, double *ans) {
	
	(*k)--;  /* since the first element of an array is 0 in C language*/
	
	double (*C_covFunctionDerivative)(const double *, const int *, const int *, const int *, const int *, const double *, const double *, const int *, const double *);

	double scf = C_covScalingFactor(*type);   /* scaling factor */ 
	
	if (strcmp(*type, "gauss") == 0)  C_covFunctionDerivative = C_covGaussDerivative;
	else if (strcmp(*type, "exp") == 0) C_covFunctionDerivative = C_covExpDerivative;
	else if (strcmp(*type, "matern3_2") == 0) C_covFunctionDerivative = C_covMatern3_2Derivative;
	else if (strcmp(*type, "matern5_2") == 0) C_covFunctionDerivative = C_covMatern5_2Derivative;
	else if (strcmp(*type, "powexp") == 0) C_covFunctionDerivative = C_covPowExpDerivative;
	
	for (int i = 0; i < *n; i++) {
		for (int j = 0; j < i; j++) {
			ans[j + *n * i] = ans[i + *n * j] = (*C_covFunctionDerivative)(X, n, d, &i, &j, param, &scf, k, C);
		}
		ans[i + *n * i] = 0;
	}

}


/* ------------------------------------------------------- */
/* covariance matrix derivatives with respect to one point */
/* ------------------------------------------------------- */

double C_covGaussDerivative_dx(const double *X, const int *n, const int *d, const int *i, const int *j, const double *param, const double *scaling_factor, const int *k, const double *C) {
  	/* derive C(X_i, X_j) with respect to h_k */
	double dlnC = 0.;
	double v = param[*k] / *scaling_factor;
	dlnC = -2*(X[*j + *n * *k] - X[*i + *n * *k]) / SQUARE(v);
	return(dlnC * C[*i + *n * *j]);
}

double C_covExpDerivative_dx(const double *X, const int *n, const int *d, const int *i, const int *j, const double *param, const double *scaling_factor, const int *k, const double *C) {
  	/* derive C(X_i, X_j) with respect to h_k */
	double ecart = X[*j + *n * *k] - X[*i + *n * *k];
	double sign_ecart = 1.;
	double dlnC = 0.;
	
	if (ecart == 0.)
      	return(0.);
	else if (ecart < 0) 
  		sign_ecart = -1.;
	
	dlnC = - sign_ecart / param[*k];
	return(dlnC * C[*i + *n * *j]);
}
	
double C_covMatern3_2Derivative_dx(const double *X, const int *n, const int *d, const int *i, const int *j, const double *param, const double *scaling_factor, const int *k, const double *C) {
  	/* derive C(X_i, X_j) with respect to h_k */
	double ecart = X[*j + *n * *k] - X[*i + *n * *k];
	double sign_ecart = 1.;
	double dlnC = 0.;
	
	if (ecart == 0.)
      	return(0.);
	else if (ecart < 0) 
  		sign_ecart = -1.;
	
	ecart = fabs(ecart) / (param[*k] / *scaling_factor);
	dlnC = - sign_ecart * ecart / (1+ecart) / (param[*k] / *scaling_factor);
	return(dlnC * C[*i + *n * *j]);
}

double C_covMatern5_2Derivative_dx(const double *X, const int *n, const int *d, const int *i, const int *j, const double *param, const double *scaling_factor, const int *k, const double *C) {
  	/* derive C(X_i, X_j) with respect to h_k */	
	double ecart = X[*j + *n * *k] - X[*i + *n * *k];
	double sign_ecart = 1.;
	double dlnC = 0.;
	
	if (ecart == 0.)
      	return(0.);
	else if (ecart < 0) 
  		sign_ecart = -1.;
	
	ecart = fabs(ecart) / (param[*k] / *scaling_factor);
	double u = 1 + ecart;
	double v = ecart/3;
	dlnC = - sign_ecart * u*v / (u+ecart*v) / (param[*k] / *scaling_factor);
	return(dlnC * C[*i + *n * *j]);		
}

double C_covPowExpDerivative_dx(const double *X, const int *n, const int *d, const int *i, const int *j, const double *param, const double *scaling_factor, const int *k, const double *C) {
  	/* derive C(X_i, X_j) with respect to h_k */
	double ecart = X[*j + *n * *k] - X[*i + *n * *k];
	double sign_ecart = 1.;
	double dlnC = 0.;
	
	if (ecart == 0.)
      	return(0.);
	else if (ecart < 0) 
  		sign_ecart = -1.;
	
	ecart = fabs(ecart) / param[*k];
	dlnC = - sign_ecart * pow(ecart, param[*k + *d] - 1) * param[*k + *d] / param[*k];
	return(dlnC * C[*i + *n * *j]);			
}


void C_covMatrixDerivative_dx(const double *X, const int *n, const int *d, const double *param, const char **type, int *k, double *C, double *ans) {
	
	(*k)--;  /* since the first element of an array is 0 in C language*/
	
	double (*C_covFunctionDerivative_dx)(const double *, const int *, const int *, const int *, const int *, const double *, const double *, const int *, const double *);
	
	double scf = C_covScalingFactor(*type);   /* scaling factor */ 
	
	if (strcmp(*type, "gauss") == 0)  C_covFunctionDerivative_dx = C_covGaussDerivative_dx;
	else if (strcmp(*type, "exp") == 0) C_covFunctionDerivative_dx = C_covExpDerivative_dx;
	else if (strcmp(*type, "matern3_2") == 0) C_covFunctionDerivative_dx = C_covMatern3_2Derivative_dx;
	else if (strcmp(*type, "matern5_2") == 0) C_covFunctionDerivative_dx = C_covMatern5_2Derivative_dx;
	else if (strcmp(*type, "powexp") == 0) C_covFunctionDerivative_dx = C_covPowExpDerivative_dx; 
	
	for (int i = 0; i < *n; i++) {
		for (int j = 0; j < i; j++) {
			ans[j + *n * i] = (*C_covFunctionDerivative_dx)(X, n, d, &i, &j, param, &scf, k, C);
			ans[i + *n * j] = - ans[j + *n * i];
		}
		ans[i + *n * i] = 0;
	}
	
}



/* ------------------------------------------------------- */
/* covariance vector derivatives with respect to one point */
/* ------------------------------------------------------- */


double C_covGauss_dx(const double *x, const double *X, const int *n, const int *d, const int *i, const int *k, const double *param, const double *scaling_factor, const double *c) {
  	/* compute the derivative of x -> C(x, X_i) with respect to x_k */
    	
  double dlnc = 0.;
  
  dlnc = -2*(x[*k] - X[*i + *n * *k]) / SQUARE(param[*k] / *scaling_factor);
  return(c[*i] * dlnc);
  
}

double C_covExp_dx(const double *x, const double *X, const int *n, const int *d, const int *i, const int *k, const double *param, const double *scaling_factor, const double *c) {
  	/* compute the derivative of x -> C(x, X_i) with respect to x_k */
    	
  double ecart = x[*k] - X[*i + *n * *k];
  double sign_ecart = 1.;
  double dlnc = 0.;
  
  if (ecart == 0.)
      	return(0.);
  else if (ecart < 0) 
  		sign_ecart = -1.;
  
  dlnc = - sign_ecart / param[*k];
  return(c[*i] * dlnc);
  
}


double C_covMatern3_2_dx(const double *x, const double *X, const int *n, const int *d, const int *i, const int *k, const double *param, const double *scaling_factor, const double *c) {
  	/* compute the derivative of x -> C(x, X_i) with respect to x_k */
    	
  double ecart = x[*k] - X[*i + *n * *k];
  double sign_ecart = 1.;
  double dlnc = 0.;
  
  if (ecart == 0.)
      	return(0.);
  else if (ecart < 0) 
  		sign_ecart = -1.;

  ecart = fabs(ecart) / (param[*k] / *scaling_factor);
  dlnc = - sign_ecart * ecart / (1+ecart) / (param[*k] / *scaling_factor);
  return(c[*i] * dlnc);
  
}


double C_covMatern5_2_dx(const double *x, const double *X, const int *n, const int *d, const int *i, const int *k, const double *param, const double *scaling_factor, const double *c) {
  	/* compute the derivative of x -> C(x, X_i) with respect to x_k */
    	
  double ecart = x[*k] - X[*i + *n * *k];
  double sign_ecart = 1.;
  double dlnc = 0.;
  
  if (ecart == 0.)
      	return(0.);
  else if (ecart < 0) 
  		sign_ecart = -1.;

  ecart = fabs(ecart) / (param[*k] / *scaling_factor);
  double u = 1 + ecart;
  double v = ecart/3;
  dlnc = - sign_ecart * u*v / (u+ecart*v) / (param[*k] / *scaling_factor);
  return(c[*i] * dlnc);
  
}


double C_covPowExp_dx(const double *x, const double *X, const int *n, const int *d, const int *i, const int *k, const double *param, const double *scaling_factor, const double *c) {
  	/* compute the derivative of x -> C(x, X_i) with respect to x_k */
    	
  double ecart = x[*k] - X[*i + *n * *k];
  double sign_ecart = 1.;
  double dlnc = 0.;
  
  if (ecart == 0.)
      	return(0.);
  else if (ecart < 0) 
  		sign_ecart = -1.;
  
  ecart = fabs(ecart) / param[*k];
  dlnc = - sign_ecart * pow(ecart, param[*k + *d] - 1) * param[*k + *d] / param[*k];
  return(c[*i] * dlnc);
  
}


void C_covVector_dx(const double *x, const double *X, const int *n, const int *d, const double *param, const char **type, double *c, double *ans) {
		
	double (*C_covFunction_dx)(const double *, const double *, const int *, const int *, const int *, const int *, const double *, const double *, const double *);
	
	double scf = C_covScalingFactor(*type);   /* scaling factor */ 

	if (strcmp(*type, "gauss") == 0)  C_covFunction_dx = C_covGauss_dx;
	else if (strcmp(*type, "exp") == 0) C_covFunction_dx = C_covExp_dx;
	else if (strcmp(*type, "matern3_2") == 0) C_covFunction_dx = C_covMatern3_2_dx;
	else if (strcmp(*type, "matern5_2") == 0) C_covFunction_dx = C_covMatern5_2_dx;
	else if (strcmp(*type, "powexp") == 0) C_covFunction_dx = C_covPowExp_dx;
	
	for (int i = 0; i < *n; i++) {
		for (int k = 0; k < *d; k++) {
			ans[i + *n * k] = (*C_covFunction_dx)(x, X, n, d, &i, &k, param, &scf, c);
		}
	}

}






