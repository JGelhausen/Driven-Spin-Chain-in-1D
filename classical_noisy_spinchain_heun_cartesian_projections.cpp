/**************************************************************
*	MAIN PROGRAM FOR A CLASSICAL SPIN CHAIN WITH NOISE
*	Jan Gelhausen - 2018
**************************************************************/

#define _USE_MATH_DEFINES

#include <sys/time.h>
#include <iostream>
#include <sstream>

// time
#include <ctime>

// Includes for file access
#include <fstream>

// math
#include <cmath>
#include <cstdlib>

// include random number generators
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

// For Matrix Diagonalisation
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_vector.h>

// For declaration of the polygamma function
#include <gsl/gsl_sf_psi.h>

#include <iostream>     // std::cout
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector


// initialise global parameters
int glob_maxit=0;

// forward declaration of functions - list of all functions
double dynamicalcorrelations(double ** spins, double k, double w, int N, int equilibrationtime, int arraysize, int stepsize);
void HeunMethod(double** spins, double* drift, double* driftpredict, double* noisevector, double* noisepredict,double noise, double h, int t, int L);
void EulerMaruyamaPredictor(double** spins, double* driftvector,double* noisevector ,double h, double noise, int t, int L);
void CalcDriftVector(double* driftvector, double* noisevector,double** Jakobi, double** spins,double h, int t, int N, int L, double J, double DeltaJ, double omegad, double lambda, double var, double noise,double* SpinChainNoise, double staggered, double J2, int cossin, double phi);
void Projecttoconservedmanifolds(double** spins, int t, int N, double staggered);
void randomize_mat_contents(gsl_matrix *matrix, int size);
void print_mat_contents(gsl_matrix *matrix, int size);
gsl_matrix *invert_a_matrix(gsl_matrix *matrix);
void print_mat_contents(gsl_matrix *matrix);
void randomize_mat_contents(gsl_matrix *matrix);
void PrintMatrix(double **Matrix, int L);
int mod(int jn, int N);
int pos(int j, int alpha, int N);
int myrandom (int i);

/**************************************************************
*
*	Main Program Routine
*
**************************************************************/

int main (int argc, char * argv[]) {
    // Printing program start ---------------------------------
	std::cout << "----------------------------------------------------------------" << std::endl;
	std::cout << "Program started" << std::endl;
	std::cout << "----------------------------------------------------------------" << std::endl;

    // get the total runtime
	clock_t clock_total = clock();


/**************************************************************
*
*	Setup Random Number Generator
*
**************************************************************/
    struct timeval tv;
    struct timezone tz;
    gettimeofday (&tv, &tz);
    gsl_rng_env_setup ();
    /* Specifies the type of the random number generator */
    /* Two types: (1) "gsl_rng_default" and (2) "gsl_rng_mt19937" Mersenne Twister */
    const gsl_rng_type * rng_type = gsl_rng_mt19937;
    /* Seed: System time */
    const size_t rng_seed = tv.tv_usec;

    fprintf (stderr,
             "Setting up the GSL random number generator with seed %lu ...\n",
             (unsigned long int) tv.tv_usec);

    /* Create the random number generator */
    gsl_rng * r = NULL;
    r = gsl_rng_alloc (rng_type);
    gsl_rng_set (r, rng_seed);

//////////////////////////////////////////////////////////////

    /* Create Random number generator */
    gsl_rng * r1 = NULL;
    r1 = gsl_rng_alloc (rng_type);
    gsl_rng_set (r1, gsl_ran_gaussian(r,1.00));

/**************************************************************
*
*	Initial System Parameters
*
**************************************************************/
int N=4;
int arraysize=1;
int L=3*N;
double h=0.0001;
double spinnorm=1.0;
int count=1;
double staggered=0.3;
int cossin=2;
int Heun=1;
int init=0;
double J=1.0;
double J2=0.4;
double DeltaJ=0.0*0.1;
double omegad=.5;
double lambda=.1;
double T=0.1;
// variance = standard deviation squared - function for the noise
double var=2.0*lambda*T*h;
double noise=0.0;
double phi=0.0;

/**************************************************************
*
*	Read-In commandline parameters, if supplied
*
**************************************************************/

if (argc==15){
    Heun=atoi(argv[1]);
    arraysize=atoi(argv[2]);
    h=atof(argv[3]);
    J=atof(argv[4]);
    DeltaJ=atof(argv[5])/100.0;
    lambda=atof(argv[6])/100.0;
    noise=atof(argv[7]);
    T=atof(argv[8])/100.0; // contains now the temperature
    init=atoi(argv[9]);
    var=2.0*lambda*T*h; // contains now the variance for the noise
    N=atoi(argv[10]);
    staggered=atof(argv[11]);
    J2=atof(argv[12])/100.0;
		cossin=atoi(argv[13]);
    phi=atof(argv[14]);
    printf("obtained parameters from the comand line:Heun=%i \t arraysize=%i \t h=%.6lf \t J=%.3lf \t DeltaJ=%.3lf \t lambda=%.3lf\t noise=%.1lf \t T=%.3lf \t init=%i \t var=%.8lf \t N=%i \t stag=%.3lf \t J2=%.3lf \t cossin=%i \t phi=%.3lf \n",Heun,arraysize,h,J,DeltaJ,lambda,noise,T,init,var,N,staggered,J2,cossin,phi);
}

//Initialise arrays
    L=3*N;
    int equilibrationtime=arraysize/20;
    double *driftvector = new double[L]();
    double *driftpredict = new double[L]();
    double *noisevector = new double[L]();
    double *noisepredict = new double[L]();
    double **spins = new double *[L]();
    double **Jakobi = new double *[L]();
    double *SpinChainNoise = new double [N]();

    printf("Equilibration time is:%i\n",equilibrationtime);
    double *scurrent = new double[arraysize]();


    for (int i = 0; i < L; i++)
    {
        spins[i] = new double[arraysize]();
        Jakobi[i] = new double[L]();
    }
// Initialise finished

/**************************************************************
*	Initialise Spins
**************************************************************/
//Method 1: Random Spins on the sphere
for(int j=0;j<N;j++){
 //Spin at site j
        spins[pos(j,0,N)][0]=gsl_ran_gaussian (r,1.00);
        spins[pos(j,1,N)][0]=gsl_ran_gaussian (r,1.00);
        spins[pos(j,2,N)][0]=gsl_ran_gaussian (r,1.00);

        spinnorm=1.0/sqrt(spins[pos(j,0,N)][0]*spins[pos(j,0,N)][0]+spins[pos(j,1,N)][0]*spins[pos(j,1,N)][0]+spins[pos(j,2,N)][0]*spins[pos(j,2,N)][0]);

    spins[pos(j,0,N)][0]*=spinnorm;
    spins[pos(j,1,N)][0]*=spinnorm;
    spins[pos(j,2,N)][0]*=spinnorm;
    printf("%.3lf\t%.3lf\t%.3lf\n",spins[pos(j,0,N)][0],spins[pos(j,1,N)][0],spins[pos(j,2,N)][0]);

}

if(init==8){
// Method 3: Fixed 20 Spins, allows comparison with Mathematica

spins[0][0]=0.007961012813583968; spins[3][0]=-0.5743810753980985; spins[6][0]=0.28533635890833375;
spins[1][0]=0.0009672038459881919;   spins[4][0]=-0.2554959435547057; spins[7][0]=0.03219546294016889;
spins[2][0]=0.9999678428788109;  spins[5][0]=-0.7776941577841604; spins[8][0]=0.9578865352697066;

spins[9][0]=0.01432311272063624; spins[12][0]=-0.40122777625103334; spins[15][0]=-0.28036867898919876;
spins[10][0]=0.011676768832543417;   spins[13][0]=-0.3230376345883874; spins[16][0]=-0.04643523530071463;
spins[11][0]=0.9998292361756698;  spins[14][0]=-0.8571248206674394; spins[17][0]=-0.9587685710140997;

spins[18][0]=0.5554168908316551; spins[21][0]=-0.5949260215443559; spins[24][0]=0.5743810753980985;
spins[19][0]=0.21612981337583856;   spins[22][0]=-0.4164347414692153; spins[25][0]=0.2554959435547059;
spins[20][0]=0.8029943842574632;  spins[23][0]=-0.6874919163065646; spins[26][0]=0.7776941577841604;

spins[27][0]=0.663709698629683; spins[30][0]=0.4012277762510332; spins[33][0]=0.3344889099562447;
spins[28][0]=0.01767541286364497;   spins[31][0]=0.3230376345883874; spins[34][0]=0.05086192139206389;
spins[29][0]=0.7477813956799106;  spins[32][0]=0.8571248206674394; spins[35][0]=0.9410261601404026;

spins[36][0]=-0.2853363589083341; spins[39][0]=-0.6637096986296832; spins[42][0]=-0.3344889099562448;
spins[37][0]=-0.03219546294016889;   spins[40][0]=-0.01767541286364489; spins[43][0]=-0.050861921392063796;
spins[38][0]=-0.9578865352697065;  spins[41][0]=-0.7477813956799104; spins[44][0]=-0.9410261601404026;

spins[45][0]=0.2803686789891984; spins[48][0]=-0.007961012813583867; spins[51][0]=0.594926021544356;
spins[46][0]=0.046435235300714556;   spins[49][0]=-0.0009672038459881769; spins[52][0]=0.4164347414692153;
spins[47][0]=0.9587685710140998;  spins[50][0]=-0.9999678428788109; spins[53][0]=0.6874919163065645;

spins[54][0]=-0.5554168908316552; spins[57][0]=-0.014323112720636338; ;
spins[55][0]=-0.21612981337583853;   spins[58][0]=-0.01167676883254349; ;
spins[56][0]=-0.8029943842574632;  spins[59][0]=-0.9998292361756698;
}

//Zero Magnetisation state
if(init==4){
    if(N%2>0){printf("ERROR: N needs to be even to have a state with zero magnetisation"); exit(1);}

// Create a vector that has as entries N integers that are randomly shuffled
std::srand ( unsigned ( std::time(0) ) );
  std::vector<int> shuffled;

  // set some values:
  for (int i=0; i<N; ++i) shuffled.push_back(i); // 1 2 3 4 5 6 7 8 9

  // using built-in random generator:
  std::random_shuffle ( shuffled.begin(), shuffled.end() );

  // using myrandom:
  std::random_shuffle ( shuffled.begin(), shuffled.end(), myrandom);

double alpha=0.123;
double beta=0.321;
double theta=0.0;
double phi=0.0;
int shuffledit=0;

alpha=gsl_ran_gaussian (r1,1.00);
beta=gsl_ran_gaussian (r1,1.00);
printf("alpha:%.5lf\t beta:%.5lf\n",alpha,beta);


for(int i=0;i<N;i++){

phi=alpha;
theta=beta;

shuffledit=shuffled[i]; // Holds a random integer 0,...,N

spins[pos(shuffledit,0,N)][0]=sin(phi)*cos(theta);
spins[pos(shuffledit,1,N)][0]=sin(theta)*sin(phi);
spins[pos(shuffledit,2,N)][0]=cos(phi);

//printf("pos:%i\t x:%.3lf\n",i,spins[pos(i,0,N)][0]);
i=i+1;
shuffledit=shuffled[i];

phi=M_PI-alpha;
theta=beta+M_PI;

spins[pos(shuffledit,0,N)][0]=sin(phi)*cos(theta);
spins[pos(shuffledit,1,N)][0]=sin(theta)*sin(phi);
spins[pos(shuffledit,2,N)][0]=cos(phi);

alpha=gsl_ran_gaussian (r1,1.00);
beta=gsl_ran_gaussian (r1,1.00);
printf("alpha:%.5lf\t beta:%.5lf\n",alpha,beta);

}
}

if(init==5){
	double eins=-1.0;
	for(int i=0;i<N;i++){
		eins*=-1.0;
		spins[pos(i,0,N)][0]=0.0;
		spins[pos(i,1,N)][0]=0.0;
		spins[pos(i,2,N)][0]=eins;
	}
}

// /**************************************************************
// *	Heun
// **************************************************************/
clock_t timer1;
double counter1=0.0,counter2=0.0,counter3=0.0,counter4=0.0,counter5=0.0;
// timer = clock();
 for(int t=0;t<arraysize-1;t++){
	 	// For every timestep draw a new noise realisation for every spin
     for(int k=0;k<N;k++){
       SpinChainNoise[k]=gsl_ran_gaussian (r,sqrt(var));
     }

// Step 1 Calculate the drift F(u_n) - modifies the arrays driftvector and noisevector
		timer1 = clock();
    CalcDriftVector(driftvector, noisevector, Jakobi, spins, h, t, N, L, J, DeltaJ, omegad, lambda, var, noise, SpinChainNoise,staggered,J2,cossin,phi);
		timer1 = clock() - timer1;
		counter1 += ((double)timer1)/CLOCKS_PER_SEC;
// Step 2 Predict the next step based on Euler ut_n+1=F(u_n)*h+dW*noisevector; ut_n+1, according to Euler Maruyama, actually modifies spins[t+1]
		timer1 = clock();
    EulerMaruyamaPredictor(spins,driftvector,noisevector,h,noise,t,L);
		timer1 = clock()-timer1;
		counter2+=((double)timer1)/CLOCKS_PER_SEC;
// Step 3 Driftpredict: F(ut_n+1) and G(ut_n+1) - modifies driftpredict and noisepredict
		timer1 = clock();
    CalcDriftVector(driftpredict,noisepredict,Jakobi,spins,h,t+1,N,L,J,DeltaJ,omegad,lambda,var,noise,SpinChainNoise,staggered,J2,cossin,phi);
		timer1 = clock() - timer1;
		counter3+=((double)timer1)/CLOCKS_PER_SEC;
//Step 4 Heun-Method - overwrites the spins[t+1] array again and fills it with Heun Prediction
		timer1 = clock();
    HeunMethod(spins,driftvector,driftpredict,noisevector,noisepredict,noise,h,t,L);
		timer1 = clock()-timer1;
		counter4+=((double)timer1)/CLOCKS_PER_SEC;
// Step 5 Projection Method: Normalise all spins, ensures that solution evolves on surface of sphere
		timer1 = clock();
    Projecttoconservedmanifolds(spins,t,N,staggered);
		timer1 = clock()-timer1;
		counter5+=((double)timer1)/CLOCKS_PER_SEC;
// Calculate the remaining time - based on the calculation duration of the first two percent,
         if(t%(arraysize/50)==0 && t>0){
             double percentage=100.0*double(t)/double(arraysize);
             printf("%.3lf percent through\n",percentage);
            if(count==1){
            double sec_total    = (double)(clock() - clock_total)/(CLOCKS_PER_SEC);

            // printing the times - after 2% of the total calculation time has passed
	        std::cout << "- total expected runtime: ( " << 100.0/percentage * sec_total/60 << " minutes )" << std::endl;
					std::cout << "minutes spent in CalcDriftVector" << (counter1+counter3)/60 << "minutes" << std::endl;
					std::cout << "minutes spent in EulerMaruyama" << counter2/60 << "minutes" << std::endl;
					std::cout << "minutes spent in HeunMethod" << counter4/60 << "minutes" << std::endl;
					std::cout << "minutes spent in Projecttoconservedmanifolds" << counter5/60 << "minutes" << std::endl;
            }
            count=2;
         }
 }
// glob_maxit counts the maximum number of iteration steps needed to project the solution back to the conserved manifild
printf("maxiterations:%i",glob_maxit);
printf("finished everything\n");

/**************************************************************
*
*	Write collected information to disk
*
**************************************************************/
//Output from Heun scheme example
            double timeincrements=0.025;
            int stepsize=(int)(timeincrements/h);
            if(stepsize>arraysize){stepsize=1;}
            printf("this is the stepsize %i\n",stepsize);
            char filename[1024];
            snprintf (filename, sizeof (filename), "dataAfinitial/Hdyn/HamiltonianDynamics_Heun=%i_init=%i_arsize=%i_N=%i_J=%.2lf_J2=%.2lf_DJ=%.3lf_od=%.3lf_lam=%.3lf_h=%.8lf_n=%.1lf_T=%3lf_10kvar=%.3lf_stag=%.2lf_cossin=%i_phi=%.3lf.txt",Heun,init,arraysize,N,J,J2,DeltaJ,omegad,lambda,h,noise,T,10000*var,staggered,cossin,phi);

            FILE *fh = NULL;
            fh = fopen (filename, "w");
            for(int l=equilibrationtime;l<2*equilibrationtime;l=l+stepsize){
                // Output norm of spin 4 and all components of the spin
            fprintf (fh, "%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.6lf\t\n",spins[0][l],spins[1][l],spins[2][l],spins[3][l],spins[4][l],spins[5][l],spins[6][l],spins[7][l],spins[8][l],spins[9][l],spins[10][l],spins[11][l],spins[12][l],spins[13][l],spins[14][l],spins[15][l],spins[16][l],spins[17][l],h*l);
            }
            fclose (fh);
// Calculate the magnetisations
            char filename1[1024];
            snprintf (filename1, sizeof (filename1), "dataAfinitial/magn/magnetisations_Heun=%i_init=%i_arsize=%i_N=%i_J=%.2lf_J2=%.2lf_DJ=%.3lf_od=%.3lf_lam=%.3lf_h=%.5lf_n=%.1lf_T=%3lf_10kvar=%.3lf_stag=%.2lf_cossin=%i_phi=%.3lf.txt",Heun,init,arraysize,N,J,J2,DeltaJ,omegad,lambda,h,noise,T,10000*var,staggered,cossin,phi);

            FILE *fh1 = NULL;
            fh1 = fopen (filename1, "w");

            for(int l=0;l<arraysize-1;l=l+stepsize){
              double Mx=0.0;
              double My=0.0;
              double Mz=0.0;

              double sx=0.0;
              double sy=0.0;
              double sz=0.0;

              double chainlength=0.0;

              for(int j=0;j<N;j++){
                  Mx+=spins[pos(j,0,N)][l];
                  My+=spins[pos(j,1,N)][l];
                  Mz+=spins[pos(j,2,N)][l];

                  sx=spins[pos(j,0,N)][l];
                  sy=spins[pos(j,1,N)][l];
                  sz=spins[pos(j,2,N)][l];

                  chainlength+=sqrt(sx*sx+sy*sy+sz*sz);

              }
              Mx*=1.0/(double)N;
              My*=1.0/(double)N;
              Mz*=1.0/(double)N;
              // Output norm of spin 3 and all components of the spin
          fprintf (fh1, "%.15lf \t %.15lf \t %.15lf \t %.15lf\t %.15lf\t %.7lf\n",Mx,My,Mz,sqrt(Mx*Mx+My*My+Mz*Mz),chainlength,h*l);
          }
          fclose (fh1);
// //Output from Heun scheme dot products
//             char filename2[1024];
//             snprintf (filename2, sizeof (filename2), "dataAfinitial/dotp/dotproducts_Heun=%i_init=%i_arsize=%i_N=%i_J=%.2lf_J2=%.2lf_DJ=%.3lf_od=%.3lf_lam=%.3lf_h=%.5lf_n=%.1lf_T=%3lf_10kvar=%.3lf_stag=%.2lf_cossin=%i_phi=%.3lf.txt",Heun,init,arraysize,N,J,J2,DeltaJ,omegad,lambda,h,noise,T,10000*var,staggered,cossin,phi);
//
//             FILE *fh2 = NULL;
//             fh2 = fopen (filename2, "w");
//             for(int l=0;l<arraysize-1;l=l+stepsize){
//                 double dotsum=0.0;
//                 for(int j=0;j<N;j++){
//                     dotsum+=spins[pos(j,0,N)][l]*spins[pos(j+1,0,N)][l]+spins[pos(j,1,N)][l]*spins[pos(j+1,1,N)][l]+spins[pos(j,2,N)][l]*spins[pos(j+1,2,N)][l];
//                 }
//                 dotsum*=1.0/N;
//                 // Output norm of spin 3 and all components of the spin
//             fprintf (fh2, "%.6lf\t%.6lf\n",dotsum,h*l);
//             }
//             fclose (fh2);

// Calculate the dynamical spin-spin correlation function
            char filename10[1024];
            snprintf (filename10, sizeof (filename10), "dataAfinitial/dynamicalspinspin/dynamicspinspin_Heun=%i_init=%i_arsize=%i_N=%i_J=%.2lf_J2=%.2lf_DJ=%.3lf_od=%.3lf_lam=%.3lf_h=%.5lf_n=%.1lf_T=%3lf_stag=%.2lf_cossin=%i_phi=%.3lf.txt",Heun,init,arraysize,N,J,J2,DeltaJ,omegad,lambda,h,noise,T,staggered,cossin,phi);

            FILE *fh10 = NULL;
            fh10 = fopen (filename10, "w");
						for(int w=equilibrationtime;w<arraysize;w+=16*stepsize){
							for(int k=0;k<N;k++){
								// Output norm of spin 3 and all components of the spin
								fprintf (fh10, "%.6lf\t%.6lf\t%.6lf\n",double((w-equilibrationtime)/(4*stepsize))*2.0*M_PI/(double(arraysize-equilibrationtime)/(4*stepsize)),double(k)*2.0*M_PI/double(N),dynamicalcorrelations(spins,double(k),0.0*double(((w-equilibrationtime)/(4*stepsize))),N,equilibrationtime,arraysize,4*stepsize));
								}
								std::cout << "w is " << (w-equilibrationtime)/(4*stepsize) << "out of " << (arraysize-equilibrationtime)/(4*stepsize) << std::endl;
						}
            fclose (fh10);
//
//Calculate the Spin-Spin correlation functions
// Choose the spins, S_i and S_j

    double correlationsum=0.0;

    for(int j=0;j<N;j++){
        correlationsum=0.0;
    // Time loop
        for(int l=equilibrationtime;l<arraysize-1;l++){
    // Calculate the spin-spin correlation function <S_0 \dot S_j> for every time step
    //  sxi*sxj+syi*syj+szi*szj;
        correlationsum+=spins[pos(0,0,N)][l]*spins[pos(j,0,N)][l]+spins[pos(0,1,N)][l]*spins[pos(j,1,N)][l]+spins[pos(0,2,N)][l]*spins[pos(j,2,N)][l];
        }
        driftvector[j]=correlationsum/(double(arraysize-1-equilibrationtime));
    }

            char filename3[1024];
            snprintf (filename3, sizeof (filename3), "dataAfinitial/correl/correlations_Heun=%i_init=%i_arsize=%i_N=%i_J=%.2lf_J2=%.2lf_DJ=%.3lf_od=%.3lf_lam=%.3lf_h=%.5lf_n=%.1lf_T=%3lf_stag=%.2lf_cossin=%i_phi=%.3lf.txt",Heun,init,arraysize,N,J,J2,DeltaJ,omegad,lambda,h,noise,T,staggered,cossin,phi);

            FILE *fh3 = NULL;
            fh3 = fopen (filename3, "w");
            for(int j=0;j<N;j=j+1){
            fprintf (fh3, "%.8lf \t %i\n",driftvector[j],j);
            }
            fclose (fh3);

// Calculate the spin-current js=sxj syj+1-syj sx_j+1

// Gather some statistics for the spin current - here the mean
    double spincurrent=0.0;
    double spincurrentmean=0.0;
    double spincurrentvariance=0.0;
    double spincateacht=0.0;
// Write the statistics on the current before the equilibration time
		for(int l=0;l<equilibrationtime;l++){
            for(int j=0;j<N;j++){
            spincurrent+=spins[pos(j,0,N)][l]*spins[pos(j+1,1,N)][l]-spins[pos(j,1,N)][l]*spins[pos(j+1,0,N)][l];
            }
						scurrent[l]=spincurrent;
						spincurrent=0.0;
    }

    for(int l=equilibrationtime;l<arraysize-1;l++){
            for(int j=0;j<N;j++){
            spincurrent+=spins[pos(j,0,N)][l]*spins[pos(j+1,1,N)][l]-spins[pos(j,1,N)][l]*spins[pos(j+1,0,N)][l];
            }
    }
        spincurrentmean=spincurrent/(double(N)*double(arraysize-1-equilibrationtime));
// Calculate the variance of the current
for(int l=equilibrationtime;l<arraysize-1;l++){
        //calculate the spincurrent
        spincateacht=0.0;
        for(int j=0;j<N;j++){
        spincateacht+=spins[pos(j,0,N)][l]*spins[pos(j+1,1,N)][l]-spins[pos(j,1,N)][l]*spins[pos(j+1,0,N)][l];
        }
        scurrent[l]=spincateacht/double(N);
        spincurrentvariance+=(spincateacht-spincurrentmean)*(spincateacht-spincurrentmean);
}
spincurrentvariance*=1.0/(double(arraysize-2-equilibrationtime));

char filename4[1024];
snprintf (filename4, sizeof (filename4), "dataAfinitial/scurr/spincurrent_Heun=%i_init=%i_arsize=%i_N=%i_J=%.2lf_J2=%.2lf_DJ=%.3lf_od=%.3lf_lam=%.3lf_h=%.5lf_n=%.1lf_T=%3lf_stag=%.2lf_cossin=%i_phi=%.3lf.txt",Heun,init,arraysize,N,J,J2,DeltaJ,omegad,lambda,h,noise,T,staggered,cossin,phi);

FILE *fh4 = NULL;
fh4 = fopen (filename4, "a");
            fprintf (fh4, "%.10lf \t  %.10lf\n",spincurrentmean,spincurrentvariance);
                fclose (fh4);

// Detailed spincurrents
char filename5[1024];
 snprintf (filename5, sizeof (filename5), "dataAfinitial/scurr/indscurrent_Heun=%i_init=%i_arsize=%i_N=%i_J=%.2lf_J2=%.2lf_DJ=%.3lf_od=%.3lf_lam=%.3lf_h=%.5lf_n=%.1lf_T=%3lf_stag=%.2lf_cossin=%i_phi=%.3lf.txt",Heun,init,arraysize,N,J,J2,DeltaJ,omegad,lambda,h,noise,T,staggered,cossin,phi);

FILE *fh5 = NULL;
fh5 = fopen (filename5, "w");
for(int l=0;l<arraysize-1;l+=stepsize){
            fprintf (fh5, "%.10lf\t %.5lf\n",scurrent[l],h*l);
}

                fclose (fh5);
// // Write the magnetic order parameter
char filename6[1024];
 snprintf (filename6, sizeof (filename6), "dataAfinitial/orderparam/orderparam_Heun=%i_init=%i_arsize=%i_N=%i_J=%.2lf_J2=%.2lf_DJ=%.3lf_od=%.3lf_lam=%.3lf_h=%.5lf_n=%.1lf_T=%3lf_stag=%.2lf_cossin=%i_phi=%.3lf.txt",Heun,init,arraysize,N,J,J2,DeltaJ,omegad,lambda,h,noise,T,staggered,cossin,phi);

FILE *fh6 = NULL;
fh6 = fopen (filename6, "w");
double orderparamx=0.0;
double orderparamy=0.0;
double orderparamz=0.0;
for(int l=equilibrationtime;l<arraysize-1;l++){
				for(int j=0;j<N;j+=2){
				orderparamx+=spins[pos(j,0,N)][l];
				orderparamy+=spins[pos(j,1,N)][l];
				orderparamz+=spins[pos(j,2,N)][l];
				}
				for(int j=1;j<N;j+=2){
					orderparamx-=spins[pos(j,0,N)][l];
					orderparamy-=spins[pos(j,1,N)][l];
					orderparamz-=spins[pos(j,2,N)][l];
				}
}
		orderparamx=orderparamx/(double(N)*double(arraysize-1-equilibrationtime));
		orderparamy=orderparamy/(double(N)*double(arraysize-1-equilibrationtime));
		orderparamz=orderparamz/(double(N)*double(arraysize-1-equilibrationtime));

            fprintf (fh6, "%.10lf\t %.10lf\t %.10lf\n",orderparamx,orderparamy,orderparamz);

                fclose (fh6);

// free memory

    for (int i = 0; i < L; i++)
{
    delete[] spins[i];
    delete[] Jakobi[i];
}
    delete[] driftvector;
    delete[] driftpredict;
    delete[] noisevector;
    delete[] noisepredict;
    delete[] SpinChainNoise;
    delete[] scurrent;


//     // total runtime

	double sec_total    = (double)(clock() - clock_total)/(CLOCKS_PER_SEC);

    // printing the times
	std::cout << "- time total: ( " << sec_total << " s )" << std::endl;

    return 0;

}

///////////////////////////////////////////////////////////////
//Declaration of functions
///////////////////////////////////////////////////////////////
/**************************************************************
*
*	DGL Solver & Projection Methods
*
**************************************************************/

// Predicts a step based on the Euler-Maruyama Scheme
void EulerMaruyamaPredictor(double** spins, double* driftvector,double* noisevector ,double h, double noise, int t, int L){
    for(int j=0;j<L;j++){
        spins[j][t+1]=spins[j][t]+driftvector[j]*h+noise*noisevector[j];
    }
}

void HeunMethod(double** spins, double* drift, double* driftpredict, double* noisevector, double* noisepredict,double noise, double h, int t, int L){
    for(int j=0;j<L;j++){
        spins[j][t+1]=spins[j][t]+0.5*(drift[j]+driftpredict[j])*h+noise*0.5*(noisevector[j]+noisepredict[j]);
        //printf("drift:%.9lf\t noise:%.9lf\n",10000*0.5*(drift[j]+driftpredict[j])*h,10000*noise*0.5*(noisevector[j]+noisepredict[j]));
        //spins[j][t+1]=spins[j][t]+drift[j]*h;
    }
}


/**************************************************************
*
*	HELPFULL FUNCTIONS
*
**************************************************************/


// random generator function:
int myrandom (int i) { return std::rand()%i;}

int pos(int j, int alpha, int N){
    if(j==N) return alpha;
    else if (j==-1) return (N-1)*3+alpha;
    else return j*3+alpha;
}

int pos2(int j, int alpha, int N){
    if(j==N) return alpha;
    else if(j==(N+1)) return 3+alpha;
    else if (j==-1) return (N-1)*3+alpha;
    else if (j==-2) return (N-2)*3+alpha;
    else return j*3+alpha;
}

int mod(int jn, int N){
  if(jn==-1) return (N-1);
  else if(jn==N) return 0;
  else return jn;
}

void PrintMatrix(double **Matrix, int L){
              for (int i=0;i<L;i++) {
        for (int j=0;j<L;j++) {
            printf( "%.4lf\t",Matrix[i][j]);
        }
           printf( "\n");
      }
}

/**************************************************************
*
*	Matrix Inversion
*
**************************************************************/


        /************************************************************
         * PROCEDURE: invert_a_matrix
         *
         * DESCRIPTION: Invert a matrix using GSL.
         *
         * RETURNS:
         *      gsl_matrix pointer
         */
gsl_matrix *invert_a_matrix(gsl_matrix *matrix, int size)
{
    gsl_permutation *p = gsl_permutation_alloc(size);
    int s;

    // Compute the LU decomposition of this matrix
    gsl_linalg_LU_decomp(matrix, p, &s);

    // Compute the  inverse of the LU decomposition
    gsl_matrix *inv = gsl_matrix_alloc(size, size);
    gsl_matrix_set_zero(inv);
    gsl_linalg_LU_invert(matrix, p, inv);

    gsl_permutation_free(p);

    return inv;
}


        /************************************************************
         * PROCEDURE: print_mat_contents
         *
         * DESCRIPTION: Print the contents of a gsl-allocated matrix
         *
         * RETURNS:
         *      None.
         */
void print_mat_contents(gsl_matrix *matrix, int size)
{
    double element;

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            element = gsl_matrix_get(matrix, i, j);
            printf("%f ", element);
        }
        printf("\n");
    }
}


        /************************************************************
         * PROCEDURE: randomize_mat_contents
         *
         * DESCRIPTION: Overwrite entries in matrix with randomly
         *              generated values.
         *
         * RETURNS:
         *      None.
         */
void randomize_mat_contents(gsl_matrix *matrix, int size)
{
    double random_value;
    double range = 1.0 * RAND_MAX;

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {

            // generate a random value
            random_value = rand() / range;

            // set entry at i, j to random_value
            gsl_matrix_set(matrix, i, j, random_value);

        }
    }
}

/**************************************************************
 *
 *	- Deterministic Time Evolution -
 *
 **************************************************************/


void CalcDriftVector(double* driftvector, double* noisevector,double** Jakobi, double** spins,double h, int t, int N, int L, double J, double DeltaJ, double omegad, double lambda, double var, double noise,double* SpinChainNoise, double staggered, double J2,int cossin, double phi){

    int jn;
    double drive=0.0;
		double drive2=0.0;
		if(cossin==1){
		// Dribing nearest-neighbours with an ordinary cos
		drive=cos(omegad*h*t)+0.0*sin(2.0*omegad*h*t);
		}
		else{
		// Driving nearest-neighbours with a sawtooth function
		drive=((sin(h*t*omegad) + sin(2.0*h*t*omegad)/2. + sin(3.0*h*t*omegad)/3. + sin(4.0*h*t*omegad)/4.)/M_PI);
		}
		drive2=sin(h*t*omegad+phi);
    // Initialise global arrays
    double *A = new double[3]();
    double *B = new double[3]();
    double *sjvec = new double[3]();
    double *sjnvec = new double[3]();
    double C [3]={};
	  double D [3]={};

    for(int j=0;j<N;j++){
        // Every time step needs a new drift- and noisevector;
        driftvector[pos(j,0,N)]=0.0;
        driftvector[pos(j,1,N)]=0.0;
        driftvector[pos(j,2,N)]=0.0;

        noisevector[pos(j,0,N)]=0.0;
        noisevector[pos(j,1,N)]=0.0;
        noisevector[pos(j,2,N)]=0.0;

        sjvec[0]=spins[pos(j,0,N)][t];
        sjvec[1]=spins[pos(j,1,N)][t];
        sjvec[2]=spins[pos(j,2,N)][t];

         // Spin at next nearest neighbouring site, jn
        jn=j-2;
        sjnvec[0]=spins[pos2(jn,0,N)][t];
        sjnvec[1]=spins[pos2(jn,1,N)][t];
        sjnvec[2]=spins[pos2(jn,2,N)][t];

        // -sj x sj-2
        C[0]= -(-sjvec[2] * sjnvec[1] + sjvec[1] * sjnvec[2]);
        C[1]= -(sjvec[2] * sjnvec[0] - sjvec[0] * sjnvec[2]);
        C[2]= -(-sjvec[1] * sjnvec[0] + sjvec[0] * sjnvec[1]);

				// Heisenberg interaction with next-nearest-neighbour drive
        driftvector[pos2(j,0,N)] += C[0]*J2*drive2;
        driftvector[pos2(j,1,N)] += C[1]*J2*drive2;
        driftvector[pos2(j,2,N)] += C[2]*J2*drive2;

        // Spin at neighbouring site, jn
        jn=j-1;
        sjnvec[0]=spins[pos(jn,0,N)][t];
        sjnvec[1]=spins[pos(jn,1,N)][t];
        sjnvec[2]=spins[pos(jn,2,N)][t];

        // -sj x sj-1
        B[0]= -(-sjvec[2] * sjnvec[1] + sjvec[1] * sjnvec[2])*(J);
        B[1]= -(sjvec[2] * sjnvec[0] - sjvec[0] * sjnvec[2])*(J);
        B[2]= -(-sjvec[1] * sjnvec[0] + sjvec[0] * sjnvec[1])*(J);

				// Heisenberg Wechselwirkung NN-Terme
        driftvector[pos(j,0,N)] += B[0];
        driftvector[pos(j,1,N)] += B[1];
        driftvector[pos(j,2,N)] += B[2];

				// Generiert den Noise für jeden Spin
        noisevector[pos(j,0,N)] += B[0] / J * SpinChainNoise[mod(jn,N)];
        noisevector[pos(j,1,N)] += B[1] / J * SpinChainNoise[mod(jn,N)];
        noisevector[pos(j,2,N)] += B[2] / J * SpinChainNoise[mod(jn,N)];

        // choose spin at position jn
        jn=j+1;
        sjnvec[0]=spins[pos(jn,0,N)][t];
        sjnvec[1]=spins[pos(jn,1,N)][t];
        sjnvec[2]=spins[pos(jn,2,N)][t];

				// -sj x sj+1
        A[0]= -(-sjvec[2] * sjnvec[1] + sjvec[1] * sjnvec[2])*(J);
        A[1]= -(sjvec[2] * sjnvec[0] - sjvec[0] * sjnvec[2])*(J);
        A[2]= -(-sjvec[1] * sjnvec[0] + sjvec[0] * sjnvec[1])*(J);

				// Heisenberg Wechselwirkung NN-Terme
        driftvector[pos(j,0,N)] += A[0];
        driftvector[pos(j,1,N)] += A[1];
        driftvector[pos(j,2,N)] += A[2];

				// generate noise for every spin
        noisevector[pos(j,0,N)] += A[0] / J * SpinChainNoise[j];
        noisevector[pos(j,1,N)] += A[1] / J * SpinChainNoise[j];
        noisevector[pos(j,2,N)] += A[2] / J * SpinChainNoise[j];

        // Spin at next nearest neighbouring site
        jn=j+2;
        sjnvec[0]=spins[pos2(jn,0,N)][t];
        sjnvec[1]=spins[pos2(jn,1,N)][t];
        sjnvec[2]=spins[pos2(jn,2,N)][t];

				// -sj x sj+2
				D[0]= -(-sjvec[2] * sjnvec[1] + sjvec[1] * sjnvec[2]);
				D[1]= -(sjvec[2] * sjnvec[0] - sjvec[0] * sjnvec[2]);
				D[2]= -(-sjvec[1] * sjnvec[0] + sjvec[0] * sjnvec[1]);

				// Heisenberg interaction with NNN interaction and drive
        driftvector[pos2(j,0,N)] += D[0]*J2*drive2;
        driftvector[pos2(j,1,N)] += D[1]*J2*drive2;
        driftvector[pos2(j,2,N)] += D[2]*J2*drive2;


        // staggered terms go here
        if(j%2==0){ // j even (sj x sj-1)-Terme;
						// gestaggertes Magnetfeld
						driftvector[pos(j,0,N)] += staggered*spins[pos(j,1,N)][t];
						driftvector[pos(j,1,N)] -= staggered*spins[pos(j,0,N)][t];

						// Heisenberg Wechselwirkung mit drive bei NN-Terme
						driftvector[pos(j,0,N)] += (-A[0]+B[0])*DeltaJ*drive;
						driftvector[pos(j,1,N)] += (-A[1]+B[1])*DeltaJ*drive;
						driftvector[pos(j,2,N)] += (-A[2]+B[2])*DeltaJ*drive;

        }
        else{ // j odd (sj x sj+1)-Terme
						// staggered magnetic field
						driftvector[pos(j,0,N)] -= staggered*spins[pos(j,1,N)][t];
						driftvector[pos(j,1,N)] += staggered*spins[pos(j,0,N)][t];

						// Heisenberg-interaction with drive contribution
						driftvector[pos(j,0,N)] += (A[0]-B[0])*DeltaJ*drive;
						driftvector[pos(j,1,N)] += (A[1]-B[1])*DeltaJ*drive;
						driftvector[pos(j,2,N)] += (A[2]-B[2])*DeltaJ*drive;
        }

        // Berechnung der Eintraege fuer die Jakobi Matrix - damping contribution
        for(int eps=0;eps<3;eps++){
            for(int delt=0;delt<N;delt++){
                for(int gam=0;gam<3;gam++){
                    if(delt==j) Jakobi[pos(j,eps,N)][pos(delt,gam,N)]= lambda*(A[eps]*spins[pos(j+1,gam,N)][t]+B[eps]*spins[pos(j-1,gam,N)][t]);
                    else if(delt==j+1) Jakobi[pos(j,eps,N)][pos(delt,gam,N)]=lambda*A[eps]*spins[pos(j,gam,N)][t];
                    else if(delt==j-1) Jakobi[pos(j,eps,N)][pos(delt,gam,N)]=lambda*B[eps]*spins[pos(j,gam,N)][t];
                    else if(delt==0 && j==(N-1)) Jakobi[pos(j,eps,N)][pos(delt,gam,N)]=lambda*A[eps]*spins[pos(j,gam,N)][t];
                    else if(delt==(N-1) && j==0) Jakobi[pos(j,eps,N)][pos(delt,gam,N)]=lambda*B[eps]*spins[pos(j,gam,N)][t];
                    //else{};
                }
            }
        }
    } // Ende der j-Schleife fuer die Spin-Positionen

    // Matrix Inversion

    gsl_matrix * mat = gsl_matrix_alloc (L, L);
    gsl_matrix_set_zero(mat);

    for (int l = 0; l < L; l++){
        for (int m = 0; m < L; m++){
            if(l==m)gsl_matrix_set(mat, l, l, 1.0-Jakobi[l][l]);
            else gsl_matrix_set(mat, l, m, -Jakobi[l][m]);
        }
    }

    gsl_matrix *inverse = invert_a_matrix(mat,L);

    double *driftsumsvec = new double[L]();
    double *noisesumvec = new double[L]();

    // now we have solved numerically for the derivatives , we calculate the deterministic contribution now
    for (int l = 0; l < L; l++){
        double sum=0.0;
        double noisesum=0.0;
      for (int m = 0; m < L; m++){
              sum+=gsl_matrix_get(inverse,l,m)*driftvector[m];
            //   if(l==1){
            //   printf("matrix:%.6lf\t noise:%.6lf\n",gsl_matrix_get(inverse,l,m),noisevector[m][t]);
            // }
              noisesum+=gsl_matrix_get(inverse,l,m)*noisevector[m];
          }
          driftsumsvec[l]=sum;
          noisesumvec[l]=noisesum;
      }

      for (int l = 0; l < L; l++){
        driftvector[l]=driftsumsvec[l];
        noisevector[l]=noisesumvec[l];
      }


    gsl_matrix_free (mat);
    gsl_matrix_free (inverse);
    delete[] driftsumsvec;
    delete[] noisesumvec;
    delete[] A;
    delete[] B;
    delete[] sjvec;
    delete[] sjnvec;

}
/**************************************************************
*
* Helper Program that calculates a frequency discretisation as arraysize-equilibrationtime= slots*ggt, where both are integers
**************************************************************/

int calcfrequencydiscretisation(int arraysize, int equilibrationtime, int h, int ggt){
	int slots = arraysize-equilibrationtime;
	int n=ggt;
	while(n<2*ggt)
	{
		if(slots%n==0){
			return n;  // has now the frequency discretisation
			// we have achieved now arraysize-equilibrationtime = n * (arraysize-equilibrationtime)/n; where both parts are integer
			// n is the desired number of frequency discretisations  closest to the initially supplied integer ggt rand
			// (arraysize-equilibrationtime)/n is the stepsize package as (arraysize-equilibrationtime)/n * h is the time interval used to evaluate the time signal
		}
		n++;

		if ( n == 2*ggt )
		{
	    std::cout << "no suitable frequency discretisation found" << std::endl;
	    std::exit( EXIT_FAILURE );
		}
	}
}
/**************************************************************
*
* Calculate dynamical spin-spin correlation function
**************************************************************/
// k is an integer between 0 and #number of spins, w is an integer between 0 and arraysize-equilibrationtime
double dynamicalcorrelations(double ** spins, double k, double w, int N, int equilibrationtime, int arraysize,int stepsize){
double doubledynamiccorrel=0.0;
double prew=2.0*M_PI/(double(arraysize-equilibrationtime)/stepsize);
double prek=2.0*M_PI/(double(N));
double szzonej=0.0;
double szzonep=0.0;

for(int n=equilibrationtime;n<arraysize;n+=stepsize){ // time loop
	for(int m=equilibrationtime;m<arraysize;m+=stepsize){ // time loop
		for(int j=0;j<N;j++){ // momentum loop
			for(int p=0;p<N;p++){ // momentum loop
				szzonej=spins[pos(0,2,N)][equilibrationtime]*spins[pos(j,2,N)][n];
				szzonep=spins[pos(0,2,N)][equilibrationtime]*spins[pos(p,2,N)][m];
				doubledynamiccorrel+=szzonej*szzonep*(cos(prek*k*j+prew*w*n)*cos(prek*k*p+prew*w*m)+sin(prek*k*j+prew*w*n)*sin(prek*k*p+prew*w*m));
			}
		}
	}
}
// Output with normalisation
return doubledynamiccorrel*=(prek*prew*prew*prek)/(16.0*M_PI*M_PI*M_PI*M_PI);
}

/**************************************************************
*
* Projection to conserved manifold
**************************************************************/

void Projecttoconservedmanifolds(double** spins, int t, int N, double staggered){
    // contains all conserved quantitites
    double *g = new double [N+3]();
    // Lagrange parameter for conserved quantities
    double *lambdavec = new double [N+3]();
    // Input for g
    double *vecinput = new double [3*N]();
    // driftvector for Newton Iteration
    double *driftnewton = new double [N+3]();
    // Measure the convergence
    double *convergencechecker = new double [N+3]();
    // this measures how much the lagrange parameters change from iteration to iteration ||\lambda_i+1-\lambda_i||
    double distance=0.0;
    // what is the norm of g - used to quantify degree of violation for the conserved Projecttoconservedmanifolds
    double normofg=0.0;
    // specify the maximum number of iterations
    int maxiterations=20;
    // Collect convergence information
    double *convergenceinformation = new double [maxiterations]();
    // count how many Newton iterations you needs
    int newtoniterations=0;
    // convergencethreshold
    double convergencethreshold = 1.0e-14;

		// allocate memory for the matrices
    gsl_matrix * gprime = gsl_matrix_alloc (N+3, 3*N);
    gsl_matrix * gprimeT = gsl_matrix_alloc (3*N, N+3);
    gsl_matrix * G = gsl_matrix_alloc (N+3, N+3);

    // initialise
    gsl_matrix_set_zero(gprime);
    gsl_matrix_set_zero(gprimeT);
    gsl_matrix_set_zero(G);

    // fill gprime
    int count=0;
    for(int i=0;i<N;i++){
        for(int j=i+count;j<(i+count+3) && j<3*N;j++){
            gsl_matrix_set(gprime,i,j,2.0*spins[j][t]);
            //printf("i:%i \t j:%i\t count:%i\n",i,j,count);
        }
        count+=2;
    }
        for(int j=0;j<3*N;j+=3){
            gsl_matrix_set(gprime,N,j,1.0);
            gsl_matrix_set(gprime,N+1,j+1,1.0);
            gsl_matrix_set(gprime,N+2,j+2,1.0);
        }
    // Transpose the matrix: Here gprimeT is constructed from gprime
    gsl_matrix_transpose_memcpy (gprimeT, gprime);
    // Calculate the matrix product gprime gprimeT
    double sum=0.0;
    for(int i=0;i<N+3;i++){
        for(int j=0;j<N+3;j++){
            sum=0.0;
                for(int l=0;l<3*N;l++){
                sum+=gsl_matrix_get(gprime,i,l)*gsl_matrix_get(gprimeT,l,j);
            }
            gsl_matrix_set(G,i,j,sum);
        }
    }

    // Invert the matrix product
    gsl_matrix *inverse = invert_a_matrix(G,N+3);

    // Start the Newton Iteration loop
    do{

    for(int i=0;i<3*N;i++){
        double sum=0.0;
        for(int l=0;l<(N+3);l++){
        sum+=gsl_matrix_get(gprimeT,i,l)*lambdavec[l];
        //printf("lambdavec[%i]:%.3lf\n",l,lambdavec[l]);
        }
        vecinput[i]=spins[i][t]+sum;
    }
    // (2) calculate the current g
    // this loop is for the conservation of the length of every spin
    for(int i=0;i<N;i++){
        g[i]=vecinput[pos(i,0,N)]*vecinput[pos(i,0,N)]+vecinput[pos(i,1,N)]*vecinput[pos(i,1,N)]+vecinput[pos(i,2,N)]*vecinput[pos(i,2,N)]-1.0;
        //g[i]=spins[pos(i,0,N)][t]*spins[pos(i,0,N)][t]+spins[pos(i,1,N)][t]*spins[pos(i,1,N)][t]+spins[pos(i,2,N)][t]*spins[pos(i,2,N)][t];
    }

    double mx=0.0;
    double my=0.0;
    double mz=0.0;

    for(int l=0;l<N;l++){
        mx+=vecinput[pos(l,0,N)];
        my+=vecinput[pos(l,1,N)];
        mz+=vecinput[pos(l,2,N)];
    }
    if(staggered!=0.0){
        g[N]=0;
        g[N+1]=0;
        g[N+2]=mz;
    }
    else{
      g[N]=mx;
      g[N+1]=my;
      g[N+2]=mz;
    }
    // Output g to check the violation of the conservation law
    normofg=0.0;
    for(int p=0;p<N+3;p++){
      //printf("g[%i]:%.15lf\n",p,g[p]);
      normofg+=g[p]*g[p];
    }
    normofg=sqrt(normofg);
    // (3) Calculate G^{-1}.g matrix times a vector
    for(int i=0;i<N+3;i++){
        double sum=0.0;
        for(int j=0;j<N+3;j++){
            sum+=gsl_matrix_get(inverse,i,j)*g[j];
        }
        driftnewton[i]=-sum;
    }

    // (4) Update the lambda vectors through one step of the Newtwon Iteration
    // check convergence
    for(int n=0;n<N+3;n++){
      convergencechecker[n]=lambdavec[n];
    }

    for(int i=0;i<N+3;i++){
        lambdavec[i]=lambdavec[i]+driftnewton[i];
        //printf("check the lambdavalues[%i]:%.15lf\n",i,lambdavec[i]);
    }
    distance=0.0;
    for(int n=0;n<N+3;n++){
      convergencechecker[n]=convergencechecker[n]-lambdavec[n];
      distance+=convergencechecker[n]*convergencechecker[n];
    }
    distance=sqrt(distance);
    //printf("Newton Iteration: distance of lambdas:%.15lf \t norm of g:%.15lf\n",distance,normofg);
    convergenceinformation[newtoniterations]=normofg;
    newtoniterations++;

} while(normofg>convergencethreshold && newtoniterations<maxiterations);
// Statistics
//printf("Norm of g:%.15lf \t Iterations needed:%i",normofg,newtoniterations);
if(newtoniterations>glob_maxit) glob_maxit=newtoniterations;

if ( glob_maxit == maxiterations )
{
    std::cout << "reached no convergence - max number of iterations reached" << std::endl;
    std::exit( EXIT_FAILURE );
}

// ////////////////////////////
// char filename4[1024];
// snprintf (filename4, sizeof (filename4), "convergence_Heun=%i_N=%i.txt",0,N);
//
// FILE *fh4 = NULL;
// fh4 = fopen (filename4, "w");
// for(int l=0;l<maxiterations;l++){
//   fprintf (fh4, "%.15lf \n",convergenceinformation[l]);
//   }
//     fclose (fh4);
// /////////////////////////////

// Update to the new spin-vectors
  for(int i=0;i<3*N;i++){
    spins[i][t]=vecinput[i];
  }

    gsl_matrix_free (gprime);
    gsl_matrix_free (gprimeT);
    gsl_matrix_free (G);
    gsl_matrix_free (inverse);

    // free memory
    delete[] g;
    delete[] lambdavec;
    delete[] convergencechecker;
}
