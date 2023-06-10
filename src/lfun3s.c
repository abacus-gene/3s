#include "3s.h"

double lfun(double x[], int np);
void setupGtreeTab();

void copyParams(double x[], int np);

double lfun_seqdata();
void setupGtreeTab_seqdata();
void priorM0_3seq();
//double priorM1DiscreteBeta_3seq(double x[], int np);
void priorM2SIM3s_3seq();
void priorM2Pro_3seq();
void priorM2ProMax_3seq();
void priorM3MSci12_3seq();
void priorM3MSci13_3seq();
void priorM3MSci23_3seq();
void priorM0_2seq();
//double priorM1DiscreteBeta_2seq(double x[], int np);
void priorM2SIM3s_2seq();
void priorM2Pro_2seq();
void priorM2ProMax_2seq();
void priorM3MSci12_2seq();
void priorM3MSci13_2seq();
void priorM3MSci23_2seq();
void update_limits();
//int GetUVrootIM3s(double U[5*5], double V[5*5], double Root[5]);
//int GetPMatIM3s(double Pt[5*5], double t, double U[5*5], double V[5*5], double Root[5]);
double lnpD_locus (int locus);
void writeGeneTreePosterior(int locus, double pGk[], double pD, double lmax, int nTrees);
// void computePMatrices(double t[2], double tau1, double * Q, int N, double * Pt0, double * Ptau1t1, double * Pt1, double * work);
void computePMatrix(double t, double * Q, int N, double * P, double * work);
int skipTree(int gtree);

double lfun_treedata();
void setupGtreeTab_treedata();
void update_GtreeTypes(int GtreeTypeCnt[]);
void preprocessM2SIM3s(int GtreeTypeCnt[]);
void preprocessM2Pro(int GtreeTypeCnt[]);
void preprocessM2ProMax(int GtreeTypeCnt[]);
double lnpD_tree(int tree);
double treeProbM0(int tree);
double treeProbM2SIM3s(int tree);
double treeProbM2Pro(int tree);
double treeProbM2ProMax(int tree);
double treeProbM3MSci12(int tree);
double treeProbM3MSci13(int tree);
double treeProbM3MSci23(int tree);

/* Quadrature using Gauss-Legendre rule.  The following transforms are used.
 x: (-1, 1) <== y: (a=0, b)  <==  t: (0, inf || taugap)
 y = (b+a)/2 + (b-a)/2*x;
 t = y/(1-y)
 */

double lfun(double x[], int np)
{
    copyParams(x, np);

    if (com.usedata == ESeqData)
        return lfun_seqdata();
    else
        return lfun_treedata();
}

//para[MAXPARAMETERS]:
//Indix:       0      1      2      3      4      5      6      7      8      9     10     11     12     13     14     15
//M0:     theta4 theta5   tau0   tau1 theta1 theta2 theta3
//M1:     theta4 theta5   tau0   tau1 theta1 theta2 theta3  qbeta
//M2:     theta4 theta5   tau0   tau1 theta1 theta2 theta3 thetaW    M12    M21    M13    M31    M23    M32    M53    M35
//M3-12:  theta4 theta5   tau0   tau1 theta1 theta2 theta3      T thetaX thetaY  phi12  phi21
//M3-13:  theta4 theta5   tau0   tau1 theta1 theta2 theta3      T thetaX thetaZ  phi13  phi31
//M3-23:  theta4 theta5   tau0   tau1 theta1 theta2 theta3      T thetaY thetaZ  phi23  phi32

void copyParams(double x[], int np) {
    int i;
    int isM3 = (com.model == M3MSci12 || com.model == M3MSci13 || com.model == M3MSci23);
    memset(para, 0, MAXPARAMETERS * sizeof(double));

    for (i = 0; i < com.maxnp; i++) {
        if (com.paraMap[i] != -1)
            para[i] = x[com.paraMap[i]];
        else if (i <= 7 || i <= 9 && isM3)
            para[i] = 1;
    }

    if (!LASTROUND) {
        para[3] *= para[2]; // tau1 from xtau1
        if (isM3)
            para[7] *= para[3]; // T from xT
    }
    else {
        if (com.paraMap[3] == -1)
            para[3] *= para[2]; // tau1 from xtau1
        if (com.paraMap[7] == -1 && isM3)
            para[7] *= para[3]; // T from xT
    }

    for (i = 0; i < com.maxnp; i++)
        if (com.paraMap[i] != -1)
            para[i] /= MULTIPLIER;
}

 // This function computes the likelihood for parameters given in x under the
// current model (in com.model). It calls different sub-functions for the
// calculation of the gene tree probabilities, depending on whether the dataset
// contains loci with two or three sequences. The likelihood per locus is then
// computed in parallel (if the code is compiled with OpenMP).
double lfun_seqdata()
{
    int locus, itau1;
    double Li, lnL=0, z;
    double parasave[MAXPARAMETERS], *Pt/*[5*5]*/, mbeta, q, p;
    char fpGkName[15];

    // fixed values for drosophila exon datasets
    //if(com.paraMap[4] != -1 && !LASTROUND) {
    //  x[com.paraMap[2]] = 0.020;
    //  x[com.paraMap[3]] = 0.65;
    //  x[com.paraMap[4]] = x[com.paraMap[5]];
    //}

    NFunCall++;

    if (com.model == M1DiscreteBeta) {
        xtoy(para, parasave, MAXPARAMETERS);

        q = parasave[7];
        mbeta = parasave[3] / parasave[2];
        p = mbeta / (1-mbeta)*q;

        Pt = com.tau1beta + com.ncatBeta;
        DiscreteBeta(Pt, com.tau1beta, p, q, com.ncatBeta, com.UseMedianBeta);

        for (itau1 = 0; itau1 < com.ncatBeta; itau1++) {
            para[3] = com.tau1beta[itau1] * para[2];

            update_limits();

            if (data.ndata - data.twoSeqLoci) {
                priorM0_3seq();
            }
            if (data.twoSeqLoci) {
                priorM0_2seq();
            }
            if (com.nthreads >= 1) {
#pragma omp parallel for default(none) shared(com, stree, data, itau1) num_threads(com.nthreads)
                for (locus = 0; locus < data.ndata; locus++) {
                    com.pDclass[itau1*data.ndata+locus] = lnpD_locus(locus);
                }
            }
            else {
#pragma omp parallel for default(none) shared(com, stree, data, itau1)
                for (locus = 0; locus < data.ndata; locus++) {
                    com.pDclass[itau1*data.ndata+locus] = lnpD_locus(locus);
                }
            }
        }
        for (locus=0, lnL=0; locus < data.ndata; locus++) {
            for (itau1=0, z=-1e300; itau1 < com.ncatBeta; itau1++) {
                z = max2(z, com.pDclass[itau1*data.ndata+locus]);
            }
            for (itau1=0, Li=0; itau1 < com.ncatBeta; itau1++) {
                Li += 1.0/com.ncatBeta * exp(com.pDclass[itau1*data.ndata+locus] - z);
            }
            lnL += z + log(Li);
        }
        para[3] = parasave[3];
    }
    else {
        update_limits();

        if(com.model == M0) {
            if (data.ndata - data.twoSeqLoci) {
                priorM0_3seq();
            }
            if (data.twoSeqLoci) {
                priorM0_2seq();
            }
        } else if(com.model == M2SIM3s) {
            if (data.ndata - data.twoSeqLoci) {
                priorM2SIM3s_3seq();
            }
            if (data.twoSeqLoci) {
                priorM2SIM3s_2seq();
            }
        } else if(com.model == M2Pro) {
            if (data.ndata - data.twoSeqLoci) {
                priorM2Pro_3seq();
            }
            if (data.twoSeqLoci) {
                priorM2Pro_2seq();
            }
        } else if(com.model == M2ProMax) {
            if (data.ndata - data.twoSeqLoci) {
                priorM2ProMax_3seq();
            }
            if (data.twoSeqLoci) {
                priorM2ProMax_2seq();
            }
        } else if(com.model == M3MSci12) {
            if (data.ndata - data.twoSeqLoci) {
                priorM3MSci12_3seq();
            }
            if (data.twoSeqLoci) {
                priorM3MSci12_2seq();
            }
        } else if(com.model == M3MSci13) {
            if (data.ndata - data.twoSeqLoci) {
                priorM3MSci13_3seq();
            }
            if (data.twoSeqLoci) {
                priorM3MSci13_2seq();
            }
        } else if(com.model == M3MSci23) {
            if (data.ndata - data.twoSeqLoci) {
                priorM3MSci23_3seq();
            }
            if (data.twoSeqLoci) {
                priorM3MSci23_2seq();
            }
        } else {
            error2("model not implemented yet!");
        }

        // output of gene tree posteriors
        if (LASTROUND == 2 && com.model != M1DiscreteBeta) {
            if (stree.speciestree)
                snprintf(fpGkName, sizeof(fpGkName), "pGk%d-%d.txt", com.extModel, stree.sptree + 1);
            else
                snprintf(fpGkName, sizeof(fpGkName), "pGk%d.txt", com.extModel);
            fpGk = gfopen(fpGkName, "w");
        }

        // now compute locus-specific likelihood
        if (com.nthreads >= 1) {
#pragma omp parallel for default(none) reduction(+:lnL) private(Li) shared(com, stree, data) num_threads(com.nthreads)
            for(locus=0; locus<data.ndata; locus++) {
                Li = lnpD_locus(locus);
                lnL += Li;
#ifdef DEBUG2
                int * n = data.Nij[stree.sptree] + locus*5;
                printf("%6d - %d\t%d\t%d\t%d\t%d\t%.6f\n", locus,n[0],n[1],n[2],n[3],n[4], Li);
#endif
            }
        }
        else {
#pragma omp parallel for default(none) reduction(+:lnL) private(Li) shared(com, stree, data)
            for(locus=0; locus<data.ndata; locus++) {
                Li = lnpD_locus(locus);
                lnL += Li;
#ifdef DEBUG2
                int * n = data.Nij[stree.sptree] + locus*5;
                printf("%6d - %d\t%d\t%d\t%d\t%d\t%.6f\n", locus,n[0],n[1],n[2],n[3],n[4], Li);
#endif
            }
        }

        if (LASTROUND == 2 && com.model != M1DiscreteBeta) {
            fclose(fpGk);
        }
    }

    return(-lnL);
}

double lfun_treedata()
{
    int tree, itau1;
    double Li, lnL=0, z;
    double parasave[MAXPARAMETERS], *Pt/*[5*5]*/, mbeta, q, p;
    enum { NGTREETYPES = 6 };
    STATIC_ASSERT(MAXGTREETYPES >= NGTREETYPES);
    int GtreeTypeCnt[NGTREETYPES + 1];

    NFunCall++;

    if (com.model == M1DiscreteBeta) {
        xtoy(para, parasave, MAXPARAMETERS);

        q = parasave[7];
        mbeta = parasave[3] / parasave[2];
        p = mbeta / (1-mbeta)*q;

        Pt = com.tau1beta + com.ncatBeta;
        DiscreteBeta(Pt, com.tau1beta, p, q, com.ncatBeta, com.UseMedianBeta);

        for (itau1 = 0; itau1 < com.ncatBeta; itau1++) {
            para[3] = com.tau1beta[itau1] * para[2];

            update_GtreeTypes(GtreeTypeCnt);

            if (com.nthreads >= 1) {
#pragma omp parallel for default(none) shared(com, stree, data, itau1) num_threads(com.nthreads)
                for (tree = 0; tree < data.ndata; tree++) {
                    com.pDclass[itau1*data.ndata+tree] = lnpD_tree(tree);
                }
            }
            else {
#pragma omp parallel for default(none) shared(com, stree, data, itau1)
                for (tree = 0; tree < data.ndata; tree++) {
                    com.pDclass[itau1*data.ndata+tree] = lnpD_tree(tree);
                }
            }
        }
        for (tree=0, lnL=0; tree < data.ndata; tree++) {
            for (itau1=0, z=-1e300; itau1 < com.ncatBeta; itau1++) {
                z = max2(z, com.pDclass[itau1*data.ndata+tree]);
            }
            for (itau1=0, Li=0; itau1 < com.ncatBeta; itau1++) {
                Li += 1.0/com.ncatBeta * exp(com.pDclass[itau1*data.ndata+tree] - z);
            }
            lnL += z + log(Li);
        }
        para[3] = parasave[3];
    }
    else {
        update_GtreeTypes(GtreeTypeCnt);

        if (com.model == M2SIM3s)
            preprocessM2SIM3s(GtreeTypeCnt);
        else if (com.model == M2Pro)
            preprocessM2Pro(GtreeTypeCnt);
        else if (com.model == M2ProMax)
            preprocessM2ProMax(GtreeTypeCnt);

        // now compute tree-specific likelihood
        if (com.nthreads >= 1) {
#pragma omp parallel for default(none) reduction(+:lnL) private(Li) shared(com, stree, data) num_threads(com.nthreads)
            for (tree = 0; tree < data.ndata; tree++) {
                Li = lnpD_tree(tree);
                lnL += Li;
            }
        }
        else {
#pragma omp parallel for default(none) reduction(+:lnL) private(Li) shared(com, stree, data)
            for (tree = 0; tree < data.ndata; tree++) {
                Li = lnpD_tree(tree);
                lnL += Li;
            }
        }
    }

    return(-lnL);
}

static inline void getParams(double *theta4, double *theta5, double *tau0, double *tau1, double *theta1, double *theta2, double *theta3) {
    *theta4 = para[0];
    *theta5 = para[1];
    *tau0 = para[2];
    *tau1 = para[3];
    *theta1 = para[4];
    *theta2 = para[5];
    *theta3 = para[6];
}

static inline void computeMultinomProbs(int Gtree, int igrid, int GtIndex, double t[], double theta12) {
    double b[2];
    int i;
    //struct BTEntry * gtt = GtreeTab[0];
    BTEntry * gtt = com.GtreeTab[0];
    
    gtt[GtIndex].b(t[0], t[1], theta12, b);
    
    if(!com.fix_locusrate) {
        p0124Fromb0b1(com.bp0124[Gtree]+(igrid/3)*5, b);
        for(i = 0; i < 5; i++) {
            if(com.bp0124[Gtree][(igrid/3)*5+i] < 0) {
                printf("just computed a p < 0\n");
                raise(SIGINT);
            }
        }
    } else {
        com.bp0124[Gtree][(igrid/3)*2+0] = b[0];
        com.bp0124[Gtree][(igrid/3)*2+1] = b[1];
    }
}

// The prior for M0 is computed in this function, based on table 2 in the notes
void priorM0_3seq() {
    double theta4, theta5, theta1, theta2, theta12, theta3, tau0, tau1, coeff;
    int j, igrid, Gtree, ixw[2], K=com.npoints;
    const double *xI=NULL, *wI=NULL;
    double /*b[2],*/ s[2], t[2], y[2], yup[2];
    BTEntry * gtt;
    
    getParams(&theta4, &theta5, &tau0, &tau1, &theta1, &theta2, &theta3);
    
    if(tau0 < tau1)
        printf("tau0 = %12.8f < tau1 = %12.8f\n", tau0, tau1);

    GaussLegendreRule(&xI, &wI, com.npoints);

    //gtt = GtreeTab[0];
    gtt = com.GtreeTab[0];
    for(Gtree = 0; Gtree < com.nGtree; Gtree++) {
        if (skipTree(Gtree)) continue;
        
        if (Gtree < 6) {
            theta12 = theta1;
        } else if (Gtree < 12) {
            theta12 = theta2;
        } else if (Gtree < 17) {
            theta12 = theta1;
        } else if (Gtree < 22) {
            theta12 = theta2;
        } else if (Gtree < 25) {
            theta12 = theta1;
        } else if (Gtree < 28) {
            theta12 = theta2;
        } else {
            theta12 = 0;
        }

        yup[0] = gtt[GtOffsetM0[Gtree]].lim[0];
        yup[1] = gtt[GtOffsetM0[Gtree]].lim[1];
        coeff = yup[0] * yup[1] * 0.5 * 0.5;
        
        for(igrid=0; igrid<3*K*K; igrid+=3) {
            ixw[0] = (igrid/3)/K; ixw[1] = (igrid/3)%K;
            for(j=0; j<2; j++) {  /* t0 (y0) and t1 (y1) */
                if(ixw[j]<K/2) { ixw[j] = K/2-1-ixw[j];  s[j]=-1; }
                else           { ixw[j] = ixw[j]-K/2;    s[j]=1;  }
                y[j] = yup[j]*(1 + s[j]*xI[ixw[j]])/2;
                t[j] = y[j]/(1-y[j]);
            }

            com.wwprior[Gtree][igrid] = wI[ixw[0]]*wI[ixw[1]]/square((1-y[0])*(1-y[1]))*coeff;
            com.wwprior[Gtree][igrid] *= exp(-gtt[GtOffsetM0[Gtree]].T(t[0], t[1], theta12)) * gtt[GtOffsetM0[Gtree]].RJ(t[0], t[1]);
            com.wwprior[Gtree][igrid+1] = com.wwprior[Gtree][igrid+2] = com.wwprior[Gtree][igrid];

            computeMultinomProbs(Gtree, igrid, GtOffsetM0[Gtree], t, theta12);
        }
    }
}

// This computes the prior for M2 (IM model).
void priorM2SIM3s_3seq() {
    double theta4, theta5, theta1, theta2, theta3, tau0, tau1, M12=0.0, M21=0.0, coeff/*, Panc*/;
    int /*i,*/ j, igrid, Gtree, ixw[2], K=com.npoints, initState;
    const double *xI=NULL, *wI=NULL;
    double /*b[2],*/ s[2], t[2], y[2], yup[2], t01[2];
    double * Q_C1, * Q_C2, * Q_C3, * Pt1, * Pt0, * Ptau1t1, *Ptau1, * Ptau1_C1, * Ptau1_C2, * Ptau1_C3, * work, PG1a, PG1a123=0, PG1a113=0, PG1a223=0;
    //double c, m, a; //, PSG;
    BTEntry * gtt;

    Q_C3 = com.space;
    if(!Q_C3) error2("oom allocating Q_C3 in priorM2SIM3s_3seq");
    Ptau1_C3 = &Q_C3[C3*C3]; Q_C1 = &Ptau1_C3[C3*C3];
    Ptau1_C1 = &Q_C1[C1*C1]; Q_C2 = &Ptau1_C1[C1*C1];
    Ptau1_C2 = &Q_C2[C2*C2]; Pt1 = &Ptau1_C2[C2*C2]; Pt0 = &Ptau1_C2[2*C2*C2]; Ptau1t1 = &Ptau1_C2[3*C2*C2]; work = &Ptau1_C2[4*C2*C2];
    com.space = Q_C2+8*C2*C2;

    getParams(&theta4, &theta5, &tau0, &tau1, &theta1, &theta2, &theta3);

#ifdef FIXM12
    M21 = M12 = 0.000001;
#else
    M12 = para[8];
    M21 = para[9];
#endif

    GaussLegendreRule(&xI, &wI, com.npoints);

    GenerateQ1SIM3S(Q_C1, C1, theta1, theta2, 0, M12, M21);
    GenerateQ1SIM3S(Q_C2, C2, theta1, theta2, 0, M12, M21);
    GenerateQ1SIM3S(Q_C3, C3, theta1, theta2, 0, M12, M21);

    computePMatrix(tau1, Q_C1, C1, Ptau1_C1, work);
    computePMatrix(tau1, Q_C2, C2, Ptau1_C2, work);
    computePMatrix(tau1, Q_C3, C3, Ptau1_C3, work);

    PG1a113 = Ptau1_C3[3];
    PG1a123 = Ptau1_C3[1*C3+3];
    PG1a223 = Ptau1_C3[2*C3+3];

    //gtt = GtreeTab[0];
    gtt = com.GtreeTab[0];

    for(Gtree = 0; Gtree < com.nGtree; Gtree++) {
        if (skipTree(Gtree)) continue;

        initState = GtreeMapM2[Gtree];

        yup[0] = gtt[GtOffsetM2[Gtree]].lim[0];
        yup[1] = gtt[GtOffsetM2[Gtree]].lim[1];
        coeff = yup[0] * yup[1] * 0.5 * 0.5;

        for(igrid=0; igrid<3*K*K; igrid+=3) {
            ixw[0] = (igrid/3)/K; ixw[1] = (igrid/3)%K;
            for(j=0; j<2; j++) {  /* t0 (y0) and t1 (y1) */
                if(ixw[j]<K/2) { ixw[j] = K/2-1-ixw[j];  s[j]=-1; }
                else           { ixw[j] = ixw[j]-K/2;    s[j]=1;  }
                y[j] = yup[j]*(1 + s[j]*xI[ixw[j]])/2;
                t[j] = y[j]/(1-y[j]);
            }

            com.wwprior[Gtree][igrid] = wI[ixw[0]]*wI[ixw[1]]/square((1-y[0])*(1-y[1]))*coeff;

            gtt[GtOffsetM2[Gtree]].t0t1(t[0], t[1], t01);

            if (Gtree < 24) {
                // create transition probability matrices
                if (Gtree < 12) { // chain 1
                    computePMatrix(t01[0], Q_C1, C1, Pt0, work);
                    computePMatrix(t01[1], Q_C1, C1, Pt1, work);
                    computePMatrix(tau1-t01[1], Q_C1, C1, Ptau1t1, work);
                    Ptau1 = Ptau1_C1;
                } else { // chain 2
                    computePMatrix(t01[0], Q_C2, C2, Pt0, work);
                    computePMatrix(t01[1], Q_C2, C2, Pt1, work);
                    computePMatrix(tau1-t01[1], Q_C2, C2, Ptau1t1, work);
                    Ptau1 = Ptau1_C2;
                }
                com.wwprior[Gtree][igrid+1] = com.wwprior[Gtree][igrid+2] = com.wwprior[Gtree][igrid];
                gtt[GtOffsetM2[Gtree]].f(t[0], t[1], initState, Pt0, Pt1, Ptau1, Ptau1t1, &com.wwprior[Gtree][igrid]);
            } else if(Gtree < 33) { // chain 3
                PG1a = (Gtree < 27) ? PG1a113 : ((Gtree < 30) ? PG1a223 : PG1a123);
                if (!(Gtree % 3)) {
                        computePMatrix(t01[1], Q_C3, C3, Pt1, work);
                        gtt[GtOffsetM2[Gtree]].f(t[0], t[1], initState, NULL, Pt1, NULL, NULL, &com.wwprior[Gtree][igrid]);
                } else {
                    com.wwprior[Gtree][igrid] *= (1-PG1a);
                    gtt[GtOffsetM2[Gtree]].f(t[0], t[1], initState, NULL, NULL, NULL, NULL, &com.wwprior[Gtree][igrid]);
                }
            } else { // chain 4 (like M0)
                com.wwprior[Gtree][igrid] *= exp(-gtt[GtOffsetM2[Gtree]].T(t[0], t[1], 0)) * gtt[GtOffsetM2[Gtree]].RJ(t[0], t[1]);
                com.wwprior[Gtree][igrid+1] = com.wwprior[Gtree][igrid+2] = com.wwprior[Gtree][igrid];
            }

            computeMultinomProbs(Gtree, igrid, GtOffsetM2[Gtree], t, (theta1+theta2)/2);
        }
    }
    com.space = Q_C3;
}

void priorM2Pro_3seq() {
    double theta4, theta5, theta1, theta2, theta3, thetaW, tau0, tau1, M12 = 0.0, M21 = 0.0, M13 = 0.0, M31 = 0.0, M23 = 0.0, M32 = 0.0, coeff;
    int i, j, igrid, Gtree, ixw[2], K = com.npoints, initState, GtreeType;
    const double *xI = NULL, *wI = NULL;
    double /*b[2],*/ s[2], t[2], y[2], yup[2], t01[2];
    double *Q_C5, *Q_C6, *Pt_C5, *Pt_C6, *Ptau1_C5, *Ptau1_C5_ex, *Pr, *work;
    double *P5 = NULL;
    double exptaugap[4];
    BTEntry * gtt;
    enum { NGTREETYPES = 6 };
    STATIC_ASSERT(MAXGTREETYPES >= NGTREETYPES);

    Q_C5 = com.space;
    if (!Q_C5) error2("oom allocating Q_C5 in priorM2Pro_3seq");
    Q_C6 = &Q_C5[C5 * C5];
    Pt_C5 = &Q_C6[C6 * C6];
    Pt_C6 = &Pt_C5[C5 * C5];
    Ptau1_C5 = &Pt_C6[C6 * C6];
    Ptau1_C5_ex = &Ptau1_C5[C5 * C5];
    Pr = &Ptau1_C5_ex[C5 * C5_ex];
    work = &Pr[max2(C6, 4)];
    com.space = work + 3 * square(max2(C5, C6));

    getParams(&theta4, &theta5, &tau0, &tau1, &theta1, &theta2, &theta3);

    thetaW = para[7];
    M12 = para[8];
    M21 = para[9];
    M13 = para[10];
    M31 = para[11];
    M23 = para[12];
    M32 = para[13];

    GaussLegendreRule(&xI, &wI, com.npoints);

    GenerateQ5(Q_C5, theta1, theta2, theta3, M12, M21, M13, M31, M23, M32);
    GenerateQ6(Q_C6, theta1, theta2, theta3, M12, M21, M13, M31, M23, M32);

    computePMatrix(tau1, Q_C5, C5, Ptau1_C5, work);

    exptaugap[0] = exp(-6 * (tau0 - tau1) / theta5);
    exptaugap[1] = exp(-2 * (tau0 - tau1) / theta5);
    exptaugap[2] = exp(-2 * (tau0 - tau1) / thetaW);
    exptaugap[3] = exp(-6 * (tau0 - tau1) / thetaW);

    for (i = 0; i < NINITIALSTATES; i++) {
        j = initStates[i];
        Ptau1_C5_ex[C5_ex * j + 0] = Ptau1_C5[C5 * j +  0] + Ptau1_C5[C5 * j +  1] + Ptau1_C5[C5 * j +  3] + Ptau1_C5[C5 * j +  4]
                                   + Ptau1_C5[C5 * j +  9] + Ptau1_C5[C5 * j + 10] + Ptau1_C5[C5 * j + 12] + Ptau1_C5[C5 * j + 13];
        Ptau1_C5_ex[C5_ex * j + 1] = Ptau1_C5[C5 * j +  2] + Ptau1_C5[C5 * j +  5] + Ptau1_C5[C5 * j + 11] + Ptau1_C5[C5 * j + 14];
        Ptau1_C5_ex[C5_ex * j + 2] = Ptau1_C5[C5 * j +  6] + Ptau1_C5[C5 * j +  7] + Ptau1_C5[C5 * j + 15] + Ptau1_C5[C5 * j + 16];
        Ptau1_C5_ex[C5_ex * j + 3] = Ptau1_C5[C5 * j + 18] + Ptau1_C5[C5 * j + 19] + Ptau1_C5[C5 * j + 21] + Ptau1_C5[C5 * j + 22];
        Ptau1_C5_ex[C5_ex * j + 4] = Ptau1_C5[C5 * j + 24] + Ptau1_C5[C5 * j + 25];
        Ptau1_C5_ex[C5_ex * j + 5] = Ptau1_C5[C5 * j + 20] + Ptau1_C5[C5 * j + 23];
        Ptau1_C5_ex[C5_ex * j + 6] = Ptau1_C5[C5 * j +  8] + Ptau1_C5[C5 * j + 17];
        Ptau1_C5_ex[C5_ex * j + 7] = Ptau1_C5[C5 * j + 26];
        Ptau1_C5_ex[C5_ex * j + 8] =  Ptau1_C5_ex[C5_ex * j + 0] * exptaugap[0]
                                   + (Ptau1_C5_ex[C5_ex * j + 1] + Ptau1_C5_ex[C5_ex * j + 2] + Ptau1_C5_ex[C5_ex * j + 3]) * exptaugap[1]
                                   + (Ptau1_C5_ex[C5_ex * j + 4] + Ptau1_C5_ex[C5_ex * j + 5] + Ptau1_C5_ex[C5_ex * j + 6]) * exptaugap[2]
                                   +  Ptau1_C5_ex[C5_ex * j + 7] * exptaugap[3];
    }

    gtt = com.GtreeTab[0];

    for (GtreeType = 1; GtreeType <= NGTREETYPES; GtreeType++) {
        yup[0] = gtt[GtreeType - 1].lim[0];
        yup[1] = gtt[GtreeType - 1].lim[1];
        coeff = yup[0] * yup[1] * 0.5 * 0.5;

        for (igrid = 0; igrid < 3 * K * K; igrid += 3) {
            ixw[0] = (igrid/3)/K; ixw[1] = (igrid/3)%K;
            for(j=0; j<2; j++) {  /* t0 (y0) and t1 (y1) */
                if(ixw[j]<K/2) { ixw[j] = K/2-1-ixw[j];  s[j]=-1; }
                else           { ixw[j] = ixw[j]-K/2;    s[j]=1;  }
                y[j] = yup[j]*(1 + s[j]*xI[ixw[j]])/2;
                t[j] = y[j]/(1-y[j]);
            }

            gtt[GtreeType - 1].t0t1(t[0], t[1], t01);

            if (GtreeType <= 3) {
                computePMatrix(t01[1], Q_C5, C5, Pt_C5, work);
                P5 = Pt_C5;
            }
            else
                P5 = Ptau1_C5_ex;

            if (GtreeType == 1)
                computePMatrix(t01[0], Q_C6, C6, Pt_C6, work);
            else if (GtreeType <= 3)
                computePMatrix(tau1 - t01[1], Q_C6, C6, Pt_C6, work);

            gtt[GtreeType - 1].helper(t[0], t[1], Pt_C6, exptaugap + 1, Pr);

            for (i = 0; i < NINITIALSTATES; i++) {
                if (!data.iStateCnt[stree.sptree][initStates[i]])
                    continue;

                initState = initStates[i];
                Gtree = NGTREETYPES * i + GtreeType - 1;

                com.wwprior[Gtree][igrid] = wI[ixw[0]] * wI[ixw[1]] / square((1 - y[0]) * (1 - y[1])) * coeff;
                com.wwprior[Gtree][igrid + 1] = com.wwprior[Gtree][igrid + 2] = com.wwprior[Gtree][igrid];

                gtt[GtreeType - 1].density(initState, P5, NULL, Pr, &com.wwprior[Gtree][igrid]);

                computeMultinomProbs(Gtree, igrid, GtreeType - 1, t, 0);
            }
        }
    }

    com.space = Q_C5;
}

void priorM2ProMax_3seq() {
    double theta4, theta5, theta1, theta2, theta3, thetaW, tau0, tau1, M12 = 0.0, M21 = 0.0, M13 = 0.0, M31 = 0.0, M23 = 0.0, M32 = 0.0, M53 = 0.0, M35 = 0.0, coeff;
    int i, j, igrid, Gtree, ixw[2], K = com.npoints, initState, GtreeType;
    const double *xI = NULL, *wI = NULL;
    double /*b[2],*/ s[2], t[2], y[2], yup[2], t01[2];
    double *Q_C5, *Q_C6, *Q_C7, *Q_C8, *Pt_C5, *Pt_C6, *Pt_C7, *Pt_C8, *Ptau1_C5, *Ptau1_C5_ex, *Ptaugap_C7, *Ptaugap_C7_ex, *Ptaugap_C8, *Ptaugap_C8_ex, *Pr, *work;
    double *P5 = NULL, *P8 = NULL;
    BTEntry * gtt;
    enum { NGTREETYPES = 6 };
    STATIC_ASSERT(MAXGTREETYPES >= NGTREETYPES);

    Q_C5 = com.space;
    if (!Q_C5) error2("oom allocating Q_C5 in priorM2ProMax_3seq");
    Q_C6 = &Q_C5[C5 * C5];
    Q_C7 = &Q_C6[C6 * C6];
    Q_C8 = &Q_C7[C7 * C7];
    Pt_C5 = &Q_C8[C8 * C8];
    Pt_C6 = &Pt_C5[C5 * C5];
    Pt_C7 = &Pt_C6[C6 * C6];
    Pt_C8 = &Pt_C7[C7 * C7];
    Ptau1_C5 = &Pt_C8[C8 * C8];
    Ptau1_C5_ex = &Ptau1_C5[C5 * C5];
    Ptaugap_C7 = &Ptau1_C5_ex[C5 * C5_ex];
    Ptaugap_C7_ex = &Ptaugap_C7[C7 * C7];
    Ptaugap_C8 = &Ptaugap_C7_ex[C7 * C7_ex];
    Ptaugap_C8_ex = &Ptaugap_C8[C8 * C8];
    Pr = &Ptaugap_C8_ex[C8 * C8_ex];
    work = &Pr[max2(max2(C6, C8), 1)];
    com.space = work + 3 * square(max2(max2(max2(C5, C6), C7), C8));

    getParams(&theta4, &theta5, &tau0, &tau1, &theta1, &theta2, &theta3);

    thetaW = para[7];
    M12 = para[8];
    M21 = para[9];
    M13 = para[10];
    M31 = para[11];
    M23 = para[12];
    M32 = para[13];
    M53 = para[14];
    M35 = para[15];

    GaussLegendreRule(&xI, &wI, com.npoints);

    GenerateQ5(Q_C5, theta1, theta2, theta3, M12, M21, M13, M31, M23, M32);
    GenerateQ6(Q_C6, theta1, theta2, theta3, M12, M21, M13, M31, M23, M32);
    GenerateQ7(Q_C7, theta5, thetaW, M53, M35);
    GenerateQ8(Q_C8, theta5, thetaW, M53, M35);

    computePMatrix(tau1, Q_C5, C5, Ptau1_C5, work);
    computePMatrix(tau0 - tau1, Q_C7, C7, Ptaugap_C7, work);
    computePMatrix(tau0 - tau1, Q_C8, C8, Ptaugap_C8, work);

    for (i = 0; i < C8 - 1; i++)
        Ptaugap_C8_ex[C8_ex * i] = 1 - Ptaugap_C8[C8 * i + C8 - 1];

    for (i = 0; i < C7 - 1; i++)
        Ptaugap_C7_ex[C7_ex * i] = 1 - Ptaugap_C7[C7 * i + C7 - 1];

    for (i = 0; i < NINITIALSTATES; i++) {
        j = initStates[i];
        Ptau1_C5_ex[C5_ex * j + 0] = Ptau1_C5[C5 * j +  0] + Ptau1_C5[C5 * j +  1] + Ptau1_C5[C5 * j +  3] + Ptau1_C5[C5 * j +  4]
                                   + Ptau1_C5[C5 * j +  9] + Ptau1_C5[C5 * j + 10] + Ptau1_C5[C5 * j + 12] + Ptau1_C5[C5 * j + 13];
        Ptau1_C5_ex[C5_ex * j + 1] = Ptau1_C5[C5 * j +  2] + Ptau1_C5[C5 * j +  5] + Ptau1_C5[C5 * j + 11] + Ptau1_C5[C5 * j + 14];
        Ptau1_C5_ex[C5_ex * j + 2] = Ptau1_C5[C5 * j +  6] + Ptau1_C5[C5 * j +  7] + Ptau1_C5[C5 * j + 15] + Ptau1_C5[C5 * j + 16];
        Ptau1_C5_ex[C5_ex * j + 3] = Ptau1_C5[C5 * j + 18] + Ptau1_C5[C5 * j + 19] + Ptau1_C5[C5 * j + 21] + Ptau1_C5[C5 * j + 22];
        Ptau1_C5_ex[C5_ex * j + 4] = Ptau1_C5[C5 * j + 24] + Ptau1_C5[C5 * j + 25];
        Ptau1_C5_ex[C5_ex * j + 5] = Ptau1_C5[C5 * j + 20] + Ptau1_C5[C5 * j + 23];
        Ptau1_C5_ex[C5_ex * j + 6] = Ptau1_C5[C5 * j +  8] + Ptau1_C5[C5 * j + 17];
        Ptau1_C5_ex[C5_ex * j + 7] = Ptau1_C5[C5 * j + 26];
        Ptau1_C5_ex[C5_ex * j + 8] = Ptau1_C5_ex[C5_ex * j + 0] * Ptaugap_C7_ex[C7_ex * 0]
                                   + Ptau1_C5_ex[C5_ex * j + 1] * Ptaugap_C7_ex[C7_ex * 1]
                                   + Ptau1_C5_ex[C5_ex * j + 2] * Ptaugap_C7_ex[C7_ex * 2]
                                   + Ptau1_C5_ex[C5_ex * j + 3] * Ptaugap_C7_ex[C7_ex * 3]
                                   + Ptau1_C5_ex[C5_ex * j + 4] * Ptaugap_C7_ex[C7_ex * 4]
                                   + Ptau1_C5_ex[C5_ex * j + 5] * Ptaugap_C7_ex[C7_ex * 5]
                                   + Ptau1_C5_ex[C5_ex * j + 6] * Ptaugap_C7_ex[C7_ex * 6]
                                   + Ptau1_C5_ex[C5_ex * j + 7] * Ptaugap_C7_ex[C7_ex * 7];
    }

    gtt = com.GtreeTab[0];

    for (GtreeType = 1; GtreeType <= NGTREETYPES; GtreeType++) {
        yup[0] = gtt[GtreeType - 1].lim[0];
        yup[1] = gtt[GtreeType - 1].lim[1];
        coeff = yup[0] * yup[1] * 0.5 * 0.5;

        for (igrid = 0; igrid < 3 * K * K; igrid += 3) {
            ixw[0] = (igrid/3)/K; ixw[1] = (igrid/3)%K;
            for(j=0; j<2; j++) {  /* t0 (y0) and t1 (y1) */
                if(ixw[j]<K/2) { ixw[j] = K/2-1-ixw[j];  s[j]=-1; }
                else           { ixw[j] = ixw[j]-K/2;    s[j]=1;  }
                y[j] = yup[j]*(1 + s[j]*xI[ixw[j]])/2;
                t[j] = y[j]/(1-y[j]);
            }

            gtt[GtreeType - 1].t0t1(t[0], t[1], t01);

            if (GtreeType <= 3) {
                computePMatrix(t01[1], Q_C5, C5, Pt_C5, work);
                P5 = Pt_C5;
            }
            else
                P5 = Ptau1_C5_ex;

            if (GtreeType == 1)
                computePMatrix(t01[0], Q_C6, C6, Pt_C6, work);
            else if (GtreeType <= 3)
                computePMatrix(tau1 - t01[1], Q_C6, C6, Pt_C6, work);
            else if (GtreeType <= 5)
                computePMatrix(t01[1], Q_C7, C7, Pt_C7, work);

            if (GtreeType == 2 || GtreeType == 4) {
                computePMatrix(t01[0], Q_C8, C8, Pt_C8, work);
                P8 = Pt_C8;
            }
            else if (GtreeType == 5) {
                computePMatrix(tau0 - tau1 - t01[1], Q_C8, C8, Pt_C8, work);
                P8 = Pt_C8;
            }
            if (GtreeType == 3)
                P8 = Ptaugap_C8_ex;

            gtt[GtreeType - 1].helper(t[0], t[1], Pt_C6, P8, Pr);

            for (i = 0; i < NINITIALSTATES; i++) {
                if (!data.iStateCnt[stree.sptree][initStates[i]])
                    continue;

                initState = initStates[i];
                Gtree = NGTREETYPES * i + GtreeType - 1;

                com.wwprior[Gtree][igrid] = wI[ixw[0]] * wI[ixw[1]] / square((1 - y[0]) * (1 - y[1])) * coeff;
                com.wwprior[Gtree][igrid + 1] = com.wwprior[Gtree][igrid + 2] = com.wwprior[Gtree][igrid];

                gtt[GtreeType - 1].density(initState, P5, Pt_C7, Pr, &com.wwprior[Gtree][igrid]);

                computeMultinomProbs(Gtree, igrid, GtreeType - 1, t, 0);
            }
        }
    }

    com.space = Q_C5;
}

void priorM3MSci12_3seq() {
    double theta4, theta5, theta1, theta2, theta3, tau0, tau1, T, thetaX, thetaY, phi12, phi21, coeff;
    int j, igrid, Gtree, ixw[2], K=com.npoints, *iStateCnt = data.iStateCnt[stree.sptree];
    const double *xI=NULL, *wI=NULL;
    double /*b[2],*/ s[2], t[2], y[2], yup[2];
    double exptau1T[4], Pr[7];
    BTEntry * gtt;

    getParams(&theta4, &theta5, &tau0, &tau1, &theta1, &theta2, &theta3);

    T = para[7];
    thetaX = para[8];
    thetaY = para[9];
    phi12 = para[10];
    phi21 = para[11];

    GaussLegendreRule(&xI, &wI, com.npoints);

    exptau1T[0] = exp(-6 * (tau1 - T) / thetaX);
    exptau1T[1] = exp(-2 * (tau1 - T) / thetaX);
    exptau1T[2] = exp(-2 * (tau1 - T) / thetaY);
    exptau1T[3] = exp(-6 * (tau1 - T) / thetaY);

    if (iStateCnt[0] || iStateCnt[2])
        Pr[0] = (1 - phi21) * (1 - phi21) * exptau1T[1] + 2 * (1 - phi21) * phi21 + phi21 * phi21 * exptau1T[2];
    if (iStateCnt[13] || iStateCnt[14])
        Pr[1] = (1 - phi12) * (1 - phi12) * exptau1T[2] + 2 * (1 - phi12) * phi12 + phi12 * phi12 * exptau1T[1];
    if (iStateCnt[0])
        Pr[2] = (1 - phi21) * (1 - phi21) * (1 - phi21) * exptau1T[0]
              + 3 * (1 - phi21) * (1 - phi21) * phi21 * exptau1T[1]
              + 3 * (1 - phi21) * phi21 * phi21 * exptau1T[2]
              + phi21 * phi21 * phi21 * exptau1T[3];
    if (iStateCnt[13])
        Pr[3] = (1 - phi12) * (1 - phi12) * (1 - phi12) * exptau1T[3]
              + 3 * (1 - phi12) * (1 - phi12) * phi12 * exptau1T[2]
              + 3 * (1 - phi12) * phi12 * phi12 * exptau1T[1]
              + phi12 * phi12 * phi12 * exptau1T[0];
    if (iStateCnt[1] || iStateCnt[12] || iStateCnt[5])
        Pr[4] = (1 - phi21) * phi12 * exptau1T[1] + (1 - phi21) * (1 - phi12) + phi21 * phi12 + phi21 * (1 - phi12) * exptau1T[2];
    if (iStateCnt[1])
        Pr[5] = (1 - phi21) * (1 - phi21) * phi12 * exptau1T[0]
              + (1 - phi21) * ((1 - phi21) * (1 - phi12) + 2 * phi21 * phi12) * exptau1T[1]
              + phi21 * (phi21 * phi12 + 2 * (1 - phi21) * (1 - phi12)) * exptau1T[2]
              + phi21 * phi21 * (1 - phi12) * exptau1T[3];
    if (iStateCnt[12])
        Pr[6] = (1 - phi12) * (1 - phi12) * phi21 * exptau1T[3]
              + (1 - phi12) * ((1 - phi12) * (1 - phi21) + 2 * phi12 * phi21) * exptau1T[2]
              + phi12 * (phi12 * phi21 + 2 * (1 - phi12) * (1 - phi21)) * exptau1T[1]
              + phi12 * phi12 * (1 - phi21) * exptau1T[0];

    gtt = com.GtreeTab[0];
    for(Gtree = 0; Gtree < com.nGtree; Gtree++) {
        if (skipTree(Gtree)) continue;

        yup[0] = gtt[GtOffsetM3MSci12[Gtree]].lim[0];
        yup[1] = gtt[GtOffsetM3MSci12[Gtree]].lim[1];
        coeff = yup[0] * yup[1] * 0.5 * 0.5;

        for(igrid=0; igrid<3*K*K; igrid+=3) {
            ixw[0] = (igrid/3)/K; ixw[1] = (igrid/3)%K;
            for(j=0; j<2; j++) {  /* t0 (y0) and t1 (y1) */
                if(ixw[j]<K/2) { ixw[j] = K/2-1-ixw[j];  s[j]=-1; }
                else           { ixw[j] = ixw[j]-K/2;    s[j]=1;  }
                y[j] = yup[j]*(1 + s[j]*xI[ixw[j]])/2;
                t[j] = y[j]/(1-y[j]);
            }

            com.wwprior[Gtree][igrid] = wI[ixw[0]] * wI[ixw[1]] / square((1 - y[0]) * (1 - y[1])) * coeff;
            com.wwprior[Gtree][igrid + 1] = com.wwprior[Gtree][igrid + 2] = com.wwprior[Gtree][igrid];

            gtt[GtOffsetM3MSci12[Gtree]].g(t[0], t[1], Pr, &com.wwprior[Gtree][igrid]);

            computeMultinomProbs(Gtree, igrid, GtOffsetM3MSci12[Gtree], t, 0);
        }
    }
}

void priorM3MSci13_3seq() {
    double theta4, theta5, theta1, theta2, theta3, tau0, tau1, T, thetaX, thetaZ, phi13, phi31, coeff;
    int j, igrid, Gtree, ixw[2], K=com.npoints, *iStateCnt = data.iStateCnt[stree.sptree];
    const double *xI=NULL, *wI=NULL;
    double /*b[2],*/ s[2], t[2], y[2], yup[2];
    double exptaugapT[3], exptau0T[2], exptaugap, Pr[11];
    BTEntry * gtt;

    getParams(&theta4, &theta5, &tau0, &tau1, &theta1, &theta2, &theta3);

    T = para[7];
    thetaX = para[8];
    thetaZ = para[9];
    phi13 = para[10];
    phi31 = para[11];

    GaussLegendreRule(&xI, &wI, com.npoints);

    exptaugapT[0] = exp(-2 * (tau1 - T) / thetaX - 2 * (tau0 - tau1) / theta5);
    exptaugapT[1] = exp(-6 * (tau1 - T) / thetaX - 6 * (tau0 - tau1) / theta5);
    exptaugapT[2] = exp(-2 * (tau1 - T) / thetaX - 6 * (tau0 - tau1) / theta5);

    exptau0T[0] = exp(-2 * (tau0 - T) / thetaZ);
    exptau0T[1] = exp(-6 * (tau0 - T) / thetaZ);

    exptaugap = exp(-2 * (tau0 - tau1) / theta5);

    if (iStateCnt[0]) {
        Pr[0] = (1 - phi31) * (1 - phi31) * exptaugapT[0] + 2 * (1 - phi31) * phi31 + phi31 * phi31 * exptau0T[0];
        Pr[1] = (1 - phi31) * (1 - phi31) * (1 - phi31) * exptaugapT[1]
              + 3 * (1 - phi31) * (1 - phi31) * phi31 * exptaugapT[0]
              + 3 * (1 - phi31) * phi31 * phi31 * exptau0T[0]
              + phi31 * phi31 * phi31 * exptau0T[1];
    }
    if (iStateCnt[1]) {
        Pr[2] = (1 - phi31) * (1 - phi31) * exptaugapT[2] + 2 * (1 - phi31) * phi31 * exptaugap + phi31 * phi31 * exptau0T[0];
    }
    if (iStateCnt[2]) {
        Pr[3] = (1 - phi31) * phi13 * exptaugapT[0] + (1 - phi31) * (1 - phi13) + phi31 * phi13 + phi31 * (1 - phi13) * exptau0T[0];
        Pr[4] = (1 - phi31) * (1 - phi31) * phi13 * exptaugapT[1]
              + (1 - phi31) * ((1 - phi31) * (1 - phi13) + 2 * phi31 * phi13) * exptaugapT[0]
              + phi31 * (phi31 * phi13 + 2 * (1 - phi31) * (1 - phi13)) * exptau0T[0]
              + phi31 * phi31 * (1 - phi13) * exptau0T[1];
    }
    if (iStateCnt[5]) {
        Pr[5] = (1 - phi31) * phi13 * exptaugapT[2] + ((1 - phi31) * (1 - phi13) + phi31 * phi13) * exptaugap + phi31 * (1 - phi13) * exptau0T[0];
    }
    if (iStateCnt[8]) {
        Pr[6] = phi13 * (1 - phi31) * exptaugapT[0] + phi13 * phi31 + (1 - phi13) * (1 - phi31) + (1 - phi13) * phi31 * exptau0T[0];
        Pr[7] = phi13 * phi13 * (1 - phi31) * exptaugapT[1]
              + phi13 * (phi13 * phi31 + 2 * (1 - phi13) * (1 - phi31)) * exptaugapT[0]
              + (1 - phi13) * ((1 - phi13) * (1 - phi31) + 2 * phi13 * phi31) * exptau0T[0]
              + (1 - phi13) * (1 - phi13) * phi31 * exptau0T[1];
    }
    if (iStateCnt[17]) {
        Pr[8] = phi13 * phi13 * exptaugapT[2] + 2 * phi13 * (1 - phi13) * exptaugap + (1 - phi13) * (1 - phi13) * exptau0T[0];
    }
    if (iStateCnt[26]) {
        Pr[9] = phi13 * phi13 * exptaugapT[0] + 2 * phi13 * (1 - phi13) + (1 - phi13) * (1 - phi13) * exptau0T[0];
        Pr[10] = phi13 * phi13 * phi13 * exptaugapT[1]
               + 3 * phi13 * phi13 * (1 - phi13) * exptaugapT[0]
               + 3 * phi13 * (1 - phi13) * (1 - phi13) * exptau0T[0]
               + (1 - phi13) * (1 - phi13) * (1 - phi13) * exptau0T[1];
    }

    gtt = com.GtreeTab[0];
    for(Gtree = 0; Gtree < com.nGtree; Gtree++) {
        if (skipTree(Gtree)) continue;

        yup[0] = gtt[GtOffsetM3MSci13[Gtree]].lim[0];
        yup[1] = gtt[GtOffsetM3MSci13[Gtree]].lim[1];
        coeff = yup[0] * yup[1] * 0.5 * 0.5;

        for(igrid=0; igrid<3*K*K; igrid+=3) {
            ixw[0] = (igrid/3)/K; ixw[1] = (igrid/3)%K;
            for(j=0; j<2; j++) {  /* t0 (y0) and t1 (y1) */
                if(ixw[j]<K/2) { ixw[j] = K/2-1-ixw[j];  s[j]=-1; }
                else           { ixw[j] = ixw[j]-K/2;    s[j]=1;  }
                y[j] = yup[j]*(1 + s[j]*xI[ixw[j]])/2;
                t[j] = y[j]/(1-y[j]);
            }

            com.wwprior[Gtree][igrid] = wI[ixw[0]] * wI[ixw[1]] / square((1 - y[0]) * (1 - y[1])) * coeff;
            com.wwprior[Gtree][igrid + 1] = com.wwprior[Gtree][igrid + 2] = com.wwprior[Gtree][igrid];

            gtt[GtOffsetM3MSci13[Gtree]].g(t[0], t[1], Pr, &com.wwprior[Gtree][igrid]);

            computeMultinomProbs(Gtree, igrid, GtOffsetM3MSci13[Gtree], t, 0);
        }
    }
}

void priorM3MSci23_3seq() {
    double theta4, theta5, theta1, theta2, theta3, tau0, tau1, T, thetaY, thetaZ, phi23, phi32, coeff;
    int j, igrid, Gtree, ixw[2], K=com.npoints, *iStateCnt = data.iStateCnt[stree.sptree];
    const double *xI=NULL, *wI=NULL;
    double /*b[2],*/ s[2], t[2], y[2], yup[2];
    double exptaugapT[3], exptau0T[2], exptaugap, Pr[11];
    BTEntry * gtt;

    getParams(&theta4, &theta5, &tau0, &tau1, &theta1, &theta2, &theta3);

    T = para[7];
    thetaY = para[8];
    thetaZ = para[9];
    phi23 = para[10];
    phi32 = para[11];

    GaussLegendreRule(&xI, &wI, com.npoints);

    exptaugapT[0] = exp(-2 * (tau1 - T) / thetaY - 2 * (tau0 - tau1) / theta5);
    exptaugapT[1] = exp(-6 * (tau1 - T) / thetaY - 6 * (tau0 - tau1) / theta5);
    exptaugapT[2] = exp(-2 * (tau1 - T) / thetaY - 6 * (tau0 - tau1) / theta5);

    exptau0T[0] = exp(-2 * (tau0 - T) / thetaZ);
    exptau0T[1] = exp(-6 * (tau0 - T) / thetaZ);

    exptaugap = exp(-2 * (tau0 - tau1) / theta5);

    if (iStateCnt[13]) {
        Pr[0] = (1 - phi32) * (1 - phi32) * exptaugapT[0] + 2 * (1 - phi32) * phi32 + phi32 * phi32 * exptau0T[0];
        Pr[1] = (1 - phi32) * (1 - phi32) * (1 - phi32) * exptaugapT[1]
              + 3 * (1 - phi32) * (1 - phi32) * phi32 * exptaugapT[0]
              + 3 * (1 - phi32) * phi32 * phi32 * exptau0T[0]
              + phi32 * phi32 * phi32 * exptau0T[1];
    }
    if (iStateCnt[12]) {
        Pr[2] = (1 - phi32) * (1 - phi32) * exptaugapT[2] + 2 * (1 - phi32) * phi32 * exptaugap + phi32 * phi32 * exptau0T[0];
    }
    if (iStateCnt[14]) {
        Pr[3] = (1 - phi32) * phi23 * exptaugapT[0] + (1 - phi32) * (1 - phi23) + phi32 * phi23 + phi32 * (1 - phi23) * exptau0T[0];
        Pr[4] = (1 - phi32) * (1 - phi32) * phi23 * exptaugapT[1]
              + (1 - phi32) * ((1 - phi32) * (1 - phi23) + 2 * phi32 * phi23) * exptaugapT[0]
              + phi32 * (phi32 * phi23 + 2 * (1 - phi32) * (1 - phi23)) * exptau0T[0]
              + phi32 * phi32 * (1 - phi23) * exptau0T[1];
    }
    if (iStateCnt[5]) {
        Pr[5] = (1 - phi32) * phi23 * exptaugapT[2] + ((1 - phi32) * (1 - phi23) + phi32 * phi23) * exptaugap + phi32 * (1 - phi23) * exptau0T[0];
    }
    if (iStateCnt[17]) {
        Pr[6] = phi23 * (1 - phi32) * exptaugapT[0] + phi23 * phi32 + (1 - phi23) * (1 - phi32) + (1 - phi23) * phi32 * exptau0T[0];
        Pr[7] = phi23 * phi23 * (1 - phi32) * exptaugapT[1]
              + phi23 * (phi23 * phi32 + 2 * (1 - phi23) * (1 - phi32)) * exptaugapT[0]
              + (1 - phi23) * ((1 - phi23) * (1 - phi32) + 2 * phi23 * phi32) * exptau0T[0]
              + (1 - phi23) * (1 - phi23) * phi32 * exptau0T[1];
    }
    if (iStateCnt[8]) {
        Pr[8] = phi23 * phi23 * exptaugapT[2] + 2 * phi23 * (1 - phi23) * exptaugap + (1 - phi23) * (1 - phi23) * exptau0T[0];
    }
    if (iStateCnt[26]) {
        Pr[9] = phi23 * phi23 * exptaugapT[0] + 2 * phi23 * (1 - phi23) + (1 - phi23) * (1 - phi23) * exptau0T[0];
        Pr[10] = phi23 * phi23 * phi23 * exptaugapT[1]
               + 3 * phi23 * phi23 * (1 - phi23) * exptaugapT[0]
               + 3 * phi23 * (1 - phi23) * (1 - phi23) * exptau0T[0]
               + (1 - phi23) * (1 - phi23) * (1 - phi23) * exptau0T[1];
    }

    gtt = com.GtreeTab[0];
    for(Gtree = 0; Gtree < com.nGtree; Gtree++) {
        if (skipTree(Gtree)) continue;

        yup[0] = gtt[GtOffsetM3MSci23[Gtree]].lim[0];
        yup[1] = gtt[GtOffsetM3MSci23[Gtree]].lim[1];
        coeff = yup[0] * yup[1] * 0.5 * 0.5;

        for(igrid=0; igrid<3*K*K; igrid+=3) {
            ixw[0] = (igrid/3)/K; ixw[1] = (igrid/3)%K;
            for(j=0; j<2; j++) {  /* t0 (y0) and t1 (y1) */
                if(ixw[j]<K/2) { ixw[j] = K/2-1-ixw[j];  s[j]=-1; }
                else           { ixw[j] = ixw[j]-K/2;    s[j]=1;  }
                y[j] = yup[j]*(1 + s[j]*xI[ixw[j]])/2;
                t[j] = y[j]/(1-y[j]);
            }

            com.wwprior[Gtree][igrid] = wI[ixw[0]] * wI[ixw[1]] / square((1 - y[0]) * (1 - y[1])) * coeff;
            com.wwprior[Gtree][igrid + 1] = com.wwprior[Gtree][igrid + 2] = com.wwprior[Gtree][igrid];

            gtt[GtOffsetM3MSci23[Gtree]].g(t[0], t[1], Pr, &com.wwprior[Gtree][igrid]);

            computeMultinomProbs(Gtree, igrid, GtOffsetM3MSci23[Gtree], t, 0);
        }
    }
}

void priorM0_2seq() {
    double theta4, theta5, theta1, theta2, theta12, theta3, tau0, tau1, coeff;
    int /*j,*/ igrid, segm, ixw, K=com.npoints, index2seq = MAXGTREES;
    const double *xI=NULL, *wI=NULL;
    double /*b,*/ s, t, y, yup;
    enum { NSEGM = 11 };
    STATIC_ASSERT(MAXGTREES2SEQ >= NSEGM);

    getParams(&theta4, &theta5, &tau0, &tau1, &theta1, &theta2, &theta3);

    GaussLegendreRule(&xI, &wI, com.npoints);

    for(segm = 0; segm < NSEGM; segm++) {
        if (segm < 3) {
            theta12 = theta1;
        } else if (segm < 6) {
            theta12 = theta2;
        } else {
            theta12 = -1;
        }
        if (segm == 0 || segm == 3) {
            yup = 2*tau1/(2*tau1 + theta12);
        } else if (segm == 1 || segm == 4 || segm == 6) {
            yup = 2*(tau0-tau1)/(2*(tau0-tau1)+theta5);
        } else if (segm == 9) {
            yup = 2*tau0/(2*tau0 + theta3);
        } else {
            yup = 1;
        }
        coeff = yup * 0.5;

        for(igrid=0; igrid<K; igrid++) {
            /* compute t (y) on the grid */
            if(igrid<K/2) { ixw = K/2-1-igrid; s=-1; }
            else          { ixw = igrid-K/2;   s= 1; }
            y = yup*(1 + s*xI[ixw])/2;
            t = y/(1-y);

            com.wwprior[index2seq][segm*K+igrid] = wI[ixw]/((1-y)*(1-y))*coeff;

            if (segm == 1 || segm == 4) {
                com.wwprior[index2seq][segm*K+igrid] *= exp(-(2/theta12*tau1+t));
            } else if (segm == 2 || segm == 5) {
                com.wwprior[index2seq][segm*K+igrid] *= exp(-(2/theta12*tau1+2/theta5*(tau0-tau1)+t));
            } else if (segm == 7) {
                com.wwprior[index2seq][segm*K+igrid] *= exp(-(2/theta5*(tau0-tau1)+t));
            } else if (segm == 10) {
                com.wwprior[index2seq][segm*K+igrid] *= exp(-(2/theta3*tau0+t));
            } else {
                com.wwprior[index2seq][segm*K+igrid] *= exp(-t);
            }

            if (segm == 0 || segm == 3) {
                com.bp0124[index2seq][segm*K+igrid] = theta12/2*t;
            } else if (segm == 1 || segm == 4 || segm == 6) {
                com.bp0124[index2seq][segm*K+igrid] = tau1 + theta5/2*t;
            } else if (segm == 9) {
                com.bp0124[index2seq][segm*K+igrid] = theta3/2*t;
            } else {
                com.bp0124[index2seq][segm*K+igrid] = tau0 + theta4/2*t;
            }
        }
    }
}

void priorM2SIM3s_2seq() {
    double theta4, theta5, theta1, theta2, theta12, theta3, tau0, tau1, M12, M21, coeff;
    int /*i, j,*/ igrid, segm, ixw, K=com.npoints, index2seq = MAXGTREES, initState;
    const double *xI=NULL, *wI=NULL;
    double /*a, b, c, m,*/ s, t, y, yup, Pc11, Pc12, Pc22, Pc/*, tmp*/;
    double * Q, * Ptau1, * Pt1, * work;
    enum { NSEGM = 12 };
    STATIC_ASSERT(MAXGTREES2SEQ >= NSEGM);

    Q = com.space;
    if(!Q) error2("oom allocating Q in priorM2SIM3s_2seq");
    Ptau1 = &Q[C3*C3]; Pt1 = &Ptau1[C3*C3]; work = &Pt1[C3*C3];
    com.space = &work[4*C3*C3];

    getParams(&theta4, &theta5, &tau0, &tau1, &theta1, &theta2, &theta3);

    theta12 = (theta1 + theta2) / 2;
#ifdef FIXM12
    M12 = 0.000001;
    M21 = 0.000001;
#else
    M12 = para[8];
    M21 = para[9];
#endif

    GaussLegendreRule(&xI, &wI, com.npoints);

    GenerateQ1SIM3S(Q, C3, theta1, theta2, 0, M12, M21);

    computePMatrix(tau1, Q, C3, Ptau1, work);

    Pc11 = Ptau1[3];
    Pc12 = Ptau1[1*C3+3];
    Pc22 = Ptau1[2*C3+3];

    for(segm = 0; segm < NSEGM; segm++) {
        if (segm < 9 && !(segm % 3)) {
            yup = 2*tau1/(2*tau1 + theta12); //y=t/(1+t)
        } else if (segm < 9 && (segm % 3 == 1)) {
            yup = 2*(tau0-tau1)/(2*(tau0-tau1)+theta5);
        } else if (segm == 10) {
            yup = 2*tau0/(2*tau0 + theta3);
        } else {
            yup = 1;
        }
        coeff = yup * 0.5;

        if (segm < 3) {
            Pc = Pc11;
            initState = 0;
        } else if (segm < 6) {
            Pc = Pc22;
            initState = 2;
        } else {
            Pc = Pc12;
            initState = 1;
        }

        for(igrid=0; igrid<K; igrid++) {
            /* compute t (y) on the grid */
            if(igrid<K/2) { ixw = K/2-1-igrid; s=-1; }
            else          { ixw = igrid-K/2;   s= 1; }
            y = yup*(1 + s*xI[ixw])/2;
            t = y/(1-y);

            com.wwprior[index2seq][segm*K+igrid] = wI[ixw]/((1-y)*(1-y))*coeff;
            if (segm < 9) {
                if (!(segm % 3)) {
                        computePMatrix(theta12/2 * t, Q, C3, Pt1, work);
                        com.wwprior[index2seq][segm*K+igrid] *= (2/theta1*Pt1[initState*C3] + 2/theta2*Pt1[initState*C3+2]) * theta12/2;
                } else if (segm % 3 == 1) {
                    com.wwprior[index2seq][segm*K+igrid] *= (1-Pc)*exp(-t);
                } else if (segm % 3 == 2) {
                    com.wwprior[index2seq][segm*K+igrid] *= (1-Pc)*exp(-(2/theta5*(tau0-tau1)+t));
                }
            } else if (segm == 11) {
                com.wwprior[index2seq][segm*K+igrid] *= exp(-(2/theta3*tau0+t));
            } else {
                com.wwprior[index2seq][segm*K+igrid] *= exp(-t);
            }

            if (segm < 9 && !(segm % 3)) {
                com.bp0124[index2seq][segm*K+igrid] = theta12/2*t;
            } else if (segm < 9 && (segm % 3 == 1)) {
                com.bp0124[index2seq][segm*K+igrid] = tau1 + theta5/2*t;
            } else if (segm == 10) {
                com.bp0124[index2seq][segm*K+igrid] = theta3/2*t;
            } else {
                com.bp0124[index2seq][segm*K+igrid] = tau0 + theta4/2*t;
            }
        }
    }
    com.space = Q;
}

void priorM2Pro_2seq() {
    double theta4, theta5, theta1, theta2, theta3, thetaW, theta123, theta5W, tau0, tau1, M12, M21, M13, M31, M23, M32, coeff;
    int i, j, igrid, segm, ixw, K = com.npoints, index2seq = MAXGTREES, initState, GtreeType;
    const double *xI = NULL, *wI = NULL;
    double /*a, b, c, m,*/ s, t, y, yup;
    double *Q_C6, *Pt_C6, *Ptau1_C6, *Ptau1_C6_ex, *work;
    double expt[3], exptaugap[2];
    enum { NGTREETYPES2SEQ = 3 };
    STATIC_ASSERT(MAXGTREETYPES2SEQ >= NGTREETYPES2SEQ);

    Q_C6 = com.space;
    if (!Q_C6) error2("oom allocating Q_C6 in priorM2Pro_2seq");
    Pt_C6 = &Q_C6[C6 * C6];
    Ptau1_C6 = &Pt_C6[C6 * C6];
    Ptau1_C6_ex = &Ptau1_C6[C6 * C6];
    work = &Ptau1_C6_ex[C6 * C6_ex];
    com.space = work + 3 * square(C6);

    getParams(&theta4, &theta5, &tau0, &tau1, &theta1, &theta2, &theta3);

    thetaW = para[7];
    M12 = para[8];
    M21 = para[9];
    M13 = para[10];
    M31 = para[11];
    M23 = para[12];
    M32 = para[13];

    theta5W = theta5 + thetaW;
    theta123 = theta1 + theta2 + theta3;

    GaussLegendreRule(&xI, &wI, com.npoints);

    GenerateQ6(Q_C6, theta1, theta2, theta3, M12, M21, M13, M31, M23, M32);

    computePMatrix(tau1, Q_C6, C6, Ptau1_C6, work);

    exptaugap[0] = exp(-2 * (tau0 - tau1) / theta5);
    exptaugap[1] = exp(-2 * (tau0 - tau1) / thetaW);

    for (i = 0; i < NINITIALSTATES2SEQ; i++) {
        j = initStates2seq[i];
        Ptau1_C6_ex[C6_ex * j + 0] = Ptau1_C6[C6 * j + 0] + Ptau1_C6[C6 * j + 1] + Ptau1_C6[C6 * j + 2];
        Ptau1_C6_ex[C6_ex * j + 1] = Ptau1_C6[C6 * j + 3] + Ptau1_C6[C6 * j + 4];
        Ptau1_C6_ex[C6_ex * j + 2] = Ptau1_C6[C6 * j + 5];
        Ptau1_C6_ex[C6_ex * j + 3] = Ptau1_C6_ex[C6_ex * j + 0] * exptaugap[0]
                                   + Ptau1_C6_ex[C6_ex * j + 1]
                                   + Ptau1_C6_ex[C6_ex * j + 2] * exptaugap[1];
    }

    for (GtreeType = 1; GtreeType <= NGTREETYPES2SEQ; GtreeType++) {
        if (GtreeType == 1)
            yup = 2 * tau1 / (2 * tau1 + theta123);
        else if (GtreeType == 2)
            yup = 2 * (tau0 - tau1) / (2 * (tau0 - tau1) + theta5W);
        else if (GtreeType == 3)
            yup = 1;

        coeff = yup * 0.5;

        for (igrid = 0; igrid < K; igrid++) {
            /* compute t (y) on the grid */
            if(igrid<K/2) { ixw = K/2-1-igrid; s=-1; }
            else          { ixw = igrid-K/2;   s= 1; }
            y = yup*(1 + s*xI[ixw])/2;
            t = y/(1-y);

            if (GtreeType == 1)
                computePMatrix(theta123 * t / 2, Q_C6, C6, Pt_C6, work);
            else if (GtreeType == 2) {
                expt[0] = exp(-theta5W * t / theta5);
                expt[1] = exp(-theta5W * t / thetaW);
            }
            else if (GtreeType == 3)
                expt[2] = exp(-t);

            for (i = 0; i < NINITIALSTATES2SEQ; i++) {
                if (!data.iStateCnt[stree.sptree][initStatesRaw2seq[i]])
                    continue;

                initState = initStates2seq[i];
                segm = NGTREETYPES2SEQ * i + GtreeType - 1;

                com.wwprior[index2seq][segm * K + igrid] = wI[ixw] / ((1 - y) * (1 - y)) * coeff;

                if (GtreeType == 1) {
                    com.wwprior[index2seq][segm * K + igrid] *= theta123 * (
                          Pt_C6[C6 * initState + 0] / theta1
                        + Pt_C6[C6 * initState + 2] / theta2
                        + Pt_C6[C6 * initState + 5] / theta3);
                }
                else if (GtreeType == 2) {
                    com.wwprior[index2seq][segm * K + igrid] *= theta5W * (
                          Ptau1_C6_ex[C6_ex * initState + 0] * expt[0] / theta5
                        + Ptau1_C6_ex[C6_ex * initState + 2] * expt[1] / thetaW);
                }
                else if (GtreeType == 3)
                    com.wwprior[index2seq][segm * K + igrid] *= Ptau1_C6_ex[C6_ex * initState + 3] * expt[2];

                if (GtreeType == 1)
                    com.bp0124[index2seq][segm * K + igrid] = theta123 * t / 2;
                else if (GtreeType == 2)
                    com.bp0124[index2seq][segm * K + igrid] = theta5W * t / 2 + tau1;
                else if (GtreeType == 3)
                    com.bp0124[index2seq][segm * K + igrid] = theta4 * t / 2 + tau0;
            }
        }
    }

    com.space = Q_C6;
}

void priorM2ProMax_2seq() {
    double theta4, theta5, theta1, theta2, theta3, thetaW, theta123, theta5W, tau0, tau1, M12, M21, M13, M31, M23, M32, M53, M35, coeff;
    int i, j, igrid, segm, ixw, K = com.npoints, index2seq = MAXGTREES, initState, GtreeType;
    const double *xI = NULL, *wI = NULL;
    double /*a, b, c, m,*/ s, t, y, yup;
    double *Q_C6, *Q_C8, *Pt_C6, *Pt_C8, *Ptau1_C6, *Ptau1_C6_ex, *Ptaugap_C8, *Ptaugap_C8_ex, *work;
    double expt;
    enum { NGTREETYPES2SEQ = 3 };
    STATIC_ASSERT(MAXGTREETYPES2SEQ >= NGTREETYPES2SEQ);

    Q_C6 = com.space;
    if (!Q_C6) error2("oom allocating Q_C6 in priorM2ProMax_2seq");
    Q_C8 = &Q_C6[C6 * C6];
    Pt_C6 = &Q_C8[C8 * C8];
    Pt_C8 = &Pt_C6[C6 * C6];
    Ptau1_C6 = &Pt_C8[C8 * C8];
    Ptau1_C6_ex = &Ptau1_C6[C6 * C6];
    Ptaugap_C8 = &Ptau1_C6_ex[C6 * C6_ex];
    Ptaugap_C8_ex = &Ptaugap_C8[C8 * C8];
    work = &Ptaugap_C8_ex[C8 * C8_ex];
    com.space = work + 3 * square(max2(C6, C8));

    getParams(&theta4, &theta5, &tau0, &tau1, &theta1, &theta2, &theta3);

    thetaW = para[7];
    M12 = para[8];
    M21 = para[9];
    M13 = para[10];
    M31 = para[11];
    M23 = para[12];
    M32 = para[13];
    M53 = para[14];
    M35 = para[15];

    theta5W = theta5 + thetaW;
    theta123 = theta1 + theta2 + theta3;

    GaussLegendreRule(&xI, &wI, com.npoints);

    GenerateQ6(Q_C6, theta1, theta2, theta3, M12, M21, M13, M31, M23, M32);
    GenerateQ8(Q_C8, theta5, thetaW, M53, M35);

    computePMatrix(tau1, Q_C6, C6, Ptau1_C6, work);
    computePMatrix(tau0 - tau1, Q_C8, C8, Ptaugap_C8, work);

    for (i = 0; i < C8 - 1; i++)
        Ptaugap_C8_ex[C8_ex * i] = 1 - Ptaugap_C8[C8 * i + C8 - 1];

    for (i = 0; i < NINITIALSTATES2SEQ; i++) {
        j = initStates2seq[i];
        Ptau1_C6_ex[C6_ex * j + 0] = Ptau1_C6[C6 * j + 0] + Ptau1_C6[C6 * j + 1] + Ptau1_C6[C6 * j + 2];
        Ptau1_C6_ex[C6_ex * j + 1] = Ptau1_C6[C6 * j + 3] + Ptau1_C6[C6 * j + 4];
        Ptau1_C6_ex[C6_ex * j + 2] = Ptau1_C6[C6 * j + 5];
        Ptau1_C6_ex[C6_ex * j + 3] = Ptau1_C6_ex[C6_ex * j + 0] * Ptaugap_C8_ex[C8_ex * 0]
                                   + Ptau1_C6_ex[C6_ex * j + 1] * Ptaugap_C8_ex[C8_ex * 1]
                                   + Ptau1_C6_ex[C6_ex * j + 2] * Ptaugap_C8_ex[C8_ex * 2];
    }

    for (GtreeType = 1; GtreeType <= NGTREETYPES2SEQ; GtreeType++) {
        if (GtreeType == 1)
            yup = 2 * tau1 / (2 * tau1 + theta123);
        else if (GtreeType == 2)
            yup = 2 * (tau0 - tau1) / (2 * (tau0 - tau1) + theta5W);
        else if (GtreeType == 3)
            yup = 1;

        coeff = yup * 0.5;

        for (igrid = 0; igrid < K; igrid++) {
            /* compute t (y) on the grid */
            if(igrid<K/2) { ixw = K/2-1-igrid; s=-1; }
            else          { ixw = igrid-K/2;   s= 1; }
            y = yup*(1 + s*xI[ixw])/2;
            t = y/(1-y);

            if (GtreeType == 1)
                computePMatrix(theta123 * t / 2, Q_C6, C6, Pt_C6, work);
            else if (GtreeType == 2)
                computePMatrix(theta5W * t / 2, Q_C8, C8, Pt_C8, work);
            else if (GtreeType == 3)
                expt = exp(-t);

            for (i = 0; i < NINITIALSTATES2SEQ; i++) {
                if (!data.iStateCnt[stree.sptree][initStatesRaw2seq[i]])
                    continue;

                initState = initStates2seq[i];
                segm = NGTREETYPES2SEQ * i + GtreeType - 1;

                com.wwprior[index2seq][segm * K + igrid] = wI[ixw] / ((1 - y) * (1 - y)) * coeff;

                if (GtreeType == 1) {
                    com.wwprior[index2seq][segm * K + igrid] *= theta123 * (
                          Pt_C6[C6 * initState + 0] / theta1
                        + Pt_C6[C6 * initState + 2] / theta2
                        + Pt_C6[C6 * initState + 5] / theta3);
                }
                else if (GtreeType == 2) {
                    com.wwprior[index2seq][segm * K + igrid] *= theta5W * ((
                          Ptau1_C6_ex[C6_ex * initState + 0] * Pt_C8[C8 * 0 + 0]
                        + Ptau1_C6_ex[C6_ex * initState + 1] * Pt_C8[C8 * 1 + 0]
                        + Ptau1_C6_ex[C6_ex * initState + 2] * Pt_C8[C8 * 2 + 0]) / theta5 + (
                          Ptau1_C6_ex[C6_ex * initState + 0] * Pt_C8[C8 * 0 + 2]
                        + Ptau1_C6_ex[C6_ex * initState + 1] * Pt_C8[C8 * 1 + 2]
                        + Ptau1_C6_ex[C6_ex * initState + 2] * Pt_C8[C8 * 2 + 2]) / thetaW);
                }
                else if (GtreeType == 3)
                    com.wwprior[index2seq][segm * K + igrid] *= Ptau1_C6_ex[C6_ex * initState + 3] * expt;

                if (GtreeType == 1)
                    com.bp0124[index2seq][segm * K + igrid] = theta123 * t / 2;
                else if (GtreeType == 2)
                    com.bp0124[index2seq][segm * K + igrid] = theta5W * t / 2 + tau1;
                else if (GtreeType == 3)
                    com.bp0124[index2seq][segm * K + igrid] = theta4 * t / 2 + tau0;
            }
        }
    }

    com.space = Q_C6;
}

void priorM3MSci12_2seq() {
    double theta4, theta5, theta1, theta2, theta3, tau0, tau1, T, thetaX, thetaY, thetaXY, phi12, phi21, coeff;
    int /*j,*/ igrid, segm, ixw, K=com.npoints, index2seq = MAXGTREES;
    const double *xI=NULL, *wI=NULL;
    double /*b,*/ s, t, y, yup;
    double expT1, expT2, exptau0, exptaugap, exp0, exp1, Pr[3];
    enum { NSEGM = 14 };
    STATIC_ASSERT(MAXGTREES2SEQ >= NSEGM);

    getParams(&theta4, &theta5, &tau0, &tau1, &theta1, &theta2, &theta3);

    T = para[7];
    thetaX = para[8];
    thetaY = para[9];
    phi12 = para[10];
    phi21 = para[11];

    thetaXY = thetaX + thetaY;
    expT1 = exp(-2 * T / theta1);
    expT2 = exp(-2 * T / theta2);
    exptau0 = exp(-2 * tau0 / theta3);
    exptaugap = exp(-2 * (tau0 - tau1) / theta5);

    exp0 = exp(-2 * T / theta1 - 2 * (tau1 - T) / thetaX);
    exp1 = exp(-2 * T / theta1 - 2 * (tau1 - T) / thetaY);
    Pr[0] = (1 - phi21) * (1 - phi21) * exp0 + 2 * (1 - phi21) * phi21 * expT1 + phi21 * phi21 * exp1;
    exp0 = exp(-2 * T / theta2 - 2 * (tau1 - T) / thetaY);
    exp1 = exp(-2 * T / theta2 - 2 * (tau1 - T) / thetaX);
    Pr[1] = (1 - phi12) * (1 - phi12) * exp0 + 2 * (1 - phi12) * phi12 * expT2 + phi12 * phi12 * exp1;
    exp0 = exp(-2 * (tau1 - T) / thetaX);
    exp1 = exp(-2 * (tau1 - T) / thetaY);
    Pr[2] = (1 - phi21) * phi12 * exp0 + (1 - phi21) * (1 - phi12) + phi21 * phi12 + phi21 * (1 - phi12) * exp1;

    GaussLegendreRule(&xI, &wI, com.npoints);

    for(segm = 0; segm < NSEGM; segm++) {
        if (segm == 0) {
            yup = 2 * T / (2 * T + theta1);
        } else if (segm == 4) {
            yup = 2 * T / (2 * T + theta2);
        } else if (segm == 12) {
            yup = 2 * tau0 / (2 * tau0 + theta3);
        } else if (segm == 1 || segm == 5 || segm == 8) {
            yup = 2 * (tau0 - tau1) / (2 * (tau0 - tau1) + theta5);
        } else if (segm == 3 || segm == 7 || segm == 10) {
            yup = 2 * (tau1 - T) / (2 * (tau1 - T) + thetaXY);
        } else { // 2, 6, 9, 11, 13
            yup = 1;
        }

        coeff = yup * 0.5;

        for(igrid=0; igrid<K; igrid++) {
            /* compute t (y) on the grid */
            if(igrid<K/2) { ixw = K/2-1-igrid; s=-1; }
            else          { ixw = igrid-K/2;   s= 1; }
            y = yup*(1 + s*xI[ixw])/2;
            t = y/(1-y);

            com.wwprior[index2seq][segm * K + igrid] = wI[ixw] / ((1 - y) * (1 - y)) * coeff;

            if (segm == 1) {
                com.wwprior[index2seq][segm * K + igrid] *= Pr[0] * exp(-t);
            } else if (segm == 2) {
                com.wwprior[index2seq][segm * K + igrid] *= Pr[0] * exptaugap * exp(-t);
            } else if (segm == 3) {
                exp0 = exp(-2 * T / theta1 - thetaXY * t / thetaX);
                exp1 = exp(-2 * T / theta1 - thetaXY * t / thetaY);
                com.wwprior[index2seq][segm * K + igrid] *= thetaXY * ((1 - phi21) * (1 - phi21) * exp0 / thetaX + phi21 * phi21 * exp1 / thetaY);
            } else if (segm == 5) {
                com.wwprior[index2seq][segm * K + igrid] *= Pr[1] * exp(-t);
            } else if (segm == 6) {
                com.wwprior[index2seq][segm * K + igrid] *= Pr[1] * exptaugap * exp(-t);
            } else if (segm == 7) {
                exp0 = exp(-2 * T / theta2 - thetaXY * t / thetaY);
                exp1 = exp(-2 * T / theta2 - thetaXY * t / thetaX);
                com.wwprior[index2seq][segm * K + igrid] *= thetaXY * ((1 - phi12) * (1 - phi12) * exp0 / thetaY + phi12 * phi12 * exp1 / thetaX);
            } else if (segm == 8) {
                com.wwprior[index2seq][segm * K + igrid] *= Pr[2] * exp(-t);
            } else if (segm == 9) {
                com.wwprior[index2seq][segm * K + igrid] *= Pr[2] * exptaugap * exp(-t);
            } else if (segm == 10) {
                exp0 = exp(-thetaXY * t / thetaX);
                exp1 = exp(-thetaXY * t / thetaY);
                com.wwprior[index2seq][segm * K + igrid] *= thetaXY * ((1 - phi21) * phi12 * exp0 / thetaX + phi21 * (1 - phi12) * exp1 / thetaY);
            } else if (segm == 13) {
                com.wwprior[index2seq][segm * K + igrid] *= exptau0 * exp(-t);
            } else { // 0, 4, 11, 12
                com.wwprior[index2seq][segm * K + igrid] *= exp(-t);
            }

            if (segm == 0) {
                com.bp0124[index2seq][segm * K + igrid] = theta1 * t / 2;
            } else if (segm == 4) {
                com.bp0124[index2seq][segm * K + igrid] = theta2 * t / 2;
            } else if (segm == 12) {
                com.bp0124[index2seq][segm * K + igrid] = theta3 * t / 2;
            } else if (segm == 1 || segm == 5 || segm == 8) {
                com.bp0124[index2seq][segm * K + igrid] = theta5 * t / 2 + tau1;
            } else if (segm == 3 || segm == 7 || segm == 10) {
                com.bp0124[index2seq][segm * K + igrid] = thetaXY * t / 2 + T;
            } else { // 2, 6, 9, 11, 13
                com.bp0124[index2seq][segm * K + igrid] = theta4 * t / 2 + tau0;
            }
        }
    }
}

void priorM3MSci13_2seq() {
    double theta4, theta5, theta1, theta2, theta3, tau0, tau1, T, thetaX, thetaZ, theta5Z, thetaXZ, phi13, phi31, coeff;
    int /*j,*/ igrid, segm, ixw, K=com.npoints, index2seq = MAXGTREES;
    const double *xI=NULL, *wI=NULL;
    double /*b,*/ s, t, y, yup;
    double expT1, expT3, exptau1, exptaugap, exp0, exp1, Pr[3];
    enum { NSEGM = 18 };
    STATIC_ASSERT(MAXGTREES2SEQ >= NSEGM);

    getParams(&theta4, &theta5, &tau0, &tau1, &theta1, &theta2, &theta3);

    T = para[7];
    thetaX = para[8];
    thetaZ = para[9];
    phi13 = para[10];
    phi31 = para[11];

    theta5Z = theta5 + thetaZ;
    thetaXZ = thetaX + thetaZ;
    expT1 = exp(-2 * T / theta1);
    expT3 = exp(-2 * T / theta3);
    exptau1 = exp(-2 * tau1 / theta2);
    exptaugap = exp(-2 * (tau0 - tau1) / theta5);

    exp0 = exp(-2 * T / theta1 - 2 * (tau1 - T) / thetaX - 2 * (tau0 - tau1) / theta5);
    exp1 = exp(-2 * T / theta1 - 2 * (tau0 - T) / thetaZ);
    Pr[0] = (1 - phi31) * (1 - phi31) * exp0 + 2 * (1 - phi31) * phi31 * expT1 + phi31 * phi31 * exp1;
    exp0 = exp(-2 * (tau1 - T) / thetaX - 2 * (tau0 - tau1) / theta5);
    exp1 = exp(-2 * (tau0 - T) / thetaZ);
    Pr[1] = (1 - phi31) * phi13 * exp0 + (1 - phi31) * (1 - phi13) + phi31 * phi13 + phi31 * (1 - phi13) * exp1;
    exp0 = exp(-2 * T / theta3 - 2 * (tau1 - T) / thetaX - 2 * (tau0 - tau1) / theta5);
    exp1 = exp(-2 * T / theta3 - 2 * (tau0 - T) / thetaZ);
    Pr[2] = phi13 * phi13 * exp0 + 2 * phi13 * (1 - phi13) * expT3 + (1 - phi13) * (1 - phi13) * exp1;

    GaussLegendreRule(&xI, &wI, com.npoints);

    for(segm = 0; segm < NSEGM; segm++) {
        if (segm == 0) {
            yup = 2 * T / (2 * T + theta1);
        } else if (segm == 4) {
            yup = 2 * tau1 / (2 * tau1 + theta2);
        } else if (segm == 14) {
            yup = 2 * T / (2 * T + theta3);
        } else if (segm == 1 || segm == 9 || segm == 15) {
            yup = 2 * (tau0 - tau1) / (2 * (tau0 - tau1) + theta5Z);
        } else if (segm == 5 || segm == 7 || segm == 12) {
            yup = 2 * (tau0 - tau1) / (2 * (tau0 - tau1) + theta5);
        } else if (segm == 3 || segm == 11 || segm == 17) {
            yup = 2 * (tau1 - T) / (2 * (tau1 - T) + thetaXZ);
        } else { // 2, 6, 8, 10, 13, 16
            yup = 1;
        }

        coeff = yup * 0.5;

        for(igrid=0; igrid<K; igrid++) {
            /* compute t (y) on the grid */
            if(igrid<K/2) { ixw = K/2-1-igrid; s=-1; }
            else          { ixw = igrid-K/2;   s= 1; }
            y = yup*(1 + s*xI[ixw])/2;
            t = y/(1-y);

            com.wwprior[index2seq][segm * K + igrid] = wI[ixw] / ((1 - y) * (1 - y)) * coeff;

            if (segm == 1) {
                exp0 = exp(-2 * T / theta1 - 2 * (tau1 - T) / thetaX - theta5Z * t / theta5);
                exp1 = exp(-2 * T / theta1 - (2 * (tau1 - T) + theta5Z * t) / thetaZ);
                com.wwprior[index2seq][segm * K + igrid] *= theta5Z * ((1 - phi31) * (1 - phi31) * exp0 / theta5 + phi31 * phi31 * exp1 / thetaZ);
            } else if (segm == 2) {
                com.wwprior[index2seq][segm * K + igrid] *= Pr[0] * exp(-t);
            } else if (segm == 3) {
                exp0 = exp(-2 * T / theta1 - thetaXZ * t / thetaX);
                exp1 = exp(-2 * T / theta1 - thetaXZ * t / thetaZ);
                com.wwprior[index2seq][segm * K + igrid] *= thetaXZ * ((1 - phi31) * (1 - phi31) * exp0 / thetaX + phi31 * phi31 * exp1 / thetaZ);
            } else if (segm == 5) {
                com.wwprior[index2seq][segm * K + igrid] *= exptau1 * exp(-t);
            } else if (segm == 6) {
                com.wwprior[index2seq][segm * K + igrid] *= exptau1 * exptaugap * exp(-t);
            } else if (segm == 7) {
                com.wwprior[index2seq][segm * K + igrid] *= (1 - phi31) * exp(-t);
            } else if (segm == 8) {
                com.wwprior[index2seq][segm * K + igrid] *= ((1 - phi31) * exptaugap + phi31) * exp(-t);
            } else if (segm == 9) {
                exp0 = exp(-2 * (tau1 - T) / thetaX - theta5Z * t / theta5);
                exp1 = exp(-(2 * (tau1 - T) + theta5Z * t) / thetaZ);
                com.wwprior[index2seq][segm * K + igrid] *= theta5Z * ((1 - phi31) * phi13 * exp0 / theta5 + phi31 * (1 - phi13) * exp1 / thetaZ);
            } else if (segm == 10) {
                com.wwprior[index2seq][segm * K + igrid] *= Pr[1] * exp(-t);
            } else if (segm == 11) {
                exp0 = exp(-thetaXZ * t / thetaX);
                exp1 = exp(-thetaXZ * t / thetaZ);
                com.wwprior[index2seq][segm * K + igrid] *= thetaXZ * ((1 - phi31) * phi13 * exp0 / thetaX + phi31 * (1 - phi13) * exp1 / thetaZ);
            } else if (segm == 12) {
                com.wwprior[index2seq][segm * K + igrid] *= phi13 * exp(-t);
            } else if (segm == 13) {
                com.wwprior[index2seq][segm * K + igrid] *= (phi13 * exptaugap + (1 - phi13)) * exp(-t);
            } else if (segm == 15) {
                exp0 = exp(-2 * T / theta3 - 2 * (tau1 - T) / thetaX - theta5Z * t / theta5);
                exp1 = exp(-2 * T / theta3 - (2 * (tau1 - T) + theta5Z * t) / thetaZ);
                com.wwprior[index2seq][segm * K + igrid] *= theta5Z * (phi13 * phi13 * exp0 / theta5 + (1 - phi13) * (1 - phi13) * exp1 / thetaZ);
            } else if (segm == 16) {
                com.wwprior[index2seq][segm * K + igrid] *= Pr[2] * exp(-t);
            } else if (segm == 17) {
                exp0 = exp(-2 * T / theta3 - thetaXZ * t / thetaX);
                exp1 = exp(-2 * T / theta3 - thetaXZ * t / thetaZ);
                com.wwprior[index2seq][segm * K + igrid] *= thetaXZ * (phi13 * phi13 * exp0 / thetaX + (1 - phi13) * (1 - phi13) * exp1 / thetaZ);
            } else { // 0, 4, 14
                com.wwprior[index2seq][segm * K + igrid] *= exp(-t);
            }

            if (segm == 0) {
                com.bp0124[index2seq][segm * K + igrid] = theta1 * t / 2;
            } else if (segm == 4) {
                com.bp0124[index2seq][segm * K + igrid] = theta2 * t / 2;
            } else if (segm == 14) {
                com.bp0124[index2seq][segm * K + igrid] = theta3 * t / 2;
            } else if (segm == 1 || segm == 9 || segm == 15) {
                com.bp0124[index2seq][segm * K + igrid] = theta5Z * t / 2 + tau1;
            } else if (segm == 5 || segm == 7 || segm == 12) {
                com.bp0124[index2seq][segm * K + igrid] = theta5 * t / 2 + tau1;
            } else if (segm == 3 || segm == 11 || segm == 17) {
                com.bp0124[index2seq][segm * K + igrid] = thetaXZ * t / 2 + T;
            } else { // 2, 6, 8, 10, 13, 16
                com.bp0124[index2seq][segm * K + igrid] = theta4 * t / 2 + tau0;
            }
        }
    }
}

void priorM3MSci23_2seq() {
    double theta4, theta5, theta1, theta2, theta3, tau0, tau1, T, thetaY, thetaZ, theta5Z, thetaYZ, phi23, phi32, coeff;
    int /*j,*/ igrid, segm, ixw, K=com.npoints, index2seq = MAXGTREES;
    const double *xI=NULL, *wI=NULL;
    double /*b,*/ s, t, y, yup;
    double expT2, expT3, exptau1, exptaugap, exp0, exp1, Pr[3];
    enum { NSEGM = 18 };
    STATIC_ASSERT(MAXGTREES2SEQ >= NSEGM);

    getParams(&theta4, &theta5, &tau0, &tau1, &theta1, &theta2, &theta3);

    T = para[7];
    thetaY = para[8];
    thetaZ = para[9];
    phi23 = para[10];
    phi32 = para[11];

    theta5Z = theta5 + thetaZ;
    thetaYZ = thetaY + thetaZ;
    expT2 = exp(-2 * T / theta2);
    expT3 = exp(-2 * T / theta3);
    exptau1 = exp(-2 * tau1 / theta1);
    exptaugap = exp(-2 * (tau0 - tau1) / theta5);

    exp0 = exp(-2 * T / theta2 - 2 * (tau1 - T) / thetaY - 2 * (tau0 - tau1) / theta5);
    exp1 = exp(-2 * T / theta2 - 2 * (tau0 - T) / thetaZ);
    Pr[0] = (1 - phi32) * (1 - phi32) * exp0 + 2 * (1 - phi32) * phi32 * expT2 + phi32 * phi32 * exp1;
    exp0 = exp(-2 * (tau1 - T) / thetaY - 2 * (tau0 - tau1) / theta5);
    exp1 = exp(-2 * (tau0 - T) / thetaZ);
    Pr[1] = (1 - phi32) * phi23 * exp0 + (1 - phi32) * (1 - phi23) + phi32 * phi23 + phi32 * (1 - phi23) * exp1;
    exp0 = exp(-2 * T / theta3 - 2 * (tau1 - T) / thetaY - 2 * (tau0 - tau1) / theta5);
    exp1 = exp(-2 * T / theta3 - 2 * (tau0 - T) / thetaZ);
    Pr[2] = phi23 * phi23 * exp0 + 2 * phi23 * (1 - phi23) * expT3 + (1 - phi23) * (1 - phi23) * exp1;

    GaussLegendreRule(&xI, &wI, com.npoints);

    for(segm = 0; segm < NSEGM; segm++) {
        if (segm == 3) {
            yup = 2 * T / (2 * T + theta2);
        } else if (segm == 0) {
            yup = 2 * tau1 / (2 * tau1 + theta1);
        } else if (segm == 14) {
            yup = 2 * T / (2 * T + theta3);
        } else if (segm == 4 || segm == 11 || segm == 15) {
            yup = 2 * (tau0 - tau1) / (2 * (tau0 - tau1) + theta5Z);
        } else if (segm == 1 || segm == 7 || segm == 9) {
            yup = 2 * (tau0 - tau1) / (2 * (tau0 - tau1) + theta5);
        } else if (segm == 6 || segm == 13 || segm == 17) {
            yup = 2 * (tau1 - T) / (2 * (tau1 - T) + thetaYZ);
        } else { // 5, 2, 8, 12, 10, 16
            yup = 1;
        }

        coeff = yup * 0.5;

        for(igrid=0; igrid<K; igrid++) {
            /* compute t (y) on the grid */
            if(igrid<K/2) { ixw = K/2-1-igrid; s=-1; }
            else          { ixw = igrid-K/2;   s= 1; }
            y = yup*(1 + s*xI[ixw])/2;
            t = y/(1-y);

            com.wwprior[index2seq][segm * K + igrid] = wI[ixw] / ((1 - y) * (1 - y)) * coeff;

            if (segm == 4) {
                exp0 = exp(-2 * T / theta2 - 2 * (tau1 - T) / thetaY - theta5Z * t / theta5);
                exp1 = exp(-2 * T / theta2 - (2 * (tau1 - T) + theta5Z * t) / thetaZ);
                com.wwprior[index2seq][segm * K + igrid] *= theta5Z * ((1 - phi32) * (1 - phi32) * exp0 / theta5 + phi32 * phi32 * exp1 / thetaZ);
            } else if (segm == 5) {
                com.wwprior[index2seq][segm * K + igrid] *= Pr[0] * exp(-t);
            } else if (segm == 6) {
                exp0 = exp(-2 * T / theta2 - thetaYZ * t / thetaY);
                exp1 = exp(-2 * T / theta2 - thetaYZ * t / thetaZ);
                com.wwprior[index2seq][segm * K + igrid] *= thetaYZ * ((1 - phi32) * (1 - phi32) * exp0 / thetaY + phi32 * phi32 * exp1 / thetaZ);
            } else if (segm == 1) {
                com.wwprior[index2seq][segm * K + igrid] *= exptau1 * exp(-t);
            } else if (segm == 2) {
                com.wwprior[index2seq][segm * K + igrid] *= exptau1 * exptaugap * exp(-t);
            } else if (segm == 7) {
                com.wwprior[index2seq][segm * K + igrid] *= (1 - phi32) * exp(-t);
            } else if (segm == 8) {
                com.wwprior[index2seq][segm * K + igrid] *= ((1 - phi32) * exptaugap + phi32) * exp(-t);
            } else if (segm == 11) {
                exp0 = exp(-2 * (tau1 - T) / thetaY - theta5Z * t / theta5);
                exp1 = exp(-(2 * (tau1 - T) + theta5Z * t) / thetaZ);
                com.wwprior[index2seq][segm * K + igrid] *= theta5Z * ((1 - phi32) * phi23 * exp0 / theta5 + phi32 * (1 - phi23) * exp1 / thetaZ);
            } else if (segm == 12) {
                com.wwprior[index2seq][segm * K + igrid] *= Pr[1] * exp(-t);
            } else if (segm == 13) {
                exp0 = exp(-thetaYZ * t / thetaY);
                exp1 = exp(-thetaYZ * t / thetaZ);
                com.wwprior[index2seq][segm * K + igrid] *= thetaYZ * ((1 - phi32) * phi23 * exp0 / thetaY + phi32 * (1 - phi23) * exp1 / thetaZ);
            } else if (segm == 9) {
                com.wwprior[index2seq][segm * K + igrid] *= phi23 * exp(-t);
            } else if (segm == 10) {
                com.wwprior[index2seq][segm * K + igrid] *= (phi23 * exptaugap + (1 - phi23)) * exp(-t);
            } else if (segm == 15) {
                exp0 = exp(-2 * T / theta3 - 2 * (tau1 - T) / thetaY - theta5Z * t / theta5);
                exp1 = exp(-2 * T / theta3 - (2 * (tau1 - T) + theta5Z * t) / thetaZ);
                com.wwprior[index2seq][segm * K + igrid] *= theta5Z * (phi23 * phi23 * exp0 / theta5 + (1 - phi23) * (1 - phi23) * exp1 / thetaZ);
            } else if (segm == 16) {
                com.wwprior[index2seq][segm * K + igrid] *= Pr[2] * exp(-t);
            } else if (segm == 17) {
                exp0 = exp(-2 * T / theta3 - thetaYZ * t / thetaY);
                exp1 = exp(-2 * T / theta3 - thetaYZ * t / thetaZ);
                com.wwprior[index2seq][segm * K + igrid] *= thetaYZ * (phi23 * phi23 * exp0 / thetaY + (1 - phi23) * (1 - phi23) * exp1 / thetaZ);
            } else { // 3, 0, 14
                com.wwprior[index2seq][segm * K + igrid] *= exp(-t);
            }

            if (segm == 3) {
                com.bp0124[index2seq][segm * K + igrid] = theta2 * t / 2;
            } else if (segm == 0) {
                com.bp0124[index2seq][segm * K + igrid] = theta1 * t / 2;
            } else if (segm == 14) {
                com.bp0124[index2seq][segm * K + igrid] = theta3 * t / 2;
            } else if (segm == 4 || segm == 11 || segm == 15) {
                com.bp0124[index2seq][segm * K + igrid] = theta5Z * t / 2 + tau1;
            } else if (segm == 1 || segm == 7 || segm == 9) {
                com.bp0124[index2seq][segm * K + igrid] = theta5 * t / 2 + tau1;
            } else if (segm == 6 || segm == 13 || segm == 17) {
                com.bp0124[index2seq][segm * K + igrid] = thetaYZ * t / 2 + T;
            } else { // 5, 2, 8, 12, 10, 16
                com.bp0124[index2seq][segm * K + igrid] = theta4 * t / 2 + tau0;
            }
        }
    }
}

double lnpD_locus (int locus)
{
    int *n = data.Nij[stree.sptree] + locus*5, K=com.npoints, /*n123max,*/ igrid, k, Gtree, /*y,*/ initState = data.initState[stree.sptree][locus], error=0, i, /*np,*/ index2seq = MAXGTREES; //, mult = 1;
    int /*chain=data.chain[locus], is=initStateMap[initState],*/ nGtree, offset;
    const int * GtOffset;
    double lnL=0, lmax=data.lnLmax[stree.sptree][locus], pD=0, pGk[MAXGTREETYPES*3] = {0}, f[3] = {0, 0, 0}, p[5], /*p12,*/ b[2], lp1, lp2, /*sump12,*/ tmp, **bp0124, **wwprior; //, curpD[3];
    char /*** GtStr,*/ errorStr[130];
    //struct BTEntry * gtt = GtreeTab[0];
    BTEntry * gtt = com.GtreeTab[0];
    /* lmax can be dynamically adjusted. */

#ifdef DEBUG_GTREE_PROB
    if(LASTROUND == 2) { n[0] = n[1] = n[2] = n[3] = n[4] = 0; lmax = 0; }
#endif
    
#ifdef DEBUG1

    if(locus== DBGLOCUS) {
        printf("lnLmax: %12.8f\n", lmax);
    }
#endif
    if (initState >= NSTATES) { // 2 sequences
        if (com.model == M0 || com.model == M1DiscreteBeta) {
            switch (initState) {
                case 27:    nGtree = 3;
                    offset = 0;
                    break;
                case 31:    nGtree = 3;
                    offset = 3;
                    break;
                case 28:    nGtree = 2;
                    offset = 6;
                    break;
                case 29:
                case 32:    nGtree = 1;
                    offset = 8;
                    break;
                case 35:    nGtree = 2;
                    offset = 9;
                    break;
                default:    nGtree = 0;
                    error2("unexpected initial state!");
            }
        }
        else if (com.model == M2SIM3s) {
            switch (initState) {
                case 27:    nGtree = 3;
                    offset = 0;
                    break;
                case 31:    nGtree = 3;
                    offset = 3;
                    break;
                case 28:    nGtree = 3;
                    offset = 6;
                    break;
                case 29:
                case 32:    nGtree = 1;
                    offset = 9;
                    break;
                case 35:    nGtree = 2;
                    offset = 10;
                    break;
                default:    nGtree = 0;
                    error2("unexpected initial state!");
            }
        }
        else if (com.model == M2Pro || com.model == M2ProMax) {
            switch (initState) {
                case 27:    nGtree = 3;
                    offset = 0;
                    break;
                case 31:    nGtree = 3;
                    offset = 3;
                    break;
                case 28:    nGtree = 3;
                    offset = 6;
                    break;
                case 29:    nGtree = 3;
                    offset = 9;
                    break;
                case 32:    nGtree = 3;
                    offset = 12;
                    break;
                case 35:    nGtree = 3;
                    offset = 15;
                    break;
                default:    nGtree = 0;
                    error2("unexpected initial state!");
            }
        }
        else if (com.model == M3MSci12) {
            switch (initState) {
                case 27:    nGtree = 4;
                    offset = 0;
                    break;
                case 31:    nGtree = 4;
                    offset = 4;
                    break;
                case 28:    nGtree = 3;
                    offset = 8;
                    break;
                case 29:
                case 32:    nGtree = 1;
                    offset = 11;
                    break;
                case 35:    nGtree = 2;
                    offset = 12;
                    break;
                default:    nGtree = 0;
                    error2("unexpected initial state!");
            }
        }
        else if (com.model == M3MSci13) {
            switch (initState) {
                case 27:    nGtree = 4;
                    offset = 0;
                    break;
                case 31:    nGtree = 3;
                    offset = 4;
                    break;
                case 28:    nGtree = 2;
                    offset = 7;
                    break;
                case 29:    nGtree = 3;
                    offset = 9;
                    break;
                case 32:    nGtree = 2;
                    offset = 12;
                    break;
                case 35:    nGtree = 4;
                    offset = 14;
                    break;
                default:    nGtree = 0;
                    error2("unexpected initial state!");
            }
        }
        else if (com.model == M3MSci23) {
            switch (initState) {
                case 27:    nGtree = 3;
                    offset = 0;
                    break;
                case 31:    nGtree = 4;
                    offset = 3;
                    break;
                case 28:    nGtree = 2;
                    offset = 7;
                    break;
                case 29:    nGtree = 2;
                    offset = 9;
                    break;
                case 32:    nGtree = 3;
                    offset = 11;
                    break;
                case 35:    nGtree = 4;
                    offset = 14;
                    break;
                default:    nGtree = 0;
                    error2("unexpected initial state!");
            }
        }
        else {
            nGtree = 0;
            error2("not implemented");
        }

        for(Gtree=0; Gtree < nGtree; Gtree++) {
            pGk[Gtree] = 0.0;
            for(igrid=0; igrid < K; igrid++) {
                if(com.fix_locusrate) {  /* p0124 is calculated only if locus rates are fixed. */
                    // TODO: check whether this is correct
                    b[0] = com.bp0124[index2seq][(offset+Gtree)*K+igrid] * data.locusrate[locus];
                } else {
                    b[0] = com.bp0124[index2seq][(offset+Gtree)*K+igrid];
                }
                f[0] = -lmax;
                if (n[1]) {
                    f[0] += n[1]*log(3.0/4.0 - 3.0/4.0*exp(-8.0*b[0]/3.0));
                }
                if (n[0]) {
                    f[0] += n[0]*log(1.0/4.0 + 3.0/4.0*exp(-8.0*b[0]/3.0));
                }
                f[0] = (f[0] < -500 ? 0 : exp(f[0])*com.wwprior[index2seq][(offset+Gtree)*K+igrid]);
                pGk[Gtree] += f[0];
                pD += f[0];
            }
        }
    }

    else { // 3 sequences
        if (com.model == M0 || com.model == M1DiscreteBeta) {
            GtOffset = &GtOffsetM0[0];
            switch (initState) {
                case 0:    nGtree = 6;
                    offset = 0;
                    break;
                case 13:   nGtree = 6;
                    offset = 6;
                    break;
                case 1:    nGtree = 5;
                    offset = 12;
                    break;
                case 12:   nGtree = 5;
                    offset = 17;
                    break;
                case 2:    nGtree = 3;
                    offset = 22;
                    break;
                case 14:   nGtree = 3;
                    offset = 25;
                    break;
                case 5:    nGtree = 2;
                    offset = 28;
                    break;
                case 8:
                case 17:   nGtree = 2;
                    offset = 30;
                    break;
                case 26:   nGtree = 3;
                    offset = 32;
                    break;
                default:   nGtree = 0;
                    error2("unexpected initial state!");
            }
        }
        else if (com.model == M2SIM3s) {
            GtOffset = &GtOffsetM2[0];
            switch (initState) {
                case 0:    nGtree = 6;
                    offset = 0;
                    break;
                case 13:   nGtree = 6;
                    offset = 6;
                    break;
                case 1:    nGtree = 6;
                    offset = 12;
                    break;
                case 12:   nGtree = 6;
                    offset = 18;
                    break;
                case 2:    nGtree = 3;
                    offset = 24;
                    break;
                case 14:   nGtree = 3;
                    offset = 27;
                    break;
                case 5:    nGtree = 3;
                    offset = 30;
                    break;
                case 8:
                case 17:   nGtree = 2;
                    offset = 33;
                    break;
                case 26:   nGtree = 3;
                    offset = 35;
                    break;
                default:   nGtree = 0;
                    error2("unexpected initial state!");
                    break;
            }
        }
        else if (com.model == M2Pro || com.model == M2ProMax) {
            GtOffset = (com.model == M2Pro) ? &GtOffsetM2Pro[0] : &GtOffsetM2ProMax[0];
            switch (initState) {
                case 0:    nGtree = 6;
                    offset = 0;
                    break;
                case 13:   nGtree = 6;
                    offset = 6;
                    break;
                case 1:    nGtree = 6;
                    offset = 12;
                    break;
                case 12:   nGtree = 6;
                    offset = 18;
                    break;
                case 2:    nGtree = 6;
                    offset = 24;
                    break;
                case 14:   nGtree = 6;
                    offset = 30;
                    break;
                case 5:    nGtree = 6;
                    offset = 36;
                    break;
                case 8:    nGtree = 6;
                    offset = 42;
                    break;
                case 17:   nGtree = 6;
                    offset = 48;
                    break;
                case 26:   nGtree = 6;
                    offset = 54;
                    break;
                default:   nGtree = 0;
                    error2("unexpected initial state!");
                    break;
            }
        }
        else if (com.model == M3MSci12) {
            GtOffset = &GtOffsetM3MSci12[0];
            switch (initState) {
                case 0:    nGtree = 10;
                    offset = 0;
                    break;
                case 13:   nGtree = 10;
                    offset = 10;
                    break;
                case 1:    nGtree = 9;
                    offset = 20;
                    break;
                case 12:   nGtree = 9;
                    offset = 29;
                    break;
                case 2:    nGtree = 4;
                    offset = 38;
                    break;
                case 14:   nGtree = 4;
                    offset = 42;
                    break;
                case 5:    nGtree = 3;
                    offset = 46;
                    break;
                case 8:
                case 17:   nGtree = 2;
                    offset = 49;
                    break;
                case 26:   nGtree = 3;
                    offset = 51;
                    break;
                default:   nGtree = 0;
                    error2("unexpected initial state!");
                    break;
            }
        }
        else if (com.model == M3MSci13) {
            GtOffset = &GtOffsetM3MSci13[0];
            switch (initState) {
                case 0:    nGtree = 10;
                    offset = 0;
                    break;
                case 13:   nGtree = 6;
                    offset = 10;
                    break;
                case 1:    nGtree = 7;
                    offset = 16;
                    break;
                case 12:   nGtree = 5;
                    offset = 23;
                    break;
                case 2:    nGtree = 9;
                    offset = 28;
                    break;
                case 14:   nGtree = 5;
                    offset = 37;
                    break;
                case 5:    nGtree = 5;
                    offset = 42;
                    break;
                case 8:    nGtree = 9;
                    offset = 47;
                    break;
                case 17:   nGtree = 7;
                    offset = 56;
                    break;
                case 26:   nGtree = 10;
                    offset = 63;
                    break;
                default:   nGtree = 0;
                    error2("unexpected initial state!");
                    break;
            }
        }
        else if (com.model == M3MSci23) {
            GtOffset = &GtOffsetM3MSci23[0];
            switch (initState) {
                case 0:    nGtree = 6;
                    offset = 0;
                    break;
                case 13:   nGtree = 10;
                    offset = 6;
                    break;
                case 1:    nGtree = 5;
                    offset = 16;
                    break;
                case 12:   nGtree = 7;
                    offset = 21;
                    break;
                case 2:    nGtree = 5;
                    offset = 28;
                    break;
                case 14:   nGtree = 9;
                    offset = 33;
                    break;
                case 5:    nGtree = 5;
                    offset = 42;
                    break;
                case 8:    nGtree = 7;
                    offset = 47;
                    break;
                case 17:   nGtree = 9;
                    offset = 54;
                    break;
                case 26:   nGtree = 10;
                    offset = 63;
                    break;
                default:   nGtree = 0;
                    error2("unexpected initial state!");
                    break;
            }
        }
        else {
            nGtree = 0;
            error2("not implemented");
        }

        bp0124 = &com.bp0124[offset];
        wwprior = &com.wwprior[offset];
        //GtStr = &GtreeStr[offset];
        GtOffset = &GtOffset[offset];

        for(Gtree=0; Gtree < nGtree; Gtree++) {
            pGk[Gtree*3] = pGk[Gtree*3+1] = pGk[Gtree*3+2] = 0.0;
            for(igrid=0; igrid < 3*K*K; igrid+=3) {
                if(com.fix_locusrate) {  /* p0124 is calculated only if locus rates are fixed. */
                    b[0] = bp0124[Gtree][(igrid/3)*2+0] * data.locusrate[locus];
                    b[1] = bp0124[Gtree][(igrid/3)*2+1] * data.locusrate[locus];
                    p0124Fromb0b1 (p, b);
                }
                else {
                    for(k=0; k<5; k++) {
                        p[k] = bp0124[Gtree][(igrid/3)*5+k];
                    }
                }
                
#ifdef TESTA1DBG
                printf("%-3s %-6d\t\t%12.8f %12.8f %12.8f %12.8f %12.8f\n", GtStr[Gtree], igrid/3, p[0], p[1], p[2], p[3], p[4]);
                printf("GtOffset: %d  gtt.nGtrees: %d  gtt.config: %d\n", GtOffset[Gtree], gtt[GtOffset[Gtree]].nGtrees, gtt[GtOffset[Gtree]].config);
#endif
                
                f[0] = -lmax;
#ifdef DEBUG1
                if (locus == DBGLOCUS) {
                    printf("f[0]: %8.6f\n",f[0]);
                }
#endif
                if(n[0]) f[0] += n[0]*log(p[0]);
                if(n[4]) f[0] += n[4]*log(p[4]);
#ifdef DEBUG1
                if (locus == DBGLOCUS) {
                    printf("f2: %8.6f\n",f[0]);
                }
#endif
                
                if(gtt[GtOffset[Gtree]].nGtrees == 1) {
                    if (gtt[GtOffset[Gtree]].config == 0) { // topology ((a,b), c)
                        if(n[1]) f[0] += n[1]*log(p[1]);
                        if(n[2]+n[3]) f[0] += (n[2]+n[3])*log(p[2]);
                        
                        f[0] = (f[0] < -500 ? 0 : exp(f[0])*wwprior[Gtree][igrid]);
//                        if(LASTROUND)
//                            printf("Gtree: %d  igrid: %03d  f[0]: %20.18f  prior: %20.18f\n", Gtree, igrid, f[0], wwprior[Gtree][igrid]);
                        pGk[Gtree*3] += f[0];
                        pD += f[0];
                    } else if(gtt[GtOffset[Gtree]].config == 1) { // topology ((a,c),b)
                        if(n[3]) f[0] += n[3]*log(p[1]);
                        if(n[1]+n[2]) f[0] += (n[1]+n[2])*log(p[2]);
                        f[0] = (f[0] < -500 ? 0 : exp(f[0])*wwprior[Gtree][igrid+1]);
                        pGk[Gtree*3+1] += f[0];
                        pD += f[0];
                    } else if(gtt[GtOffset[Gtree]].config == 2) { // topology ((b,c), a)
                        if(n[2]) f[0] += n[2]*log(p[1]);
                        if(n[1]+n[3]) f[0] += (n[1]+n[3])*log(p[2]);
                        f[0] = (f[0] < -500 ? 0 : exp(f[0])*wwprior[Gtree][igrid+2]);
                        pGk[Gtree*3+2] += f[0];
                        pD += f[0];
                    }
                } else { // all topologies
                    // before sorting: f[0] = ((a,b),c); f[1] = ((a,c),b); f[2] = ((b,c),a)
                    f[1] = f[2] = f[0];
                    
                    
                    lp1 = log(p[1]); lp2 = log(p[2]);
                    for (k=0; k<3; k++) {
                        if(n[(2-k+1)%3+1]) f[k] += n[(2-k+1)%3+1]*lp1;
                        if(n[(3-k+1)%3+1]+n[3-k]) f[k] += (n[(3-k+1)%3+1]+n[3-k])*lp2;
                    }
                    for (k=0; k<3; k++) {
                        f[k] = (f[k] < -500 ? 0 : exp(f[k])*wwprior[Gtree][igrid+k]);
                        pGk[Gtree*3+k] += f[k];
                    }
                    
                    //sort f values
                    if (f[0] < f[1]) {
                        if (f[2] < f[0]) {
                            tmp = f[0]; f[0] = f[2]; f[2] = tmp;
                        }
                    } else {
                        if (f[1] < f[2]) {
                            tmp = f[0]; f[0] = f[1]; f[1] = tmp;
                        } else {
                            tmp = f[0]; f[0] = f[2]; f[2] = tmp;
                        }
                    }
                    if (f[2] < f[1]) {
                        tmp = f[1]; f[1] = f[2]; f[2] = tmp;
                    }
                    f[0] += f[1] + f[2];
                    pD += f[0];
                } // if(chain)
                
            }  /* for(igrid) over the grid */
        }     /* for(Gtree) */
    }

    if(pD < 1e-300) {
        error = -1;
        printf("\nat locus %2d, pD = %.6g\n", locus+1, pD);
        lnL += -1e100 + lmax;
    }
    else {
        if (LASTROUND == 2) {
            if (com.model != M1DiscreteBeta) {
#pragma omp critical
                {
                    writeGeneTreePosterior(locus, pGk, pD, lmax, nGtree);
                }
            }
        }
        lnL += log(pD) + lmax;
        if(lnL != lnL) {
            snprintf(errorStr, sizeof(errorStr), "lnL for locus %d is NaN! Try re-running the analysis with different initial parameters or smaller value for Small_Diff.\n", locus+1);
            error2(errorStr);
        }
    }

    if(error) {
        printf("\n");
        for (i = 0; i < MAXPARAMETERS; i++)
            if (com.paraMap[i] != -1)
                printf(" %11.6f", para[i]);
        printf("\nlocus %d\n", locus+1);
        puts("floating point problem");
    }

    return(lnL);
}

void writeGeneTreePosterior(int locus, double pGk[], double pD, double lmax, int nTrees) {
    int i, istart = 0, iend, initState = data.initState[stree.sptree][locus];
    int fw = 7, p = 5, fwz, fws;

#ifdef DEBUG_GTREE_PROB
    double _pGk[MAXGTREETYPES * 3];
    if (com.model == M3MSci12 || com.model == M3MSci13 || com.model == M3MSci23)
        memcpy(_pGk, pGk, MAXGTREETYPES * 3 * sizeof(double));
    fw = 14;
    p = 12;
#endif

    fws = p + 1;
    fwz = (fw > fws) ? fw - fws : 0;

    if (com.model == M3MSci12) {
        switch (initState)
        {
        case 0:
        case 13:
            pGk[0] += pGk[18] + pGk[21];
            pGk[1] += pGk[19] + pGk[22];
            pGk[2] += pGk[20] + pGk[23];
            pGk[3] += pGk[24];
            pGk[4] += pGk[25];
            pGk[5] += pGk[26];
            pGk[6] += pGk[27];
            pGk[7] += pGk[28];
            pGk[8] += pGk[29];
            nTrees = 6;
            break;
        case 1:
        case 12:
            pGk[18] += pGk[15];
            pGk[21] += pGk[0];
            pGk[24] += pGk[3];
            memmove(pGk + 9, pGk + 6, 9 * sizeof(double));
            memcpy(pGk, pGk + 18, 9 * sizeof(double));
            nTrees = 6;
            break;
        case 2:
        case 14:
            pGk[0] += pGk[9];
            nTrees = 3;
            break;
        case 5:
            memmove(pGk + 3, pGk, 9 * sizeof(double));
            pGk[0] = pGk[9];
            nTrees = 3;
            break;
        case 27:
        case 31:
            pGk[0] += pGk[3];
            nTrees = 3;
            break;
        case 28:
            memmove(pGk + 1, pGk, 3 * sizeof(double));
            pGk[0] = pGk[3];
            nTrees = 3;
            break;
        default:;
        }
    }
    else if (com.model == M3MSci13) {
        switch (initState)
        {
        case 0:
        case 26:
            pGk[0] += pGk[18] + pGk[21];
            pGk[1] += pGk[19] + pGk[22];
            pGk[2] += pGk[20] + pGk[23];
            pGk[3] += pGk[24];
            pGk[4] += pGk[25];
            pGk[5] += pGk[26];
            pGk[6] += pGk[27];
            pGk[7] += pGk[28];
            pGk[8] += pGk[29];
            nTrees = 6;
            break;
        case 1:
            pGk[0] += pGk[15];
            pGk[3] += pGk[18];
            nTrees = 5;
            break;
        case 2:
            pGk[18] += pGk[15];
            pGk[21] += pGk[0];
            pGk[24] += pGk[3];
            memmove(pGk + 9, pGk + 6, 9 * sizeof(double));
            memcpy(pGk, pGk + 18, 9 * sizeof(double));
            nTrees = 6;
            break;
        case 5:
            memmove(pGk + 6, pGk, 15 * sizeof(double));
            memcpy(pGk, pGk + 15, 6 * sizeof(double));
            nTrees = 5;
            break;
        case 8:
            pGk[20] += pGk[17];
            pGk[23] += pGk[2];
            pGk[26] += pGk[5];
            memmove(pGk + 9, pGk + 6, 9 * sizeof(double));
            memcpy(pGk, pGk + 18, 9 * sizeof(double));
            nTrees = 6;
            break;
        case 17:
            pGk[2] += pGk[17];
            pGk[5] += pGk[20];
            nTrees = 5;
            break;
        case 27:
        case 35:
            pGk[0] += pGk[3];
            nTrees = 3;
            break;
        case 29:
            memmove(pGk + 1, pGk, 3 * sizeof(double));
            pGk[0] = pGk[3];
            nTrees = 3;
            break;
        default:;
        }
    }
    else if (com.model == M3MSci23) {
        switch (initState)
        {
        case 13:
        case 26:
            pGk[0] += pGk[18] + pGk[21];
            pGk[1] += pGk[19] + pGk[22];
            pGk[2] += pGk[20] + pGk[23];
            pGk[3] += pGk[24];
            pGk[4] += pGk[25];
            pGk[5] += pGk[26];
            pGk[6] += pGk[27];
            pGk[7] += pGk[28];
            pGk[8] += pGk[29];
            nTrees = 6;
            break;
        case 12:
            pGk[0] += pGk[15];
            pGk[3] += pGk[18];
            nTrees = 5;
            break;
        case 14:
            pGk[18] += pGk[15];
            pGk[21] += pGk[0];
            pGk[24] += pGk[3];
            memmove(pGk + 9, pGk + 6, 9 * sizeof(double));
            memcpy(pGk, pGk + 18, 9 * sizeof(double));
            nTrees = 6;
            break;
        case 5:
            memmove(pGk + 6, pGk, 15 * sizeof(double));
            memcpy(pGk, pGk + 15, 6 * sizeof(double));
            nTrees = 5;
            break;
        case 8:
            pGk[2] += pGk[17];
            pGk[5] += pGk[20];
            nTrees = 5;
            break;
        case 17:
            pGk[20] += pGk[17];
            pGk[23] += pGk[2];
            pGk[26] += pGk[5];
            memmove(pGk + 9, pGk + 6, 9 * sizeof(double));
            memcpy(pGk, pGk + 18, 9 * sizeof(double));
            nTrees = 6;
            break;
        case 31:
        case 35:
            pGk[0] += pGk[3];
            nTrees = 3;
            break;
        case 32:
            memmove(pGk + 1, pGk, 3 * sizeof(double));
            pGk[0] = pGk[3];
            nTrees = 3;
            break;
        default:;
        }
    }

#ifndef DEBUG_GTREE_PROB
    for (i = 0; i < 18; i++) {
        pGk[i] /= pD;
    }
#endif

    fprintf(fpGk, "%d\t%s", locus + 1, stateStr[initState]);

#ifdef DEBUG_GTREE_PROB
    fprintf(fpGk, "\t%*.*f", fw, p, pD);
#endif

    if (nTrees == 5) { // 112/122 for M0
        fprintf(fpGk, "\t%*d%*c\t%*d%*c\t%*d%*c", fwz, 0, fws, ' ', fwz, 0, fws, ' ', fwz, 0, fws, ' ');
        iend = 15;
    } else if (nTrees == 6) { // 111/222, 112/122
        iend = 18;
    } else if (initState >= NSTATES) { // 2 sequences
        if (nTrees == 2 && initState == 35) {
            fprintf(fpGk, "\t%*.*f", fw, p, pGk[0]);
            fprintf(fpGk, "\t%*d%*c", fwz, 0, fws, ' ');
            istart = 1;
            iend = 2;
        }
        else {
            for (i = 0; i < 3 - nTrees; i++)
                fprintf(fpGk, "\t%*d%*c", fwz, 0, fws, ' ');
            iend = nTrees;
        }
    } else if (nTrees == 2) { // 133/233; 123 for M0
        if (initState == 5) {
            for (i = 0; i < 4; i++)
                fprintf(fpGk, "\t%*d%*c\t%*d%*c\t%*d%*c", fwz, 0, fws, ' ', fwz, 0, fws, ' ', fwz, 0, fws, ' ');
            iend = 6;
        }
        else {
            for (i = 0; i < 2; i++)
                fprintf(fpGk, "\t%*d%*c\t%*d%*c\t%*d%*c", fwz, 0, fws, ' ', fwz, 0, fws, ' ', fwz, 0, fws, ' ');
            for (i = 0; i < 3; i++) {
                fprintf(fpGk, "\t%*.*f", fw, p, pGk[i]);
            }
            for (i = 0; i < 2; i++)
                fprintf(fpGk, "\t%*d%*c\t%*d%*c\t%*d%*c", fwz, 0, fws, ' ', fwz, 0, fws, ' ', fwz, 0, fws, ' ');
            istart = 3;
            iend = 6;
        }
    } else if (initState == 26) { // 333
        for (i = 0; i < 3; i++) {
            fprintf(fpGk, "\t%*.*f", fw, p, pGk[i]);
        }
        fprintf(fpGk, "\t%*d%*c\t%*d%*c\t%*d%*c", fwz, 0, fws, ' ', fwz, 0, fws, ' ', fwz, 0, fws, ' ');
        for (i = 3; i < 6; i++) {
            fprintf(fpGk, "\t%*.*f", fw, p, pGk[i]);
        }
        for (i = 0; i < 2; i++)
            fprintf(fpGk, "\t%*d%*c\t%*d%*c\t%*d%*c", fwz, 0, fws, ' ', fwz, 0, fws, ' ', fwz, 0, fws, ' ');
        istart = 6;
        iend = 9;
    } else { // 113/123/223
        for (i = 0; i < 2; i++)
            fprintf(fpGk, "\t%*d%*c\t%*d%*c\t%*d%*c", fwz, 0, fws, ' ', fwz, 0, fws, ' ', fwz, 0, fws, ' ');
        for (i = 0; i < 3; i++) {
            fprintf(fpGk, "\t%*.*f", fw, p, pGk[i]);
        }
        fprintf(fpGk, "\t%*d%*c\t%*d%*c\t%*d%*c", fwz, 0, fws, ' ', fwz, 0, fws, ' ', fwz, 0, fws, ' ');
        istart = 3;
        iend = 9;
    }
    for (i = istart; i < iend; i++) {
        fprintf(fpGk, "\t%*.*f", fw, p, pGk[i]);
    }
    fprintf(fpGk, "\n");

#ifdef DEBUG_GTREE_PROB
    if (com.model == M3MSci12) {
        switch (initState)
        {
        case 0:
        case 13:
            fprintf(fpGk, "\t\t\tG1\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 0], fw, p, _pGk[ 1], fw, p, _pGk[ 2]);
            fprintf(fpGk, "\t\t\tG1\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[18], fw, p, _pGk[19], fw, p, _pGk[20]);
            fprintf(fpGk, "\t\t\tG1\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[21], fw, p, _pGk[22], fw, p, _pGk[23]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 3], fw, p, _pGk[ 4], fw, p, _pGk[ 5]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[24], fw, p, _pGk[25], fw, p, _pGk[26]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 6], fw, p, _pGk[ 7], fw, p, _pGk[ 8]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[27], fw, p, _pGk[28], fw, p, _pGk[29]);
            fprintf(fpGk, "\t\t\tG4\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 9], fw, p, _pGk[10], fw, p, _pGk[11]);
            fprintf(fpGk, "\t\t\tG5\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[12], fw, p, _pGk[13], fw, p, _pGk[14]);
            fprintf(fpGk, "\t\t\tG6\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[15], fw, p, _pGk[16], fw, p, _pGk[17]);
            break;
        case 1:
        case 12:
            fprintf(fpGk, "\t\t\tG1\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[15], fw, p, _pGk[16], fw, p, _pGk[17]);
            fprintf(fpGk, "\t\t\tG1\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[18], fw, p, _pGk[19], fw, p, _pGk[20]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 0], fw, p, _pGk[ 1], fw, p, _pGk[ 2]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[21], fw, p, _pGk[22], fw, p, _pGk[23]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 3], fw, p, _pGk[ 4], fw, p, _pGk[ 5]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[24], fw, p, _pGk[25], fw, p, _pGk[26]);
            fprintf(fpGk, "\t\t\tG4\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 6], fw, p, _pGk[ 7], fw, p, _pGk[ 8]);
            fprintf(fpGk, "\t\t\tG5\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 9], fw, p, _pGk[10], fw, p, _pGk[11]);
            fprintf(fpGk, "\t\t\tG6\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[12], fw, p, _pGk[13], fw, p, _pGk[14]);
            break;
        case 2:
        case 14:
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 0], fw, p, _pGk[ 1], fw, p, _pGk[ 2]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 9], fw, p, _pGk[10], fw, p, _pGk[11]);
            fprintf(fpGk, "\t\t\tG5\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 3], fw, p, _pGk[ 4], fw, p, _pGk[ 5]);
            fprintf(fpGk, "\t\t\tG6\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 6], fw, p, _pGk[ 7], fw, p, _pGk[ 8]);
            break;
        case 5:
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 6], fw, p, _pGk[ 7], fw, p, _pGk[ 8]);
            fprintf(fpGk, "\t\t\tG5\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 0], fw, p, _pGk[ 1], fw, p, _pGk[ 2]);
            fprintf(fpGk, "\t\t\tG6\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 3], fw, p, _pGk[ 4], fw, p, _pGk[ 5]);
            break;
        case 8:
        case 17:
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 0], fw, p, _pGk[ 1], fw, p, _pGk[ 2]);
            fprintf(fpGk, "\t\t\tG6\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 3], fw, p, _pGk[ 4], fw, p, _pGk[ 5]);
            break;
        case 26:
            fprintf(fpGk, "\t\t\tG1\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 0], fw, p, _pGk[ 1], fw, p, _pGk[ 2]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 3], fw, p, _pGk[ 4], fw, p, _pGk[ 5]);
            fprintf(fpGk, "\t\t\tG6\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 6], fw, p, _pGk[ 7], fw, p, _pGk[ 8]);
            break;
        case 27:
        case 31:
            fprintf(fpGk, "\t\t\tG1\t%*.*f\n", fw, p, _pGk[0]);
            fprintf(fpGk, "\t\t\tG1\t%*.*f\n", fw, p, _pGk[3]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\n", fw, p, _pGk[1]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\n", fw, p, _pGk[2]);
            break;
        case 28:
            fprintf(fpGk, "\t\t\tG1\t%*.*f\n", fw, p, _pGk[2]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\n", fw, p, _pGk[0]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\n", fw, p, _pGk[1]);
            break;
        case 29:
        case 32:
            fprintf(fpGk, "\t\t\tG3\t%*.*f\n", fw, p, _pGk[0]);
            break;
        case 35:
            fprintf(fpGk, "\t\t\tG1\t%*.*f\n", fw, p, _pGk[0]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\n", fw, p, _pGk[1]);
            break;
        default:;
        }
    }
    else if (com.model == M3MSci13) {
        switch (initState)
        {
        case 0:
        case 26:
            fprintf(fpGk, "\t\t\tG1\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 0], fw, p, _pGk[ 1], fw, p, _pGk[ 2]);
            fprintf(fpGk, "\t\t\tG1\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[18], fw, p, _pGk[19], fw, p, _pGk[20]);
            fprintf(fpGk, "\t\t\tG1\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[21], fw, p, _pGk[22], fw, p, _pGk[23]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 3], fw, p, _pGk[ 4], fw, p, _pGk[ 5]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[24], fw, p, _pGk[25], fw, p, _pGk[26]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 6], fw, p, _pGk[ 7], fw, p, _pGk[ 8]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[27], fw, p, _pGk[28], fw, p, _pGk[29]);
            fprintf(fpGk, "\t\t\tG4\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 9], fw, p, _pGk[10], fw, p, _pGk[11]);
            fprintf(fpGk, "\t\t\tG5\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[12], fw, p, _pGk[13], fw, p, _pGk[14]);
            fprintf(fpGk, "\t\t\tG6\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[15], fw, p, _pGk[16], fw, p, _pGk[17]);
            break;
        case 13:
            fprintf(fpGk, "\t\t\tG1\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 0], fw, p, _pGk[ 1], fw, p, _pGk[ 2]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 3], fw, p, _pGk[ 4], fw, p, _pGk[ 5]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 6], fw, p, _pGk[ 7], fw, p, _pGk[ 8]);
            fprintf(fpGk, "\t\t\tG4\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 9], fw, p, _pGk[10], fw, p, _pGk[11]);
            fprintf(fpGk, "\t\t\tG5\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[12], fw, p, _pGk[13], fw, p, _pGk[14]);
            fprintf(fpGk, "\t\t\tG6\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[15], fw, p, _pGk[16], fw, p, _pGk[17]);
            break;
        case 1:
        case 17:
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 0], fw, p, _pGk[ 1], fw, p, _pGk[ 2]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[15], fw, p, _pGk[16], fw, p, _pGk[17]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 3], fw, p, _pGk[ 4], fw, p, _pGk[ 5]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[18], fw, p, _pGk[19], fw, p, _pGk[20]);
            fprintf(fpGk, "\t\t\tG4\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 6], fw, p, _pGk[ 7], fw, p, _pGk[ 8]);
            fprintf(fpGk, "\t\t\tG5\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 9], fw, p, _pGk[10], fw, p, _pGk[11]);
            fprintf(fpGk, "\t\t\tG6\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[12], fw, p, _pGk[13], fw, p, _pGk[14]);
            break;
        case 12:
        case 14:
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 0], fw, p, _pGk[ 1], fw, p, _pGk[ 2]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 3], fw, p, _pGk[ 4], fw, p, _pGk[ 5]);
            fprintf(fpGk, "\t\t\tG4\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 6], fw, p, _pGk[ 7], fw, p, _pGk[ 8]);
            fprintf(fpGk, "\t\t\tG5\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 9], fw, p, _pGk[10], fw, p, _pGk[11]);
            fprintf(fpGk, "\t\t\tG6\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[12], fw, p, _pGk[13], fw, p, _pGk[14]);
            break;
        case 2:
        case 8:
            fprintf(fpGk, "\t\t\tG1\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[15], fw, p, _pGk[16], fw, p, _pGk[17]);
            fprintf(fpGk, "\t\t\tG1\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[18], fw, p, _pGk[19], fw, p, _pGk[20]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 0], fw, p, _pGk[ 1], fw, p, _pGk[ 2]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[21], fw, p, _pGk[22], fw, p, _pGk[23]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 3], fw, p, _pGk[ 4], fw, p, _pGk[ 5]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[24], fw, p, _pGk[25], fw, p, _pGk[26]);
            fprintf(fpGk, "\t\t\tG4\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 6], fw, p, _pGk[ 7], fw, p, _pGk[ 8]);
            fprintf(fpGk, "\t\t\tG5\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 9], fw, p, _pGk[10], fw, p, _pGk[11]);
            fprintf(fpGk, "\t\t\tG6\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[12], fw, p, _pGk[13], fw, p, _pGk[14]);
            break;
        case 5:
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 9], fw, p, _pGk[10], fw, p, _pGk[11]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[12], fw, p, _pGk[13], fw, p, _pGk[14]);
            fprintf(fpGk, "\t\t\tG4\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 0], fw, p, _pGk[ 1], fw, p, _pGk[ 2]);
            fprintf(fpGk, "\t\t\tG5\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 3], fw, p, _pGk[ 4], fw, p, _pGk[ 5]);
            fprintf(fpGk, "\t\t\tG6\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 6], fw, p, _pGk[ 7], fw, p, _pGk[ 8]);
            break;
        case 27:
        case 35:
            fprintf(fpGk, "\t\t\tG1\t%*.*f\n", fw, p, _pGk[0]);
            fprintf(fpGk, "\t\t\tG1\t%*.*f\n", fw, p, _pGk[3]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\n", fw, p, _pGk[1]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\n", fw, p, _pGk[2]);
            break;
        case 31:
            fprintf(fpGk, "\t\t\tG1\t%*.*f\n", fw, p, _pGk[0]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\n", fw, p, _pGk[1]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\n", fw, p, _pGk[2]);
            break;
        case 28:
        case 32:
            fprintf(fpGk, "\t\t\tG2\t%*.*f\n", fw, p, _pGk[0]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\n", fw, p, _pGk[1]);
            break;
        case 29:
            fprintf(fpGk, "\t\t\tG1\t%*.*f\n", fw, p, _pGk[2]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\n", fw, p, _pGk[0]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\n", fw, p, _pGk[1]);
            break;
        default:;
        }
    }
    else if (com.model == M3MSci23) {
        switch (initState)
        {
        case 0:
            fprintf(fpGk, "\t\t\tG1\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 0], fw, p, _pGk[ 1], fw, p, _pGk[ 2]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 3], fw, p, _pGk[ 4], fw, p, _pGk[ 5]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 6], fw, p, _pGk[ 7], fw, p, _pGk[ 8]);
            fprintf(fpGk, "\t\t\tG4\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 9], fw, p, _pGk[10], fw, p, _pGk[11]);
            fprintf(fpGk, "\t\t\tG5\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[12], fw, p, _pGk[13], fw, p, _pGk[14]);
            fprintf(fpGk, "\t\t\tG6\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[15], fw, p, _pGk[16], fw, p, _pGk[17]);
            break;
        case 13:
        case 26:
            fprintf(fpGk, "\t\t\tG1\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 0], fw, p, _pGk[ 1], fw, p, _pGk[ 2]);
            fprintf(fpGk, "\t\t\tG1\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[18], fw, p, _pGk[19], fw, p, _pGk[20]);
            fprintf(fpGk, "\t\t\tG1\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[21], fw, p, _pGk[22], fw, p, _pGk[23]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 3], fw, p, _pGk[ 4], fw, p, _pGk[ 5]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[24], fw, p, _pGk[25], fw, p, _pGk[26]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 6], fw, p, _pGk[ 7], fw, p, _pGk[ 8]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[27], fw, p, _pGk[28], fw, p, _pGk[29]);
            fprintf(fpGk, "\t\t\tG4\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 9], fw, p, _pGk[10], fw, p, _pGk[11]);
            fprintf(fpGk, "\t\t\tG5\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[12], fw, p, _pGk[13], fw, p, _pGk[14]);
            fprintf(fpGk, "\t\t\tG6\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[15], fw, p, _pGk[16], fw, p, _pGk[17]);
            break;
        case 1:
        case 2:
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 0], fw, p, _pGk[ 1], fw, p, _pGk[ 2]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 3], fw, p, _pGk[ 4], fw, p, _pGk[ 5]);
            fprintf(fpGk, "\t\t\tG4\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 6], fw, p, _pGk[ 7], fw, p, _pGk[ 8]);
            fprintf(fpGk, "\t\t\tG5\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 9], fw, p, _pGk[10], fw, p, _pGk[11]);
            fprintf(fpGk, "\t\t\tG6\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[12], fw, p, _pGk[13], fw, p, _pGk[14]);
            break;
        case 12:
        case 8:
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 0], fw, p, _pGk[ 1], fw, p, _pGk[ 2]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[15], fw, p, _pGk[16], fw, p, _pGk[17]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 3], fw, p, _pGk[ 4], fw, p, _pGk[ 5]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[18], fw, p, _pGk[19], fw, p, _pGk[20]);
            fprintf(fpGk, "\t\t\tG4\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 6], fw, p, _pGk[ 7], fw, p, _pGk[ 8]);
            fprintf(fpGk, "\t\t\tG5\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 9], fw, p, _pGk[10], fw, p, _pGk[11]);
            fprintf(fpGk, "\t\t\tG6\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[12], fw, p, _pGk[13], fw, p, _pGk[14]);
            break;
        case 14:
        case 17:
            fprintf(fpGk, "\t\t\tG1\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[15], fw, p, _pGk[16], fw, p, _pGk[17]);
            fprintf(fpGk, "\t\t\tG1\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[18], fw, p, _pGk[19], fw, p, _pGk[20]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 0], fw, p, _pGk[ 1], fw, p, _pGk[ 2]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[21], fw, p, _pGk[22], fw, p, _pGk[23]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 3], fw, p, _pGk[ 4], fw, p, _pGk[ 5]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[24], fw, p, _pGk[25], fw, p, _pGk[26]);
            fprintf(fpGk, "\t\t\tG4\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 6], fw, p, _pGk[ 7], fw, p, _pGk[ 8]);
            fprintf(fpGk, "\t\t\tG5\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 9], fw, p, _pGk[10], fw, p, _pGk[11]);
            fprintf(fpGk, "\t\t\tG6\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[12], fw, p, _pGk[13], fw, p, _pGk[14]);
            break;
        case 5:
            fprintf(fpGk, "\t\t\tG2\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 9], fw, p, _pGk[10], fw, p, _pGk[11]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[12], fw, p, _pGk[13], fw, p, _pGk[14]);
            fprintf(fpGk, "\t\t\tG4\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 0], fw, p, _pGk[ 1], fw, p, _pGk[ 2]);
            fprintf(fpGk, "\t\t\tG5\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 3], fw, p, _pGk[ 4], fw, p, _pGk[ 5]);
            fprintf(fpGk, "\t\t\tG6\t%*.*f\t%*.*f\t%*.*f\n", fw, p, _pGk[ 6], fw, p, _pGk[ 7], fw, p, _pGk[ 8]);
            break;
        case 27:
            fprintf(fpGk, "\t\t\tG1\t%*.*f\n", fw, p, _pGk[0]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\n", fw, p, _pGk[1]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\n", fw, p, _pGk[2]);
            break;
        case 31:
        case 35:
            fprintf(fpGk, "\t\t\tG1\t%*.*f\n", fw, p, _pGk[0]);
            fprintf(fpGk, "\t\t\tG1\t%*.*f\n", fw, p, _pGk[3]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\n", fw, p, _pGk[1]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\n", fw, p, _pGk[2]);
            break;
        case 28:
        case 29:
            fprintf(fpGk, "\t\t\tG2\t%*.*f\n", fw, p, _pGk[0]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\n", fw, p, _pGk[1]);
            break;
        case 32:
            fprintf(fpGk, "\t\t\tG1\t%*.*f\n", fw, p, _pGk[2]);
            fprintf(fpGk, "\t\t\tG2\t%*.*f\n", fw, p, _pGk[0]);
            fprintf(fpGk, "\t\t\tG3\t%*.*f\n", fw, p, _pGk[1]);
            break;
        default:;
        }
    }
#endif
}

void computePMatrix(double t, double * Q, int N, double * P, double * work) {
    int i;
    for(i = 0; i < N*N; i++) {
        P[i] = Q[i] * t;
    }
    
#ifdef USE_GSL
    matexpGSL(P, N, work);
#else
    matexp(P, N, NTAYLORTERMS, NSQUARES, work);
#endif
}

int skipTree(int gtree) {
    int lociSum = 0;
    int* iStateCnt = data.iStateCnt[stree.sptree];
    if(com.model == M0 || com.model == M1DiscreteBeta) {
        if (gtree < 6) {
            lociSum = iStateCnt[0];
        } else if (gtree < 12) {
            lociSum = iStateCnt[13];
        } else if(gtree < 17) {
            lociSum = iStateCnt[1];
        } else if(gtree < 22) {
            lociSum = iStateCnt[12];
        } else if(gtree < 25) {
            lociSum = iStateCnt[2];
        } else if(gtree < 28) {
            lociSum = iStateCnt[14];
        } else if(gtree < 30) {
            lociSum = iStateCnt[5];
        } else if(gtree < 32) {
            lociSum = iStateCnt[8] + iStateCnt[17];
        } else if(gtree < 35) {
            lociSum = iStateCnt[26];
        } else {
            error2("unknown gene tree");
        }
    } else if(com.model == M2SIM3s) {
        if (gtree < 6) {
            lociSum = iStateCnt[0];
        } else if(gtree < 12) {
            lociSum = iStateCnt[13];
        } else if(gtree < 18) {
            lociSum = iStateCnt[1];
        } else if(gtree < 24) {
            lociSum = iStateCnt[12];
        } else if(gtree < 27) {
            lociSum = iStateCnt[2];
        } else if(gtree < 30) {
            lociSum = iStateCnt[14];
        } else if(gtree < 33) {
            lociSum = iStateCnt[5];
        } else if(gtree < 35) {
            lociSum = iStateCnt[8] + iStateCnt[17];
        } else if(gtree < 38) {
            lociSum = iStateCnt[26];
        } else {
            error2("unknown gene tree");
        }
    } else if(com.model == M2Pro || com.model == M2ProMax) {
        if (gtree < 6) {
            lociSum = iStateCnt[0];
        } else if(gtree < 12) {
            lociSum = iStateCnt[13];
        } else if(gtree < 18) {
            lociSum = iStateCnt[1];
        } else if(gtree < 24) {
            lociSum = iStateCnt[12];
        } else if(gtree < 30) {
            lociSum = iStateCnt[2];
        } else if(gtree < 36) {
            lociSum = iStateCnt[14];
        } else if(gtree < 42) {
            lociSum = iStateCnt[5];
        } else if(gtree < 48) {
            lociSum = iStateCnt[8];
        } else if(gtree < 54) {
            lociSum = iStateCnt[17];
        } else if(gtree < 60) {
            lociSum = iStateCnt[26];
        } else {
            error2("unknown gene tree");
        }
    } else if(com.model == M3MSci12) {
        if (gtree < 10) {
            lociSum = iStateCnt[0];
        } else if(gtree < 20) {
            lociSum = iStateCnt[13];
        } else if(gtree < 29) {
            lociSum = iStateCnt[1];
        } else if(gtree < 38) {
            lociSum = iStateCnt[12];
        } else if(gtree < 42) {
            lociSum = iStateCnt[2];
        } else if(gtree < 46) {
            lociSum = iStateCnt[14];
        } else if(gtree < 49) {
            lociSum = iStateCnt[5];
        } else if(gtree < 51) {
            lociSum = iStateCnt[8] + iStateCnt[17];
        } else if(gtree < 54) {
            lociSum = iStateCnt[26];
        } else {
            error2("unknown gene tree");
        }
    } else if(com.model == M3MSci13) {
        if (gtree < 10) {
            lociSum = iStateCnt[0];
        } else if(gtree < 16) {
            lociSum = iStateCnt[13];
        } else if(gtree < 23) {
            lociSum = iStateCnt[1];
        } else if(gtree < 28) {
            lociSum = iStateCnt[12];
        } else if(gtree < 37) {
            lociSum = iStateCnt[2];
        } else if(gtree < 42) {
            lociSum = iStateCnt[14];
        } else if(gtree < 47) {
            lociSum = iStateCnt[5];
        } else if(gtree < 56) {
            lociSum = iStateCnt[8];
        } else if(gtree < 63) {
            lociSum = iStateCnt[17];
        } else if(gtree < 73) {
            lociSum = iStateCnt[26];
        } else {
            error2("unknown gene tree");
        }
    } else if(com.model == M3MSci23) {
        if (gtree < 6) {
            lociSum = iStateCnt[0];
        } else if(gtree < 16) {
            lociSum = iStateCnt[13];
        } else if(gtree < 21) {
            lociSum = iStateCnt[1];
        } else if(gtree < 28) {
            lociSum = iStateCnt[12];
        } else if(gtree < 33) {
            lociSum = iStateCnt[2];
        } else if(gtree < 42) {
            lociSum = iStateCnt[14];
        } else if(gtree < 47) {
            lociSum = iStateCnt[5];
        } else if(gtree < 54) {
            lociSum = iStateCnt[8];
        } else if(gtree < 63) {
            lociSum = iStateCnt[17];
        } else if(gtree < 73) {
            lociSum = iStateCnt[26];
        } else {
            error2("unknown gene tree");
        }
    }
    return !lociSum;
}

void update_GtreeTypes(int GtreeTypeCnt[]) {
    int tree;
    double *b, tau0, tau1, treeHight;

    tau0 = para[2];
    tau1 = para[3];

    for (tree = 0, b = data.Bij; tree < data.ndata; tree++, b += 2) {
        if (data.initState[stree.sptree][tree] >= NSTATES)
            error2("not implemented");

        treeHight = b[0] + b[1];

        if (treeHight > tau0) {
            if (b[1] > tau0)
                data.GtreeType[tree] = 6;
            else if (b[1] > tau1)
                data.GtreeType[tree] = 5;
            else
                data.GtreeType[tree] = 3;
        }
        else if (treeHight > tau1) {
            if (b[1] > tau1)
                data.GtreeType[tree] = 4;
            else
                data.GtreeType[tree] = 2;
        }
        else
            data.GtreeType[tree] = 1;

        GtreeTypeCnt[data.GtreeType[tree]]++;
    }
}

void preprocessM2SIM3s(int GtreeTypeCnt[]) {
    double theta4, theta5, theta1, theta2, theta3, tau0, tau1, M12=0.0, M21=0.0;
    double * Q_C1, * Q_C2, * Q_C3, * Pt1, * Pt0, * Ptau1t1, /**Ptau1,*/ * Ptau1_C1, * Ptau1_C2, * Ptau1_C3, * work, /*PG1a,*/ PG1a123=0, PG1a113=0, PG1a223=0;

    Q_C3 = com.space;
    if(!Q_C3) error2("oom allocating Q_C3 in preprocessM2SIM3s");
    Ptau1_C3 = &Q_C3[C3*C3]; Q_C1 = &Ptau1_C3[C3*C3];
    Ptau1_C1 = &Q_C1[C1*C1]; Q_C2 = &Ptau1_C1[C1*C1];
    Ptau1_C2 = &Q_C2[C2*C2]; Pt1 = &Ptau1_C2[C2*C2]; Pt0 = &Ptau1_C2[2*C2*C2]; Ptau1t1 = &Ptau1_C2[3*C2*C2]; work = &Ptau1_C2[4*C2*C2];
    com.space = Q_C2+8*C2*C2;

    getParams(&theta4, &theta5, &tau0, &tau1, &theta1, &theta2, &theta3);

#ifdef FIXM12
    M21 = M12 = 0.000001;
#else
    M12 = para[8];
    M21 = para[9];
#endif

    GenerateQ1SIM3S(Q_C1, C1, theta1, theta2, 0, M12, M21);
    GenerateQ1SIM3S(Q_C2, C2, theta1, theta2, 0, M12, M21);
    GenerateQ1SIM3S(Q_C3, C3, theta1, theta2, 0, M12, M21);

    computePMatrix(tau1, Q_C1, C1, Ptau1_C1, work);
    computePMatrix(tau1, Q_C2, C2, Ptau1_C2, work);
    computePMatrix(tau1, Q_C3, C3, Ptau1_C3, work);

    com.space = Q_C3;
}

void preprocessM2Pro(int GtreeTypeCnt[]) {
    double theta4, theta5, theta1, theta2, theta3, thetaW, tau0, tau1, M12, M21, M13, M31, M23, M32;
    double *Q_C5, *Q_C6, *Ptau1_C5, *Ptau1_C5_ex, *exptaugap, *work;
    int i, j;

    Q_C5 = com.space;
    if (!Q_C5) error2("oom allocating Q_C5 in preprocessM2Pro");
    Q_C6 = &Q_C5[C5 * C5];
    Ptau1_C5 = &Q_C6[C6 * C6];
    Ptau1_C5_ex = &Ptau1_C5[C5 * C5];
    exptaugap = &Ptau1_C5_ex[C5 * C5_ex];
    work = &exptaugap[4];
    com.space = work + 3 * square(max2(C5, C6));

    getParams(&theta4, &theta5, &tau0, &tau1, &theta1, &theta2, &theta3);

    thetaW = para[7];
    M12 = para[8];
    M21 = para[9];
    M13 = para[10];
    M31 = para[11];
    M23 = para[12];
    M32 = para[13];

    GenerateQ5(Q_C5, theta1, theta2, theta3, M12, M21, M13, M31, M23, M32);
    GenerateQ6(Q_C6, theta1, theta2, theta3, M12, M21, M13, M31, M23, M32);

    if (GtreeTypeCnt[3] + GtreeTypeCnt[4] + GtreeTypeCnt[5] + GtreeTypeCnt[6]) {
        exptaugap[0] = exp(-6 * (tau0 - tau1) / theta5);
        exptaugap[1] = exp(-2 * (tau0 - tau1) / theta5);
        exptaugap[2] = exp(-2 * (tau0 - tau1) / thetaW);
        exptaugap[3] = exp(-6 * (tau0 - tau1) / thetaW);

        if (GtreeTypeCnt[4] + GtreeTypeCnt[5] + GtreeTypeCnt[6]) {
            computePMatrix(tau1, Q_C5, C5, Ptau1_C5, work);

            for (i = 0; i < NINITIALSTATES; i++) {
                j = initStates[i];
                Ptau1_C5_ex[C5_ex * j + 0] = Ptau1_C5[C5 * j +  0] + Ptau1_C5[C5 * j +  1] + Ptau1_C5[C5 * j +  3] + Ptau1_C5[C5 * j +  4]
                                           + Ptau1_C5[C5 * j +  9] + Ptau1_C5[C5 * j + 10] + Ptau1_C5[C5 * j + 12] + Ptau1_C5[C5 * j + 13];
                Ptau1_C5_ex[C5_ex * j + 1] = Ptau1_C5[C5 * j +  2] + Ptau1_C5[C5 * j +  5] + Ptau1_C5[C5 * j + 11] + Ptau1_C5[C5 * j + 14];
                Ptau1_C5_ex[C5_ex * j + 2] = Ptau1_C5[C5 * j +  6] + Ptau1_C5[C5 * j +  7] + Ptau1_C5[C5 * j + 15] + Ptau1_C5[C5 * j + 16];
                Ptau1_C5_ex[C5_ex * j + 3] = Ptau1_C5[C5 * j + 18] + Ptau1_C5[C5 * j + 19] + Ptau1_C5[C5 * j + 21] + Ptau1_C5[C5 * j + 22];
                Ptau1_C5_ex[C5_ex * j + 4] = Ptau1_C5[C5 * j + 24] + Ptau1_C5[C5 * j + 25];
                Ptau1_C5_ex[C5_ex * j + 5] = Ptau1_C5[C5 * j + 20] + Ptau1_C5[C5 * j + 23];
                Ptau1_C5_ex[C5_ex * j + 6] = Ptau1_C5[C5 * j +  8] + Ptau1_C5[C5 * j + 17];
                Ptau1_C5_ex[C5_ex * j + 7] = Ptau1_C5[C5 * j + 26];
                Ptau1_C5_ex[C5_ex * j + 8] =  Ptau1_C5_ex[C5_ex * j + 0] * exptaugap[0]
                                           + (Ptau1_C5_ex[C5_ex * j + 1] + Ptau1_C5_ex[C5_ex * j + 2] + Ptau1_C5_ex[C5_ex * j + 3]) * exptaugap[1]
                                           + (Ptau1_C5_ex[C5_ex * j + 4] + Ptau1_C5_ex[C5_ex * j + 5] + Ptau1_C5_ex[C5_ex * j + 6]) * exptaugap[2]
                                           +  Ptau1_C5_ex[C5_ex * j + 7] * exptaugap[3];
            }
        }
    }

    com.space = Q_C5;
}

void preprocessM2ProMax(int GtreeTypeCnt[]) {
    double theta4, theta5, theta1, theta2, theta3, thetaW, tau0, tau1, M12, M21, M13, M31, M23, M32, M53, M35;
    double *Q_C5, *Q_C6, *Q_C7, *Q_C8, *Ptau1_C5, *Ptau1_C5_ex, *Ptaugap_C7, *Ptaugap_C7_ex, *Ptaugap_C8, *Ptaugap_C8_ex, *work;
    int i, j;

    Q_C5 = com.space;
    if (!Q_C5) error2("oom allocating Q_C5 in preprocessM2ProMax");
    Q_C6 = &Q_C5[C5 * C5];
    Q_C7 = &Q_C6[C6 * C6];
    Q_C8 = &Q_C7[C7 * C7];
    Ptau1_C5 = &Q_C8[C8 * C8];
    Ptau1_C5_ex = &Ptau1_C5[C5 * C5];
    Ptaugap_C7 = &Ptau1_C5_ex[C5 * C5_ex];
    Ptaugap_C7_ex = &Ptaugap_C7[C7 * C7];
    Ptaugap_C8 = &Ptaugap_C7_ex[C7 * C7_ex];
    Ptaugap_C8_ex = &Ptaugap_C8[C8 * C8];
    work = &Ptaugap_C8_ex[C8 * C8_ex];
    com.space = work + 3 * square(max2(max2(max2(C5, C6), C7), C8));

    getParams(&theta4, &theta5, &tau0, &tau1, &theta1, &theta2, &theta3);

    thetaW = para[7];
    M12 = para[8];
    M21 = para[9];
    M13 = para[10];
    M31 = para[11];
    M23 = para[12];
    M32 = para[13];
    M53 = para[14];
    M35 = para[15];

    GenerateQ5(Q_C5, theta1, theta2, theta3, M12, M21, M13, M31, M23, M32);
    GenerateQ6(Q_C6, theta1, theta2, theta3, M12, M21, M13, M31, M23, M32);
    GenerateQ7(Q_C7, theta5, thetaW, M53, M35);
    GenerateQ8(Q_C8, theta5, thetaW, M53, M35);

    if (GtreeTypeCnt[3] + GtreeTypeCnt[4] + GtreeTypeCnt[5] + GtreeTypeCnt[6]) {
        computePMatrix(tau0 - tau1, Q_C8, C8, Ptaugap_C8, work);

        for (i = 0; i < C8 - 1; i++)
            Ptaugap_C8_ex[C8_ex * i] = 1 - Ptaugap_C8[C8 * i + C8 - 1];

        if (GtreeTypeCnt[4] + GtreeTypeCnt[5] + GtreeTypeCnt[6]) {
            computePMatrix(tau1, Q_C5, C5, Ptau1_C5, work);
            computePMatrix(tau0 - tau1, Q_C7, C7, Ptaugap_C7, work);

            for (i = 0; i < C7 - 1; i++)
                Ptaugap_C7_ex[C7_ex * i] = 1 - Ptaugap_C7[C7 * i + C7 - 1];

            for (i = 0; i < NINITIALSTATES; i++) {
                j = initStates[i];
                Ptau1_C5_ex[C5_ex * j + 0] = Ptau1_C5[C5 * j +  0] + Ptau1_C5[C5 * j +  1] + Ptau1_C5[C5 * j +  3] + Ptau1_C5[C5 * j +  4]
                                           + Ptau1_C5[C5 * j +  9] + Ptau1_C5[C5 * j + 10] + Ptau1_C5[C5 * j + 12] + Ptau1_C5[C5 * j + 13];
                Ptau1_C5_ex[C5_ex * j + 1] = Ptau1_C5[C5 * j +  2] + Ptau1_C5[C5 * j +  5] + Ptau1_C5[C5 * j + 11] + Ptau1_C5[C5 * j + 14];
                Ptau1_C5_ex[C5_ex * j + 2] = Ptau1_C5[C5 * j +  6] + Ptau1_C5[C5 * j +  7] + Ptau1_C5[C5 * j + 15] + Ptau1_C5[C5 * j + 16];
                Ptau1_C5_ex[C5_ex * j + 3] = Ptau1_C5[C5 * j + 18] + Ptau1_C5[C5 * j + 19] + Ptau1_C5[C5 * j + 21] + Ptau1_C5[C5 * j + 22];
                Ptau1_C5_ex[C5_ex * j + 4] = Ptau1_C5[C5 * j + 24] + Ptau1_C5[C5 * j + 25];
                Ptau1_C5_ex[C5_ex * j + 5] = Ptau1_C5[C5 * j + 20] + Ptau1_C5[C5 * j + 23];
                Ptau1_C5_ex[C5_ex * j + 6] = Ptau1_C5[C5 * j +  8] + Ptau1_C5[C5 * j + 17];
                Ptau1_C5_ex[C5_ex * j + 7] = Ptau1_C5[C5 * j + 26];
                Ptau1_C5_ex[C5_ex * j + 8] = Ptau1_C5_ex[C5_ex * j + 0] * Ptaugap_C7_ex[C7_ex * 0]
                                           + Ptau1_C5_ex[C5_ex * j + 1] * Ptaugap_C7_ex[C7_ex * 1]
                                           + Ptau1_C5_ex[C5_ex * j + 2] * Ptaugap_C7_ex[C7_ex * 2]
                                           + Ptau1_C5_ex[C5_ex * j + 3] * Ptaugap_C7_ex[C7_ex * 3]
                                           + Ptau1_C5_ex[C5_ex * j + 4] * Ptaugap_C7_ex[C7_ex * 4]
                                           + Ptau1_C5_ex[C5_ex * j + 5] * Ptaugap_C7_ex[C7_ex * 5]
                                           + Ptau1_C5_ex[C5_ex * j + 6] * Ptaugap_C7_ex[C7_ex * 6]
                                           + Ptau1_C5_ex[C5_ex * j + 7] * Ptaugap_C7_ex[C7_ex * 7];
            }
        }
    }

    com.space = Q_C5;
}

double lnpD_tree(int tree) {
    double pD, lnL;
    char errorStr[130];

    if (com.model == M0)
        pD = treeProbM0(tree);
    else if (com.model == M2SIM3s)
        pD = treeProbM2SIM3s(tree);
    else if (com.model == M2Pro)
        pD = treeProbM2Pro(tree);
    else if (com.model == M2ProMax)
        pD = treeProbM2ProMax(tree);
    else if (com.model == M3MSci12)
        pD = treeProbM3MSci12(tree);
    else if (com.model == M3MSci13)
        pD = treeProbM3MSci13(tree);
    else if (com.model == M3MSci23)
        pD = treeProbM3MSci23(tree);
    else
        error2("not implemented");

    lnL = (pD < 1e-300) ? -1e10 : log(pD);

    if (lnL != lnL) {
        snprintf(errorStr, sizeof(errorStr), "lnL for tree %d is NaN! Try re-running the analysis with different initial parameters or smaller value for Small_Diff.\n", tree + 1);
        error2(errorStr);
    }

    return lnL;
}

double treeProbM0(int tree) {
    int start, offset, Gtree, zeroProb = 0;
    double theta12;
    double theta1 = para[4];
    double theta2 = para[5];
    double t[2], t01[2], prob[3];
    double *b = data.Bij + tree * 2;
    int initState = data.initState[stree.sptree][tree];
    int topology = data.topology[stree.sptree][tree];
    int GtreeType = data.GtreeType[tree];
    BTEntry *gtt = com.GtreeTab[0];

    switch (initState)
    {
    case 0:
        start = 0;
        offset = GtreeType - 1;
        break;
    case 13:
        start = 6;
        offset = GtreeType - 1;
        break;
    case 1:
        start = 12;
        if ((GtreeType == 2 || GtreeType == 3) && !topology || GtreeType >= 4) offset = GtreeType - 2;
        else zeroProb = 1;
        break;
    case 12:
        start = 17;
        if ((GtreeType == 2 || GtreeType == 3) && !topology || GtreeType >= 4) offset = GtreeType - 2;
        else zeroProb = 1;
        break;
    case 2:
        start = 22;
        if (GtreeType == 3 && !topology) offset = 0;
        else if ((GtreeType == 5 && !topology) || GtreeType == 6) offset = GtreeType - 4;
        else zeroProb = 1;
        break;
    case 14:
        start = 25;
        if (GtreeType == 3 && !topology) offset = 0;
        else if ((GtreeType == 5 && !topology) || GtreeType == 6) offset = GtreeType - 4;
        else zeroProb = 1;
        break;
    case 5:
        start = 28;
        if ((GtreeType == 5 && !topology) || GtreeType == 6) offset = GtreeType - 5;
        else zeroProb = 1;
        break;
    case 8:
    case 17:
        start = 30;
        if ((GtreeType == 3 || GtreeType == 5) && topology == 2) offset = 0;
        else if (GtreeType == 6) offset = 1;
        else zeroProb = 1;
        break;
    case 26:
        start = 32;
        if (GtreeType == 1 || GtreeType == 2 || GtreeType == 4) offset = 0;
        else if (GtreeType == 3 || GtreeType == 5) offset = 1;
        else offset = 2;
        break;
    default:
        error2("unexpected initial state!");
    }

    if (zeroProb) return 0;

    Gtree = start + offset;

    if (Gtree < 6) {
        theta12 = theta1;
    } else if (Gtree < 12) {
        theta12 = theta2;
    } else if (Gtree < 17) {
        theta12 = theta1;
    } else if (Gtree < 22) {
        theta12 = theta2;
    } else if (Gtree < 25) {
        theta12 = theta1;
    } else if (Gtree < 28) {
        theta12 = theta2;
    } else {
        theta12 = 0;
    }

    gtt[GtOffsetM0[Gtree]].bTot(b, t01, t);

    prob[topology] = exp(-gtt[GtOffsetM0[Gtree]].T(t[0], t[1], theta12)) * gtt[GtOffsetM0[Gtree]].RJ(t[0], t[1]) / gtt[GtOffsetM0[Gtree]].detJ(t[0], t[1]);

    return prob[topology];
}

double treeProbM2SIM3s(int tree) {
    int start, offset, Gtree, zeroProb = 0;
    double tau1 = para[3];
    double * Q_C1, * Q_C2, * Q_C3, * Pt1, * Pt0, * Ptau1t1, *Ptau1, * Ptau1_C1, * Ptau1_C2, * Ptau1_C3, * work, PG1a, PG1a123=0, PG1a113=0, PG1a223=0;
    double t[2], t01[2], prob[3];
    double *b = data.Bij + tree * 2;
    int initState = data.initState[stree.sptree][tree];
    int topology = data.topology[stree.sptree][tree];
    int GtreeType = data.GtreeType[tree];
    BTEntry *gtt = com.GtreeTab[0];

    switch (initState)
    {
    case 0:
        start = 0;
        offset = GtreeType - 1;
        break;
    case 13:
        start = 6;
        offset = GtreeType - 1;
        break;
    case 1:
        start = 12;
        offset = GtreeType - 1;
        break;
    case 12:
        start = 18;
        offset = GtreeType - 1;
        break;
    case 2:
        start = 24;
        if (GtreeType == 3 && !topology) offset = 0;
        else if ((GtreeType == 5 && !topology) || GtreeType == 6) offset = GtreeType - 4;
        else zeroProb = 1;
        break;
    case 14:
        start = 27;
        if (GtreeType == 3 && !topology) offset = 0;
        else if ((GtreeType == 5 && !topology) || GtreeType == 6) offset = GtreeType - 4;
        else zeroProb = 1;
        break;
    case 5:
        start = 30;
        if (GtreeType == 3 && !topology) offset = 0;
        else if ((GtreeType == 5 && !topology) || GtreeType == 6) offset = GtreeType - 4;
        else zeroProb = 1;
        break;
    case 8:
    case 17:
        start = 33;
        if ((GtreeType == 3 || GtreeType == 5) && topology == 2) offset = 0;
        else if (GtreeType == 6) offset = 1;
        else zeroProb = 1;
        break;
    case 26:
        start = 35;
        if (GtreeType == 1 || GtreeType == 2 || GtreeType == 4) offset = 0;
        else if (GtreeType == 3 || GtreeType == 5) offset = 1;
        else offset = 2;
        break;
    default:
        error2("unexpected initial state!");
    }

    if (zeroProb) return 0;

    Gtree = start + offset;

    Q_C3 = com.space;
    Ptau1_C3 = &Q_C3[C3*C3]; Q_C1 = &Ptau1_C3[C3*C3];
    Ptau1_C1 = &Q_C1[C1*C1]; Q_C2 = &Ptau1_C1[C1*C1];
    Ptau1_C2 = &Q_C2[C2*C2];

#ifdef _OPENMP
    if ((Pt1 = (double*)malloc((6 * C2 * C2) * sizeof(double))) == NULL)
        error2("oom allocating Pt1 in treeProbM2SIM3s");
#else
    Pt1 = &Ptau1_C2[C2*C2];
#endif

    Pt0 = &Pt1[C2*C2];
    Ptau1t1 = &Pt0[C2*C2];
    work = &Ptau1t1[C2*C2];

    PG1a113 = Ptau1_C3[3];
    PG1a123 = Ptau1_C3[1 * C3 + 3];
    PG1a223 = Ptau1_C3[2 * C3 + 3];

    initState = GtreeMapM2[Gtree];

    prob[0] = 1;

    gtt[GtOffsetM2[Gtree]].bTot(b, t01, t);

    if (Gtree < 24) {
        // create transition probability matrices
        if (Gtree < 12) { // chain 1
            computePMatrix(t01[0], Q_C1, C1, Pt0, work);
            computePMatrix(t01[1], Q_C1, C1, Pt1, work);
            computePMatrix(tau1-t01[1], Q_C1, C1, Ptau1t1, work);
            Ptau1 = Ptau1_C1;
        } else { // chain 2
            computePMatrix(t01[0], Q_C2, C2, Pt0, work);
            computePMatrix(t01[1], Q_C2, C2, Pt1, work);
            computePMatrix(tau1-t01[1], Q_C2, C2, Ptau1t1, work);
            Ptau1 = Ptau1_C2;
        }
        prob[1] = prob[2] = prob[0];
        gtt[GtOffsetM2[Gtree]].f(t[0], t[1], initState, Pt0, Pt1, Ptau1, Ptau1t1, prob);
    } else if(Gtree < 33) { // chain 3
        PG1a = (Gtree < 27) ? PG1a113 : ((Gtree < 30) ? PG1a223 : PG1a123);
        if (!(Gtree % 3)) {
                computePMatrix(t01[1], Q_C3, C3, Pt1, work);
                gtt[GtOffsetM2[Gtree]].f(t[0], t[1], initState, NULL, Pt1, NULL, NULL, prob);
        } else {
            prob[0] *= (1-PG1a);
            gtt[GtOffsetM2[Gtree]].f(t[0], t[1], initState, NULL, NULL, NULL, NULL, prob);
        }
    } else { // chain 4 (like M0)
        prob[0] *= exp(-gtt[GtOffsetM2[Gtree]].T(t[0], t[1], 0)) * gtt[GtOffsetM2[Gtree]].RJ(t[0], t[1]);
        prob[1] = prob[2] = prob[0];
    }

    prob[topology] /= gtt[GtOffsetM2[Gtree]].detJ(t[0], t[1]);

#ifdef _OPENMP
    free(Pt1);
#endif

    return prob[topology];
}

double treeProbM2Pro(int tree) {
    double tau1 = para[3];
    double *Q_C5, *Q_C6, *Ptau1_C5, *Ptau1_C5_ex, *exptaugap;
    double *Pt_C5, *Pt_C6, *Pr, *work;
    double *P5 = NULL;
    double t[2], t01[2], prob[3];
    double *b = data.Bij + tree * 2;
    int initState = data.initState[stree.sptree][tree];
    int topology = data.topology[stree.sptree][tree];
    int GtreeType = data.GtreeType[tree];
    BTEntry *gtt = com.GtreeTab[0];

    Q_C5 = com.space;
    Q_C6 = &Q_C5[C5 * C5];
    Ptau1_C5 = &Q_C6[C6 * C6];
    Ptau1_C5_ex = &Ptau1_C5[C5 * C5];
    exptaugap = &Ptau1_C5_ex[C5 * C5_ex];

#ifdef _OPENMP
    if ((Pt_C5 = (double*)malloc((C5 * C5 + C6 * C6 + max2(C6, 4) + 3 * square(max2(C5, C6))) * sizeof(double))) == NULL)
        error2("oom allocating Pt_C5 in treeProbM2Pro");
#else
    Pt_C5 = &exptaugap[4];
#endif

    Pt_C6 = &Pt_C5[C5 * C5];
    Pr = &Pt_C6[C6 * C6];
    work = &Pr[max2(C6, 4)];

    gtt[GtreeType - 1].bTot(b, t01, t);

    if (GtreeType <= 3) {
        computePMatrix(t01[1], Q_C5, C5, Pt_C5, work);
        P5 = Pt_C5;
    }
    else
        P5 = Ptau1_C5_ex;

    if (GtreeType == 1)
        computePMatrix(t01[0], Q_C6, C6, Pt_C6, work);
    else if (GtreeType <= 3)
        computePMatrix(tau1 - t01[1], Q_C6, C6, Pt_C6, work);

    gtt[GtreeType - 1].helper(t[0], t[1], Pt_C6, exptaugap + 1, Pr);

    prob[0] = prob[1] = prob[2] = 1;

    gtt[GtreeType - 1].density(initState, P5, NULL, Pr, prob);

    prob[topology] /= gtt[GtreeType - 1].detJ(t[0], t[1]);

#ifdef _OPENMP
    free(Pt_C5);
#endif

    return prob[topology];
}

double treeProbM2ProMax(int tree) {
    double tau0 = para[2];
    double tau1 = para[3];
    double *Q_C5, *Q_C6, *Q_C7, *Q_C8, *Ptau1_C5, *Ptau1_C5_ex, *Ptaugap_C7, *Ptaugap_C7_ex, *Ptaugap_C8, *Ptaugap_C8_ex;
    double *Pt_C5, *Pt_C6, *Pt_C7, *Pt_C8, *Pr, *work;
    double *P5 = NULL, *P8 = NULL;
    double t[2], t01[2], prob[3];
    double *b = data.Bij + tree * 2;
    int initState = data.initState[stree.sptree][tree];
    int topology = data.topology[stree.sptree][tree];
    int GtreeType = data.GtreeType[tree];
    BTEntry *gtt = com.GtreeTab[0];

    Q_C5 = com.space;
    Q_C6 = &Q_C5[C5 * C5];
    Q_C7 = &Q_C6[C6 * C6];
    Q_C8 = &Q_C7[C7 * C7];
    Ptau1_C5 = &Q_C8[C8 * C8];
    Ptau1_C5_ex = &Ptau1_C5[C5 * C5];
    Ptaugap_C7 = &Ptau1_C5_ex[C5 * C5_ex];
    Ptaugap_C7_ex = &Ptaugap_C7[C7 * C7];
    Ptaugap_C8 = &Ptaugap_C7_ex[C7 * C7_ex];
    Ptaugap_C8_ex = &Ptaugap_C8[C8 * C8];

#ifdef _OPENMP
    if ((Pt_C5 = (double*)malloc((C5 * C5 + C6 * C6 + C7 * C7 + C8 * C8 + max2(max2(C6, C8), 1) + 3 * square(max2(max2(max2(C5, C6), C7), C8))) * sizeof(double))) == NULL)
        error2("oom allocating Pt_C5 in treeProbM2ProMax");
#else
    Pt_C5 = &Ptaugap_C8_ex[C8 * C8_ex];
#endif

    Pt_C6 = &Pt_C5[C5 * C5];
    Pt_C7 = &Pt_C6[C6 * C6];
    Pt_C8 = &Pt_C7[C7 * C7];
    Pr = &Pt_C8[C8 * C8];
    work = &Pr[max2(max2(C6, C8), 1)];

    gtt[GtreeType - 1].bTot(b, t01, t);

    if (GtreeType <= 3) {
        computePMatrix(t01[1], Q_C5, C5, Pt_C5, work);
        P5 = Pt_C5;
    }
    else
        P5 = Ptau1_C5_ex;

    if (GtreeType == 1)
        computePMatrix(t01[0], Q_C6, C6, Pt_C6, work);
    else if (GtreeType <= 3)
        computePMatrix(tau1 - t01[1], Q_C6, C6, Pt_C6, work);
    else if (GtreeType <= 5)
        computePMatrix(t01[1], Q_C7, C7, Pt_C7, work);

    if (GtreeType == 2 || GtreeType == 4) {
        computePMatrix(t01[0], Q_C8, C8, Pt_C8, work);
        P8 = Pt_C8;
    }
    else if (GtreeType == 5) {
        computePMatrix(tau0 - tau1 - t01[1], Q_C8, C8, Pt_C8, work);
        P8 = Pt_C8;
    }
    if (GtreeType == 3)
        P8 = Ptaugap_C8_ex;

    gtt[GtreeType - 1].helper(t[0], t[1], Pt_C6, P8, Pr);

    prob[0] = prob[1] = prob[2] = 1;

    gtt[GtreeType - 1].density(initState, P5, Pt_C7, Pr, prob);

    prob[topology] /= gtt[GtreeType - 1].detJ(t[0], t[1]);

#ifdef _OPENMP
    free(Pt_C5);
#endif

    return prob[topology];
}

double treeProbM3MSci12(int tree) {
    int start, offset, Gtree, zeroProb = 0;
    double tau1 = para[3];
    double T = para[7];
    double thetaX = para[8];
    double thetaY = para[9];
    double phi12 = para[10];
    double phi21 = para[11];
    double exptau1T[4], Pr[7];
    double t[2], t01[2], prob[3];
    double *b = data.Bij + tree * 2;
    int initState = data.initState[stree.sptree][tree];
    int topology = data.topology[stree.sptree][tree];
    int GtreeType = data.GtreeType[tree];
    BTEntry *gtt = com.GtreeTab[0];

    if (GtreeType == 1)
        GtreeType += (b[1] > T) ? 7 : ((b[0] + b[1] > T) ? 6 : 0);
    else if (GtreeType == 2 || GtreeType == 3)
        GtreeType += ((b[1] > T) ? 7 : 0);

    switch (initState)
    {
    case 0:
        start = 0;
        offset = GtreeType - 1;
        break;
    case 13:
        start = 10;
        offset = GtreeType - 1;
        break;
    case 1:
        start = 20;
        if ((GtreeType == 2 || GtreeType == 3 || GtreeType == 7) && !topology
            || GtreeType >= 4 && GtreeType <= 6
            || GtreeType >= 8 && GtreeType <= 10)
            offset = GtreeType - 2;
        else zeroProb = 1;
        break;
    case 12:
        start = 29;
        if ((GtreeType == 2 || GtreeType == 3 || GtreeType == 7) && !topology
            || GtreeType >= 4 && GtreeType <= 6
            || GtreeType >= 8 && GtreeType <= 10)
            offset = GtreeType - 2;
        else zeroProb = 1;
        break;
    case 2:
        start = 38;
        if (GtreeType == 3 && !topology) offset = 0;
        else if ((GtreeType == 5 && !topology) || GtreeType == 6) offset = GtreeType - 4;
        else if (GtreeType == 10 && !topology) offset = 3;
        else zeroProb = 1;
        break;
    case 14:
        start = 42;
        if (GtreeType == 3 && !topology) offset = 0;
        else if ((GtreeType == 5 && !topology) || GtreeType == 6) offset = GtreeType - 4;
        else if (GtreeType == 10 && !topology) offset = 3;
        else zeroProb = 1;
        break;
    case 5:
        start = 46;
        if ((GtreeType == 5 && !topology) || GtreeType == 6) offset = GtreeType - 5;
        else if (GtreeType == 10 && !topology) offset = 2;
        else zeroProb = 1;
        break;
    case 8:
    case 17:
        start = 49;
        if ((GtreeType == 3 || GtreeType == 5) && topology == 2) offset = 0;
        else if (GtreeType == 6) offset = 1;
        else zeroProb = 1;
        break;
    case 26:
        start = 51;
        if (GtreeType == 1 || GtreeType == 2 || GtreeType == 4) offset = 0;
        else if (GtreeType == 3 || GtreeType == 5) offset = 1;
        else offset = 2;
        break;
    default:
        error2("unexpected initial state!");
    }

    if (zeroProb) return 0;

    Gtree = start + offset;

    exptau1T[0] = exp(-6 * (tau1 - T) / thetaX);
    exptau1T[1] = exp(-2 * (tau1 - T) / thetaX);
    exptau1T[2] = exp(-2 * (tau1 - T) / thetaY);
    exptau1T[3] = exp(-6 * (tau1 - T) / thetaY);

    if (initState == 0 || initState == 2)
        Pr[0] = (1 - phi21) * (1 - phi21) * exptau1T[1] + 2 * (1 - phi21) * phi21 + phi21 * phi21 * exptau1T[2];
    if (initState == 13 || initState == 14)
        Pr[1] = (1 - phi12) * (1 - phi12) * exptau1T[2] + 2 * (1 - phi12) * phi12 + phi12 * phi12 * exptau1T[1];
    if (initState == 0)
        Pr[2] = (1 - phi21) * (1 - phi21) * (1 - phi21) * exptau1T[0]
              + 3 * (1 - phi21) * (1 - phi21) * phi21 * exptau1T[1]
              + 3 * (1 - phi21) * phi21 * phi21 * exptau1T[2]
              + phi21 * phi21 * phi21 * exptau1T[3];
    if (initState == 13)
        Pr[3] = (1 - phi12) * (1 - phi12) * (1 - phi12) * exptau1T[3]
              + 3 * (1 - phi12) * (1 - phi12) * phi12 * exptau1T[2]
              + 3 * (1 - phi12) * phi12 * phi12 * exptau1T[1]
              + phi12 * phi12 * phi12 * exptau1T[0];
    if (initState == 1 || initState == 12 || initState == 5)
        Pr[4] = (1 - phi21) * phi12 * exptau1T[1] + (1 - phi21) * (1 - phi12) + phi21 * phi12 + phi21 * (1 - phi12) * exptau1T[2];
    if (initState == 1)
        Pr[5] = (1 - phi21) * (1 - phi21) * phi12 * exptau1T[0]
              + (1 - phi21) * ((1 - phi21) * (1 - phi12) + 2 * phi21 * phi12) * exptau1T[1]
              + phi21 * (phi21 * phi12 + 2 * (1 - phi21) * (1 - phi12)) * exptau1T[2]
              + phi21 * phi21 * (1 - phi12) * exptau1T[3];
    if (initState == 12)
        Pr[6] = (1 - phi12) * (1 - phi12) * phi21 * exptau1T[3]
              + (1 - phi12) * ((1 - phi12) * (1 - phi21) + 2 * phi12 * phi21) * exptau1T[2]
              + phi12 * (phi12 * phi21 + 2 * (1 - phi12) * (1 - phi21)) * exptau1T[1]
              + phi12 * phi12 * (1 - phi21) * exptau1T[0];

    gtt[GtOffsetM3MSci12[Gtree]].bTot(b, t01, t);

    prob[0] = prob[1] = prob[2] = 1;

    gtt[GtOffsetM3MSci12[Gtree]].g(t[0], t[1], Pr, prob);

    prob[topology] /= gtt[GtOffsetM3MSci12[Gtree]].detJ(t[0], t[1]);

    return prob[topology];
}

double treeProbM3MSci13(int tree) {
    int start, offset, Gtree, zeroProb = 0;
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetaX = para[8];
    double thetaZ = para[9];
    double phi13 = para[10];
    double phi31 = para[11];
    double exptaugapT[3], exptau0T[2], exptaugap, Pr[11];
    double t[2], t01[2], prob[3];
    double *b = data.Bij + tree * 2;
    int initState = data.initState[stree.sptree][tree];
    int topology = data.topology[stree.sptree][tree];
    int GtreeType = data.GtreeType[tree];
    BTEntry *gtt = com.GtreeTab[0];

    if (GtreeType == 1)
        GtreeType += (b[1] > T) ? 7 : ((b[0] + b[1] > T) ? 6 : 0);
    else if (GtreeType == 2 || GtreeType == 3)
        GtreeType += ((b[1] > T) ? 7 : 0);

    switch (initState)
    {
    case 0:
        start = 0;
        offset = GtreeType - 1;
        break;
    case 13:
        start = 10;
        if (GtreeType >= 1 && GtreeType <= 6) offset = GtreeType - 1;
        else zeroProb = 1;
        break;
    case 1:
        start = 16;
        if ((GtreeType == 2 || GtreeType == 3) && !topology || GtreeType >= 4 && GtreeType <= 6) offset = GtreeType - 2;
        else if ((GtreeType == 9 || GtreeType == 10) && !topology) offset = GtreeType - 4;
        else zeroProb = 1;
        break;
    case 12:
        start = 23;
        if ((GtreeType == 2 || GtreeType == 3) && !topology || GtreeType >= 4 && GtreeType <= 6) offset = GtreeType - 2;
        else zeroProb = 1;
        break;
    case 2:
        start = 28;
        if ((GtreeType == 2 || GtreeType == 3 || GtreeType == 7) && !topology
            || GtreeType >= 4 && GtreeType <= 6
            || GtreeType >= 8 && GtreeType <= 10)
            offset = GtreeType - 2;
        else zeroProb = 1;
        break;
    case 14:
        start = 37;
        if ((GtreeType == 2 || GtreeType == 3) && !topology || GtreeType >= 4 && GtreeType <= 6) offset = GtreeType - 2;
        else zeroProb = 1;
        break;
    case 5:
        start = 42;
        if (GtreeType >= 4 && GtreeType <= 6) offset = GtreeType - 4;
        else if ((GtreeType == 9 || GtreeType == 10) && topology == 1) offset = GtreeType - 6;
        else zeroProb = 1;
        break;
    case 8:
        start = 47;
        if ((GtreeType == 2 || GtreeType == 3 || GtreeType == 7) && topology == 2
            || GtreeType >= 4 && GtreeType <= 6
            || GtreeType >= 8 && GtreeType <= 10)
            offset = GtreeType - 2;
        else zeroProb = 1;
        break;
    case 17:
        start = 56;
        if ((GtreeType == 2 || GtreeType == 3) && topology == 2 || GtreeType >= 4 && GtreeType <= 6) offset = GtreeType - 2;
        else if ((GtreeType == 9 || GtreeType == 10) && topology == 2) offset = GtreeType - 4;
        else zeroProb = 1;
        break;
    case 26:
        start = 63;
        offset = GtreeType - 1;
        break;
    default:
        error2("unexpected initial state!");
    }

    if (zeroProb) return 0;

    Gtree = start + offset;

    exptaugapT[0] = exp(-2 * (tau1 - T) / thetaX - 2 * (tau0 - tau1) / theta5);
    exptaugapT[1] = exp(-6 * (tau1 - T) / thetaX - 6 * (tau0 - tau1) / theta5);
    exptaugapT[2] = exp(-2 * (tau1 - T) / thetaX - 6 * (tau0 - tau1) / theta5);

    exptau0T[0] = exp(-2 * (tau0 - T) / thetaZ);
    exptau0T[1] = exp(-6 * (tau0 - T) / thetaZ);

    exptaugap = exp(-2 * (tau0 - tau1) / theta5);

    if (initState == 0) {
        Pr[0] = (1 - phi31) * (1 - phi31) * exptaugapT[0] + 2 * (1 - phi31) * phi31 + phi31 * phi31 * exptau0T[0];
        Pr[1] = (1 - phi31) * (1 - phi31) * (1 - phi31) * exptaugapT[1]
              + 3 * (1 - phi31) * (1 - phi31) * phi31 * exptaugapT[0]
              + 3 * (1 - phi31) * phi31 * phi31 * exptau0T[0]
              + phi31 * phi31 * phi31 * exptau0T[1];
    }
    else if (initState == 1) {
        Pr[2] = (1 - phi31) * (1 - phi31) * exptaugapT[2] + 2 * (1 - phi31) * phi31 * exptaugap + phi31 * phi31 * exptau0T[0];
    }
    else if (initState == 2) {
        Pr[3] = (1 - phi31) * phi13 * exptaugapT[0] + (1 - phi31) * (1 - phi13) + phi31 * phi13 + phi31 * (1 - phi13) * exptau0T[0];
        Pr[4] = (1 - phi31) * (1 - phi31) * phi13 * exptaugapT[1]
              + (1 - phi31) * ((1 - phi31) * (1 - phi13) + 2 * phi31 * phi13) * exptaugapT[0]
              + phi31 * (phi31 * phi13 + 2 * (1 - phi31) * (1 - phi13)) * exptau0T[0]
              + phi31 * phi31 * (1 - phi13) * exptau0T[1];
    }
    else if (initState == 5) {
        Pr[5] = (1 - phi31) * phi13 * exptaugapT[2] + ((1 - phi31) * (1 - phi13) + phi31 * phi13) * exptaugap + phi31 * (1 - phi13) * exptau0T[0];
    }
    else if (initState == 8) {
        Pr[6] = phi13 * (1 - phi31) * exptaugapT[0] + phi13 * phi31 + (1 - phi13) * (1 - phi31) + (1 - phi13) * phi31 * exptau0T[0];
        Pr[7] = phi13 * phi13 * (1 - phi31) * exptaugapT[1]
              + phi13 * (phi13 * phi31 + 2 * (1 - phi13) * (1 - phi31)) * exptaugapT[0]
              + (1 - phi13) * ((1 - phi13) * (1 - phi31) + 2 * phi13 * phi31) * exptau0T[0]
              + (1 - phi13) * (1 - phi13) * phi31 * exptau0T[1];
    }
    else if (initState == 17) {
        Pr[8] = phi13 * phi13 * exptaugapT[2] + 2 * phi13 * (1 - phi13) * exptaugap + (1 - phi13) * (1 - phi13) * exptau0T[0];
    }
    else if (initState == 26) {
        Pr[9] = phi13 * phi13 * exptaugapT[0] + 2 * phi13 * (1 - phi13) + (1 - phi13) * (1 - phi13) * exptau0T[0];
        Pr[10] = phi13 * phi13 * phi13 * exptaugapT[1]
               + 3 * phi13 * phi13 * (1 - phi13) * exptaugapT[0]
               + 3 * phi13 * (1 - phi13) * (1 - phi13) * exptau0T[0]
               + (1 - phi13) * (1 - phi13) * (1 - phi13) * exptau0T[1];
    }

    gtt[GtOffsetM3MSci13[Gtree]].bTot(b, t01, t);

    prob[0] = prob[1] = prob[2] = 1;

    gtt[GtOffsetM3MSci13[Gtree]].g(t[0], t[1], Pr, prob);

    prob[topology] /= gtt[GtOffsetM3MSci13[Gtree]].detJ(t[0], t[1]);

    return prob[topology];
}

double treeProbM3MSci23(int tree) {
    int start, offset, Gtree, zeroProb = 0;
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetaY = para[8];
    double thetaZ = para[9];
    double phi23 = para[10];
    double phi32 = para[11];
    double exptaugapT[3], exptau0T[2], exptaugap, Pr[11];
    double t[2], t01[2], prob[3];
    double *b = data.Bij + tree * 2;
    int initState = data.initState[stree.sptree][tree];
    int topology = data.topology[stree.sptree][tree];
    int GtreeType = data.GtreeType[tree];
    BTEntry *gtt = com.GtreeTab[0];

    if (GtreeType == 1)
        GtreeType += (b[1] > T) ? 7 : ((b[0] + b[1] > T) ? 6 : 0);
    else if (GtreeType == 2 || GtreeType == 3)
        GtreeType += ((b[1] > T) ? 7 : 0);

    switch (initState)
    {
    case 13:
        start = 6;
        offset = GtreeType - 1;
        break;
    case 0:
        start = 0;
        if (GtreeType >= 1 && GtreeType <= 6) offset = GtreeType - 1;
        else zeroProb = 1;
        break;
    case 12:
        start = 21;
        if ((GtreeType == 2 || GtreeType == 3) && !topology || GtreeType >= 4 && GtreeType <= 6) offset = GtreeType - 2;
        else if ((GtreeType == 9 || GtreeType == 10) && !topology) offset = GtreeType - 4;
        else zeroProb = 1;
        break;
    case 1:
        start = 16;
        if ((GtreeType == 2 || GtreeType == 3) && !topology || GtreeType >= 4 && GtreeType <= 6) offset = GtreeType - 2;
        else zeroProb = 1;
        break;
    case 14:
        start = 33;
        if ((GtreeType == 2 || GtreeType == 3 || GtreeType == 7) && !topology
            || GtreeType >= 4 && GtreeType <= 6
            || GtreeType >= 8 && GtreeType <= 10)
            offset = GtreeType - 2;
        else zeroProb = 1;
        break;
    case 2:
        start = 28;
        if ((GtreeType == 2 || GtreeType == 3) && !topology || GtreeType >= 4 && GtreeType <= 6) offset = GtreeType - 2;
        else zeroProb = 1;
        break;
    case 5:
        start = 42;
        if (GtreeType >= 4 && GtreeType <= 6) offset = GtreeType - 4;
        else if ((GtreeType == 9 || GtreeType == 10) && topology == 2) offset = GtreeType - 6;
        else zeroProb = 1;
        break;
    case 17:
        start = 54;
        if ((GtreeType == 2 || GtreeType == 3 || GtreeType == 7) && topology == 2
            || GtreeType >= 4 && GtreeType <= 6
            || GtreeType >= 8 && GtreeType <= 10)
            offset = GtreeType - 2;
        else zeroProb = 1;
        break;
    case 8:
        start = 47;
        if ((GtreeType == 2 || GtreeType == 3) && topology == 2 || GtreeType >= 4 && GtreeType <= 6) offset = GtreeType - 2;
        else if ((GtreeType == 9 || GtreeType == 10) && topology == 2) offset = GtreeType - 4;
        else zeroProb = 1;
        break;
    case 26:
        start = 63;
        offset = GtreeType - 1;
        break;
    default:
        error2("unexpected initial state!");
    }

    if (zeroProb) return 0;

    Gtree = start + offset;

    exptaugapT[0] = exp(-2 * (tau1 - T) / thetaY - 2 * (tau0 - tau1) / theta5);
    exptaugapT[1] = exp(-6 * (tau1 - T) / thetaY - 6 * (tau0 - tau1) / theta5);
    exptaugapT[2] = exp(-2 * (tau1 - T) / thetaY - 6 * (tau0 - tau1) / theta5);

    exptau0T[0] = exp(-2 * (tau0 - T) / thetaZ);
    exptau0T[1] = exp(-6 * (tau0 - T) / thetaZ);

    exptaugap = exp(-2 * (tau0 - tau1) / theta5);

    if (initState == 13) {
        Pr[0] = (1 - phi32) * (1 - phi32) * exptaugapT[0] + 2 * (1 - phi32) * phi32 + phi32 * phi32 * exptau0T[0];
        Pr[1] = (1 - phi32) * (1 - phi32) * (1 - phi32) * exptaugapT[1]
              + 3 * (1 - phi32) * (1 - phi32) * phi32 * exptaugapT[0]
              + 3 * (1 - phi32) * phi32 * phi32 * exptau0T[0]
              + phi32 * phi32 * phi32 * exptau0T[1];
    }
    else if (initState == 12) {
        Pr[2] = (1 - phi32) * (1 - phi32) * exptaugapT[2] + 2 * (1 - phi32) * phi32 * exptaugap + phi32 * phi32 * exptau0T[0];
    }
    else if (initState == 14) {
        Pr[3] = (1 - phi32) * phi23 * exptaugapT[0] + (1 - phi32) * (1 - phi23) + phi32 * phi23 + phi32 * (1 - phi23) * exptau0T[0];
        Pr[4] = (1 - phi32) * (1 - phi32) * phi23 * exptaugapT[1]
              + (1 - phi32) * ((1 - phi32) * (1 - phi23) + 2 * phi32 * phi23) * exptaugapT[0]
              + phi32 * (phi32 * phi23 + 2 * (1 - phi32) * (1 - phi23)) * exptau0T[0]
              + phi32 * phi32 * (1 - phi23) * exptau0T[1];
    }
    else if (initState == 5) {
        Pr[5] = (1 - phi32) * phi23 * exptaugapT[2] + ((1 - phi32) * (1 - phi23) + phi32 * phi23) * exptaugap + phi32 * (1 - phi23) * exptau0T[0];
    }
    else if (initState == 17) {
        Pr[6] = phi23 * (1 - phi32) * exptaugapT[0] + phi23 * phi32 + (1 - phi23) * (1 - phi32) + (1 - phi23) * phi32 * exptau0T[0];
        Pr[7] = phi23 * phi23 * (1 - phi32) * exptaugapT[1]
              + phi23 * (phi23 * phi32 + 2 * (1 - phi23) * (1 - phi32)) * exptaugapT[0]
              + (1 - phi23) * ((1 - phi23) * (1 - phi32) + 2 * phi23 * phi32) * exptau0T[0]
              + (1 - phi23) * (1 - phi23) * phi32 * exptau0T[1];
    }
    else if (initState == 8) {
        Pr[8] = phi23 * phi23 * exptaugapT[2] + 2 * phi23 * (1 - phi23) * exptaugap + (1 - phi23) * (1 - phi23) * exptau0T[0];
    }
    else if (initState == 26) {
        Pr[9] = phi23 * phi23 * exptaugapT[0] + 2 * phi23 * (1 - phi23) + (1 - phi23) * (1 - phi23) * exptau0T[0];
        Pr[10] = phi23 * phi23 * phi23 * exptaugapT[1]
               + 3 * phi23 * phi23 * (1 - phi23) * exptaugapT[0]
               + 3 * phi23 * (1 - phi23) * (1 - phi23) * exptau0T[0]
               + (1 - phi23) * (1 - phi23) * (1 - phi23) * exptau0T[1];
    }

    gtt[GtOffsetM3MSci23[Gtree]].bTot(b, t01, t);

    prob[0] = prob[1] = prob[2] = 1;

    gtt[GtOffsetM3MSci23[Gtree]].g(t[0], t[1], Pr, prob);

    prob[topology] /= gtt[GtOffsetM3MSci23[Gtree]].detJ(t[0], t[1]);

    return prob[topology];
}


#define DEFINE_WRAPPER(wrapper, function, i) \
void wrapper(double x0, double x1, double* Pr, double wwprior[3]) { \
    function(i, x0, x1, Pr, wwprior); \
} \


// Variable transformations

static inline void t0t1_type_1(double x0, double x1, double theta, double t[2]) {
    t[1] = theta * x0 * x1 / 2;
    t[0] = theta * x0 * (1 - x1) / 2;
}

static inline void t0t1_type_2(double x0, double x1, double thetai, double thetaj, double t[2]) {
    t[1] = thetai * x1 / 2;
    t[0] = thetaj * x0 / 2;
}

static inline void t0t1_type_3(double x0, double x1, double theta, double t[2]) {
    t[1] = theta * x1 / 2;
    t[0] = theta * x0 / 2;
}

static inline void t0t1_type_4(double x0, double x1, double thetai, double thetaj, double t[2]) {
    double T = para[7];
    t[1] = thetai * x1 / 2;
    t[0] = thetaj * x0 / 2 + T - t[1];
}

static inline void t0t1_type_5(double x0, double x1, double theta, double t[2]) {
    double T = para[7];
    t[1] = theta * x0 * x1 / 2 + T;
    t[0] = theta * x0 * (1 - x1) / 2;
}

static inline void t0t1_type_6(double x0, double x1, double thetai, double thetaj, double t[2]) {
    double T = para[7];
    t[1] = thetai * x1 / 2 + T;
    t[0] = thetaj * x0 / 2;
}

static inline double detJ_type_1(double x0, double theta) {
    return x0 * theta * theta / 4;
}

static inline double detJ_type_2(double thetai, double thetaj) {
    return thetai * thetaj / 4;
}

static inline double detJ_type_3(double theta) {
    return theta * theta / 4;
}

static inline void b_type_1(double x0, double x1, double theta, double b[2]) {
    b[1] = theta * x0 * x1 / 2;
    b[0] = theta * x0 * (1 - x1) / 2;
}

static inline void b_type_2(double x0, double x1, double thetai, double thetaj, double b[2]) {
    double tau1 = para[3];
    b[1] = thetai * x1 / 2;
    b[0] = thetaj * x0 / 2 + tau1 - b[1];
}

static inline void b_type_3(double x0, double x1, double thetai, double thetaj, double b[2]) {
    double tau0 = para[2];
    b[1] = thetai * x1 / 2;
    b[0] = thetaj * x0 / 2 + tau0 - b[1];
}

static inline void b_type_4(double x0, double x1, double theta, double b[2]) {
    double tau1 = para[3];
    b[1] = theta * x0 * x1 / 2 + tau1;
    b[0] = theta * x0 * (1 - x1) / 2;
}

static inline void b_type_5(double x0, double x1, double thetai, double thetaj, double b[2]) {
    double tau0 = para[2];
    double tau1 = para[3];
    b[1] = thetai * x1 / 2 + tau1;
    b[0] = thetaj * x0 / 2 + tau0 - b[1];
}

static inline void b_type_6(double x0, double x1, double theta, double b[2]) {
    double tau0 = para[2];
    b[1] = theta * x1 / 2 + tau0;
    b[0] = theta * x0 / 2;
}

static inline void b_type_7(double x0, double x1, double thetai, double thetaj, double b[2]) {
    double T = para[7];
    b[1] = thetai * x1 / 2;
    b[0] = thetaj * x0 / 2 + T - b[1];
}

static inline void b_type_8(double x0, double x1, double theta, double b[2]) {
    double T = para[7];
    b[1] = theta * x0 * x1 / 2 + T;
    b[0] = theta * x0 * (1 - x1) / 2;
}

static inline void b_type_9(double x0, double x1, double thetai, double thetaj, double b[2]) {
    double tau1 = para[3];
    double T = para[7];
    b[1] = thetai * x1 / 2 + T;
    b[0] = thetaj * x0 / 2 + tau1 - b[1];
}

static inline void b_type_10(double x0, double x1, double thetai, double thetaj, double b[2]) {
    double tau0 = para[2];
    double T = para[7];
    b[1] = thetai * x1 / 2 + T;
    b[0] = thetaj * x0 / 2 + tau0 - b[1];
}

static inline void bTot_type_1(double b[2], double theta, double t[2], double x[2]) {
    t[1] = b[1];
    t[0] = b[0];
    x[1] = t[1] / (t[0] + t[1]);
    x[0] = 2 * (t[0] + t[1]) / theta;
}

static inline void bTot_type_2(double b[2], double thetai, double thetaj, double t[2], double x[2]) {
    double tau1 = para[3];
    t[1] = b[1];
    t[0] = b[0] + b[1] - tau1;
    x[1] = 2 * t[1] / thetai;
    x[0] = 2 * t[0] / thetaj;
}

static inline void bTot_type_3(double b[2], double thetai, double thetaj, double t[2], double x[2]) {
    double tau0 = para[2];
    t[1] = b[1];
    t[0] = b[0] + b[1] - tau0;
    x[1] = 2 * t[1] / thetai;
    x[0] = 2 * t[0] / thetaj;
}

static inline void bTot_type_4(double b[2], double theta, double t[2], double x[2]) {
    double tau1 = para[3];
    t[1] = b[1] - tau1;
    t[0] = b[0];
    x[1] = t[1] / (t[0] + t[1]);
    x[0] = 2 * (t[0] + t[1]) / theta;
}

static inline void bTot_type_5(double b[2], double thetai, double thetaj, double t[2], double x[2]) {
    double tau0 = para[2];
    double tau1 = para[3];
    t[1] = b[1] - tau1;
    t[0] = b[0] + b[1] - tau0;
    x[1] = 2 * t[1] / thetai;
    x[0] = 2 * t[0] / thetaj;
}

static inline void bTot_type_6(double b[2], double theta, double t[2], double x[2]) {
    double tau0 = para[2];
    t[1] = b[1] - tau0;
    t[0] = b[0];
    x[1] = 2 * t[1] / theta;
    x[0] = 2 * t[0] / theta;
}

static inline void bTot_type_7(double b[2], double thetai, double thetaj, double t[2], double x[2]) {
    double T = para[7];
    t[1] = b[1];
    t[0] = b[0];
    x[1] = 2 * t[1] / thetai;
    x[0] = 2 * (t[0] + t[1] - T) / thetaj;
}

static inline void bTot_type_8(double b[2], double theta, double t[2], double x[2]) {
    double T = para[7];
    t[1] = b[1];
    t[0] = b[0];
    x[1] = (t[1] - T) / (t[0] + t[1] - T);
    x[0] = 2 * (t[0] + t[1] - T) / theta;
}

static inline void bTot_type_9(double b[2], double thetai, double thetaj, double t[2], double x[2]) {
    double tau1 = para[3];
    double T = para[7];
    t[1] = b[1];
    t[0] = b[0] + b[1] - tau1;
    x[1] = 2 * (t[1] - T) / thetai;
    x[0] = 2 * t[0] / thetaj;
}

static inline void bTot_type_10(double b[2], double thetai, double thetaj, double t[2], double x[2]) {
    double tau0 = para[2];
    double T = para[7];
    t[1] = b[1];
    t[0] = b[0] + b[1] - tau0;
    x[1] = 2 * (t[1] - T) / thetai;
    x[0] = 2 * t[0] / thetaj;
}


// Density functions for M0 and M2SIM3s

// exponents of prior likelihoods for M0 (see Table S2 in Dalquen et al. 2016)
double T_G1_111(double x0, double x1, double theta12) {
    return 2*x0*x1 + x0;
}

double T_G2_111(double x0, double x1, double theta12) {
    return 2*x1 + x0 + 2/theta12*para[3];
}

double T_G3_111(double x0, double x1, double theta12) {
    return 2*x1 + x0 + 2*para[3]/theta12 + 2*(para[2] - para[3])/para[1];
}

double T_G4_111(double x0, double x1, double theta12) {
    return 6/theta12*para[3] + 2*x0*x1 + x0;
}

double T_G5_111(double x0, double x1, double theta12) {
    return 2*x1 + x0 + 2*(para[2] - para[3])/para[1] + 6*para[3]/theta12;
}

double T_G6_111(double x0, double x1, double theta12) {
    return 3*x1 + x0 + 6*para[3]/theta12 + 6*(para[2] - para[3])/para[1];
}

double T_G2_112(double x0, double x1, double theta12) {
    return x1 + x0;
}

double T_G3_112(double x0, double x1, double theta12) {
    return T_G2_112(x0, x1, theta12) + 2/para[1]*(para[2] - para[3]);
}

double T_G4_112(double x0, double x1, double theta12) {
    return T_G1_111(x0, x1, theta12) + 2/theta12*para[3];
}

double T_G5_112(double x0, double x1, double theta12) {
    return T_G3_111(x0, x1, theta12);
}

double T_G6_112(double x0, double x1, double theta12) {
    return 3*x1 + x0 + 2/theta12*para[3] + 6/para[1]*(para[2] - para[3]);
}

double T_G3_113(double x0, double x1, double theta12) {
    return T_G2_112(x0, x1, theta12);
}

double T_G5_113(double x0, double x1, double theta12) {
    return T_G2_112(x0, x1, theta12) + 2/theta12*para[3];
}

double T_G6_113(double x0, double x1, double theta12) {
    return 3*x1 + x0 + 2/theta12*para[3] + 2/para[1]*(para[2] - para[3]);
}

double T_G5_123(double x0, double x1, double theta12) {
    return T_G2_112(x0, x1, theta12);
}

double T_G6_123(double x0, double x1, double theta12) {
    return 3*x1 + x0 + 2/para[1]*(para[2] - para[3]);
}

double T_G35_133(double x0, double x1, double theta12) {
    return T_G2_112(x0, x1, theta12);
}

double T_G6_133(double x0, double x1, double theta12) {
    return 3*x1 + x0 + 2/para[6]*para[2];
}

double T_G124_333(double x0, double x1, double theta12) {
    return T_G1_111(x0, x1, theta12);
}

double T_G35_333(double x0, double x1, double theta12) {
    return 2*x1 + x0 + 2/para[6]*para[2];
}

double T_G6_333(double x0, double x1, double theta12) {
    return 3*x1 + x0 + 6/para[6]*para[2];
}

// rate/jacobi factor for M0
double RJ_normal(double x0, double x1) {
    return 1;
}

double RJ_special(double x0, double x1) {
    return x0;
}

// branch lengths for different initial states (see Dalquen et al. 2016)
void b_G1_111(double x0, double x1, double theta12, double b[2]) {
    b[0] = theta12*x0*(1-x1)/2;
    b[1] = theta12*x0*x1/2;
}

void b_G2_111(double x0, double x1, double theta12, double b[2]) {
    b[0] = (para[1]*x0 - theta12*x1)/2 + para[3];
    b[1] = theta12*x1/2;
}

void b_G3_111(double x0, double x1, double theta12, double b[2]) {
    b[0] = (para[0]*x0 - theta12*x1)/2 + para[2];
    b[1] = theta12*x1/2;
}

void b_G4_111(double x0, double x1, double theta12, double b[2]) {
    b[0] = para[1]*x0*(1-x1)/2;
    b[1] = para[1]*x0*x1/2 + para[3];
}

void b_G5_111(double x0, double x1, double theta12, double b[2]) {
    b[0] = (para[0]*x0 - para[1]*x1)/2 + para[2] - para[3];
    b[1] = para[1]*x1/2 + para[3];
}

void b_G6_111(double x0, double x1, double theta12, double b[2]) {
    b[0] = para[0]*x0/2;
    b[1] = para[0]*x1/2 + para[2];
}

void b_G1_333(double x0, double x1, double theta12, double b[2]) {
    b[0] = para[6]*x0*(1-x1)/2;
    b[1] = para[6]*x0*x1/2;
}

void b_G3_333(double x0, double x1, double theta12, double b[2]) {
    b[0] = (para[0]*x0 - para[6]*x1)/2 + para[2];
    b[1] = para[6]*x1/2;
}

// for M2, we have different functions to compute the gene tree probabilities
void f_G1_111(double x0, double x1, int s, double * Pt0, double * Pt1, double * Ptau1, double * Ptau1t1, double wwprior[3]) {
    double theta1r = 1/para[4], theta2r = 1/para[5];
//    double theta12 = 2/theta1r + 2/theta2r;
    double theta12 = (para[4]+para[5])/2;
    double f = (3*theta1r*Pt1[C1*s]*(theta1r*Pt0[C1*4+4]+theta2r*Pt0[C1*4+6])
                + (theta1r*Pt1[C1*s+1]+theta2r*Pt1[C1*s+2])*(theta1r*Pt0[C1*5+4]+theta2r*Pt0[C1*5+6])
                + 3*theta2r*Pt1[C1*s+3]*(theta1r*Pt0[C1*6+4]+theta2r*Pt0[C1*6+6])
               ) * (theta12*theta12)*x0/3;
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

void f_G2_111(double x0, double x1, int s, double * Pt0, double * Pt1, double * Ptau1, double * Ptau1t1, double wwprior[3]) {
    double theta1r = 1/para[4], theta2r = 1/para[5];
//    double theta12 = 2/theta1r + 2/theta2r;
    double theta12 = (para[4]+para[5])/2;
    double f = (3*theta1r*Pt1[C1*s]*(Ptau1t1[C1*4+4]+Ptau1t1[C1*4+5]+Ptau1t1[C1*4+6])
                + (theta1r*Pt1[C1*s+1]+theta2r*Pt1[C1*s+2])*(Ptau1t1[C1*5+4]+Ptau1t1[C1*5+5]+Ptau1t1[C1*5+6])
                + 3*theta2r*Pt1[C1*s+3]*(Ptau1t1[C1*6+4]+Ptau1t1[C1*6+5]+Ptau1t1[C1*6+6])
               ) * theta12*exp(-x0)/3;
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

void f_G3_111(double x0, double x1, int s, double * Pt0, double * Pt1, double * Ptau1, double * Ptau1t1, double wwprior[3]) {
    double theta1r = 1/para[4], theta2r = 1/para[5];
//    double theta12 = 2/theta1r + 2/theta2r;
    double theta12 = (para[4]+para[5])/2;
    double f = (3*theta1r*Pt1[C1*s]*(Ptau1t1[C1*4+4]+Ptau1t1[C1*4+5]+Ptau1t1[C1*4+6])
                + (theta1r*Pt1[C1*s+1]+theta2r*Pt1[C1*s+2])*(Ptau1t1[C1*5+4]+Ptau1t1[C1*5+5]+Ptau1t1[C1*5+6])
                + 3*theta2r*Pt1[C1*s+3]*(Ptau1t1[C1*6+4]+Ptau1t1[C1*6+5]+Ptau1t1[C1*6+6])
               ) * theta12*exp(-(x0+2/para[1]*(para[2]-para[3])))/3;
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

void f_G4_111(double x0, double x1, int s, double * Pt0, double * Pt1, double * Ptau1, double * Ptau1t1, double wwprior[3]) {
    double f = (Ptau1[C1*s]+Ptau1[C1*s+1]+Ptau1[C1*s+2]+Ptau1[C1*s+3])*exp(-(2*x0*x1+x0))*x0;
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

void f_G5_111(double x0, double x1, int s, double * Pt0, double * Pt1, double * Ptau1, double * Ptau1t1, double wwprior[3]) {
    double f = (Ptau1[C1*s]+Ptau1[C1*s+1]+Ptau1[C1*s+2]+Ptau1[C1*s+3])*exp(-(2*x1+x0+2/para[1]*(para[2]-para[3])));
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

void f_G6_111(double x0, double x1, int s, double * Pt0, double * Pt1, double * Ptau1, double * Ptau1t1, double wwprior[3]) {
    double f = (Ptau1[C1*s]+Ptau1[C1*s+1]+Ptau1[C1*s+2]+Ptau1[C1*s+3])*exp(-(x0+3*x1+6/para[1]*(para[2]-para[3])));
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

void f_G1_112(double x0, double x1, int s, double * Pt0, double * Pt1, double * Ptau1, double * Ptau1t1, double wwprior[3]) {
    double theta1r = 1/para[4], theta2r = 1/para[5];
//    double theta12 = 2/theta1r + 2/theta2r;
    double theta12 = (para[4]+para[5])/2;
    wwprior[0] *= (theta1r*Pt1[C2*s]*(theta1r*Pt0[C2*10+10]+theta2r*Pt0[C2*10+19])
                   + theta1r*Pt1[C2*s+1]*(theta1r*Pt0[C2*13+10]+theta2r*Pt0[C2*13+19])
                   + theta2r*Pt1[C2*s+6]*(theta1r*Pt0[C2*16+10]+theta2r*Pt0[C2*16+19])
                   + theta2r*Pt1[C2*s+7]*(theta1r*Pt0[C2*19+10]+theta2r*Pt0[C2*19+19]))*(theta12*theta12)*x0;
    wwprior[1] *= (theta1r*Pt1[C2*s]*(theta1r*Pt0[C2* 9+ 9]+theta2r*Pt0[C2* 9+18])
                   + theta1r*Pt1[C2*s+2]*(theta1r*Pt0[C2*12+ 9]+theta2r*Pt0[C2*12+18])
                   + theta2r*Pt1[C2*s+5]*(theta1r*Pt0[C2*15+ 9]+theta2r*Pt0[C2*15+18])
                   + theta2r*Pt1[C2*s+7]*(theta1r*Pt0[C2*18+ 9]+theta2r*Pt0[C2*18+18]))*(theta12*theta12)*x0;
    wwprior[2] *= (theta1r*Pt1[C2*s]*(theta1r*Pt0[C2* 8+ 8]+theta2r*Pt0[C2* 8+17])
                   + theta1r*Pt1[C2*s+3]*(theta1r*Pt0[C2*11+ 8]+theta2r*Pt0[C2*11+17])
                   + theta2r*Pt1[C2*s+4]*(theta1r*Pt0[C2*14+ 8]+theta2r*Pt0[C2*14+17])
                   + theta2r*Pt1[C2*s+7]*(theta1r*Pt0[C2*17+ 8]+theta2r*Pt0[C2*17+17]))*(theta12*theta12)*x0;
}

void f_G2_112(double x0, double x1, int s, double * Pt0, double * Pt1, double * Ptau1, double * Ptau1t1, double wwprior[3]) {
    int i, j;
    double f1, f2, f3, f4;
    double theta1r = 1/para[4], theta2r = 1/para[5];
//    double theta12 = 2/theta1r + 2/theta2r;
    double theta12 = (para[4]+para[5])/2;
    for (j=0; j<3; j++) {
        f1 = f2 = f3 = f4 = 0;
        for (i=8; i < 20; i++) {
            f1 += Ptau1t1[C2*(10-j)+i];
            f2 += Ptau1t1[C2*(13-j)+i];
            f3 += Ptau1t1[C2*(16-j)+i];
            f4 += Ptau1t1[C2*(19-j)+i];
        }
        wwprior[j] *= (theta1r*Pt1[C2*s]*f1 + theta1r*Pt1[C2*s+(j+1)]*f2 + theta2r*Pt1[C2*s+(6-j)]*f3 + theta2r*Pt1[C2*s+7]*f4) * theta12*exp(-x0);
    }
}

void f_G3_112(double x0, double x1, int s, double * Pt0, double * Pt1, double * Ptau1, double * Ptau1t1, double wwprior[3]) {
    int i, j;
    double f1, f2, f3, f4;
    double theta1r = 1/para[4], theta2r = 1/para[5];
//    double theta12 = 2/theta1r + 2/theta2r;
    double theta12 = (para[4]+para[5])/2;
    for (j=0; j<3; j++) {
        f1 = f2 = f3 = f4 = 0;
        for (i=8; i < 20; i++) {
            f1 += Ptau1t1[C2*(10-j)+i];
            f2 += Ptau1t1[C2*(13-j)+i];
            f3 += Ptau1t1[C2*(16-j)+i];
            f4 += Ptau1t1[C2*(19-j)+i];
        }
        wwprior[j] *= (theta1r*Pt1[C2*s]*f1 + theta1r*Pt1[C2*s+(j+1)]*f2 + theta2r*Pt1[C2*s+(6-j)]*f3 + theta2r*Pt1[C2*s+7]*f4) *theta12* exp(-(x0+2/para[1]*(para[2]-para[3])));
    }
}

void f_G4_112(double x0, double x1, int s, double * Pt0, double * Pt1, double * Ptau1, double * Ptau1t1, double wwprior[3]) {
    int i;
    double f = 0;
    for (i=0; i<8; i++) {
        f += Ptau1[C2*s+i];
    }
    wwprior[0] = wwprior[1] = wwprior[2] *= f * exp(-(2*x0*x1+x0))*x0;
}

void f_G5_112(double x0, double x1, int s, double * Pt0, double * Pt1, double * Ptau1, double * Ptau1t1, double wwprior[3]) {
    int i;
    double f = 0;
    for (i=0; i<8; i++) {
        f += Ptau1[C2*s+i];
    }
    wwprior[0] = wwprior[1] = wwprior[2] *= f * exp(-(2*x1+x0+2/para[1]*(para[2]-para[3])));
}

void f_G6_112(double x0, double x1, int s, double * Pt0, double * Pt1, double * Ptau1, double * Ptau1t1, double wwprior[3]) {
    int i;
    double f = 0;
    for (i=0; i<8; i++) {
        f += Ptau1[C2*s+i];
    }
    wwprior[0] = wwprior[1] = wwprior[2] *= f * exp(-(x0+3*x1+6/para[1]*(para[2]-para[3])));
}

void f_G3_113(double x0, double x1, int s, double * Pt0, double * Pt1, double * Ptau1, double * Ptau1t1, double wwprior[3]) {
    double theta1 = para[4], theta2 = para[5];
    double theta12 = (theta1 + theta2) / 2;
    wwprior[0] *= (2/theta1*Pt1[s*C3] + 2/theta2*Pt1[s*C3+2]) * theta12/2*exp(-x0);
}

void f_G5_113(double x0, double x1, int s, double * Pt0, double * Pt1, double * Ptau1, double * Ptau1t1, double wwprior[3]) {
    wwprior[0] *= exp(-x0-x1);
}

void f_G6_113(double x0, double x1, int s, double * Pt0, double * Pt1, double * Ptau1, double * Ptau1t1, double wwprior[3]) {
    double theta5 = para[1], tau0 = para[2], tau1 = para[3];
    wwprior[2] = wwprior[1] = wwprior[0] *= exp(-2/theta5*(tau0-tau1))*exp(-3*x1-x0);
}

void t0t1_G1_111(double x0, double x1, double t[2]) {
    double theta12 = (para[4] + para[5])/2;
    t[0] = theta12/2 * x0*(1-x1);
    t[1] = theta12/2 * x0*x1;
}

void t0t1_G2_111(double x0, double x1, double t[2]) {
    double theta12 = (para[4] + para[5])/2;
    t[0] = para[1]/2 * x0;
    t[1] = theta12/2 * x1;
}

void t0t1_G3_111(double x0, double x1, double t[2]) {
    double theta12 = (para[4] + para[5])/2;
    t[0] = para[0]/2 * x0;
    t[1] = theta12/2 * x1;
}

void t0t1_G4_111(double x0, double x1, double t[2]) {
    t[0] = para[1]/2 * x0*(1-x1);
    t[1] = para[1]/2 * x0*x1;
}

void t0t1_G5_111(double x0, double x1, double t[2]) {
    t[0] = para[0]/2 * x0;
    t[1] = para[1]/2 * x1;
}

void t0t1_G6_111(double x0, double x1, double t[2]) {
    t[0] = para[0]/2 * x0;
    t[1] = para[0]/2 * x1;
}

void t0t1_G1_333(double x0, double x1, double t[2]) {
    t[0] = para[6]/2 * x0*(1-x1);
    t[1] = para[6]/2 * x0*x1;
}

void t0t1_G3_333(double x0, double x1, double t[2]) {
    t[0] = para[0]/2 * x0;
    t[1] = para[6]/2 * x1;
}


// Density functions for M2Pro and M2ProMax

void helper_G1(double x0, double x1, double* P6, double* exptaugap, double* Pr) {
    double theta1 = para[4];
    double theta2 = para[5];
    double theta3 = para[6];
    double theta123 = para[4] + para[5] + para[6];
    int i;

    for (i = 0; i < C6 - 1; i++) {
        Pr[i] = theta123 * x0 * (
              P6[C6 * i + 0] / theta1
            + P6[C6 * i + 2] / theta2
            + P6[C6 * i + 5] / theta3);
    }
}

void helper_G2(double x0, double x1, double* P6, double* exptaugap, double* Pr) {
    double theta5 = para[1];
    double thetaW = para[7];
    double theta5W = para[1] + para[7];
    double exp0 = exp(-theta5W * x0 / theta5);
    double exp1 = exp(-theta5W * x0 / thetaW);
    double P6_ex[C6_ex - 1];
    int i;

    for (i = 0; i < C6 - 1; i++) {
        P6_ex[0] = P6[C6 * i + 0] + P6[C6 * i + 1] + P6[C6 * i + 2];
        P6_ex[2] = P6[C6 * i + 5];

        Pr[i] = theta5W * (
              P6_ex[0] * exp0 / theta5
            + P6_ex[2] * exp1 / thetaW);
    }
}

void helper_G3(double x0, double x1, double* P6, double* exptaugap, double* Pr) {
    double exp0 = exp(-x0);
    double P6_ex[C6_ex - 1];
    int i;

    for (i = 0; i < C6 - 1; i++) {
        P6_ex[0] = P6[C6 * i + 0] + P6[C6 * i + 1] + P6[C6 * i + 2];
        P6_ex[1] = P6[C6 * i + 3] + P6[C6 * i + 4];
        P6_ex[2] = P6[C6 * i + 5];

        Pr[i] = (P6_ex[0] * exptaugap[0]
               + P6_ex[1]
               + P6_ex[2] * exptaugap[1]
               ) * exp0;
    }
}

void helper_G4(double x0, double x1, double* P6, double* exptaugap, double* Pr) {
    double theta5 = para[1];
    double thetaW = para[7];
    double theta5W = para[1] + para[7];

    Pr[0] = theta5W * x0 * exp(-theta5W * x0 * (2 * x1 + 1) / theta5) / theta5;
    Pr[1] = theta5W * x0 * exp(-theta5W * x0 * (2 * x1 + 1) / thetaW) / thetaW;
}

void helper_G5(double x0, double x1, double* P6, double* exptaugap, double* Pr) {
    double tau0 = para[2];
    double tau1 = para[3];
    double theta5 = para[1];
    double thetaW = para[7];
    double theta5W = para[1] + para[7];

    Pr[0] = exp(-2 * (tau0 - tau1 + theta5W * x1) / theta5 - x0);
    Pr[1] = exp(-theta5W * x1 / theta5 - x0);
    Pr[2] = exp(-theta5W * x1 / thetaW - x0);
    Pr[3] = exp(-2 * (tau0 - tau1 + theta5W * x1) / thetaW - x0);
}

void helper_G6(double x0, double x1, double* P6, double* exptaugap, double* Pr) {
    Pr[0] = exp(-3 * x1 - x0);
}

void helper_G1_PM(double x0, double x1, double* P6, double* P8, double* Pr) {
    double theta1 = para[4];
    double theta2 = para[5];
    double theta3 = para[6];
    double theta123 = para[4] + para[5] + para[6];
    int i;

    for (i = 0; i < C6 - 1; i++) {
        Pr[i] = theta123 * x0 * (
              P6[C6 * i + 0] / theta1
            + P6[C6 * i + 2] / theta2
            + P6[C6 * i + 5] / theta3);
    }
}

void helper_G2_PM(double x0, double x1, double* P6, double* P8, double* Pr) {
    double theta5 = para[1];
    double thetaW = para[7];
    double theta5W = para[1] + para[7];
    double P6_ex[C6_ex - 1];
    int i;

    for (i = 0; i < C6 - 1; i++) {
        P6_ex[0] = P6[C6 * i + 0] + P6[C6 * i + 1] + P6[C6 * i + 2];
        P6_ex[1] = P6[C6 * i + 3] + P6[C6 * i + 4];
        P6_ex[2] = P6[C6 * i + 5];

        Pr[i] = theta5W * ((
              P6_ex[0] * P8[C8 * 0 + 0]
            + P6_ex[1] * P8[C8 * 1 + 0]
            + P6_ex[2] * P8[C8 * 2 + 0]) / theta5 + (
              P6_ex[0] * P8[C8 * 0 + 2]
            + P6_ex[1] * P8[C8 * 1 + 2]
            + P6_ex[2] * P8[C8 * 2 + 2]) / thetaW);
    }
}

void helper_G3_PM(double x0, double x1, double* P6, double* P8, double* Pr) {
    double exp0 = exp(-x0);
    double P6_ex[C6_ex - 1];
    int i;

    for (i = 0; i < C6 - 1; i++) {
        P6_ex[0] = P6[C6 * i + 0] + P6[C6 * i + 1] + P6[C6 * i + 2];
        P6_ex[1] = P6[C6 * i + 3] + P6[C6 * i + 4];
        P6_ex[2] = P6[C6 * i + 5];

        Pr[i] = (P6_ex[0] * P8[C8_ex * 0]
               + P6_ex[1] * P8[C8_ex * 1]
               + P6_ex[2] * P8[C8_ex * 2]
               ) * exp0;
    }
}

void helper_G4_PM(double x0, double x1, double* P6, double* P8, double* Pr) {
    double theta5 = para[1];
    double thetaW = para[7];
    double theta5W = para[1] + para[7];
    int i;

    for (i = 0; i < C8 - 1; i++) {
        Pr[i] = theta5W * x0 * (
              P8[C8 * i + 0] / theta5
            + P8[C8 * i + 2] / thetaW);
    }
}

void helper_G5_PM(double x0, double x1, double* P6, double* P8, double* Pr) {
    double exp0 = exp(-x0);
    int i;

    for (i = 0; i < C8 - 1; i++) {
        Pr[i] = (1 - P8[C8 * i + C8 - 1]) * exp0;
    }
}

void helper_G6_PM(double x0, double x1, double* P6, double* P8, double* Pr) {
    Pr[0] = exp(-3 * x1 - x0);
}

void density_G123(int s, double* P5, double* P7, double* Pr, double wwprior[3]) {
    double theta1 = para[4];
    double theta2 = para[5];
    double theta3 = para[6];
    double theta123 = para[4] + para[5] + para[6];

    wwprior[0] *= theta123 * (
          (P5[C5 * s +  0] * Pr[0] + P5[C5 * s +  1] * Pr[1] + P5[C5 * s +  2] * Pr[3]) / theta1
        + (P5[C5 * s + 12] * Pr[1] + P5[C5 * s + 13] * Pr[2] + P5[C5 * s + 14] * Pr[4]) / theta2
        + (P5[C5 * s + 24] * Pr[3] + P5[C5 * s + 25] * Pr[4] + P5[C5 * s + 26] * Pr[5]) / theta3);

    wwprior[1] *= theta123 * (
          (P5[C5 * s +  0] * Pr[0] + P5[C5 * s +  3] * Pr[1] + P5[C5 * s +  6] * Pr[3]) / theta1
        + (P5[C5 * s + 10] * Pr[1] + P5[C5 * s + 13] * Pr[2] + P5[C5 * s + 16] * Pr[4]) / theta2
        + (P5[C5 * s + 20] * Pr[3] + P5[C5 * s + 23] * Pr[4] + P5[C5 * s + 26] * Pr[5]) / theta3);

    wwprior[2] *= theta123 * (
          (P5[C5 * s +  0] * Pr[0] + P5[C5 * s +  9] * Pr[1] + P5[C5 * s + 18] * Pr[3]) / theta1
        + (P5[C5 * s +  4] * Pr[1] + P5[C5 * s + 13] * Pr[2] + P5[C5 * s + 22] * Pr[4]) / theta2
        + (P5[C5 * s +  8] * Pr[3] + P5[C5 * s + 17] * Pr[4] + P5[C5 * s + 26] * Pr[5]) / theta3);
}

void density_G45(int s, double* P5, double* P7, double* Pr, double wwprior[3]) {
    double theta5 = para[1];
    double thetaW = para[7];
    double theta5W = para[1] + para[7];
    double P5P7[C7 - 1];
    int i;

    for (i = 0; i < C7 - 1; i++) {
        P5P7[i] = P5[C5_ex * s + 0] * P7[C7 * 0 + i]
                + P5[C5_ex * s + 1] * P7[C7 * 1 + i]
                + P5[C5_ex * s + 2] * P7[C7 * 2 + i]
                + P5[C5_ex * s + 3] * P7[C7 * 3 + i]
                + P5[C5_ex * s + 4] * P7[C7 * 4 + i]
                + P5[C5_ex * s + 5] * P7[C7 * 5 + i]
                + P5[C5_ex * s + 6] * P7[C7 * 6 + i]
                + P5[C5_ex * s + 7] * P7[C7 * 7 + i];
    }

    wwprior[0] *= theta5W * (
          (P5P7[0] * Pr[0] + P5P7[1] * Pr[1]) / theta5
        + (P5P7[4] * Pr[1] + P5P7[7] * Pr[2]) / thetaW);

    wwprior[1] *= theta5W * (
          (P5P7[0] * Pr[0] + P5P7[2] * Pr[1]) / theta5
        + (P5P7[5] * Pr[1] + P5P7[7] * Pr[2]) / thetaW);

    wwprior[2] *= theta5W * (
          (P5P7[0] * Pr[0] + P5P7[3] * Pr[1]) / theta5
        + (P5P7[6] * Pr[1] + P5P7[7] * Pr[2]) / thetaW);
}

void density_G4(int s, double* P5, double* P7, double* Pr, double wwprior[3]) {
    double theta5 = para[1];
    double thetaW = para[7];
    double theta5W = para[1] + para[7];

    double f = theta5W * (
          P5[C5_ex * s + 0] * Pr[0] / theta5
        + P5[C5_ex * s + 7] * Pr[1] / thetaW);

    wwprior[0] *= f;
    wwprior[1] *= f;
    wwprior[2] *= f;
}

void density_G5(int s, double* P5, double* P7, double* Pr, double wwprior[3]) {
    double theta5 = para[1];
    double thetaW = para[7];
    double theta5W = para[1] + para[7];

    wwprior[0] *= theta5W * (
          (P5[C5_ex * s + 0] * Pr[0] + P5[C5_ex * s + 1] * Pr[1]) / theta5
        + (P5[C5_ex * s + 4] * Pr[2] + P5[C5_ex * s + 7] * Pr[3]) / thetaW);

    wwprior[1] *= theta5W * (
          (P5[C5_ex * s + 0] * Pr[0] + P5[C5_ex * s + 2] * Pr[1]) / theta5
        + (P5[C5_ex * s + 5] * Pr[2] + P5[C5_ex * s + 7] * Pr[3]) / thetaW);

    wwprior[2] *= theta5W * (
          (P5[C5_ex * s + 0] * Pr[0] + P5[C5_ex * s + 3] * Pr[1]) / theta5
        + (P5[C5_ex * s + 6] * Pr[2] + P5[C5_ex * s + 7] * Pr[3]) / thetaW);
}

void density_G6(int s, double* P5, double* P7, double* Pr, double wwprior[3]) {
    double f = P5[C5_ex * s + 8] * Pr[0];

    wwprior[0] *= f;
    wwprior[1] *= f;
    wwprior[2] *= f;
}

void t0t1_G1_123_123(double x0, double x1, double t[2]) {
    double theta123 = para[4] + para[5] + para[6];
    t0t1_type_1(x0, x1, theta123, t);
}

void t0t1_G2_123_5W(double x0, double x1, double t[2]) {
    double theta123 = para[4] + para[5] + para[6];
    double theta5W = para[1] + para[7];
    t0t1_type_2(x0, x1, theta123, theta5W, t);
}

void t0t1_G3_123_4(double x0, double x1, double t[2]) {
    double theta123 = para[4] + para[5] + para[6];
    double theta4 = para[0];
    t0t1_type_2(x0, x1, theta123, theta4, t);
}

void t0t1_G4_5W_5W(double x0, double x1, double t[2]) {
    double theta5W = para[1] + para[7];
    t0t1_type_1(x0, x1, theta5W, t);
}

void t0t1_G5_5W_4(double x0, double x1, double t[2]) {
    double theta5W = para[1] + para[7];
    double theta4 = para[0];
    t0t1_type_2(x0, x1, theta5W, theta4, t);
}

void t0t1_G6_4_4(double x0, double x1, double t[2]) {
    double theta4 = para[0];
    t0t1_type_3(x0, x1, theta4, t);
}

void b_G1_123_123(double x0, double x1, double theta12, double b[2]) {
    double theta123 = para[4] + para[5] + para[6];
    b_type_1(x0, x1, theta123, b);
}

void b_G2_123_5W(double x0, double x1, double theta12, double b[2]) {
    double theta123 = para[4] + para[5] + para[6];
    double theta5W = para[1] + para[7];
    b_type_2(x0, x1, theta123, theta5W, b);
}

void b_G3_123_4(double x0, double x1, double theta12, double b[2]) {
    double theta123 = para[4] + para[5] + para[6];
    double theta4 = para[0];
    b_type_3(x0, x1, theta123, theta4, b);
}

void b_G4_5W_5W(double x0, double x1, double theta12, double b[2]) {
    double theta5W = para[1] + para[7];
    b_type_4(x0, x1, theta5W, b);
}

void b_G5_5W_4(double x0, double x1, double theta12, double b[2]) {
    double theta5W = para[1] + para[7];
    double theta4 = para[0];
    b_type_5(x0, x1, theta5W, theta4, b);
}

void b_G6_4_4(double x0, double x1, double theta12, double b[2]) {
    double theta4 = para[0];
    b_type_6(x0, x1, theta4, b);
}


// Density functions for M3MSci12

static inline void g12_G1_p_iii(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double f = x0 * exp(-2 * x0 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void g12_G1_z_iii(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double thetai = para[3 + i];
    double thetaI = para[7 + i];
    double thetaJ = para[10 - i];
    double thetaIJ = para[8] + para[9];
    double phiJI = para[12 - i];
    double exp0 = exp(-2 * T / thetai - 2 * x1 - thetaIJ * x0 / thetaI);
    double exp1 = exp(-2 * T / thetai - 2 * x1 - thetaIJ * x0 / thetaJ);
    double f = thetaIJ * ((1 - phiJI) * (1 - phiJI) * exp0 / thetaI + phiJI * phiJI * exp1 / thetaJ);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void g12_G1_m_iii(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double thetai = para[3 + i];
    double thetaI = para[7 + i];
    double thetaJ = para[10 - i];
    double thetaIJ = para[8] + para[9];
    double phiJI = para[12 - i];
    double exp0 = exp(-6 * T / thetai - thetaIJ * (2 * x0 * x1 + x0) / thetaI);
    double exp1 = exp(-6 * T / thetai - thetaIJ * (2 * x0 * x1 + x0) / thetaJ);
    double f = thetaIJ * thetaIJ * x0 * ((1 - phiJI) * (1 - phiJI) * (1 - phiJI) * exp0 / (thetaI * thetaI) + phiJI * phiJI * phiJI * exp1 / (thetaJ * thetaJ));
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void g12_G2_p_iii(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double thetai = para[3 + i];
    double f = Pr[i - 1] * exp(-2 * T / thetai - 2 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void g12_G2_m_iii(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double T = para[7];
    double thetai = para[3 + i];
    double thetaI = para[7 + i];
    double thetaJ = para[10 - i];
    double thetaIJ = para[8] + para[9];
    double phiJI = para[12 - i];
    double exp0 = exp(-6 * T / thetai - 2 * (tau1 - T + thetaIJ * x1) / thetaI - x0);
    double exp1 = exp(-6 * T / thetai - thetaIJ * x1 / thetaI - x0);
    double exp2 = exp(-6 * T / thetai - thetaIJ * x1 / thetaJ - x0);
    double exp3 = exp(-6 * T / thetai - 2 * (tau1 - T + thetaIJ * x1) / thetaJ - x0);
    double f = thetaIJ * ((1 - phiJI) * (1 - phiJI) * ((1 - phiJI) * exp0 + phiJI * exp1) / thetaI + phiJI * phiJI * ((1 - phiJI) * exp2 + phiJI * exp3) / thetaJ);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void g12_G3_p_iii(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetai = para[3 + i];
    double f = Pr[i - 1] * exp(-2 * T / thetai - 2 * (tau0 - tau1) / theta5 - 2 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void g12_G3_m_iii(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetai = para[3 + i];
    double thetaI = para[7 + i];
    double thetaJ = para[10 - i];
    double thetaIJ = para[8] + para[9];
    double phiJI = para[12 - i];
    double exp0 = exp(-6 * T / thetai - 2 * (tau0 - tau1) / theta5 - 2 * (tau1 - T + thetaIJ * x1) / thetaI - x0);
    double exp1 = exp(-6 * T / thetai - 2 * (tau0 - tau1) / theta5 - thetaIJ * x1 / thetaI - x0);
    double exp2 = exp(-6 * T / thetai - 2 * (tau0 - tau1) / theta5 - thetaIJ * x1 / thetaJ - x0);
    double exp3 = exp(-6 * T / thetai - 2 * (tau0 - tau1) / theta5 - 2 * (tau1 - T + thetaIJ * x1) / thetaJ - x0);
    double f = thetaIJ * ((1 - phiJI) * (1 - phiJI) * ((1 - phiJI) * exp0 + phiJI * exp1) / thetaI + phiJI * phiJI * ((1 - phiJI) * exp2 + phiJI * exp3) / thetaJ);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void g12_G4_iii(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double thetai = para[3 + i];
    double f = Pr[1 + i] * x0 * exp(-6 * T / thetai - 2 * x0 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void g12_G5_iii(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetai = para[3 + i];
    double f = Pr[1 + i] * exp(-6 * T / thetai - 2 * (tau0 - tau1) / theta5 - 2 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void g12_G6_iii(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetai = para[3 + i];
    double f = Pr[1 + i] * exp(-6 * T / thetai - 6 * (tau0 - tau1) / theta5 - 3 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void g12_G1_z_iij(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double thetai = para[3 + i];
    double thetaI = para[7 + i];
    double thetaJ = para[10 - i];
    double thetaIJ = para[8] + para[9];
    double phiIJ = para[9 + i];
    double phiJI = para[12 - i];
    double exp0 = exp(-x1 - thetaIJ * x0 / thetaI);
    double exp1 = exp(-x1 - thetaIJ * x0 / thetaJ);
    double f = thetaIJ * ((1 - phiJI) * phiIJ * exp0 / thetaI + phiJI * (1 - phiIJ) * exp1 / thetaJ);
    wwprior[0] *= f; wwprior[1] *= 0; wwprior[2] *= 0;
}

static inline void g12_G1_m_iij(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double thetai = para[3 + i];
    double thetaI = para[7 + i];
    double thetaJ = para[10 - i];
    double thetaIJ = para[8] + para[9];
    double phiIJ = para[9 + i];
    double phiJI = para[12 - i];
    double exp0 = exp(-2 * T / thetai - thetaIJ * (2 * x0 * x1 + x0) / thetaI);
    double exp1 = exp(-2 * T / thetai - thetaIJ * (2 * x0 * x1 + x0) / thetaJ);
    double f = thetaIJ * thetaIJ * x0 * ((1 - phiJI) * (1 - phiJI) * phiIJ * exp0 / (thetaI * thetaI) + phiJI * phiJI * (1 - phiIJ) * exp1 / (thetaJ * thetaJ));
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void g12_G2_p_iij(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double f = Pr[4] * exp(-x1 - x0);
    wwprior[0] *= f; wwprior[1] *= 0; wwprior[2] *= 0;
}

static inline void g12_G2_m_iij(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double T = para[7];
    double thetai = para[3 + i];
    double thetaI = para[7 + i];
    double thetaJ = para[10 - i];
    double thetaIJ = para[8] + para[9];
    double phiIJ = para[9 + i];
    double phiJI = para[12 - i];
    double exp0 = exp(-2 * T / thetai - 2 * (tau1 - T + thetaIJ * x1) / thetaI - x0);
    double exp1 = exp(-2 * T / thetai - thetaIJ * x1 / thetaI - x0);
    double exp2 = exp(-2 * T / thetai - thetaIJ * x1 / thetaJ - x0);
    double exp3 = exp(-2 * T / thetai - 2 * (tau1 - T + thetaIJ * x1) / thetaJ - x0);
    double fc = thetaIJ * ((1 - phiJI) * (1 - phiJI) * (phiIJ * exp0 + (1 - phiIJ) * exp1) / thetaI + phiJI * phiJI * (phiIJ * exp2 + (1 - phiIJ) * exp3) / thetaJ);
    double fab = thetaIJ * ((1 - phiJI) * phiIJ * ((1 - phiJI) * exp0 + phiJI * exp1) / thetaI + phiJI * (1 - phiIJ) * ((1 - phiJI) * exp2 + phiJI * exp3) / thetaJ);
    wwprior[0] *= fc; wwprior[1] *= fab; wwprior[2] *= fab;
}

static inline void g12_G3_p_iij(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double theta5 = para[1];
    double f = Pr[4] * exp(-2 * (tau0 - tau1) / theta5 - x1 - x0);
    wwprior[0] *= f; wwprior[1] *= 0; wwprior[2] *= 0;
}

static inline void g12_G3_m_iij(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetai = para[3 + i];
    double thetaI = para[7 + i];
    double thetaJ = para[10 - i];
    double thetaIJ = para[8] + para[9];
    double phiIJ = para[9 + i];
    double phiJI = para[12 - i];
    double exp0 = exp(-2 * T / thetai - 2 * (tau0 - tau1) / theta5 - 2 * (tau1 - T + thetaIJ * x1) / thetaI - x0);
    double exp1 = exp(-2 * T / thetai - 2 * (tau0 - tau1) / theta5 - thetaIJ * x1 / thetaI - x0);
    double exp2 = exp(-2 * T / thetai - 2 * (tau0 - tau1) / theta5 - thetaIJ * x1 / thetaJ - x0);
    double exp3 = exp(-2 * T / thetai - 2 * (tau0 - tau1) / theta5 - 2 * (tau1 - T + thetaIJ * x1) / thetaJ - x0);
    double fc = thetaIJ * ((1 - phiJI) * (1 - phiJI) * (phiIJ * exp0 + (1 - phiIJ) * exp1) / thetaI + phiJI * phiJI * (phiIJ * exp2 + (1 - phiIJ) * exp3) / thetaJ);
    double fab = thetaIJ * ((1 - phiJI) * phiIJ * ((1 - phiJI) * exp0 + phiJI * exp1) / thetaI + phiJI * (1 - phiIJ) * ((1 - phiJI) * exp2 + phiJI * exp3) / thetaJ);
    wwprior[0] *= fc; wwprior[1] *= fab; wwprior[2] *= fab;
}

static inline void g12_G4_iij(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double thetai = para[3 + i];
    double f = Pr[4 + i] * x0 * exp(-2 * T / thetai - 2 * x0 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void g12_G5_iij(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetai = para[3 + i];
    double f = Pr[4 + i] * exp(-2 * T / thetai - 2 * (tau0 - tau1) / theta5 - 2 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void g12_G6_iij(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetai = para[3 + i];
    double f = Pr[4 + i] * exp(-2 * T / thetai - 6 * (tau0 - tau1) / theta5 - 3 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void g12_G3_p_ii3(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double f = exp(-x1 - x0);
    wwprior[0] *= f; wwprior[1] *= 0; wwprior[2] *= 0;
}

static inline void g12_G3_m_ii3(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double thetai = para[3 + i];
    double thetaI = para[7 + i];
    double thetaJ = para[10 - i];
    double thetaIJ = para[8] + para[9];
    double phiJI = para[12 - i];
    double exp0 = exp(-2 * T / thetai - thetaIJ * x1 / thetaI - x0);
    double exp1 = exp(-2 * T / thetai - thetaIJ * x1 / thetaJ - x0);
    double f = thetaIJ * ((1 - phiJI) * (1 - phiJI) * exp0 / thetaI + phiJI * phiJI * exp1 / thetaJ);
    wwprior[0] *= f; wwprior[1] *= 0; wwprior[2] *= 0;
}

static inline void g12_G5_ii3(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double thetai = para[3 + i];
    double f = Pr[i - 1] * exp(-2 * T / thetai - x1 - x0);
    wwprior[0] *= f; wwprior[1] *= 0; wwprior[2] *= 0;
}

static inline void g12_G6_ii3(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetai = para[3 + i];
    double f = Pr[i - 1] * exp(-2 * T / thetai - 2 * (tau0 - tau1) / theta5 - 3 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

DEFINE_WRAPPER(g12_G1_p_111, g12_G1_p_iii, 1)
DEFINE_WRAPPER(g12_G1_z_111, g12_G1_z_iii, 1)
DEFINE_WRAPPER(g12_G1_m_111, g12_G1_m_iii, 1)
DEFINE_WRAPPER(g12_G2_p_111, g12_G2_p_iii, 1)
DEFINE_WRAPPER(g12_G2_m_111, g12_G2_m_iii, 1)
DEFINE_WRAPPER(g12_G3_p_111, g12_G3_p_iii, 1)
DEFINE_WRAPPER(g12_G3_m_111, g12_G3_m_iii, 1)
DEFINE_WRAPPER(g12_G4_111, g12_G4_iii, 1)
DEFINE_WRAPPER(g12_G5_111, g12_G5_iii, 1)
DEFINE_WRAPPER(g12_G6_111, g12_G6_iii, 1)
DEFINE_WRAPPER(g12_G1_p_222, g12_G1_p_iii, 2)
DEFINE_WRAPPER(g12_G1_z_222, g12_G1_z_iii, 2)
DEFINE_WRAPPER(g12_G1_m_222, g12_G1_m_iii, 2)
DEFINE_WRAPPER(g12_G2_p_222, g12_G2_p_iii, 2)
DEFINE_WRAPPER(g12_G2_m_222, g12_G2_m_iii, 2)
DEFINE_WRAPPER(g12_G3_p_222, g12_G3_p_iii, 2)
DEFINE_WRAPPER(g12_G3_m_222, g12_G3_m_iii, 2)
DEFINE_WRAPPER(g12_G4_222, g12_G4_iii, 2)
DEFINE_WRAPPER(g12_G5_222, g12_G5_iii, 2)
DEFINE_WRAPPER(g12_G6_222, g12_G6_iii, 2)
DEFINE_WRAPPER(g12_G1_z_112, g12_G1_z_iij, 1)
DEFINE_WRAPPER(g12_G1_m_112, g12_G1_m_iij, 1)
DEFINE_WRAPPER(g12_G2_p_112, g12_G2_p_iij, 1)
DEFINE_WRAPPER(g12_G2_m_112, g12_G2_m_iij, 1)
DEFINE_WRAPPER(g12_G3_p_112, g12_G3_p_iij, 1)
DEFINE_WRAPPER(g12_G3_m_112, g12_G3_m_iij, 1)
DEFINE_WRAPPER(g12_G4_112, g12_G4_iij, 1)
DEFINE_WRAPPER(g12_G5_112, g12_G5_iij, 1)
DEFINE_WRAPPER(g12_G6_112, g12_G6_iij, 1)
DEFINE_WRAPPER(g12_G1_z_221, g12_G1_z_iij, 2)
DEFINE_WRAPPER(g12_G1_m_221, g12_G1_m_iij, 2)
DEFINE_WRAPPER(g12_G2_p_221, g12_G2_p_iij, 2)
DEFINE_WRAPPER(g12_G2_m_221, g12_G2_m_iij, 2)
DEFINE_WRAPPER(g12_G3_p_221, g12_G3_p_iij, 2)
DEFINE_WRAPPER(g12_G3_m_221, g12_G3_m_iij, 2)
DEFINE_WRAPPER(g12_G4_221, g12_G4_iij, 2)
DEFINE_WRAPPER(g12_G5_221, g12_G5_iij, 2)
DEFINE_WRAPPER(g12_G6_221, g12_G6_iij, 2)
DEFINE_WRAPPER(g12_G3_p_113, g12_G3_p_ii3, 1)
DEFINE_WRAPPER(g12_G3_m_113, g12_G3_m_ii3, 1)
DEFINE_WRAPPER(g12_G5_113, g12_G5_ii3, 1)
DEFINE_WRAPPER(g12_G6_113, g12_G6_ii3, 1)
DEFINE_WRAPPER(g12_G3_p_223, g12_G3_p_ii3, 2)
DEFINE_WRAPPER(g12_G3_m_223, g12_G3_m_ii3, 2)
DEFINE_WRAPPER(g12_G5_223, g12_G5_ii3, 2)
DEFINE_WRAPPER(g12_G6_223, g12_G6_ii3, 2)

void g12_G3_m_123(double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double thetaX = para[8];
    double thetaY = para[9];
    double thetaXY = para[8] + para[9];
    double phi12 = para[10];
    double phi21 = para[11];
    double exp0 = exp(-thetaXY * x1 / thetaX - x0);
    double exp1 = exp(-thetaXY * x1 / thetaY - x0);
    double f = thetaXY * ((1 - phi21) * phi12 * exp0 / thetaX + phi21 * (1 - phi12) * exp1 / thetaY);
    wwprior[0] *= f; wwprior[1] *= 0; wwprior[2] *= 0;
}

void g12_G5_123(double x0, double x1, double* Pr, double wwprior[3]) {
    double f = Pr[4] * exp(-x1 - x0);
    wwprior[0] *= f; wwprior[1] *= 0; wwprior[2] *= 0;
}

void g12_G6_123(double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double theta5 = para[1];
    double f = Pr[4] * exp(-2 * (tau0 - tau1) / theta5 - 3 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

void g12_G3_133(double x0, double x1, double* Pr, double wwprior[3]) {
    double f = exp(-x1 - x0);
    wwprior[0] *= 0; wwprior[1] *= 0; wwprior[2] *= f;
}

void g12_G6_133(double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double theta3 = para[6];
    double f = exp(-2 * tau0 / theta3 - 3 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

void g12_G1_333(double x0, double x1, double* Pr, double wwprior[3]) {
    double f = x0 * exp(-2 * x0 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

void g12_G3_333(double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double theta3 = para[6];
    double f = exp(-2 * tau0 / theta3 - 2 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

void g12_G6_333(double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double theta3 = para[6];
    double f = exp(-6 * tau0 / theta3 - 3 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

void b_G1_1_1(double x0, double x1, double theta12, double b[2]) {
    double theta1 = para[4];
    b_type_1(x0, x1, theta1, b);
}

void b_G1_2_2(double x0, double x1, double theta12, double b[2]) {
    double theta2 = para[5];
    b_type_1(x0, x1, theta2, b);
}

void b_G1_3_3(double x0, double x1, double theta12, double b[2]) {
    double theta3 = para[6];
    b_type_1(x0, x1, theta3, b);
}

void b_G2_1_5(double x0, double x1, double theta12, double b[2]) {
    double theta1 = para[4];
    double theta5 = para[1];
    b_type_2(x0, x1, theta1, theta5, b);
}

void b_G2_2_5(double x0, double x1, double theta12, double b[2]) {
    double theta2 = para[5];
    double theta5 = para[1];
    b_type_2(x0, x1, theta2, theta5, b);
}

void b_G3_1_4(double x0, double x1, double theta12, double b[2]) {
    double theta1 = para[4];
    double theta4 = para[0];
    b_type_3(x0, x1, theta1, theta4, b);
}

void b_G3_2_4(double x0, double x1, double theta12, double b[2]) {
    double theta2 = para[5];
    double theta4 = para[0];
    b_type_3(x0, x1, theta2, theta4, b);
}

void b_G3_3_4(double x0, double x1, double theta12, double b[2]) {
    double theta3 = para[6];
    double theta4 = para[0];
    b_type_3(x0, x1, theta3, theta4, b);
}

void b_G4_5_5(double x0, double x1, double theta12, double b[2]) {
    double theta5 = para[1];
    b_type_4(x0, x1, theta5, b);
}

void b_G5_5_4(double x0, double x1, double theta12, double b[2]) {
    double theta5 = para[1];
    double theta4 = para[0];
    b_type_5(x0, x1, theta5, theta4, b);
}

void b_G1_1_XY(double x0, double x1, double theta12, double b[2]) {
    double theta1 = para[4];
    double thetaXY = para[8] + para[9];
    b_type_7(x0, x1, theta1, thetaXY, b);
}

void b_G1_2_XY(double x0, double x1, double theta12, double b[2]) {
    double theta2 = para[5];
    double thetaXY = para[8] + para[9];
    b_type_7(x0, x1, theta2, thetaXY, b);
}

void b_G1_XY_XY(double x0, double x1, double theta12, double b[2]) {
    double thetaXY = para[8] + para[9];
    b_type_8(x0, x1, thetaXY, b);
}

void b_G2_XY_5(double x0, double x1, double theta12, double b[2]) {
    double thetaXY = para[8] + para[9];
    double theta5 = para[1];
    b_type_9(x0, x1, thetaXY, theta5, b);
}

void b_G3_XY_4(double x0, double x1, double theta12, double b[2]) {
    double thetaXY = para[8] + para[9];
    double theta4 = para[0];
    b_type_10(x0, x1, thetaXY, theta4, b);
}


// Density functions for M3MSci13 and M3MSci23

static inline void gi3_G1_p_iii(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double f = x0 * exp(-2 * x0 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G1_z_iii(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double thetai = para[3 + i];
    double thetaI = para[8];
    double thetaZ = para[9];
    double thetaIZ = para[8] + para[9];
    double phi3I = para[11];
    double exp0 = exp(-2 * T / thetai - 2 * x1 - thetaIZ * x0 / thetaI);
    double exp1 = exp(-2 * T / thetai - 2 * x1 - thetaIZ * x0 / thetaZ);
    double f = thetaIZ * ((1 - phi3I) * (1 - phi3I) * exp0 / thetaI + phi3I * phi3I * exp1 / thetaZ);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G1_m_iii(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double thetai = para[3 + i];
    double thetaI = para[8];
    double thetaZ = para[9];
    double thetaIZ = para[8] + para[9];
    double phi3I = para[11];
    double exp0 = exp(-6 * T / thetai - thetaIZ * (2 * x0 * x1 + x0) / thetaI);
    double exp1 = exp(-6 * T / thetai - thetaIZ * (2 * x0 * x1 + x0) / thetaZ);
    double f = thetaIZ * thetaIZ * x0 * ((1 - phi3I) * (1 - phi3I) * (1 - phi3I) * exp0 / (thetaI * thetaI) + phi3I * phi3I * phi3I * exp1 / (thetaZ * thetaZ));
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G2_p_iii(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetai = para[3 + i];
    double thetaI = para[8];
    double thetaZ = para[9];
    double theta5Z = para[1] + para[9];
    double phi3I = para[11];
    double exp0 = exp(-2 * T / thetai - 2 * x1 - 2 * (tau1 - T) / thetaI - theta5Z * x0 / theta5);
    double exp1 = exp(-2 * T / thetai - 2 * x1 - (2 * (tau1 - T) + theta5Z * x0) / thetaZ);
    double f = theta5Z * ((1 - phi3I) * (1 - phi3I) * exp0 / theta5 + phi3I * phi3I * exp1 / thetaZ);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G2_m_iii(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetai = para[3 + i];
    double thetaI = para[8];
    double thetaZ = para[9];
    double thetaIZ = para[8] + para[9];
    double theta5Z = para[1] + para[9];
    double phi3I = para[11];
    double exp0 = exp(-6 * T / thetai - 2 * (tau1 - T + thetaIZ * x1) / thetaI - theta5Z * x0 / theta5);
    double exp1 = exp(-6 * T / thetai - (2 * (tau1 - T + thetaIZ * x1) + theta5Z * x0) / thetaZ);
    double f = thetaIZ * theta5Z * ((1 - phi3I) * (1 - phi3I) * (1 - phi3I) * exp0 / (thetaI * theta5) + phi3I * phi3I * phi3I * exp1 / (thetaZ * thetaZ));
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G3_p_iii(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double thetai = para[3 + i];
    double f = Pr[0] * exp(-2 * T / thetai - 2 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G3_m_iii(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetai = para[3 + i];
    double thetaI = para[8];
    double thetaZ = para[9];
    double thetaIZ = para[8] + para[9];
    double phi3I = para[11];
    double exp0 = exp(-6 * T / thetai - 2 * (tau1 - T + thetaIZ * x1) / thetaI - 2 * (tau0 - tau1) / theta5 - x0);
    double exp1 = exp(-6 * T / thetai - thetaIZ * x1 / thetaI - x0);
    double exp2 = exp(-6 * T / thetai - thetaIZ * x1 / thetaZ - x0);
    double exp3 = exp(-6 * T / thetai - 2 * (tau0 - T + thetaIZ * x1) / thetaZ - x0);
    double f = thetaIZ * ((1 - phi3I) * (1 - phi3I) * ((1 - phi3I) * exp0 + phi3I * exp1) / thetaI + phi3I * phi3I * ((1 - phi3I) * exp2 + phi3I * exp3) / thetaZ);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G4_iii(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetai = para[3 + i];
    double thetaI = para[8];
    double thetaZ = para[9];
    double theta5Z = para[1] + para[9];
    double phi3I = para[11];
    double exp0 = exp(-6 * T / thetai - 6 * (tau1 - T) / thetaI - theta5Z * (2 * x0 * x1 + x0) / theta5);
    double exp1 = exp(-6 * T / thetai - (6 * (tau1 - T) + theta5Z * (2 * x0 * x1 + x0)) / thetaZ);
    double f = theta5Z * theta5Z * x0 * ((1 - phi3I) * (1 - phi3I) * (1 - phi3I) * exp0 / (theta5 * theta5) + phi3I * phi3I * phi3I * exp1 / (thetaZ * thetaZ));
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G5_iii(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetai = para[3 + i];
    double thetaI = para[8];
    double thetaZ = para[9];
    double theta5Z = para[1] + para[9];
    double phi3I = para[11];
    double exp0 = exp(-6 * T / thetai - 6 * (tau1 - T) / thetaI - 2 * (tau0 - tau1 + theta5Z * x1) / theta5 - x0);
    double exp1 = exp(-6 * T / thetai - 2 * (tau1 - T) / thetaI - theta5Z * x1 / theta5 - x0);
    double exp2 = exp(-6 * T / thetai - (2 * (tau1 - T) + theta5Z * x1) / thetaZ - x0);
    double exp3 = exp(-6 * T / thetai - (6 * (tau1 - T) + 2 * (tau0 - tau1 + theta5Z * x1)) / thetaZ - x0);
    double f = theta5Z * ((1 - phi3I) * (1 - phi3I) * ((1 - phi3I) * exp0 + phi3I * exp1) / theta5 + phi3I * phi3I * ((1 - phi3I) * exp2 + phi3I * exp3) / thetaZ);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G6_iii(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double thetai = para[3 + i];
    double f = Pr[1] * exp(-6 * T / thetai - 3 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G1_jjj(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double f = x0 * exp(-2 * x0 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G2_jjj(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double thetaj = para[6 - i];
    double f = exp(-2 * tau1 / thetaj - 2 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G3_jjj(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double theta5 = para[1];
    double thetaj = para[6 - i];
    double f = exp(-2 * tau1 / thetaj - 2 * (tau0 - tau1) / theta5 - 2 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G4_jjj(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double thetaj = para[6 - i];
    double f = x0 * exp(-6 * tau1 / thetaj - 2 * x0 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G5_jjj(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double theta5 = para[1];
    double thetaj = para[6 - i];
    double f = exp(-6 * tau1 / thetaj - 2 * (tau0 - tau1) / theta5 - 2 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G6_jjj(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double theta5 = para[1];
    double thetaj = para[6 - i];
    double f = exp(-6 * tau1 / thetaj - 6 * (tau0 - tau1) / theta5 - 3 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G2_p_iij(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double phi3I = para[11];
    double f = (1 - phi3I) * exp(-x1 - x0);
    wwprior[0] *= f; wwprior[1] *= 0; wwprior[2] *= 0;
}

static inline void gi3_G2_m_iij(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double thetai = para[3 + i];
    double phi3I = para[11];
    double f = (1 - phi3I) * (1 - phi3I) * exp(-2 * T / thetai - x1 - x0);
    wwprior[0] *= f; wwprior[1] *= 0; wwprior[2] *= 0;
}

static inline void gi3_G3_p_iij(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double theta5 = para[1];
    double phi3I = para[11];
    double exp0 = exp(-2 * (tau0 - tau1) / theta5 - x1 - x0);
    double exp1 = exp(-x1 - x0);
    double f = (1 - phi3I) * exp0 + phi3I * exp1;
    wwprior[0] *= f; wwprior[1] *= 0; wwprior[2] *= 0;
}

static inline void gi3_G3_m_iij(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetai = para[3 + i];
    double thetaI = para[8];
    double thetaZ = para[9];
    double thetaIZ = para[8] + para[9];
    double phi3I = para[11];
    double exp0 = exp(-2 * T / thetai - thetaIZ * x1 / thetaI - 2 * (tau0 - tau1) / theta5 - x0);
    double exp1 = exp(-2 * T / thetai - thetaIZ * x1 / thetaZ - x0);
    double f = thetaIZ * ((1 - phi3I) * (1 - phi3I) * exp0 / thetaI + phi3I * phi3I * exp1 / thetaZ);
    wwprior[0] *= f; wwprior[1] *= 0; wwprior[2] *= 0;
}

static inline void gi3_G4_iij(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double T = para[7];
    double thetai = para[3 + i];
    double thetaI = para[8];
    double phi3I = para[11];
    double f = x0 * (1 - phi3I) * (1 - phi3I) * exp(-2 * T / thetai - 2 * (tau1 - T) / thetaI - 2 * x0 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G5_iij(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetai = para[3 + i];
    double thetaI = para[8];
    double thetaZ = para[9];
    double theta5Z = para[1] + para[9];
    double phi3I = para[11];
    double exp0 = exp(-2 * T / thetai - 2 * (tau1 - T) / thetaI - 2 * (tau0 - tau1 + theta5Z * x1) / theta5 - x0);
    double exp1 = exp(-2 * T / thetai - theta5Z * x1 / theta5 - x0);
    double exp2 = exp(-2 * T / thetai - (2 * (tau1 - T) + theta5Z * x1) / thetaZ - x0);
    double fc = theta5Z * ((1 - phi3I) * (1 - phi3I) * exp0 / theta5 + phi3I * phi3I * exp2 / thetaZ);
    double fab = theta5Z * ((1 - phi3I) * (1 - phi3I) * exp0 + (1 - phi3I) * phi3I * exp1) / theta5;
    wwprior[0] *= fc; wwprior[1] *= fab; wwprior[2] *= fab;
}

static inline void gi3_G6_iij(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double thetai = para[3 + i];
    double f = Pr[2] * exp(-2 * T / thetai - 3 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G2_jji(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double phi3I = para[11];
    double f = (1 - phi3I) * exp(-x1 - x0);
    wwprior[0] *= f; wwprior[1] *= 0; wwprior[2] *= 0;
}

static inline void gi3_G3_jji(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double theta5 = para[1];
    double phi3I = para[11];
    double exp0 = exp(-2 * (tau0 - tau1) / theta5 - x1 - x0);
    double exp1 = exp(-x1 - x0);
    double f = (1 - phi3I) * exp0 + phi3I * exp1;
    wwprior[0] *= f; wwprior[1] *= 0; wwprior[2] *= 0;
}

static inline void gi3_G4_jji(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double thetaj = para[6 - i];
    double phi3I = para[11];
    double f = x0 * (1 - phi3I) * exp(-2 * tau1 / thetaj - 2 * x0 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G5_jji(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double theta5 = para[1];
    double thetaj = para[6 - i];
    double phi3I = para[11];
    double exp0 = exp(-2 * tau1 / thetaj - 2 * (tau0 - tau1) / theta5 - 2 * x1 - x0);
    double exp1 = exp(-2 * tau1 / thetaj - x1 - x0);
    double fc = (1 - phi3I) * exp0 + phi3I * exp1;
    double fab = (1 - phi3I) * exp0;
    wwprior[0] *= fc; wwprior[1] *= fab; wwprior[2] *= fab;
}

static inline void gi3_G6_jji(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double theta5 = para[1];
    double thetaj = para[6 - i];
    double phi3I = para[11];
    double exp0 = exp(-2 * tau1 / thetaj - 6 * (tau0 - tau1) / theta5 - 3 * x1 - x0);
    double exp1 = exp(-2 * tau1 / thetaj - 2 * (tau0 - tau1) / theta5 - 3 * x1 - x0);
    double f = (1 - phi3I) * exp0 + phi3I * exp1;
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G1_z_ii3(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double thetaI = para[8];
    double thetaZ = para[9];
    double thetaIZ = para[8] + para[9];
    double phiI3 = para[10];
    double phi3I = para[11];
    double exp0 = exp(-x1 - thetaIZ * x0 / thetaI);
    double exp1 = exp(-x1 - thetaIZ * x0 / thetaZ);
    double f = thetaIZ * ((1 - phi3I) * phiI3 * exp0 / thetaI + phi3I * (1 - phiI3) * exp1 / thetaZ);
    wwprior[0] *= f; wwprior[1] *= 0; wwprior[2] *= 0;
}

static inline void gi3_G1_m_ii3(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double thetai = para[3 + i];
    double thetaI = para[8];
    double thetaZ = para[9];
    double thetaIZ = para[8] + para[9];
    double phiI3 = para[10];
    double phi3I = para[11];
    double exp0 = exp(-2 * T / thetai - thetaIZ * (2 * x0 * x1 + x0) / thetaI);
    double exp1 = exp(-2 * T / thetai - thetaIZ * (2 * x0 * x1 + x0) / thetaZ);
    double f = thetaIZ * thetaIZ * x0 * ((1 - phi3I) * (1 - phi3I) * phiI3 * exp0 / (thetaI * thetaI) + phi3I * phi3I * (1 - phiI3) * exp1 / (thetaZ * thetaZ));
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G2_p_ii3(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetaI = para[8];
    double thetaZ = para[9];
    double theta5Z = para[1] + para[9];
    double phiI3 = para[10];
    double phi3I = para[11];
    double exp0 = exp(-x1 - 2 * (tau1 - T) / thetaI - theta5Z * x0 / theta5);
    double exp1 = exp(-x1 - (2 * (tau1 - T) + theta5Z * x0) / thetaZ);
    double f = theta5Z * ((1 - phi3I) * phiI3 * exp0 / theta5 + phi3I * (1 - phiI3) * exp1 / thetaZ);
    wwprior[0] *= f; wwprior[1] *= 0; wwprior[2] *= 0;
}

static inline void gi3_G2_m_ii3(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetai = para[3 + i];
    double thetaI = para[8];
    double thetaZ = para[9];
    double thetaIZ = para[8] + para[9];
    double theta5Z = para[1] + para[9];
    double phiI3 = para[10];
    double phi3I = para[11];
    double exp0 = exp(-2 * T / thetai - 2 * (tau1 - T + thetaIZ * x1) / thetaI - theta5Z * x0 / theta5);
    double exp1 = exp(-2 * T / thetai - (2 * (tau1 - T + thetaIZ * x1) + theta5Z * x0) / thetaZ);
    double f = thetaIZ * theta5Z * ((1 - phi3I) * (1 - phi3I) * phiI3 * exp0 / (thetaI * theta5) + phi3I * phi3I * (1 - phiI3) * exp1 / (thetaZ * thetaZ));
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G3_p_ii3(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double f = Pr[3] * exp(-x1 - x0);
    wwprior[0] *= f; wwprior[1] *= 0; wwprior[2] *= 0;
}

static inline void gi3_G3_m_ii3(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetai = para[3 + i];
    double thetaI = para[8];
    double thetaZ = para[9];
    double thetaIZ = para[8] + para[9];
    double phiI3 = para[10];
    double phi3I = para[11];
    double exp0 = exp(-2 * T / thetai - 2 * (tau1 - T + thetaIZ * x1) / thetaI - 2 * (tau0 - tau1) / theta5 - x0);
    double exp1 = exp(-2 * T / thetai - thetaIZ * x1 / thetaI - x0);
    double exp2 = exp(-2 * T / thetai - thetaIZ * x1 / thetaZ - x0);
    double exp3 = exp(-2 * T / thetai - 2 * (tau0 - T + thetaIZ * x1) / thetaZ - x0);
    double fc = thetaIZ * ((1 - phi3I) * (1 - phi3I) * (phiI3 * exp0 + (1 - phiI3) * exp1) / thetaI + phi3I * phi3I * (phiI3 * exp2 + (1 - phiI3) * exp3) / thetaZ);
    double fab = thetaIZ * ((1 - phi3I) * phiI3 * ((1 - phi3I) * exp0 + phi3I * exp1) / thetaI + phi3I * (1 - phiI3) * ((1 - phi3I) * exp2 + phi3I * exp3) / thetaZ);
    wwprior[0] *= fc; wwprior[1] *= fab; wwprior[2] *= fab;
}

static inline void gi3_G4_ii3(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetai = para[3 + i];
    double thetaI = para[8];
    double thetaZ = para[9];
    double theta5Z = para[1] + para[9];
    double phiI3 = para[10];
    double phi3I = para[11];
    double exp0 = exp(-2 * T / thetai - 6 * (tau1 - T) / thetaI - theta5Z * (2 * x0 * x1 + x0) / theta5);
    double exp1 = exp(-2 * T / thetai - (6 * (tau1 - T) + theta5Z * (2 * x0 * x1 + x0)) / thetaZ);
    double f = theta5Z * theta5Z * x0 * ((1 - phi3I) * (1 - phi3I) * phiI3 * exp0 / (theta5 * theta5) + phi3I * phi3I * (1 - phiI3) * exp1 / (thetaZ * thetaZ));
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G5_ii3(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetai = para[3 + i];
    double thetaI = para[8];
    double thetaZ = para[9];
    double theta5Z = para[1] + para[9];
    double phiI3 = para[10];
    double phi3I = para[11];
    double exp0 = exp(-2 * T / thetai - 6 * (tau1 - T) / thetaI - 2 * (tau0 - tau1 + theta5Z * x1) / theta5 - x0);
    double exp1 = exp(-2 * T / thetai - 2 * (tau1 - T) / thetaI - theta5Z * x1 / theta5 - x0);
    double exp2 = exp(-2 * T / thetai - 2 * (tau1 - T) / thetaZ - theta5Z * x1 / thetaZ - x0);
    double exp3 = exp(-2 * T / thetai - (6 * (tau1 - T) + 2 * (tau0 - tau1 + theta5Z * x1)) / thetaZ - x0);
    double fc = theta5Z * ((1 - phi3I) * (1 - phi3I) * (phiI3 * exp0 + (1 - phiI3) * exp1) / theta5 + phi3I * phi3I * (phiI3 * exp2 + (1 - phiI3) * exp3) / thetaZ);
    double fab = theta5Z * ((1 - phi3I) * phiI3 * ((1 - phi3I) * exp0 + phi3I * exp1) / theta5 + phi3I * (1 - phiI3) * ((1 - phi3I) * exp2 + phi3I * exp3) / thetaZ);
    wwprior[0] *= fc; wwprior[1] *= fab; wwprior[2] *= fab;
}

static inline void gi3_G6_ii3(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double thetai = para[3 + i];
    double f = Pr[4] * exp(-2 * T / thetai - 3 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G2_jj3(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double phiI3 = para[10];
    double f = phiI3 * exp(-x1 - x0);
    wwprior[0] *= f; wwprior[1] *= 0; wwprior[2] *= 0;
}

static inline void gi3_G3_jj3(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double theta5 = para[1];
    double phiI3 = para[10];
    double exp0 = exp(-2 * (tau0 - tau1) / theta5 - x1 - x0);
    double exp1 = exp(-x1 - x0);
    double f = phiI3 * exp0 + (1 - phiI3) * exp1;
    wwprior[0] *= f; wwprior[1] *= 0; wwprior[2] *= 0;
}

static inline void gi3_G4_jj3(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double thetaj = para[6 - i];
    double phiI3 = para[10];
    double f = x0 * phiI3 * exp(-2 * tau1 / thetaj - 2 * x0 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G5_jj3(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double theta5 = para[1];
    double thetaj = para[6 - i];
    double phiI3 = para[10];
    double exp0 = exp(-2 * tau1 / thetaj - 2 * (tau0 - tau1) / theta5 - 2 * x1 - x0);
    double exp1 = exp(-2 * tau1 / thetaj - x1 - x0);
    double fc = phiI3 * exp0 + (1 - phiI3) * exp1;
    double fab = phiI3 * exp0;
    wwprior[0] *= fc; wwprior[1] *= fab; wwprior[2] *= fab;
}

static inline void gi3_G6_jj3(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double theta5 = para[1];
    double thetaj = para[6 - i];
    double phiI3 = para[10];
    double exp0 = exp(-2 * tau1 / thetaj - 6 * (tau0 - tau1) / theta5 - 3 * x1 - x0);
    double exp1 = exp(-2 * tau1 / thetaj - 2 * (tau0 - tau1) / theta5 - 3 * x1 - x0);
    double f = phiI3 * exp0 + (1 - phiI3) * exp1;
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G2_m_123(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double phiI3 = para[10];
    double phi3I = para[11];
    double f = (1 - phi3I) * phiI3 * exp(-x1 - x0);
    wwprior[0] *= 0; wwprior[i] *= f; wwprior[3 - i] *= 0;
}

static inline void gi3_G3_m_123(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double theta5 = para[1];
    double thetaI = para[8];
    double thetaZ = para[9];
    double thetaIZ = para[8] + para[9];
    double phiI3 = para[10];
    double phi3I = para[11];
    double exp0 = exp(-thetaIZ * x1 / thetaI - 2 * (tau0 - tau1) / theta5 - x0);
    double exp1 = exp(-thetaIZ * x1 / thetaZ - x0);
    double f = thetaIZ * ((1 - phi3I) * phiI3 * exp0 / thetaI + phi3I * (1 - phiI3) * exp1 / thetaZ);
    wwprior[0] *= 0; wwprior[i] *= f; wwprior[3 - i] *= 0;
}

static inline void gi3_G4_123(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double T = para[7];
    double thetaI = para[8];
    double phiI3 = para[10];
    double phi3I = para[11];
    double f = x0 * (1 - phi3I) * phiI3 * exp(-2 * (tau1 - T) / thetaI - 2 * x0 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G5_123(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetaI = para[8];
    double thetaZ = para[9];
    double theta5Z = para[1] + para[9];
    double phiI3 = para[10];
    double phi3I = para[11];
    double exp0 = exp(-2 * (tau1 - T) / thetaI - 2 * (tau0 - tau1 + theta5Z * x1) / theta5 - x0);
    double exp1 = exp(-theta5Z * x1 / theta5 - x0);
    double exp2 = exp(-(2 * (tau1 - T) + theta5Z * x1) / thetaZ - x0);
    double fc = theta5Z * ((1 - phi3I) * phiI3 * exp0 + (1 - phi3I) * (1 - phiI3) * exp1) / theta5;
    double fi = theta5Z * ((1 - phi3I) * phiI3 * exp0 / theta5 + phi3I * (1 - phiI3) * exp2 / thetaZ);
    double fj = theta5Z * ((1 - phi3I) * phiI3 * exp0 + phi3I * phiI3 * exp1) / theta5;
    wwprior[0] *= fc; wwprior[i] *= fi; wwprior[3 - i] *= fj;
}

static inline void gi3_G6_123(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double f = Pr[5] * exp(-3 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G1_z_i33(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double thetaI = para[8];
    double thetaZ = para[9];
    double thetaIZ = para[8] + para[9];
    double phiI3 = para[10];
    double phi3I = para[11];
    double exp0 = exp(-x1 - thetaIZ * x0 / thetaI);
    double exp1 = exp(-x1 - thetaIZ * x0 / thetaZ);
    double f = thetaIZ * (phiI3 * (1 - phi3I) * exp0 / thetaI + (1 - phiI3) * phi3I * exp1 / thetaZ);
    wwprior[0] *= 0; wwprior[1] *= 0; wwprior[2] *= f;
}

static inline void gi3_G1_m_i33(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double theta3 = para[6];
    double thetaI = para[8];
    double thetaZ = para[9];
    double thetaIZ = para[8] + para[9];
    double phiI3 = para[10];
    double phi3I = para[11];
    double exp0 = exp(-2 * T / theta3 - thetaIZ * (2 * x0 * x1 + x0) / thetaI);
    double exp1 = exp(-2 * T / theta3 - thetaIZ * (2 * x0 * x1 + x0) / thetaZ);
    double f = thetaIZ * thetaIZ * x0 * (phiI3 * phiI3 * (1 - phi3I) * exp0 / (thetaI * thetaI) + (1 - phiI3) * (1 - phiI3) * phi3I * exp1 / (thetaZ * thetaZ));
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G2_p_i33(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double thetaI = para[8];
    double thetaZ = para[9];
    double theta5Z = para[1] + para[9];
    double phiI3 = para[10];
    double phi3I = para[11];
    double exp0 = exp(-x1 - 2 * (tau1 - T) / thetaI - theta5Z * x0 / theta5);
    double exp1 = exp(-x1 - (2 * (tau1 - T) + theta5Z * x0) / thetaZ);
    double f = theta5Z * (phiI3 * (1 - phi3I) * exp0 / theta5 + (1 - phiI3) * phi3I * exp1 / thetaZ);
    wwprior[0] *= 0; wwprior[1] *= 0; wwprior[2] *= f;
}

static inline void gi3_G2_m_i33(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double theta3 = para[6];
    double thetaI = para[8];
    double thetaZ = para[9];
    double thetaIZ = para[8] + para[9];
    double theta5Z = para[1] + para[9];
    double phiI3 = para[10];
    double phi3I = para[11];
    double exp0 = exp(-2 * T / theta3 - 2 * (tau1 - T + thetaIZ * x1) / thetaI - theta5Z * x0 / theta5);
    double exp1 = exp(-2 * T / theta3 - (2 * (tau1 - T + thetaIZ * x1) + theta5Z * x0) / thetaZ);
    double f = thetaIZ * theta5Z * (phiI3 * phiI3 * (1 - phi3I) * exp0 / (thetaI * theta5) + (1 - phiI3) * (1 - phiI3) * phi3I * exp1 / (thetaZ * thetaZ));
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G3_p_i33(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double f = Pr[6] * exp(-x1 - x0);
    wwprior[0] *= 0; wwprior[1] *= 0; wwprior[2] *= f;
}

static inline void gi3_G3_m_i33(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double theta3 = para[6];
    double thetaI = para[8];
    double thetaZ = para[9];
    double thetaIZ = para[8] + para[9];
    double phiI3 = para[10];
    double phi3I = para[11];
    double exp0 = exp(-2 * T / theta3 - 2 * (tau1 - T + thetaIZ * x1) / thetaI - 2 * (tau0 - tau1) / theta5 - x0);
    double exp1 = exp(-2 * T / theta3 - thetaIZ * x1 / thetaI - x0);
    double exp2 = exp(-2 * T / theta3 - thetaIZ * x1 / thetaZ - x0);
    double exp3 = exp(-2 * T / theta3 - 2 * (tau0 - T + thetaIZ * x1) / thetaZ - x0);
    double fa = thetaIZ * (phiI3 * phiI3 * ((1 - phi3I) * exp0 + phi3I * exp1) / thetaI + (1 - phiI3) * (1 - phiI3) * ((1 - phi3I) * exp2 + phi3I * exp3) / thetaZ);
    double fbc = thetaIZ * (phiI3 * (1 - phi3I) * (phiI3 * exp0 + (1 - phiI3) * exp1) / thetaI + (1 - phiI3) * phi3I * (phiI3 * exp2 + (1 - phiI3) * exp3) / thetaZ);
    wwprior[0] *= fbc; wwprior[1] *= fbc; wwprior[2] *= fa;
}

static inline void gi3_G4_i33(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double theta3 = para[6];
    double thetaI = para[8];
    double thetaZ = para[9];
    double theta5Z = para[1] + para[9];
    double phiI3 = para[10];
    double phi3I = para[11];
    double exp0 = exp(-2 * T / theta3 - 6 * (tau1 - T) / thetaI - theta5Z * (2 * x0 * x1 + x0) / theta5);
    double exp1 = exp(-2 * T / theta3 - (6 * (tau1 - T) + theta5Z * (2 * x0 * x1 + x0)) / thetaZ);
    double f = theta5Z * theta5Z * x0 * (phiI3 * phiI3 * (1 - phi3I) * exp0 / (theta5 * theta5) + (1 - phiI3) * (1 - phiI3) * phi3I * exp1 / (thetaZ * thetaZ));
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G5_i33(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double theta3 = para[6];
    double thetaI = para[8];
    double thetaZ = para[9];
    double theta5Z = para[1] + para[9];
    double phiI3 = para[10];
    double phi3I = para[11];
    double exp0 = exp(-2 * T / theta3 - 6 * (tau1 - T) / thetaI - 2 * (tau0 - tau1 + theta5Z * x1) / theta5 - x0);
    double exp1 = exp(-2 * T / theta3 - 2 * (tau1 - T) / thetaI - theta5Z * x1 / theta5 - x0);
    double exp2 = exp(-2 * T / theta3 - 2 * (tau1 - T) / thetaZ - theta5Z * x1 / thetaZ - x0);
    double exp3 = exp(-2 * T / theta3 - (6 * (tau1 - T) + 2 * (tau0 - tau1 + theta5Z * x1)) / thetaZ - x0);
    double fa = theta5Z * (phiI3 * phiI3 * ((1 - phi3I) * exp0 + phi3I * exp1) / theta5 + (1 - phiI3) * (1 - phiI3) * ((1 - phi3I) * exp2 + phi3I * exp3) / thetaZ);
    double fbc = theta5Z * (phiI3 * (1 - phi3I) * (phiI3 * exp0 + (1 - phiI3) * exp1) / theta5 + (1 - phiI3) * phi3I * (phiI3 * exp2 + (1 - phiI3) * exp3) / thetaZ);
    wwprior[0] *= fbc; wwprior[1] *= fbc; wwprior[2] *= fa;
}

static inline void gi3_G6_i33(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double theta3 = para[6];
    double f = Pr[7] * exp(-2 * T / theta3 - 3 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G2_p_j33(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double phiI3 = para[10];
    double f = phiI3 * exp(-x1 - x0);
    wwprior[0] *= 0; wwprior[1] *= 0; wwprior[2] *= f;
}

static inline void gi3_G2_m_j33(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double theta3 = para[6];
    double phiI3 = para[10];
    double f = phiI3 * phiI3 * exp(-2 * T / theta3 - x1 - x0);
    wwprior[0] *= 0; wwprior[1] *= 0; wwprior[2] *= f;
}

static inline void gi3_G3_p_j33(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double theta5 = para[1];
    double phiI3 = para[10];
    double exp0 = exp(-2 * (tau0 - tau1) / theta5 - x1 - x0);
    double exp1 = exp(-x1 - x0);
    double f = phiI3 * exp0 + (1 - phiI3) * exp1;
    wwprior[0] *= 0; wwprior[1] *= 0; wwprior[2] *= f;
}

static inline void gi3_G3_m_j33(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double theta3 = para[6];
    double thetaI = para[8];
    double thetaZ = para[9];
    double thetaIZ = para[8] + para[9];
    double phiI3 = para[10];
    double exp0 = exp(-2 * T / theta3 - thetaIZ * x1 / thetaI - 2 * (tau0 - tau1) / theta5 - x0);
    double exp1 = exp(-2 * T / theta3 - thetaIZ * x1 / thetaZ - x0);
    double f = thetaIZ * (phiI3 * phiI3 * exp0 / thetaI + (1 - phiI3) * (1 - phiI3) * exp1 / thetaZ);
    wwprior[0] *= 0; wwprior[1] *= 0; wwprior[2] *= f;
}

static inline void gi3_G4_j33(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double T = para[7];
    double theta3 = para[6];
    double thetaI = para[8];
    double phiI3 = para[10];
    double f = x0 * phiI3 * phiI3 * exp(-2 * T / theta3 - 2 * (tau1 - T) / thetaI - 2 * x0 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G5_j33(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double theta3 = para[6];
    double thetaI = para[8];
    double thetaZ = para[9];
    double theta5Z = para[1] + para[9];
    double phiI3 = para[10];
    double exp0 = exp(-2 * T / theta3 - 2 * (tau1 - T) / thetaI - 2 * (tau0 - tau1 + theta5Z * x1) / theta5 - x0);
    double exp1 = exp(-2 * T / theta3 - theta5Z * x1 / theta5 - x0);
    double exp2 = exp(-2 * T / theta3 - (2 * (tau1 - T) + theta5Z * x1) / thetaZ - x0);
    double fa = theta5Z * (phiI3 * phiI3 * exp0 / theta5 + (1 - phiI3) * (1 - phiI3) * exp2 / thetaZ);
    double fbc = theta5Z * (phiI3 * phiI3 * exp0 + phiI3 * (1 - phiI3) * exp1) / theta5;
    wwprior[0] *= fbc; wwprior[1] *= fbc; wwprior[2] *= fa;
}

static inline void gi3_G6_j33(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double theta3 = para[6];
    double f = Pr[8] * exp(-2 * T / theta3 - 3 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G1_p_333(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double f = x0 * exp(-2 * x0 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G1_z_333(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double theta3 = para[6];
    double thetaI = para[8];
    double thetaZ = para[9];
    double thetaIZ = para[8] + para[9];
    double phiI3 = para[10];
    double exp0 = exp(-2 * T / theta3 - 2 * x1 - thetaIZ * x0 / thetaI);
    double exp1 = exp(-2 * T / theta3 - 2 * x1 - thetaIZ * x0 / thetaZ);
    double f = thetaIZ * (phiI3 * phiI3 * exp0 / thetaI + (1 - phiI3) * (1 - phiI3) * exp1 / thetaZ);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G1_m_333(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double theta3 = para[6];
    double thetaI = para[8];
    double thetaZ = para[9];
    double thetaIZ = para[8] + para[9];
    double phiI3 = para[10];
    double exp0 = exp(-6 * T / theta3 - thetaIZ * (2 * x0 * x1 + x0) / thetaI);
    double exp1 = exp(-6 * T / theta3 - thetaIZ * (2 * x0 * x1 + x0) / thetaZ);
    double f = thetaIZ * thetaIZ * x0 * (phiI3 * phiI3 * phiI3 * exp0 / (thetaI * thetaI) + (1 - phiI3) * (1 - phiI3) * (1 - phiI3) * exp1 / (thetaZ * thetaZ));
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G2_p_333(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double theta3 = para[6];
    double thetaI = para[8];
    double thetaZ = para[9];
    double theta5Z = para[1] + para[9];
    double phiI3 = para[10];
    double exp0 = exp(-2 * T / theta3 - 2 * x1 - 2 * (tau1 - T) / thetaI - theta5Z * x0 / theta5);
    double exp1 = exp(-2 * T / theta3 - 2 * x1 - (2 * (tau1 - T) + theta5Z * x0) / thetaZ);
    double f = theta5Z * (phiI3 * phiI3 * exp0 / theta5 + (1 - phiI3) * (1 - phiI3) * exp1 / thetaZ);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G2_m_333(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double theta3 = para[6];
    double thetaI = para[8];
    double thetaZ = para[9];
    double thetaIZ = para[8] + para[9];
    double theta5Z = para[1] + para[9];
    double phiI3 = para[10];
    double exp0 = exp(-6 * T / theta3 - 2 * (tau1 - T + thetaIZ * x1) / thetaI - theta5Z * x0 / theta5);
    double exp1 = exp(-6 * T / theta3 - (2 * (tau1 - T + thetaIZ * x1) + theta5Z * x0) / thetaZ);
    double f = thetaIZ * theta5Z * (phiI3 * phiI3 * phiI3 * exp0 / (thetaI * theta5) + (1 - phiI3) * (1 - phiI3) * (1 - phiI3) * exp1 / (thetaZ * thetaZ));
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G3_p_333(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double theta3 = para[6];
    double f = Pr[9] * exp(-2 * T / theta3 - 2 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G3_m_333(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double theta3 = para[6];
    double thetaI = para[8];
    double thetaZ = para[9];
    double thetaIZ = para[8] + para[9];
    double phiI3 = para[10];
    double exp0 = exp(-6 * T / theta3 - 2 * (tau1 - T + thetaIZ * x1) / thetaI - 2 * (tau0 - tau1) / theta5 - x0);
    double exp1 = exp(-6 * T / theta3 - thetaIZ * x1 / thetaI - x0);
    double exp2 = exp(-6 * T / theta3 - thetaIZ * x1 / thetaZ - x0);
    double exp3 = exp(-6 * T / theta3 - 2 * (tau0 - T + thetaIZ * x1) / thetaZ - x0);
    double f = thetaIZ * (phiI3 * phiI3 * (phiI3 * exp0 + (1 - phiI3) * exp1) / thetaI + (1 - phiI3) * (1 - phiI3) * (phiI3 * exp2 + (1 - phiI3) * exp3) / thetaZ);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G4_333(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double theta3 = para[6];
    double thetaI = para[8];
    double thetaZ = para[9];
    double theta5Z = para[1] + para[9];
    double phiI3 = para[10];
    double exp0 = exp(-6 * T / theta3 - 6 * (tau1 - T) / thetaI - theta5Z * (2 * x0 * x1 + x0) / theta5);
    double exp1 = exp(-6 * T / theta3 - (6 * (tau1 - T) + theta5Z * (2 * x0 * x1 + x0)) / thetaZ);
    double f = theta5Z * theta5Z * x0 * (phiI3 * phiI3 * phiI3 * exp0 / (theta5 * theta5) + (1 - phiI3) * (1 - phiI3) * (1 - phiI3) * exp1 / (thetaZ * thetaZ));
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G5_333(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double tau0 = para[2];
    double tau1 = para[3];
    double T = para[7];
    double theta5 = para[1];
    double theta3 = para[6];
    double thetaI = para[8];
    double thetaZ = para[9];
    double theta5Z = para[1] + para[9];
    double phiI3 = para[10];
    double exp0 = exp(-6 * T / theta3 - 6 * (tau1 - T) / thetaI - 2 * (tau0 - tau1 + theta5Z * x1) / theta5 - x0);
    double exp1 = exp(-6 * T / theta3 - 2 * (tau1 - T) / thetaI - theta5Z * x1 / theta5 - x0);
    double exp2 = exp(-6 * T / theta3 - (2 * (tau1 - T) + theta5Z * x1) / thetaZ - x0);
    double exp3 = exp(-6 * T / theta3 - (6 * (tau1 - T) + 2 * (tau0 - tau1 + theta5Z * x1)) / thetaZ - x0);
    double f = theta5Z * (phiI3 * phiI3 * (phiI3 * exp0 + (1 - phiI3) * exp1) / theta5 + (1 - phiI3) * (1 - phiI3) * (phiI3 * exp2 + (1 - phiI3) * exp3) / thetaZ);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

static inline void gi3_G6_333(int i, double x0, double x1, double* Pr, double wwprior[3]) {
    double T = para[7];
    double theta3 = para[6];
    double f = Pr[10] * exp(-6 * T / theta3 - 3 * x1 - x0);
    wwprior[0] *= f; wwprior[1] *= f; wwprior[2] *= f;
}

DEFINE_WRAPPER(g13_G1_p_111, gi3_G1_p_iii, 1)
DEFINE_WRAPPER(g13_G1_z_111, gi3_G1_z_iii, 1)
DEFINE_WRAPPER(g13_G1_m_111, gi3_G1_m_iii, 1)
DEFINE_WRAPPER(g13_G2_p_111, gi3_G2_p_iii, 1)
DEFINE_WRAPPER(g13_G2_m_111, gi3_G2_m_iii, 1)
DEFINE_WRAPPER(g13_G3_p_111, gi3_G3_p_iii, 1)
DEFINE_WRAPPER(g13_G3_m_111, gi3_G3_m_iii, 1)
DEFINE_WRAPPER(g13_G4_111, gi3_G4_iii, 1)
DEFINE_WRAPPER(g13_G5_111, gi3_G5_iii, 1)
DEFINE_WRAPPER(g13_G6_111, gi3_G6_iii, 1)
DEFINE_WRAPPER(g13_G1_222, gi3_G1_jjj, 1)
DEFINE_WRAPPER(g13_G2_222, gi3_G2_jjj, 1)
DEFINE_WRAPPER(g13_G3_222, gi3_G3_jjj, 1)
DEFINE_WRAPPER(g13_G4_222, gi3_G4_jjj, 1)
DEFINE_WRAPPER(g13_G5_222, gi3_G5_jjj, 1)
DEFINE_WRAPPER(g13_G6_222, gi3_G6_jjj, 1)
DEFINE_WRAPPER(g13_G2_p_112, gi3_G2_p_iij, 1)
DEFINE_WRAPPER(g13_G2_m_112, gi3_G2_m_iij, 1)
DEFINE_WRAPPER(g13_G3_p_112, gi3_G3_p_iij, 1)
DEFINE_WRAPPER(g13_G3_m_112, gi3_G3_m_iij, 1)
DEFINE_WRAPPER(g13_G4_112, gi3_G4_iij, 1)
DEFINE_WRAPPER(g13_G5_112, gi3_G5_iij, 1)
DEFINE_WRAPPER(g13_G6_112, gi3_G6_iij, 1)
DEFINE_WRAPPER(g13_G2_221, gi3_G2_jji, 1)
DEFINE_WRAPPER(g13_G3_221, gi3_G3_jji, 1)
DEFINE_WRAPPER(g13_G4_221, gi3_G4_jji, 1)
DEFINE_WRAPPER(g13_G5_221, gi3_G5_jji, 1)
DEFINE_WRAPPER(g13_G6_221, gi3_G6_jji, 1)
DEFINE_WRAPPER(g13_G1_z_113, gi3_G1_z_ii3, 1)
DEFINE_WRAPPER(g13_G1_m_113, gi3_G1_m_ii3, 1)
DEFINE_WRAPPER(g13_G2_p_113, gi3_G2_p_ii3, 1)
DEFINE_WRAPPER(g13_G2_m_113, gi3_G2_m_ii3, 1)
DEFINE_WRAPPER(g13_G3_p_113, gi3_G3_p_ii3, 1)
DEFINE_WRAPPER(g13_G3_m_113, gi3_G3_m_ii3, 1)
DEFINE_WRAPPER(g13_G4_113, gi3_G4_ii3, 1)
DEFINE_WRAPPER(g13_G5_113, gi3_G5_ii3, 1)
DEFINE_WRAPPER(g13_G6_113, gi3_G6_ii3, 1)
DEFINE_WRAPPER(g13_G2_223, gi3_G2_jj3, 1)
DEFINE_WRAPPER(g13_G3_223, gi3_G3_jj3, 1)
DEFINE_WRAPPER(g13_G4_223, gi3_G4_jj3, 1)
DEFINE_WRAPPER(g13_G5_223, gi3_G5_jj3, 1)
DEFINE_WRAPPER(g13_G6_223, gi3_G6_jj3, 1)
DEFINE_WRAPPER(g13_G2_m_123, gi3_G2_m_123, 1)
DEFINE_WRAPPER(g13_G3_m_123, gi3_G3_m_123, 1)
DEFINE_WRAPPER(g13_G4_123, gi3_G4_123, 1)
DEFINE_WRAPPER(g13_G5_123, gi3_G5_123, 1)
DEFINE_WRAPPER(g13_G6_123, gi3_G6_123, 1)
DEFINE_WRAPPER(g13_G1_z_133, gi3_G1_z_i33, 1)
DEFINE_WRAPPER(g13_G1_m_133, gi3_G1_m_i33, 1)
DEFINE_WRAPPER(g13_G2_p_133, gi3_G2_p_i33, 1)
DEFINE_WRAPPER(g13_G2_m_133, gi3_G2_m_i33, 1)
DEFINE_WRAPPER(g13_G3_p_133, gi3_G3_p_i33, 1)
DEFINE_WRAPPER(g13_G3_m_133, gi3_G3_m_i33, 1)
DEFINE_WRAPPER(g13_G4_133, gi3_G4_i33, 1)
DEFINE_WRAPPER(g13_G5_133, gi3_G5_i33, 1)
DEFINE_WRAPPER(g13_G6_133, gi3_G6_i33, 1)
DEFINE_WRAPPER(g13_G2_p_233, gi3_G2_p_j33, 1)
DEFINE_WRAPPER(g13_G2_m_233, gi3_G2_m_j33, 1)
DEFINE_WRAPPER(g13_G3_p_233, gi3_G3_p_j33, 1)
DEFINE_WRAPPER(g13_G3_m_233, gi3_G3_m_j33, 1)
DEFINE_WRAPPER(g13_G4_233, gi3_G4_j33, 1)
DEFINE_WRAPPER(g13_G5_233, gi3_G5_j33, 1)
DEFINE_WRAPPER(g13_G6_233, gi3_G6_j33, 1)
DEFINE_WRAPPER(g13_G1_p_333, gi3_G1_p_333, 1)
DEFINE_WRAPPER(g13_G1_z_333, gi3_G1_z_333, 1)
DEFINE_WRAPPER(g13_G1_m_333, gi3_G1_m_333, 1)
DEFINE_WRAPPER(g13_G2_p_333, gi3_G2_p_333, 1)
DEFINE_WRAPPER(g13_G2_m_333, gi3_G2_m_333, 1)
DEFINE_WRAPPER(g13_G3_p_333, gi3_G3_p_333, 1)
DEFINE_WRAPPER(g13_G3_m_333, gi3_G3_m_333, 1)
DEFINE_WRAPPER(g13_G4_333, gi3_G4_333, 1)
DEFINE_WRAPPER(g13_G5_333, gi3_G5_333, 1)
DEFINE_WRAPPER(g13_G6_333, gi3_G6_333, 1)

DEFINE_WRAPPER(g23_G1_p_222, gi3_G1_p_iii, 2)
DEFINE_WRAPPER(g23_G1_z_222, gi3_G1_z_iii, 2)
DEFINE_WRAPPER(g23_G1_m_222, gi3_G1_m_iii, 2)
DEFINE_WRAPPER(g23_G2_p_222, gi3_G2_p_iii, 2)
DEFINE_WRAPPER(g23_G2_m_222, gi3_G2_m_iii, 2)
DEFINE_WRAPPER(g23_G3_p_222, gi3_G3_p_iii, 2)
DEFINE_WRAPPER(g23_G3_m_222, gi3_G3_m_iii, 2)
DEFINE_WRAPPER(g23_G4_222, gi3_G4_iii, 2)
DEFINE_WRAPPER(g23_G5_222, gi3_G5_iii, 2)
DEFINE_WRAPPER(g23_G6_222, gi3_G6_iii, 2)
DEFINE_WRAPPER(g23_G1_111, gi3_G1_jjj, 2)
DEFINE_WRAPPER(g23_G2_111, gi3_G2_jjj, 2)
DEFINE_WRAPPER(g23_G3_111, gi3_G3_jjj, 2)
DEFINE_WRAPPER(g23_G4_111, gi3_G4_jjj, 2)
DEFINE_WRAPPER(g23_G5_111, gi3_G5_jjj, 2)
DEFINE_WRAPPER(g23_G6_111, gi3_G6_jjj, 2)
DEFINE_WRAPPER(g23_G2_p_221, gi3_G2_p_iij, 2)
DEFINE_WRAPPER(g23_G2_m_221, gi3_G2_m_iij, 2)
DEFINE_WRAPPER(g23_G3_p_221, gi3_G3_p_iij, 2)
DEFINE_WRAPPER(g23_G3_m_221, gi3_G3_m_iij, 2)
DEFINE_WRAPPER(g23_G4_221, gi3_G4_iij, 2)
DEFINE_WRAPPER(g23_G5_221, gi3_G5_iij, 2)
DEFINE_WRAPPER(g23_G6_221, gi3_G6_iij, 2)
DEFINE_WRAPPER(g23_G2_112, gi3_G2_jji, 2)
DEFINE_WRAPPER(g23_G3_112, gi3_G3_jji, 2)
DEFINE_WRAPPER(g23_G4_112, gi3_G4_jji, 2)
DEFINE_WRAPPER(g23_G5_112, gi3_G5_jji, 2)
DEFINE_WRAPPER(g23_G6_112, gi3_G6_jji, 2)
DEFINE_WRAPPER(g23_G1_z_223, gi3_G1_z_ii3, 2)
DEFINE_WRAPPER(g23_G1_m_223, gi3_G1_m_ii3, 2)
DEFINE_WRAPPER(g23_G2_p_223, gi3_G2_p_ii3, 2)
DEFINE_WRAPPER(g23_G2_m_223, gi3_G2_m_ii3, 2)
DEFINE_WRAPPER(g23_G3_p_223, gi3_G3_p_ii3, 2)
DEFINE_WRAPPER(g23_G3_m_223, gi3_G3_m_ii3, 2)
DEFINE_WRAPPER(g23_G4_223, gi3_G4_ii3, 2)
DEFINE_WRAPPER(g23_G5_223, gi3_G5_ii3, 2)
DEFINE_WRAPPER(g23_G6_223, gi3_G6_ii3, 2)
DEFINE_WRAPPER(g23_G2_113, gi3_G2_jj3, 2)
DEFINE_WRAPPER(g23_G3_113, gi3_G3_jj3, 2)
DEFINE_WRAPPER(g23_G4_113, gi3_G4_jj3, 2)
DEFINE_WRAPPER(g23_G5_113, gi3_G5_jj3, 2)
DEFINE_WRAPPER(g23_G6_113, gi3_G6_jj3, 2)
DEFINE_WRAPPER(g23_G2_m_123, gi3_G2_m_123, 2)
DEFINE_WRAPPER(g23_G3_m_123, gi3_G3_m_123, 2)
DEFINE_WRAPPER(g23_G4_123, gi3_G4_123, 2)
DEFINE_WRAPPER(g23_G5_123, gi3_G5_123, 2)
DEFINE_WRAPPER(g23_G6_123, gi3_G6_123, 2)
DEFINE_WRAPPER(g23_G1_z_233, gi3_G1_z_i33, 2)
DEFINE_WRAPPER(g23_G1_m_233, gi3_G1_m_i33, 2)
DEFINE_WRAPPER(g23_G2_p_233, gi3_G2_p_i33, 2)
DEFINE_WRAPPER(g23_G2_m_233, gi3_G2_m_i33, 2)
DEFINE_WRAPPER(g23_G3_p_233, gi3_G3_p_i33, 2)
DEFINE_WRAPPER(g23_G3_m_233, gi3_G3_m_i33, 2)
DEFINE_WRAPPER(g23_G4_233, gi3_G4_i33, 2)
DEFINE_WRAPPER(g23_G5_233, gi3_G5_i33, 2)
DEFINE_WRAPPER(g23_G6_233, gi3_G6_i33, 2)
DEFINE_WRAPPER(g23_G2_p_133, gi3_G2_p_j33, 2)
DEFINE_WRAPPER(g23_G2_m_133, gi3_G2_m_j33, 2)
DEFINE_WRAPPER(g23_G3_p_133, gi3_G3_p_j33, 2)
DEFINE_WRAPPER(g23_G3_m_133, gi3_G3_m_j33, 2)
DEFINE_WRAPPER(g23_G4_133, gi3_G4_j33, 2)
DEFINE_WRAPPER(g23_G5_133, gi3_G5_j33, 2)
DEFINE_WRAPPER(g23_G6_133, gi3_G6_j33, 2)
DEFINE_WRAPPER(g23_G1_p_333, gi3_G1_p_333, 2)
DEFINE_WRAPPER(g23_G1_z_333, gi3_G1_z_333, 2)
DEFINE_WRAPPER(g23_G1_m_333, gi3_G1_m_333, 2)
DEFINE_WRAPPER(g23_G2_p_333, gi3_G2_p_333, 2)
DEFINE_WRAPPER(g23_G2_m_333, gi3_G2_m_333, 2)
DEFINE_WRAPPER(g23_G3_p_333, gi3_G3_p_333, 2)
DEFINE_WRAPPER(g23_G3_m_333, gi3_G3_m_333, 2)
DEFINE_WRAPPER(g23_G4_333, gi3_G4_333, 2)
DEFINE_WRAPPER(g23_G5_333, gi3_G5_333, 2)
DEFINE_WRAPPER(g23_G6_333, gi3_G6_333, 2)

void b_G2_1_5Z(double x0, double x1, double theta12, double b[2]) {
    double theta1 = para[4];
    double theta5Z = para[1] + para[9];
    b_type_2(x0, x1, theta1, theta5Z, b);
}

void b_G2_2_5Z(double x0, double x1, double theta12, double b[2]) {
    double theta2 = para[5];
    double theta5Z = para[1] + para[9];
    b_type_2(x0, x1, theta2, theta5Z, b);
}

void b_G2_3_5Z(double x0, double x1, double theta12, double b[2]) {
    double theta3 = para[6];
    double theta5Z = para[1] + para[9];
    b_type_2(x0, x1, theta3, theta5Z, b);
}

void b_G2_3_5(double x0, double x1, double theta12, double b[2]) {
    double theta3 = para[6];
    double theta5 = para[1];
    b_type_2(x0, x1, theta3, theta5, b);
}

void b_G4_5Z_5Z(double x0, double x1, double theta12, double b[2]) {
    double theta5Z = para[1] + para[9];
    b_type_4(x0, x1, theta5Z, b);
}

void b_G5_5Z_4(double x0, double x1, double theta12, double b[2]) {
    double theta5Z = para[1] + para[9];
    double theta4 = para[0];
    b_type_5(x0, x1, theta5Z, theta4, b);
}

void b_G1_1_XZ(double x0, double x1, double theta12, double b[2]) {
    double theta1 = para[4];
    double thetaXZ = para[8] + para[9];
    b_type_7(x0, x1, theta1, thetaXZ, b);
}

void b_G1_3_XZ(double x0, double x1, double theta12, double b[2]) {
    double theta3 = para[6];
    double thetaXZ = para[8] + para[9];
    b_type_7(x0, x1, theta3, thetaXZ, b);
}

void b_G1_XZ_XZ(double x0, double x1, double theta12, double b[2]) {
    double thetaXZ = para[8] + para[9];
    b_type_8(x0, x1, thetaXZ, b);
}

void b_G2_XZ_5Z(double x0, double x1, double theta12, double b[2]) {
    double thetaXZ = para[8] + para[9];
    double theta5Z = para[1] + para[9];
    b_type_9(x0, x1, thetaXZ, theta5Z, b);
}

void b_G2_X_5(double x0, double x1, double theta12, double b[2]) {
    double thetaX = para[8];
    double theta5 = para[1];
    b_type_9(x0, x1, thetaX, theta5, b);
}

void b_G3_XZ_4(double x0, double x1, double theta12, double b[2]) {
    double thetaXZ = para[8] + para[9];
    double theta4 = para[0];
    b_type_10(x0, x1, thetaXZ, theta4, b);
}

void b_G1_2_YZ(double x0, double x1, double theta12, double b[2]) {
    double theta2 = para[5];
    double thetaYZ = para[8] + para[9];
    b_type_7(x0, x1, theta2, thetaYZ, b);
}

void b_G1_3_YZ(double x0, double x1, double theta12, double b[2]) {
    double theta3 = para[6];
    double thetaYZ = para[8] + para[9];
    b_type_7(x0, x1, theta3, thetaYZ, b);
}

void b_G1_YZ_YZ(double x0, double x1, double theta12, double b[2]) {
    double thetaYZ = para[8] + para[9];
    b_type_8(x0, x1, thetaYZ, b);
}

void b_G2_YZ_5Z(double x0, double x1, double theta12, double b[2]) {
    double thetaYZ = para[8] + para[9];
    double theta5Z = para[1] + para[9];
    b_type_9(x0, x1, thetaYZ, theta5Z, b);
}

void b_G2_Y_5(double x0, double x1, double theta12, double b[2]) {
    double thetaY = para[8];
    double theta5 = para[1];
    b_type_9(x0, x1, thetaY, theta5, b);
}

void b_G3_YZ_4(double x0, double x1, double theta12, double b[2]) {
    double thetaYZ = para[8] + para[9];
    double theta4 = para[0];
    b_type_10(x0, x1, thetaYZ, theta4, b);
}


// Gene tree as data

void bTot_G1_1_1(double b[2], double t[2], double x[2]) {
    double theta1 = para[4];
    bTot_type_1(b, theta1, t, x);
}

void bTot_G1_2_2(double b[2], double t[2], double x[2]) {
    double theta2 = para[5];
    bTot_type_1(b, theta2, t, x);
}

void bTot_G1_3_3(double b[2], double t[2], double x[2]) {
    double theta3 = para[6];
    bTot_type_1(b, theta3, t, x);
}

void bTot_G2_1_5(double b[2], double t[2], double x[2]) {
    double theta1 = para[4];
    double theta5 = para[1];
    bTot_type_2(b, theta1, theta5, t, x);
}

void bTot_G2_2_5(double b[2], double t[2], double x[2]) {
    double theta2 = para[5];
    double theta5 = para[1];
    bTot_type_2(b, theta2, theta5, t, x);
}

void bTot_G3_1_4(double b[2], double t[2], double x[2]) {
    double theta1 = para[4];
    double theta4 = para[0];
    bTot_type_3(b, theta1, theta4, t, x);
}

void bTot_G3_2_4(double b[2], double t[2], double x[2]) {
    double theta2 = para[5];
    double theta4 = para[0];
    bTot_type_3(b, theta2, theta4, t, x);
}

void bTot_G3_3_4(double b[2], double t[2], double x[2]) {
    double theta3 = para[6];
    double theta4 = para[0];
    bTot_type_3(b, theta3, theta4, t, x);
}

void bTot_G4_5_5(double b[2], double t[2], double x[2]) {
    double theta5 = para[1];
    bTot_type_4(b, theta5, t, x);
}

void bTot_G5_5_4(double b[2], double t[2], double x[2]) {
    double theta5 = para[1];
    double theta4 = para[0];
    bTot_type_5(b, theta5, theta4, t, x);
}

void bTot_G6_4_4(double b[2], double t[2], double x[2]) {
    double theta4 = para[0];
    bTot_type_6(b, theta4, t, x);
}

void bTot_G1_12_12(double b[2], double t[2], double x[2]) {
    double theta12 = (para[4] + para[5]) / 2;
    bTot_type_1(b, theta12, t, x);
}

void bTot_G2_12_5(double b[2], double t[2], double x[2]) {
    double theta12 = (para[4] + para[5]) / 2;
    double theta5 = para[1];
    bTot_type_2(b, theta12, theta5, t, x);
}

void bTot_G3_12_4(double b[2], double t[2], double x[2]) {
    double theta12 = (para[4] + para[5]) / 2;
    double theta4 = para[0];
    bTot_type_3(b, theta12, theta4, t, x);
}

void bTot_G1_123_123(double b[2], double t[2], double x[2]) {
    double theta123 = para[4] + para[5] + para[6];
    bTot_type_1(b, theta123, t, x);
}

void bTot_G2_123_5W(double b[2], double t[2], double x[2]) {
    double theta123 = para[4] + para[5] + para[6];
    double theta5W = para[1] + para[7];
    bTot_type_2(b, theta123, theta5W, t, x);
}

void bTot_G3_123_4(double b[2], double t[2], double x[2]) {
    double theta123 = para[4] + para[5] + para[6];
    double theta4 = para[0];
    bTot_type_3(b, theta123, theta4, t, x);
}

void bTot_G4_5W_5W(double b[2], double t[2], double x[2]) {
    double theta5W = para[1] + para[7];
    bTot_type_4(b, theta5W, t, x);
}

void bTot_G5_5W_4(double b[2], double t[2], double x[2]) {
    double theta5W = para[1] + para[7];
    double theta4 = para[0];
    bTot_type_5(b, theta5W, theta4, t, x);
}

void bTot_G1_1_XY(double b[2], double t[2], double x[2]) {
    double theta1 = para[4];
    double thetaXY = para[8] + para[9];
    bTot_type_7(b, theta1, thetaXY, t, x);
}

void bTot_G1_2_XY(double b[2], double t[2], double x[2]) {
    double theta2 = para[5];
    double thetaXY = para[8] + para[9];
    bTot_type_7(b, theta2, thetaXY, t, x);
}

void bTot_G1_XY_XY(double b[2], double t[2], double x[2]) {
    double thetaXY = para[8] + para[9];
    bTot_type_8(b, thetaXY, t, x);
}

void bTot_G2_XY_5(double b[2], double t[2], double x[2]) {
    double thetaXY = para[8] + para[9];
    double theta5 = para[1];
    bTot_type_9(b, thetaXY, theta5, t, x);
}

void bTot_G3_XY_4(double b[2], double t[2], double x[2]) {
    double thetaXY = para[8] + para[9];
    double theta4 = para[0];
    bTot_type_10(b, thetaXY, theta4, t, x);
}

void bTot_G2_1_5Z(double b[2], double t[2], double x[2]) {
    double theta1 = para[4];
    double theta5Z = para[1] + para[9];
    bTot_type_2(b, theta1, theta5Z, t, x);
}

void bTot_G2_2_5Z(double b[2], double t[2], double x[2]) {
    double theta2 = para[5];
    double theta5Z = para[1] + para[9];
    bTot_type_2(b, theta2, theta5Z, t, x);
}

void bTot_G2_3_5Z(double b[2], double t[2], double x[2]) {
    double theta3 = para[6];
    double theta5Z = para[1] + para[9];
    bTot_type_2(b, theta3, theta5Z, t, x);
}

void bTot_G2_3_5(double b[2], double t[2], double x[2]) {
    double theta3 = para[6];
    double theta5 = para[1];
    bTot_type_2(b, theta3, theta5, t, x);
}

void bTot_G4_5Z_5Z(double b[2], double t[2], double x[2]) {
    double theta5Z = para[1] + para[9];
    bTot_type_4(b, theta5Z, t, x);
}

void bTot_G5_5Z_4(double b[2], double t[2], double x[2]) {
    double theta5Z = para[1] + para[9];
    double theta4 = para[0];
    bTot_type_5(b, theta5Z, theta4, t, x);
}

void bTot_G1_1_XZ(double b[2], double t[2], double x[2]) {
    double theta1 = para[4];
    double thetaXZ = para[8] + para[9];
    bTot_type_7(b, theta1, thetaXZ, t, x);
}

void bTot_G1_3_XZ(double b[2], double t[2], double x[2]) {
    double theta3 = para[6];
    double thetaXZ = para[8] + para[9];
    bTot_type_7(b, theta3, thetaXZ, t, x);
}

void bTot_G1_XZ_XZ(double b[2], double t[2], double x[2]) {
    double thetaXZ = para[8] + para[9];
    bTot_type_8(b, thetaXZ, t, x);
}

void bTot_G2_XZ_5Z(double b[2], double t[2], double x[2]) {
    double thetaXZ = para[8] + para[9];
    double theta5Z = para[1] + para[9];
    bTot_type_9(b, thetaXZ, theta5Z, t, x);
}

void bTot_G2_X_5(double b[2], double t[2], double x[2]) {
    double thetaX = para[8];
    double theta5 = para[1];
    bTot_type_9(b, thetaX, theta5, t, x);
}

void bTot_G3_XZ_4(double b[2], double t[2], double x[2]) {
    double thetaXZ = para[8] + para[9];
    double theta4 = para[0];
    bTot_type_10(b, thetaXZ, theta4, t, x);
}

void bTot_G1_2_YZ(double b[2], double t[2], double x[2]) {
    double theta2 = para[5];
    double thetaYZ = para[8] + para[9];
    bTot_type_7(b, theta2, thetaYZ, t, x);
}

void bTot_G1_3_YZ(double b[2], double t[2], double x[2]) {
    double theta3 = para[6];
    double thetaYZ = para[8] + para[9];
    bTot_type_7(b, theta3, thetaYZ, t, x);
}

void bTot_G1_YZ_YZ(double b[2], double t[2], double x[2]) {
    double thetaYZ = para[8] + para[9];
    bTot_type_8(b, thetaYZ, t, x);
}

void bTot_G2_YZ_5Z(double b[2], double t[2], double x[2]) {
    double thetaYZ = para[8] + para[9];
    double theta5Z = para[1] + para[9];
    bTot_type_9(b, thetaYZ, theta5Z, t, x);
}

void bTot_G2_Y_5(double b[2], double t[2], double x[2]) {
    double thetaY = para[8];
    double theta5 = para[1];
    bTot_type_9(b, thetaY, theta5, t, x);
}

void bTot_G3_YZ_4(double b[2], double t[2], double x[2]) {
    double thetaYZ = para[8] + para[9];
    double theta4 = para[0];
    bTot_type_10(b, thetaYZ, theta4, t, x);
}

double detJ_G1_1_1(double x0, double x1) {
    double theta1 = para[4];
    return detJ_type_1(x0, theta1);
}

double detJ_G1_2_2(double x0, double x1) {
    double theta2 = para[5];
    return detJ_type_1(x0, theta2);
}

double detJ_G1_3_3(double x0, double x1) {
    double theta3 = para[6];
    return detJ_type_1(x0, theta3);
}

double detJ_G2_1_5(double x0, double x1) {
    double theta1 = para[4];
    double theta5 = para[1];
    return detJ_type_2(theta1, theta5);
}

double detJ_G2_2_5(double x0, double x1) {
    double theta2 = para[5];
    double theta5 = para[1];
    return detJ_type_2(theta2, theta5);
}

double detJ_G3_1_4(double x0, double x1) {
    double theta1 = para[4];
    double theta4 = para[0];
    return detJ_type_2(theta1, theta4);
}

double detJ_G3_2_4(double x0, double x1) {
    double theta2 = para[5];
    double theta4 = para[0];
    return detJ_type_2(theta2, theta4);
}

double detJ_G3_3_4(double x0, double x1) {
    double theta3 = para[6];
    double theta4 = para[0];
    return detJ_type_2(theta3, theta4);
}

double detJ_G4_5_5(double x0, double x1) {
    double theta5 = para[1];
    return detJ_type_1(x0, theta5);
}

double detJ_G5_5_4(double x0, double x1) {
    double theta5 = para[1];
    double theta4 = para[0];
    return detJ_type_2(theta5, theta4);
}

double detJ_G6_4_4(double x0, double x1) {
    double theta4 = para[0];
    return detJ_type_3(theta4);
}

double detJ_G1_12_12(double x0, double x1) {
    double theta12 = (para[4] + para[5]) / 2;
    return detJ_type_1(x0, theta12);
}

double detJ_G2_12_5(double x0, double x1) {
    double theta12 = (para[4] + para[5]) / 2;
    double theta5 = para[1];
    return detJ_type_2(theta12, theta5);
}

double detJ_G3_12_4(double x0, double x1) {
    double theta12 = (para[4] + para[5]) / 2;
    double theta4 = para[0];
    return detJ_type_2(theta12, theta4);
}

double detJ_G1_123_123(double x0, double x1) {
    double theta123 = para[4] + para[5] + para[6];
    return detJ_type_1(x0, theta123);
}

double detJ_G2_123_5W(double x0, double x1) {
    double theta123 = para[4] + para[5] + para[6];
    double theta5W = para[1] + para[7];
    return detJ_type_2(theta123, theta5W);
}

double detJ_G3_123_4(double x0, double x1) {
    double theta123 = para[4] + para[5] + para[6];
    double theta4 = para[0];
    return detJ_type_2(theta123, theta4);
}

double detJ_G4_5W_5W(double x0, double x1) {
    double theta5W = para[1] + para[7];
    return detJ_type_1(x0, theta5W);
}

double detJ_G5_5W_4(double x0, double x1) {
    double theta5W = para[1] + para[7];
    double theta4 = para[0];
    return detJ_type_2(theta5W, theta4);
}

double detJ_G1_1_XY(double x0, double x1) {
    double theta1 = para[4];
    double thetaXY = para[8] + para[9];
    return detJ_type_2(theta1, thetaXY);
}

double detJ_G1_2_XY(double x0, double x1) {
    double theta2 = para[5];
    double thetaXY = para[8] + para[9];
    return detJ_type_2(theta2, thetaXY);
}

double detJ_G1_XY_XY(double x0, double x1) {
    double thetaXY = para[8] + para[9];
    return detJ_type_1(x0, thetaXY);
}

double detJ_G2_XY_5(double x0, double x1) {
    double thetaXY = para[8] + para[9];
    double theta5 = para[1];
    return detJ_type_2(thetaXY, theta5);
}

double detJ_G3_XY_4(double x0, double x1) {
    double thetaXY = para[8] + para[9];
    double theta4 = para[0];
    return detJ_type_2(thetaXY, theta4);
}

double detJ_G2_1_5Z(double x0, double x1) {
    double theta1 = para[4];
    double theta5Z = para[1] + para[9];
    return detJ_type_2(theta1, theta5Z);
}

double detJ_G2_2_5Z(double x0, double x1) {
    double theta2 = para[5];
    double theta5Z = para[1] + para[9];
    return detJ_type_2(theta2, theta5Z);
}

double detJ_G2_3_5Z(double x0, double x1) {
    double theta3 = para[6];
    double theta5Z = para[1] + para[9];
    return detJ_type_2(theta3, theta5Z);
}

double detJ_G2_3_5(double x0, double x1) {
    double theta3 = para[6];
    double theta5 = para[1];
    return detJ_type_2(theta3, theta5);
}

double detJ_G4_5Z_5Z(double x0, double x1) {
    double theta5Z = para[1] + para[9];
    return detJ_type_1(x0, theta5Z);
}

double detJ_G5_5Z_4(double x0, double x1) {
    double theta5Z = para[1] + para[9];
    double theta4 = para[0];
    return detJ_type_2(theta5Z, theta4);
}

double detJ_G1_1_XZ(double x0, double x1) {
    double theta1 = para[4];
    double thetaXZ = para[8] + para[9];
    return detJ_type_2(theta1, thetaXZ);
}

double detJ_G1_3_XZ(double x0, double x1) {
    double theta3 = para[6];
    double thetaXZ = para[8] + para[9];
    return detJ_type_2(theta3, thetaXZ);
}

double detJ_G1_XZ_XZ(double x0, double x1) {
    double thetaXZ = para[8] + para[9];
    return detJ_type_1(x0, thetaXZ);
}

double detJ_G2_XZ_5Z(double x0, double x1) {
    double thetaXZ = para[8] + para[9];
    double theta5Z = para[1] + para[9];
    return detJ_type_2(thetaXZ, theta5Z);
}

double detJ_G2_X_5(double x0, double x1) {
    double thetaX = para[8];
    double theta5 = para[1];
    return detJ_type_2(thetaX, theta5);
}

double detJ_G3_XZ_4(double x0, double x1) {
    double thetaXZ = para[8] + para[9];
    double theta4 = para[0];
    return detJ_type_2(thetaXZ, theta4);
}

double detJ_G1_2_YZ(double x0, double x1) {
    double theta2 = para[5];
    double thetaYZ = para[8] + para[9];
    return detJ_type_2(theta2, thetaYZ);
}

double detJ_G1_3_YZ(double x0, double x1) {
    double theta3 = para[6];
    double thetaYZ = para[8] + para[9];
    return detJ_type_2(theta3, thetaYZ);
}

double detJ_G1_YZ_YZ(double x0, double x1) {
    double thetaYZ = para[8] + para[9];
    return detJ_type_1(x0, thetaYZ);
}

double detJ_G2_YZ_5Z(double x0, double x1) {
    double thetaYZ = para[8] + para[9];
    double theta5Z = para[1] + para[9];
    return detJ_type_2(thetaYZ, theta5Z);
}

double detJ_G2_Y_5(double x0, double x1) {
    double thetaY = para[8];
    double theta5 = para[1];
    return detJ_type_2(thetaY, theta5);
}

double detJ_G3_YZ_4(double x0, double x1) {
    double thetaYZ = para[8] + para[9];
    double theta4 = para[0];
    return detJ_type_2(thetaYZ, theta4);
}


/* 
 * This function initializes the lookup table used to simplify the likelihood
 * calculation. It is based on tables S2 and S4 from Dalquen et al. 2016
 */
void update_limits() {
    double theta4, theta5, tau0, tau1, theta1=0.0, theta2=0.0, theta12, theta3=0.0, taugap, ltau0, ltau1, ltau2;
    double theta5W, theta123, thetaX, thetaY, thetaXY, theta5Z, thetaXZ, thetaYZ, T;
    double lT1, lT2, lT3, ltaugap, ltaugap5, ltaugap5Z, ltau1T, ltau1TX, ltau1TXZ, ltau1TY, ltau1TYZ;
    int i;
    BTEntry ** gtt = com.GtreeTab;

    getParams(&theta4, &theta5, &tau0, &tau1, &theta1, &theta2, &theta3);

    if(tau0<tau1)
        printf("tau0 = %12.8f < tau1 = %12.8f\n", tau0, tau1);

    if (com.model == M2Pro || com.model == M2ProMax) {
        theta5W = para[1] + para[7];
        theta123 = theta1 + theta2 + theta3;
        ltau1 = 2 * tau1 / (2 * tau1 + theta123);
        ltaugap = 2 * (tau0 - tau1) / (2 * (tau0 - tau1) + theta5W);

        gtt[0][0].lim[0] = ltau1;
        gtt[0][0].lim[1] = 0.5;
        gtt[0][1].lim[0] = ltaugap;
        gtt[0][1].lim[1] = ltau1;
        gtt[0][2].lim[0] = 1;
        gtt[0][2].lim[1] = ltau1;
        gtt[0][3].lim[0] = ltaugap;
        gtt[0][3].lim[1] = 0.5;
        gtt[0][4].lim[0] = 1;
        gtt[0][4].lim[1] = ltaugap;
        gtt[0][5].lim[0] = 1;
        gtt[0][5].lim[1] = 1;
    }

    else if (com.model == M3MSci12) {
        T = para[7];
        thetaXY = para[8] + para[9];
        lT1 = 2 * T / (2 * T + theta1);
        lT2 = 2 * T / (2 * T + theta2);
        ltau0 = 2 * tau0 / (2 * tau0 + theta3);
        ltaugap = 2 * (tau0 - tau1) / (2 * (tau0 - tau1) + theta5);
        ltau1T = 2 * (tau1 - T) / (2 * (tau1 - T) + thetaXY);

        gtt[ 0][0].lim[0] = lT1;
        gtt[ 0][0].lim[1] = 0.5;
        gtt[13][0].lim[0] = lT2;
        gtt[13][0].lim[1] = 0.5;
        gtt[26][0].lim[0] = ltau0;
        gtt[26][0].lim[1] = 0.5;

        gtt[ 0][1].lim[0] = ltaugap;
        gtt[ 0][1].lim[1] = lT1;
        gtt[13][1].lim[0] = ltaugap;
        gtt[13][1].lim[1] = lT2;
        gtt[ 1][1].lim[0] = ltaugap;
        gtt[ 1][1].lim[1] = lT1;
        gtt[12][1].lim[0] = ltaugap;
        gtt[12][1].lim[1] = lT2;

        gtt[ 0][2].lim[0] = 1;
        gtt[ 0][2].lim[1] = lT1;
        gtt[13][2].lim[0] = 1;
        gtt[13][2].lim[1] = lT2;
        gtt[ 1][2].lim[0] = 1;
        gtt[ 1][2].lim[1] = lT1;
        gtt[12][2].lim[0] = 1;
        gtt[12][2].lim[1] = lT2;
        gtt[ 2][2].lim[0] = 1;
        gtt[ 2][2].lim[1] = lT1;
        gtt[14][2].lim[0] = 1;
        gtt[14][2].lim[1] = lT2;
        gtt[ 8][2].lim[0] = 1;
        gtt[ 8][2].lim[1] = ltau0;
        gtt[26][2].lim[0] = 1;
        gtt[26][2].lim[1] = ltau0;

        gtt[ 0][3].lim[0] = ltaugap;
        gtt[ 0][3].lim[1] = 0.5;
        gtt[13][3].lim[0] = ltaugap;
        gtt[13][3].lim[1] = 0.5;
        gtt[ 1][3].lim[0] = ltaugap;
        gtt[ 1][3].lim[1] = 0.5;
        gtt[12][3].lim[0] = ltaugap;
        gtt[12][3].lim[1] = 0.5;

        gtt[ 0][4].lim[0] = 1;
        gtt[ 0][4].lim[1] = ltaugap;
        gtt[13][4].lim[0] = 1;
        gtt[13][4].lim[1] = ltaugap;
        gtt[ 1][4].lim[0] = 1;
        gtt[ 1][4].lim[1] = ltaugap;
        gtt[12][4].lim[0] = 1;
        gtt[12][4].lim[1] = ltaugap;
        gtt[ 2][4].lim[0] = 1;
        gtt[ 2][4].lim[1] = ltaugap;
        gtt[14][4].lim[0] = 1;
        gtt[14][4].lim[1] = ltaugap;
        gtt[ 5][4].lim[0] = 1;
        gtt[ 5][4].lim[1] = ltaugap;

        gtt[ 0][5].lim[0] = 1;
        gtt[ 0][5].lim[1] = 1;
        gtt[13][5].lim[0] = 1;
        gtt[13][5].lim[1] = 1;
        gtt[ 1][5].lim[0] = 1;
        gtt[ 1][5].lim[1] = 1;
        gtt[12][5].lim[0] = 1;
        gtt[12][5].lim[1] = 1;
        gtt[ 2][5].lim[0] = 1;
        gtt[ 2][5].lim[1] = 1;
        gtt[14][5].lim[0] = 1;
        gtt[14][5].lim[1] = 1;
        gtt[ 5][5].lim[0] = 1;
        gtt[ 5][5].lim[1] = 1;
        gtt[ 8][5].lim[0] = 1;
        gtt[ 8][5].lim[1] = 1;
        gtt[26][5].lim[0] = 1;
        gtt[26][5].lim[1] = 1;

        gtt[ 0][6].lim[0] = ltau1T;
        gtt[ 0][6].lim[1] = lT1;
        gtt[13][6].lim[0] = ltau1T;
        gtt[13][6].lim[1] = lT2;
        gtt[ 1][6].lim[0] = ltau1T;
        gtt[ 1][6].lim[1] = lT1;
        gtt[12][6].lim[0] = ltau1T;
        gtt[12][6].lim[1] = lT2;

        gtt[ 0][7].lim[0] = ltau1T;
        gtt[ 0][7].lim[1] = 0.5;
        gtt[13][7].lim[0] = ltau1T;
        gtt[13][7].lim[1] = 0.5;
        gtt[ 1][7].lim[0] = ltau1T;
        gtt[ 1][7].lim[1] = 0.5;
        gtt[12][7].lim[0] = ltau1T;
        gtt[12][7].lim[1] = 0.5;

        gtt[ 0][8].lim[0] = ltaugap;
        gtt[ 0][8].lim[1] = ltau1T;
        gtt[13][8].lim[0] = ltaugap;
        gtt[13][8].lim[1] = ltau1T;
        gtt[ 1][8].lim[0] = ltaugap;
        gtt[ 1][8].lim[1] = ltau1T;
        gtt[12][8].lim[0] = ltaugap;
        gtt[12][8].lim[1] = ltau1T;

        gtt[ 0][9].lim[0] = 1;
        gtt[ 0][9].lim[1] = ltau1T;
        gtt[13][9].lim[0] = 1;
        gtt[13][9].lim[1] = ltau1T;
        gtt[ 1][9].lim[0] = 1;
        gtt[ 1][9].lim[1] = ltau1T;
        gtt[12][9].lim[0] = 1;
        gtt[12][9].lim[1] = ltau1T;
        gtt[ 2][9].lim[0] = 1;
        gtt[ 2][9].lim[1] = ltau1T;
        gtt[14][9].lim[0] = 1;
        gtt[14][9].lim[1] = ltau1T;
        gtt[ 5][9].lim[0] = 1;
        gtt[ 5][9].lim[1] = ltau1T;
    }

    else if (com.model == M3MSci13) {
        T = para[7];
        thetaX = para[8];
        theta5Z = para[1] + para[9];
        thetaXZ = para[8] + para[9];
        lT1 = 2 * T / (2 * T + theta1);
        lT3 = 2 * T / (2 * T + theta3);
        ltau1 = 2 * tau1 / (2 * tau1 + theta2);
        ltaugap5Z = 2 * (tau0 - tau1) / (2 * (tau0 - tau1) + theta5Z);
        ltaugap5 = 2 * (tau0 - tau1) / (2 * (tau0 - tau1) + theta5);
        ltau1TXZ = 2 * (tau1 - T) / (2 * (tau1 - T) + thetaXZ);
        ltau1TX = 2 * (tau1 - T) / (2 * (tau1 - T) + thetaX);

        gtt[ 0][0].lim[0] = lT1;
        gtt[ 0][0].lim[1] = 0.5;
        gtt[13][0].lim[0] = ltau1;
        gtt[13][0].lim[1] = 0.5;
        gtt[26][0].lim[0] = lT3;
        gtt[26][0].lim[1] = 0.5;

        gtt[ 0][1].lim[0] = ltaugap5Z;
        gtt[ 0][1].lim[1] = lT1;
        gtt[13][1].lim[0] = ltaugap5;
        gtt[13][1].lim[1] = ltau1;
        gtt[ 1][1].lim[0] = ltaugap5;
        gtt[ 1][1].lim[1] = lT1;
        gtt[12][1].lim[0] = ltaugap5;
        gtt[12][1].lim[1] = ltau1;
        gtt[ 2][1].lim[0] = ltaugap5Z;
        gtt[ 2][1].lim[1] = lT1;
        gtt[14][1].lim[0] = ltaugap5;
        gtt[14][1].lim[1] = ltau1;
        gtt[ 8][1].lim[0] = ltaugap5Z;
        gtt[ 8][1].lim[1] = lT3;
        gtt[17][1].lim[0] = ltaugap5;
        gtt[17][1].lim[1] = lT3;
        gtt[26][1].lim[0] = ltaugap5Z;
        gtt[26][1].lim[1] = lT3;

        gtt[ 0][2].lim[0] = 1;
        gtt[ 0][2].lim[1] = lT1;
        gtt[13][2].lim[0] = 1;
        gtt[13][2].lim[1] = ltau1;
        gtt[ 1][2].lim[0] = 1;
        gtt[ 1][2].lim[1] = lT1;
        gtt[12][2].lim[0] = 1;
        gtt[12][2].lim[1] = ltau1;
        gtt[ 2][2].lim[0] = 1;
        gtt[ 2][2].lim[1] = lT1;
        gtt[14][2].lim[0] = 1;
        gtt[14][2].lim[1] = ltau1;
        gtt[ 8][2].lim[0] = 1;
        gtt[ 8][2].lim[1] = lT3;
        gtt[17][2].lim[0] = 1;
        gtt[17][2].lim[1] = lT3;
        gtt[26][2].lim[0] = 1;
        gtt[26][2].lim[1] = lT3;

        gtt[ 0][3].lim[0] = ltaugap5Z;
        gtt[ 0][3].lim[1] = 0.5;
        gtt[13][3].lim[0] = ltaugap5;
        gtt[13][3].lim[1] = 0.5;
        gtt[ 1][3].lim[0] = ltaugap5;
        gtt[ 1][3].lim[1] = 0.5;
        gtt[12][3].lim[0] = ltaugap5;
        gtt[12][3].lim[1] = 0.5;
        gtt[ 2][3].lim[0] = ltaugap5Z;
        gtt[ 2][3].lim[1] = 0.5;
        gtt[14][3].lim[0] = ltaugap5;
        gtt[14][3].lim[1] = 0.5;
        gtt[ 5][3].lim[0] = ltaugap5;
        gtt[ 5][3].lim[1] = 0.5;
        gtt[ 8][3].lim[0] = ltaugap5Z;
        gtt[ 8][3].lim[1] = 0.5;
        gtt[17][3].lim[0] = ltaugap5;
        gtt[17][3].lim[1] = 0.5;
        gtt[26][3].lim[0] = ltaugap5Z;
        gtt[26][3].lim[1] = 0.5;

        gtt[ 0][4].lim[0] = 1;
        gtt[ 0][4].lim[1] = ltaugap5Z;
        gtt[13][4].lim[0] = 1;
        gtt[13][4].lim[1] = ltaugap5;
        gtt[ 1][4].lim[0] = 1;
        gtt[ 1][4].lim[1] = ltaugap5Z;
        gtt[12][4].lim[0] = 1;
        gtt[12][4].lim[1] = ltaugap5;
        gtt[ 2][4].lim[0] = 1;
        gtt[ 2][4].lim[1] = ltaugap5Z;
        gtt[14][4].lim[0] = 1;
        gtt[14][4].lim[1] = ltaugap5;
        gtt[ 5][4].lim[0] = 1;
        gtt[ 5][4].lim[1] = ltaugap5Z;
        gtt[ 8][4].lim[0] = 1;
        gtt[ 8][4].lim[1] = ltaugap5Z;
        gtt[17][4].lim[0] = 1;
        gtt[17][4].lim[1] = ltaugap5Z;
        gtt[26][4].lim[0] = 1;
        gtt[26][4].lim[1] = ltaugap5Z;

        gtt[ 0][5].lim[0] = 1;
        gtt[ 0][5].lim[1] = 1;
        gtt[13][5].lim[0] = 1;
        gtt[13][5].lim[1] = 1;
        gtt[ 1][5].lim[0] = 1;
        gtt[ 1][5].lim[1] = 1;
        gtt[12][5].lim[0] = 1;
        gtt[12][5].lim[1] = 1;
        gtt[ 2][5].lim[0] = 1;
        gtt[ 2][5].lim[1] = 1;
        gtt[14][5].lim[0] = 1;
        gtt[14][5].lim[1] = 1;
        gtt[ 5][5].lim[0] = 1;
        gtt[ 5][5].lim[1] = 1;
        gtt[ 8][5].lim[0] = 1;
        gtt[ 8][5].lim[1] = 1;
        gtt[17][5].lim[0] = 1;
        gtt[17][5].lim[1] = 1;
        gtt[26][5].lim[0] = 1;
        gtt[26][5].lim[1] = 1;

        gtt[ 0][6].lim[0] = ltau1TXZ;
        gtt[ 0][6].lim[1] = lT1;
        gtt[ 2][6].lim[0] = ltau1TXZ;
        gtt[ 2][6].lim[1] = lT1;
        gtt[ 8][6].lim[0] = ltau1TXZ;
        gtt[ 8][6].lim[1] = lT3;
        gtt[26][6].lim[0] = ltau1TXZ;
        gtt[26][6].lim[1] = lT3;

        gtt[ 0][7].lim[0] = ltau1TXZ;
        gtt[ 0][7].lim[1] = 0.5;
        gtt[ 2][7].lim[0] = ltau1TXZ;
        gtt[ 2][7].lim[1] = 0.5;
        gtt[ 8][7].lim[0] = ltau1TXZ;
        gtt[ 8][7].lim[1] = 0.5;
        gtt[26][7].lim[0] = ltau1TXZ;
        gtt[26][7].lim[1] = 0.5;

        gtt[ 0][8].lim[0] = ltaugap5Z;
        gtt[ 0][8].lim[1] = ltau1TXZ;
        gtt[ 1][8].lim[0] = ltaugap5;
        gtt[ 1][8].lim[1] = ltau1TX;
        gtt[ 2][8].lim[0] = ltaugap5Z;
        gtt[ 2][8].lim[1] = ltau1TXZ;
        gtt[ 5][8].lim[0] = ltaugap5;
        gtt[ 5][8].lim[1] = ltau1TX;
        gtt[ 8][8].lim[0] = ltaugap5Z;
        gtt[ 8][8].lim[1] = ltau1TXZ;
        gtt[17][8].lim[0] = ltaugap5;
        gtt[17][8].lim[1] = ltau1TX;
        gtt[26][8].lim[0] = ltaugap5Z;
        gtt[26][8].lim[1] = ltau1TXZ;

        gtt[ 0][9].lim[0] = 1;
        gtt[ 0][9].lim[1] = ltau1TXZ;
        gtt[ 1][9].lim[0] = 1;
        gtt[ 1][9].lim[1] = ltau1TXZ;
        gtt[ 2][9].lim[0] = 1;
        gtt[ 2][9].lim[1] = ltau1TXZ;
        gtt[ 5][9].lim[0] = 1;
        gtt[ 5][9].lim[1] = ltau1TXZ;
        gtt[ 8][9].lim[0] = 1;
        gtt[ 8][9].lim[1] = ltau1TXZ;
        gtt[17][9].lim[0] = 1;
        gtt[17][9].lim[1] = ltau1TXZ;
        gtt[26][9].lim[0] = 1;
        gtt[26][9].lim[1] = ltau1TXZ;
    }

    else if (com.model == M3MSci23) {
        T = para[7];
        thetaY = para[8];
        theta5Z = para[1] + para[9];
        thetaYZ = para[8] + para[9];
        lT2 = 2 * T / (2 * T + theta2);
        lT3 = 2 * T / (2 * T + theta3);
        ltau1 = 2 * tau1 / (2 * tau1 + theta1);
        ltaugap5Z = 2 * (tau0 - tau1) / (2 * (tau0 - tau1) + theta5Z);
        ltaugap5 = 2 * (tau0 - tau1) / (2 * (tau0 - tau1) + theta5);
        ltau1TYZ = 2 * (tau1 - T) / (2 * (tau1 - T) + thetaYZ);
        ltau1TY = 2 * (tau1 - T) / (2 * (tau1 - T) + thetaY);

        gtt[13][0].lim[0] = lT2;
        gtt[13][0].lim[1] = 0.5;
        gtt[ 0][0].lim[0] = ltau1;
        gtt[ 0][0].lim[1] = 0.5;
        gtt[26][0].lim[0] = lT3;
        gtt[26][0].lim[1] = 0.5;

        gtt[13][1].lim[0] = ltaugap5Z;
        gtt[13][1].lim[1] = lT2;
        gtt[ 0][1].lim[0] = ltaugap5;
        gtt[ 0][1].lim[1] = ltau1;
        gtt[12][1].lim[0] = ltaugap5;
        gtt[12][1].lim[1] = lT2;
        gtt[ 1][1].lim[0] = ltaugap5;
        gtt[ 1][1].lim[1] = ltau1;
        gtt[14][1].lim[0] = ltaugap5Z;
        gtt[14][1].lim[1] = lT2;
        gtt[ 2][1].lim[0] = ltaugap5;
        gtt[ 2][1].lim[1] = ltau1;
        gtt[17][1].lim[0] = ltaugap5Z;
        gtt[17][1].lim[1] = lT3;
        gtt[ 8][1].lim[0] = ltaugap5;
        gtt[ 8][1].lim[1] = lT3;
        gtt[26][1].lim[0] = ltaugap5Z;
        gtt[26][1].lim[1] = lT3;

        gtt[13][2].lim[0] = 1;
        gtt[13][2].lim[1] = lT2;
        gtt[ 0][2].lim[0] = 1;
        gtt[ 0][2].lim[1] = ltau1;
        gtt[12][2].lim[0] = 1;
        gtt[12][2].lim[1] = lT2;
        gtt[ 1][2].lim[0] = 1;
        gtt[ 1][2].lim[1] = ltau1;
        gtt[14][2].lim[0] = 1;
        gtt[14][2].lim[1] = lT2;
        gtt[ 2][2].lim[0] = 1;
        gtt[ 2][2].lim[1] = ltau1;
        gtt[17][2].lim[0] = 1;
        gtt[17][2].lim[1] = lT3;
        gtt[ 8][2].lim[0] = 1;
        gtt[ 8][2].lim[1] = lT3;
        gtt[26][2].lim[0] = 1;
        gtt[26][2].lim[1] = lT3;

        gtt[13][3].lim[0] = ltaugap5Z;
        gtt[13][3].lim[1] = 0.5;
        gtt[ 0][3].lim[0] = ltaugap5;
        gtt[ 0][3].lim[1] = 0.5;
        gtt[12][3].lim[0] = ltaugap5;
        gtt[12][3].lim[1] = 0.5;
        gtt[ 1][3].lim[0] = ltaugap5;
        gtt[ 1][3].lim[1] = 0.5;
        gtt[14][3].lim[0] = ltaugap5Z;
        gtt[14][3].lim[1] = 0.5;
        gtt[ 2][3].lim[0] = ltaugap5;
        gtt[ 2][3].lim[1] = 0.5;
        gtt[ 5][3].lim[0] = ltaugap5;
        gtt[ 5][3].lim[1] = 0.5;
        gtt[17][3].lim[0] = ltaugap5Z;
        gtt[17][3].lim[1] = 0.5;
        gtt[ 8][3].lim[0] = ltaugap5;
        gtt[ 8][3].lim[1] = 0.5;
        gtt[26][3].lim[0] = ltaugap5Z;
        gtt[26][3].lim[1] = 0.5;

        gtt[13][4].lim[0] = 1;
        gtt[13][4].lim[1] = ltaugap5Z;
        gtt[ 0][4].lim[0] = 1;
        gtt[ 0][4].lim[1] = ltaugap5;
        gtt[12][4].lim[0] = 1;
        gtt[12][4].lim[1] = ltaugap5Z;
        gtt[ 1][4].lim[0] = 1;
        gtt[ 1][4].lim[1] = ltaugap5;
        gtt[14][4].lim[0] = 1;
        gtt[14][4].lim[1] = ltaugap5Z;
        gtt[ 2][4].lim[0] = 1;
        gtt[ 2][4].lim[1] = ltaugap5;
        gtt[ 5][4].lim[0] = 1;
        gtt[ 5][4].lim[1] = ltaugap5Z;
        gtt[17][4].lim[0] = 1;
        gtt[17][4].lim[1] = ltaugap5Z;
        gtt[ 8][4].lim[0] = 1;
        gtt[ 8][4].lim[1] = ltaugap5Z;
        gtt[26][4].lim[0] = 1;
        gtt[26][4].lim[1] = ltaugap5Z;

        gtt[13][5].lim[0] = 1;
        gtt[13][5].lim[1] = 1;
        gtt[ 0][5].lim[0] = 1;
        gtt[ 0][5].lim[1] = 1;
        gtt[12][5].lim[0] = 1;
        gtt[12][5].lim[1] = 1;
        gtt[ 1][5].lim[0] = 1;
        gtt[ 1][5].lim[1] = 1;
        gtt[14][5].lim[0] = 1;
        gtt[14][5].lim[1] = 1;
        gtt[ 2][5].lim[0] = 1;
        gtt[ 2][5].lim[1] = 1;
        gtt[ 5][5].lim[0] = 1;
        gtt[ 5][5].lim[1] = 1;
        gtt[17][5].lim[0] = 1;
        gtt[17][5].lim[1] = 1;
        gtt[ 8][5].lim[0] = 1;
        gtt[ 8][5].lim[1] = 1;
        gtt[26][5].lim[0] = 1;
        gtt[26][5].lim[1] = 1;

        gtt[13][6].lim[0] = ltau1TYZ;
        gtt[13][6].lim[1] = lT2;
        gtt[14][6].lim[0] = ltau1TYZ;
        gtt[14][6].lim[1] = lT2;
        gtt[17][6].lim[0] = ltau1TYZ;
        gtt[17][6].lim[1] = lT3;
        gtt[26][6].lim[0] = ltau1TYZ;
        gtt[26][6].lim[1] = lT3;

        gtt[13][7].lim[0] = ltau1TYZ;
        gtt[13][7].lim[1] = 0.5;
        gtt[14][7].lim[0] = ltau1TYZ;
        gtt[14][7].lim[1] = 0.5;
        gtt[17][7].lim[0] = ltau1TYZ;
        gtt[17][7].lim[1] = 0.5;
        gtt[26][7].lim[0] = ltau1TYZ;
        gtt[26][7].lim[1] = 0.5;

        gtt[13][8].lim[0] = ltaugap5Z;
        gtt[13][8].lim[1] = ltau1TYZ;
        gtt[12][8].lim[0] = ltaugap5;
        gtt[12][8].lim[1] = ltau1TY;
        gtt[14][8].lim[0] = ltaugap5Z;
        gtt[14][8].lim[1] = ltau1TYZ;
        gtt[ 5][8].lim[0] = ltaugap5;
        gtt[ 5][8].lim[1] = ltau1TY;
        gtt[17][8].lim[0] = ltaugap5Z;
        gtt[17][8].lim[1] = ltau1TYZ;
        gtt[ 8][8].lim[0] = ltaugap5;
        gtt[ 8][8].lim[1] = ltau1TY;
        gtt[26][8].lim[0] = ltaugap5Z;
        gtt[26][8].lim[1] = ltau1TYZ;

        gtt[13][9].lim[0] = 1;
        gtt[13][9].lim[1] = ltau1TYZ;
        gtt[12][9].lim[0] = 1;
        gtt[12][9].lim[1] = ltau1TYZ;
        gtt[14][9].lim[0] = 1;
        gtt[14][9].lim[1] = ltau1TYZ;
        gtt[ 5][9].lim[0] = 1;
        gtt[ 5][9].lim[1] = ltau1TYZ;
        gtt[17][9].lim[0] = 1;
        gtt[17][9].lim[1] = ltau1TYZ;
        gtt[ 8][9].lim[0] = 1;
        gtt[ 8][9].lim[1] = ltau1TYZ;
        gtt[26][9].lim[0] = 1;
        gtt[26][9].lim[1] = ltau1TYZ;
    }

    else {
        theta12 = (theta1+theta2)/2;
        if (com.model == M2SIM3s) {
            theta1 = theta2 = theta12;
        }

        taugap = 2*(tau0 - tau1)/theta5;
        if (com.model == M2SIM3s) {
            ltau1 = ltau2 = 2*tau1/(theta12+2*tau1);
        } else {
            ltau1 = 2*tau1/(2*tau1+theta1);
            ltau2 = 2*tau1/(2*tau1+theta2);
        }
    //    ltau1 = 2*tau1/(theta12*(1+2*tau1/theta12));
        ltau0 = 2*tau0/(2*tau0+theta3);

        // initial states 111/222
        // integration limits
        gtt[ 0][0].lim[1] = gtt[ 0][3].lim[1] = 0.5;
        gtt[13][0].lim[1] = gtt[13][3].lim[1] = 0.5;
        gtt[ 0][0].lim[0] = gtt[ 0][1].lim[1] = gtt[ 0][2].lim[1] = ltau1; // transformation y = t/(t+1)
        gtt[13][0].lim[0] = gtt[13][1].lim[1] = gtt[13][2].lim[1] = ltau2; // transformation y = t/(t+1)
        gtt[ 0][1].lim[0] = gtt[ 0][3].lim[0] = gtt[ 0][4].lim[1] = (taugap)/(1+taugap);
        gtt[13][1].lim[0] = gtt[13][3].lim[0] = gtt[13][4].lim[1] = (taugap)/(1+taugap);
        gtt[ 0][2].lim[0] = gtt[ 0][4].lim[0] = gtt[ 0][5].lim[0] = gtt[ 0][5].lim[1] = 1;
        gtt[13][2].lim[0] = gtt[13][4].lim[0] = gtt[13][5].lim[0] = gtt[13][5].lim[1] = 1;

        /* ------ */

        // initial states 112/122
        for(i = 1; i < 6; i++) {
            gtt[ 1][i].lim[0] = gtt[ 0][i].lim[0];
            gtt[ 1][i].lim[1] = gtt[ 0][i].lim[1];
            gtt[12][i].lim[0] = gtt[13][i].lim[0];
            gtt[12][i].lim[1] = gtt[13][i].lim[1];
        }

        if(com.model == M2SIM3s) {
            gtt[ 1][0].lim[0] = gtt[ 0][0].lim[0];
            gtt[ 1][0].lim[1] = gtt[ 0][0].lim[1];
            gtt[12][0].lim[0] = gtt[13][0].lim[0];
            gtt[12][0].lim[1] = gtt[13][0].lim[1];
        }

        /* ------ */

        // initial states 113/223/123
        gtt[ 2][2].lim[1] = ltau1;
        gtt[14][2].lim[1] = ltau2;
        gtt[ 2][4].lim[1] = gtt[14][4].lim[1] = gtt[ 5][4].lim[1] = taugap/(1+taugap);
        gtt[ 2][2].lim[0] = gtt[ 2][4].lim[0] = gtt[ 2][5].lim[0] = gtt[ 2][5].lim[1] = 1;
        gtt[14][2].lim[0] = gtt[14][4].lim[0] = gtt[14][5].lim[0] = gtt[14][5].lim[1] = 1;
        gtt[ 5][4].lim[0] = gtt[ 5][5].lim[0] = gtt[ 5][5].lim[1] = 1;
        if (com.model == M2SIM3s) {
            gtt[5][2].lim[0] = 1;
            gtt[5][2].lim[1] = ltau1;
        }

        /* ------ */

        // initial states 133/233
        gtt[8][2].lim[0] = gtt[8][5].lim[0] = gtt[8][5].lim[1] = 1;
        gtt[8][2].lim[1] = ltau0;

        // initial state 333
        gtt[26][0].lim[1] = 0.5;
        gtt[26][0].lim[0] = gtt[26][2].lim[1] = ltau0;
        gtt[26][2].lim[0] = gtt[26][5].lim[0] = gtt[26][5].lim[1] = 1;
    }
}

void setupGtreeTab() {
    if (com.usedata == ESeqData)
        setupGtreeTab_seqdata();
    else
        setupGtreeTab_treedata();
}

void setupGtreeTab_seqdata() {
    BTEntry ** gtt = com.GtreeTab;

    if (com.model == M2Pro || com.model == M2ProMax) {
        if (com.model == M2Pro) {
            gtt[0][0].helper = helper_G1;
            gtt[0][1].helper = helper_G2;
            gtt[0][2].helper = helper_G3;
            gtt[0][3].helper = helper_G4;
            gtt[0][4].helper = helper_G5;
            gtt[0][5].helper = helper_G6;

            gtt[0][0].density = density_G123;
            gtt[0][1].density = density_G123;
            gtt[0][2].density = density_G123;
            gtt[0][3].density = density_G4;
            gtt[0][4].density = density_G5;
            gtt[0][5].density = density_G6;
        }
        else {
            gtt[0][0].helper = helper_G1_PM;
            gtt[0][1].helper = helper_G2_PM;
            gtt[0][2].helper = helper_G3_PM;
            gtt[0][3].helper = helper_G4_PM;
            gtt[0][4].helper = helper_G5_PM;
            gtt[0][5].helper = helper_G6_PM;

            gtt[0][0].density = density_G123;
            gtt[0][1].density = density_G123;
            gtt[0][2].density = density_G123;
            gtt[0][3].density = density_G45;
            gtt[0][4].density = density_G45;
            gtt[0][5].density = density_G6;
        }

        gtt[0][0].t0t1 = t0t1_G1_123_123;
        gtt[0][1].t0t1 = t0t1_G2_123_5W;
        gtt[0][2].t0t1 = t0t1_G3_123_4;
        gtt[0][3].t0t1 = t0t1_G4_5W_5W;
        gtt[0][4].t0t1 = t0t1_G5_5W_4;
        gtt[0][5].t0t1 = t0t1_G6_4_4;

        gtt[0][0].b = b_G1_123_123;
        gtt[0][1].b = b_G2_123_5W;
        gtt[0][2].b = b_G3_123_4;
        gtt[0][3].b = b_G4_5W_5W;
        gtt[0][4].b = b_G5_5W_4;
        gtt[0][5].b = b_G6_4_4;
    }

    else if (com.model == M3MSci12) {
        gtt[0][0].g = g12_G1_p_111;
        gtt[0][1].g = g12_G2_p_111;
        gtt[0][2].g = g12_G3_p_111;
        gtt[0][3].g = g12_G4_111;
        gtt[0][4].g = g12_G5_111;
        gtt[0][5].g = g12_G6_111;
        gtt[0][6].g = g12_G1_z_111;
        gtt[0][7].g = g12_G1_m_111;
        gtt[0][8].g = g12_G2_m_111;
        gtt[0][9].g = g12_G3_m_111;

        gtt[13][0].g = g12_G1_p_222;
        gtt[13][1].g = g12_G2_p_222;
        gtt[13][2].g = g12_G3_p_222;
        gtt[13][3].g = g12_G4_222;
        gtt[13][4].g = g12_G5_222;
        gtt[13][5].g = g12_G6_222;
        gtt[13][6].g = g12_G1_z_222;
        gtt[13][7].g = g12_G1_m_222;
        gtt[13][8].g = g12_G2_m_222;
        gtt[13][9].g = g12_G3_m_222;

        gtt[1][1].g = g12_G2_p_112;
        gtt[1][2].g = g12_G3_p_112;
        gtt[1][3].g = g12_G4_112;
        gtt[1][4].g = g12_G5_112;
        gtt[1][5].g = g12_G6_112;
        gtt[1][6].g = g12_G1_z_112;
        gtt[1][7].g = g12_G1_m_112;
        gtt[1][8].g = g12_G2_m_112;
        gtt[1][9].g = g12_G3_m_112;

        gtt[12][1].g = g12_G2_p_221;
        gtt[12][2].g = g12_G3_p_221;
        gtt[12][3].g = g12_G4_221;
        gtt[12][4].g = g12_G5_221;
        gtt[12][5].g = g12_G6_221;
        gtt[12][6].g = g12_G1_z_221;
        gtt[12][7].g = g12_G1_m_221;
        gtt[12][8].g = g12_G2_m_221;
        gtt[12][9].g = g12_G3_m_221;

        gtt[2][2].g = g12_G3_p_113;
        gtt[2][4].g = g12_G5_113;
        gtt[2][5].g = g12_G6_113;
        gtt[2][9].g = g12_G3_m_113;

        gtt[14][2].g = g12_G3_p_223;
        gtt[14][4].g = g12_G5_223;
        gtt[14][5].g = g12_G6_223;
        gtt[14][9].g = g12_G3_m_223;

        gtt[5][4].g = g12_G5_123;
        gtt[5][5].g = g12_G6_123;
        gtt[5][9].g = g12_G3_m_123;

        gtt[8][2].g = g12_G3_133;
        gtt[8][5].g = g12_G6_133;

        gtt[26][0].g = g12_G1_333;
        gtt[26][2].g = g12_G3_333;
        gtt[26][5].g = g12_G6_333;

        gtt[ 0][0].b = b_G1_1_1;
        gtt[13][0].b = b_G1_2_2;
        gtt[26][0].b = b_G1_3_3;

        gtt[ 0][1].b = b_G2_1_5;
        gtt[13][1].b = b_G2_2_5;
        gtt[ 1][1].b = b_G2_1_5;
        gtt[12][1].b = b_G2_2_5;

        gtt[ 0][2].b = b_G3_1_4;
        gtt[13][2].b = b_G3_2_4;
        gtt[ 1][2].b = b_G3_1_4;
        gtt[12][2].b = b_G3_2_4;
        gtt[ 2][2].b = b_G3_1_4;
        gtt[14][2].b = b_G3_2_4;
        gtt[ 8][2].b = b_G3_3_4;
        gtt[26][2].b = b_G3_3_4;

        gtt[ 0][3].b = b_G4_5_5;
        gtt[13][3].b = b_G4_5_5;
        gtt[ 1][3].b = b_G4_5_5;
        gtt[12][3].b = b_G4_5_5;

        gtt[ 0][4].b = b_G5_5_4;
        gtt[13][4].b = b_G5_5_4;
        gtt[ 1][4].b = b_G5_5_4;
        gtt[12][4].b = b_G5_5_4;
        gtt[ 2][4].b = b_G5_5_4;
        gtt[14][4].b = b_G5_5_4;
        gtt[ 5][4].b = b_G5_5_4;

        gtt[ 0][5].b = b_G6_4_4;
        gtt[13][5].b = b_G6_4_4;
        gtt[ 1][5].b = b_G6_4_4;
        gtt[12][5].b = b_G6_4_4;
        gtt[ 2][5].b = b_G6_4_4;
        gtt[14][5].b = b_G6_4_4;
        gtt[ 5][5].b = b_G6_4_4;
        gtt[ 8][5].b = b_G6_4_4;
        gtt[26][5].b = b_G6_4_4;

        gtt[ 0][6].b = b_G1_1_XY;
        gtt[13][6].b = b_G1_2_XY;
        gtt[ 1][6].b = b_G1_1_XY;
        gtt[12][6].b = b_G1_2_XY;

        gtt[ 0][7].b = b_G1_XY_XY;
        gtt[13][7].b = b_G1_XY_XY;
        gtt[ 1][7].b = b_G1_XY_XY;
        gtt[12][7].b = b_G1_XY_XY;

        gtt[ 0][8].b = b_G2_XY_5;
        gtt[13][8].b = b_G2_XY_5;
        gtt[ 1][8].b = b_G2_XY_5;
        gtt[12][8].b = b_G2_XY_5;

        gtt[ 0][9].b = b_G3_XY_4;
        gtt[13][9].b = b_G3_XY_4;
        gtt[ 1][9].b = b_G3_XY_4;
        gtt[12][9].b = b_G3_XY_4;
        gtt[ 2][9].b = b_G3_XY_4;
        gtt[14][9].b = b_G3_XY_4;
        gtt[ 5][9].b = b_G3_XY_4;
    }

    else if (com.model == M3MSci13) {
        gtt[0][0].g = g13_G1_p_111;
        gtt[0][1].g = g13_G2_p_111;
        gtt[0][2].g = g13_G3_p_111;
        gtt[0][3].g = g13_G4_111;
        gtt[0][4].g = g13_G5_111;
        gtt[0][5].g = g13_G6_111;
        gtt[0][6].g = g13_G1_z_111;
        gtt[0][7].g = g13_G1_m_111;
        gtt[0][8].g = g13_G2_m_111;
        gtt[0][9].g = g13_G3_m_111;

        gtt[13][0].g = g13_G1_222;
        gtt[13][1].g = g13_G2_222;
        gtt[13][2].g = g13_G3_222;
        gtt[13][3].g = g13_G4_222;
        gtt[13][4].g = g13_G5_222;
        gtt[13][5].g = g13_G6_222;

        gtt[1][1].g = g13_G2_p_112;
        gtt[1][2].g = g13_G3_p_112;
        gtt[1][3].g = g13_G4_112;
        gtt[1][4].g = g13_G5_112;
        gtt[1][5].g = g13_G6_112;
        gtt[1][8].g = g13_G2_m_112;
        gtt[1][9].g = g13_G3_m_112;

        gtt[12][1].g = g13_G2_221;
        gtt[12][2].g = g13_G3_221;
        gtt[12][3].g = g13_G4_221;
        gtt[12][4].g = g13_G5_221;
        gtt[12][5].g = g13_G6_221;

        gtt[2][1].g = g13_G2_p_113;
        gtt[2][2].g = g13_G3_p_113;
        gtt[2][3].g = g13_G4_113;
        gtt[2][4].g = g13_G5_113;
        gtt[2][5].g = g13_G6_113;
        gtt[2][6].g = g13_G1_z_113;
        gtt[2][7].g = g13_G1_m_113;
        gtt[2][8].g = g13_G2_m_113;
        gtt[2][9].g = g13_G3_m_113;

        gtt[14][1].g = g13_G2_223;
        gtt[14][2].g = g13_G3_223;
        gtt[14][3].g = g13_G4_223;
        gtt[14][4].g = g13_G5_223;
        gtt[14][5].g = g13_G6_223;

        gtt[5][3].g = g13_G4_123;
        gtt[5][4].g = g13_G5_123;
        gtt[5][5].g = g13_G6_123;
        gtt[5][8].g = g13_G2_m_123;
        gtt[5][9].g = g13_G3_m_123;

        gtt[8][1].g = g13_G2_p_133;
        gtt[8][2].g = g13_G3_p_133;
        gtt[8][3].g = g13_G4_133;
        gtt[8][4].g = g13_G5_133;
        gtt[8][5].g = g13_G6_133;
        gtt[8][6].g = g13_G1_z_133;
        gtt[8][7].g = g13_G1_m_133;
        gtt[8][8].g = g13_G2_m_133;
        gtt[8][9].g = g13_G3_m_133;

        gtt[17][1].g = g13_G2_p_233;
        gtt[17][2].g = g13_G3_p_233;
        gtt[17][3].g = g13_G4_233;
        gtt[17][4].g = g13_G5_233;
        gtt[17][5].g = g13_G6_233;
        gtt[17][8].g = g13_G2_m_233;
        gtt[17][9].g = g13_G3_m_233;

        gtt[26][0].g = g13_G1_p_333;
        gtt[26][1].g = g13_G2_p_333;
        gtt[26][2].g = g13_G3_p_333;
        gtt[26][3].g = g13_G4_333;
        gtt[26][4].g = g13_G5_333;
        gtt[26][5].g = g13_G6_333;
        gtt[26][6].g = g13_G1_z_333;
        gtt[26][7].g = g13_G1_m_333;
        gtt[26][8].g = g13_G2_m_333;
        gtt[26][9].g = g13_G3_m_333;

        gtt[ 0][0].b = b_G1_1_1;
        gtt[13][0].b = b_G1_2_2;
        gtt[26][0].b = b_G1_3_3;

        gtt[ 0][1].b = b_G2_1_5Z;
        gtt[13][1].b = b_G2_2_5;
        gtt[ 1][1].b = b_G2_1_5;
        gtt[12][1].b = b_G2_2_5;
        gtt[ 2][1].b = b_G2_1_5Z;
        gtt[14][1].b = b_G2_2_5;
        gtt[ 8][1].b = b_G2_3_5Z;
        gtt[17][1].b = b_G2_3_5;
        gtt[26][1].b = b_G2_3_5Z;

        gtt[ 0][2].b = b_G3_1_4;
        gtt[13][2].b = b_G3_2_4;
        gtt[ 1][2].b = b_G3_1_4;
        gtt[12][2].b = b_G3_2_4;
        gtt[ 2][2].b = b_G3_1_4;
        gtt[14][2].b = b_G3_2_4;
        gtt[ 8][2].b = b_G3_3_4;
        gtt[17][2].b = b_G3_3_4;
        gtt[26][2].b = b_G3_3_4;

        gtt[ 0][3].b = b_G4_5Z_5Z;
        gtt[13][3].b = b_G4_5_5;
        gtt[ 1][3].b = b_G4_5_5;
        gtt[12][3].b = b_G4_5_5;
        gtt[ 2][3].b = b_G4_5Z_5Z;
        gtt[14][3].b = b_G4_5_5;
        gtt[ 5][3].b = b_G4_5_5;
        gtt[ 8][3].b = b_G4_5Z_5Z;
        gtt[17][3].b = b_G4_5_5;
        gtt[26][3].b = b_G4_5Z_5Z;

        gtt[ 0][4].b = b_G5_5Z_4;
        gtt[13][4].b = b_G5_5_4;
        gtt[ 1][4].b = b_G5_5Z_4;
        gtt[12][4].b = b_G5_5_4;
        gtt[ 2][4].b = b_G5_5Z_4;
        gtt[14][4].b = b_G5_5_4;
        gtt[ 5][4].b = b_G5_5Z_4;
        gtt[ 8][4].b = b_G5_5Z_4;
        gtt[17][4].b = b_G5_5Z_4;
        gtt[26][4].b = b_G5_5Z_4;

        gtt[ 0][5].b = b_G6_4_4;
        gtt[13][5].b = b_G6_4_4;
        gtt[ 1][5].b = b_G6_4_4;
        gtt[12][5].b = b_G6_4_4;
        gtt[ 2][5].b = b_G6_4_4;
        gtt[14][5].b = b_G6_4_4;
        gtt[ 5][5].b = b_G6_4_4;
        gtt[ 8][5].b = b_G6_4_4;
        gtt[17][5].b = b_G6_4_4;
        gtt[26][5].b = b_G6_4_4;

        gtt[ 0][6].b = b_G1_1_XZ;
        gtt[ 2][6].b = b_G1_1_XZ;
        gtt[ 8][6].b = b_G1_3_XZ;
        gtt[26][6].b = b_G1_3_XZ;

        gtt[ 0][7].b = b_G1_XZ_XZ;
        gtt[ 2][7].b = b_G1_XZ_XZ;
        gtt[ 8][7].b = b_G1_XZ_XZ;
        gtt[26][7].b = b_G1_XZ_XZ;

        gtt[ 0][8].b = b_G2_XZ_5Z;
        gtt[ 1][8].b = b_G2_X_5;
        gtt[ 2][8].b = b_G2_XZ_5Z;
        gtt[ 5][8].b = b_G2_X_5;
        gtt[ 8][8].b = b_G2_XZ_5Z;
        gtt[17][8].b = b_G2_X_5;
        gtt[26][8].b = b_G2_XZ_5Z;

        gtt[ 0][9].b = b_G3_XZ_4;
        gtt[ 1][9].b = b_G3_XZ_4;
        gtt[ 2][9].b = b_G3_XZ_4;
        gtt[ 5][9].b = b_G3_XZ_4;
        gtt[ 8][9].b = b_G3_XZ_4;
        gtt[17][9].b = b_G3_XZ_4;
        gtt[26][9].b = b_G3_XZ_4;
    }

    else if (com.model == M3MSci23) {
        gtt[13][0].g = g23_G1_p_222;
        gtt[13][1].g = g23_G2_p_222;
        gtt[13][2].g = g23_G3_p_222;
        gtt[13][3].g = g23_G4_222;
        gtt[13][4].g = g23_G5_222;
        gtt[13][5].g = g23_G6_222;
        gtt[13][6].g = g23_G1_z_222;
        gtt[13][7].g = g23_G1_m_222;
        gtt[13][8].g = g23_G2_m_222;
        gtt[13][9].g = g23_G3_m_222;

        gtt[0][0].g = g23_G1_111;
        gtt[0][1].g = g23_G2_111;
        gtt[0][2].g = g23_G3_111;
        gtt[0][3].g = g23_G4_111;
        gtt[0][4].g = g23_G5_111;
        gtt[0][5].g = g23_G6_111;

        gtt[12][1].g = g23_G2_p_221;
        gtt[12][2].g = g23_G3_p_221;
        gtt[12][3].g = g23_G4_221;
        gtt[12][4].g = g23_G5_221;
        gtt[12][5].g = g23_G6_221;
        gtt[12][8].g = g23_G2_m_221;
        gtt[12][9].g = g23_G3_m_221;

        gtt[1][1].g = g23_G2_112;
        gtt[1][2].g = g23_G3_112;
        gtt[1][3].g = g23_G4_112;
        gtt[1][4].g = g23_G5_112;
        gtt[1][5].g = g23_G6_112;

        gtt[14][1].g = g23_G2_p_223;
        gtt[14][2].g = g23_G3_p_223;
        gtt[14][3].g = g23_G4_223;
        gtt[14][4].g = g23_G5_223;
        gtt[14][5].g = g23_G6_223;
        gtt[14][6].g = g23_G1_z_223;
        gtt[14][7].g = g23_G1_m_223;
        gtt[14][8].g = g23_G2_m_223;
        gtt[14][9].g = g23_G3_m_223;

        gtt[2][1].g = g23_G2_113;
        gtt[2][2].g = g23_G3_113;
        gtt[2][3].g = g23_G4_113;
        gtt[2][4].g = g23_G5_113;
        gtt[2][5].g = g23_G6_113;

        gtt[5][3].g = g23_G4_123;
        gtt[5][4].g = g23_G5_123;
        gtt[5][5].g = g23_G6_123;
        gtt[5][8].g = g23_G2_m_123;
        gtt[5][9].g = g23_G3_m_123;

        gtt[17][1].g = g23_G2_p_233;
        gtt[17][2].g = g23_G3_p_233;
        gtt[17][3].g = g23_G4_233;
        gtt[17][4].g = g23_G5_233;
        gtt[17][5].g = g23_G6_233;
        gtt[17][6].g = g23_G1_z_233;
        gtt[17][7].g = g23_G1_m_233;
        gtt[17][8].g = g23_G2_m_233;
        gtt[17][9].g = g23_G3_m_233;

        gtt[8][1].g = g23_G2_p_133;
        gtt[8][2].g = g23_G3_p_133;
        gtt[8][3].g = g23_G4_133;
        gtt[8][4].g = g23_G5_133;
        gtt[8][5].g = g23_G6_133;
        gtt[8][8].g = g23_G2_m_133;
        gtt[8][9].g = g23_G3_m_133;

        gtt[26][0].g = g23_G1_p_333;
        gtt[26][1].g = g23_G2_p_333;
        gtt[26][2].g = g23_G3_p_333;
        gtt[26][3].g = g23_G4_333;
        gtt[26][4].g = g23_G5_333;
        gtt[26][5].g = g23_G6_333;
        gtt[26][6].g = g23_G1_z_333;
        gtt[26][7].g = g23_G1_m_333;
        gtt[26][8].g = g23_G2_m_333;
        gtt[26][9].g = g23_G3_m_333;

        gtt[13][0].b = b_G1_2_2;
        gtt[ 0][0].b = b_G1_1_1;
        gtt[26][0].b = b_G1_3_3;

        gtt[13][1].b = b_G2_2_5Z;
        gtt[ 0][1].b = b_G2_1_5;
        gtt[12][1].b = b_G2_2_5;
        gtt[ 1][1].b = b_G2_1_5;
        gtt[14][1].b = b_G2_2_5Z;
        gtt[ 2][1].b = b_G2_1_5;
        gtt[17][1].b = b_G2_3_5Z;
        gtt[ 8][1].b = b_G2_3_5;
        gtt[26][1].b = b_G2_3_5Z;

        gtt[13][2].b = b_G3_2_4;
        gtt[ 0][2].b = b_G3_1_4;
        gtt[12][2].b = b_G3_2_4;
        gtt[ 1][2].b = b_G3_1_4;
        gtt[14][2].b = b_G3_2_4;
        gtt[ 2][2].b = b_G3_1_4;
        gtt[17][2].b = b_G3_3_4;
        gtt[ 8][2].b = b_G3_3_4;
        gtt[26][2].b = b_G3_3_4;

        gtt[13][3].b = b_G4_5Z_5Z;
        gtt[ 0][3].b = b_G4_5_5;
        gtt[12][3].b = b_G4_5_5;
        gtt[ 1][3].b = b_G4_5_5;
        gtt[14][3].b = b_G4_5Z_5Z;
        gtt[ 2][3].b = b_G4_5_5;
        gtt[ 5][3].b = b_G4_5_5;
        gtt[17][3].b = b_G4_5Z_5Z;
        gtt[ 8][3].b = b_G4_5_5;
        gtt[26][3].b = b_G4_5Z_5Z;

        gtt[13][4].b = b_G5_5Z_4;
        gtt[ 0][4].b = b_G5_5_4;
        gtt[12][4].b = b_G5_5Z_4;
        gtt[ 1][4].b = b_G5_5_4;
        gtt[14][4].b = b_G5_5Z_4;
        gtt[ 2][4].b = b_G5_5_4;
        gtt[ 5][4].b = b_G5_5Z_4;
        gtt[17][4].b = b_G5_5Z_4;
        gtt[ 8][4].b = b_G5_5Z_4;
        gtt[26][4].b = b_G5_5Z_4;

        gtt[13][5].b = b_G6_4_4;
        gtt[ 0][5].b = b_G6_4_4;
        gtt[12][5].b = b_G6_4_4;
        gtt[ 1][5].b = b_G6_4_4;
        gtt[14][5].b = b_G6_4_4;
        gtt[ 2][5].b = b_G6_4_4;
        gtt[ 5][5].b = b_G6_4_4;
        gtt[17][5].b = b_G6_4_4;
        gtt[ 8][5].b = b_G6_4_4;
        gtt[26][5].b = b_G6_4_4;

        gtt[13][6].b = b_G1_2_YZ;
        gtt[14][6].b = b_G1_2_YZ;
        gtt[17][6].b = b_G1_3_YZ;
        gtt[26][6].b = b_G1_3_YZ;

        gtt[13][7].b = b_G1_YZ_YZ;
        gtt[14][7].b = b_G1_YZ_YZ;
        gtt[17][7].b = b_G1_YZ_YZ;
        gtt[26][7].b = b_G1_YZ_YZ;

        gtt[13][8].b = b_G2_YZ_5Z;
        gtt[12][8].b = b_G2_Y_5;
        gtt[14][8].b = b_G2_YZ_5Z;
        gtt[ 5][8].b = b_G2_Y_5;
        gtt[17][8].b = b_G2_YZ_5Z;
        gtt[ 8][8].b = b_G2_Y_5;
        gtt[26][8].b = b_G2_YZ_5Z;

        gtt[13][9].b = b_G3_YZ_4;
        gtt[12][9].b = b_G3_YZ_4;
        gtt[14][9].b = b_G3_YZ_4;
        gtt[ 5][9].b = b_G3_YZ_4;
        gtt[17][9].b = b_G3_YZ_4;
        gtt[ 8][9].b = b_G3_YZ_4;
        gtt[26][9].b = b_G3_YZ_4;
    }

    else {
        // initial states 111/222
        // exponent
        gtt[0][0].T = gtt[13][0].T = T_G1_111;
        gtt[0][1].T = gtt[13][1].T = T_G2_111;
        gtt[0][2].T = gtt[13][2].T = T_G3_111;
        gtt[0][3].T = gtt[13][3].T = T_G4_111;
        gtt[0][4].T = gtt[13][4].T = T_G5_111;
        gtt[0][5].T = gtt[13][5].T = T_G6_111;
        // rate/jacobi factor
        gtt[ 0][0].RJ = gtt[ 0][3].RJ = RJ_special;
        gtt[13][0].RJ = gtt[13][3].RJ = RJ_special;
        gtt[ 0][1].RJ = gtt[ 0][2].RJ = gtt[ 0][4].RJ = gtt[ 0][5].RJ = RJ_normal;
        gtt[13][1].RJ = gtt[13][2].RJ = gtt[13][4].RJ = gtt[13][5].RJ = RJ_normal;

        if(com.model == M2SIM3s) {
            // gene tree probability
            gtt[0][0].f = gtt[13][0].f = f_G1_111;
            gtt[0][1].f = gtt[13][1].f = f_G2_111;
            gtt[0][2].f = gtt[13][2].f = f_G3_111;
            gtt[0][3].f = gtt[13][3].f = f_G4_111;
            gtt[0][4].f = gtt[13][4].f = f_G5_111;
            gtt[0][5].f = gtt[13][5].f = f_G6_111;
            // computation of t0 and t1 in the manuscript
            gtt[0][0].t0t1 = gtt[1][0].t0t1 = gtt[13][0].t0t1 = gtt[12][0].t0t1 = t0t1_G1_111;
            gtt[0][1].t0t1 = gtt[1][1].t0t1 = gtt[13][1].t0t1 = gtt[12][1].t0t1 = t0t1_G2_111;
            gtt[0][2].t0t1 = gtt[1][2].t0t1 = gtt[13][2].t0t1 = gtt[12][2].t0t1 = t0t1_G3_111;
            gtt[0][3].t0t1 = gtt[1][3].t0t1 = gtt[13][3].t0t1 = gtt[12][3].t0t1 = t0t1_G4_111;
            gtt[0][4].t0t1 = gtt[1][4].t0t1 = gtt[13][4].t0t1 = gtt[12][4].t0t1 = t0t1_G5_111;
            gtt[0][5].t0t1 = gtt[1][5].t0t1 = gtt[13][5].t0t1 = gtt[12][5].t0t1 = t0t1_G6_111;
        }

        /* ------ */

        // initial states 112/122
        // exponent
        gtt[1][1].T = gtt[12][1].T = T_G2_112;
        gtt[1][2].T = gtt[12][2].T = T_G3_112;
        gtt[1][3].T = gtt[12][3].T = T_G4_112;
        gtt[1][4].T = gtt[12][4].T = T_G5_112;
        gtt[1][5].T = gtt[12][5].T = T_G6_112;
        // rate/jacobi factor
        gtt[ 1][3].RJ = gtt[12][3].RJ = RJ_special;
        gtt[ 1][1].RJ = gtt[ 1][2].RJ = gtt[ 1][4].RJ = gtt[ 1][5].RJ = RJ_normal;
        gtt[12][1].RJ = gtt[12][2].RJ = gtt[12][4].RJ = gtt[12][5].RJ = RJ_normal;

        if(com.model == M2SIM3s) {
            // gene tree probability
            gtt[1][0].f = gtt[12][0].f = f_G1_112;
            gtt[1][1].f = gtt[12][1].f = f_G2_112;
            gtt[1][2].f = gtt[12][2].f = f_G3_112;
            gtt[1][3].f = gtt[12][3].f = f_G4_112;
            gtt[1][4].f = gtt[12][4].f = f_G5_112;
            gtt[1][5].f = gtt[12][5].f = f_G6_112;
        }

        /* ------ */

        // initial states 113/223/123
        if (com.model == M2SIM3s) {
            // gene tree probability
            gtt[2][2].f = gtt[14][2].f = gtt[5][2].f = f_G3_113;
            gtt[2][4].f = gtt[14][4].f = gtt[5][4].f = f_G5_113;
            gtt[2][5].f = gtt[14][5].f = gtt[5][5].f = f_G6_113;
            // computation of t0 and t1 in the manuscript
            gtt[2][2].t0t1 = gtt[14][2].t0t1 = gtt[5][2].t0t1 = t0t1_G3_111;
            gtt[2][4].t0t1 = gtt[14][4].t0t1 = gtt[5][4].t0t1 = t0t1_G5_111;
            gtt[2][5].t0t1 = gtt[14][5].t0t1 = gtt[5][5].t0t1 = t0t1_G6_111;
        }

        // exponent
        gtt[2][2].T = gtt[14][2].T = T_G3_113;
        gtt[2][4].T = gtt[14][4].T = T_G5_113;
        gtt[2][5].T = gtt[14][5].T = T_G6_113;
        gtt[5][4].T = T_G5_123; gtt[5][5].T = T_G6_123;
        // rate/jacobi factor
        gtt[2][2].RJ = gtt[2][4].RJ = gtt[2][5].RJ = gtt[14][2].RJ = gtt[14][4].RJ = gtt[14][5].RJ = gtt[5][4].RJ = gtt[5][5].RJ = RJ_normal;

        /* ------ */

        // initial states 133/233
        // exponent
        gtt[8][2].T = T_G35_133; gtt[8][5].T = T_G6_133;
        // rate/jacobi factor
        gtt[8][2].RJ = gtt[8][5].RJ = RJ_normal;
        // computation of t0 and t1 in the manuscript
        gtt[8][2].t0t1 = t0t1_G3_333;
        gtt[8][5].t0t1 = t0t1_G6_111;

        // initial state 333
        // exponent
        gtt[26][0].T = T_G124_333; gtt[26][2].T = T_G35_333; gtt[26][5].T = T_G6_333;
        // rate/jacobi factor
        gtt[26][0].RJ = RJ_special;
        gtt[26][2].RJ = gtt[26][5].RJ = RJ_normal;
        // computation of t0 and t1 in the manuscript
        gtt[26][0].t0t1 = t0t1_G1_333;
        gtt[26][2].t0t1 = t0t1_G3_333;
        gtt[26][5].t0t1 = t0t1_G6_111;


        // branch lengths
        gtt[0][0].b = gtt[13][0].b = b_G1_111;
        gtt[0][1].b = gtt[13][1].b = gtt[1][1].b = gtt[12][1].b = b_G2_111;
        gtt[0][2].b = gtt[13][2].b = gtt[1][2].b = gtt[12][2].b = gtt[2][2].b = gtt[14][2].b = b_G3_111;
        gtt[0][3].b = gtt[13][3].b = gtt[1][3].b = gtt[12][3].b = b_G4_111;
        gtt[0][4].b = gtt[13][4].b = gtt[1][4].b = gtt[12][4].b = gtt[2][4].b = gtt[14][4].b = gtt[5][4].b = b_G5_111;
        gtt[0][5].b = gtt[13][5].b = gtt[1][5].b = gtt[12][5].b = gtt[2][5].b = gtt[14][5].b = gtt[5][5].b = b_G6_111;

        gtt[26][0].b = b_G1_333;
        gtt[8][2].b = gtt[26][2].b = b_G3_333;
        gtt[8][5].b = gtt[26][5].b = b_G6_111;

        if (com.model == M2SIM3s) {
            gtt[1][0].b = b_G1_111;
            gtt[12][0].b = b_G1_111;
            gtt[5][2].b = b_G3_111;
        }
    }
}

void setupGtreeTab_treedata() {
    BTEntry ** gtt = com.GtreeTab;

    if (com.model == M2Pro || com.model == M2ProMax) {
        if (com.model == M2Pro) {
            gtt[0][0].helper = helper_G1;
            gtt[0][1].helper = helper_G2;
            gtt[0][2].helper = helper_G3;
            gtt[0][3].helper = helper_G4;
            gtt[0][4].helper = helper_G5;
            gtt[0][5].helper = helper_G6;

            gtt[0][0].density = density_G123;
            gtt[0][1].density = density_G123;
            gtt[0][2].density = density_G123;
            gtt[0][3].density = density_G4;
            gtt[0][4].density = density_G5;
            gtt[0][5].density = density_G6;
        }
        else {
            gtt[0][0].helper = helper_G1_PM;
            gtt[0][1].helper = helper_G2_PM;
            gtt[0][2].helper = helper_G3_PM;
            gtt[0][3].helper = helper_G4_PM;
            gtt[0][4].helper = helper_G5_PM;
            gtt[0][5].helper = helper_G6_PM;

            gtt[0][0].density = density_G123;
            gtt[0][1].density = density_G123;
            gtt[0][2].density = density_G123;
            gtt[0][3].density = density_G45;
            gtt[0][4].density = density_G45;
            gtt[0][5].density = density_G6;
        }

        gtt[0][0].bTot = bTot_G1_123_123;
        gtt[0][1].bTot = bTot_G2_123_5W;
        gtt[0][2].bTot = bTot_G3_123_4;
        gtt[0][3].bTot = bTot_G4_5W_5W;
        gtt[0][4].bTot = bTot_G5_5W_4;
        gtt[0][5].bTot = bTot_G6_4_4;

        gtt[0][0].detJ = detJ_G1_123_123;
        gtt[0][1].detJ = detJ_G2_123_5W;
        gtt[0][2].detJ = detJ_G3_123_4;
        gtt[0][3].detJ = detJ_G4_5W_5W;
        gtt[0][4].detJ = detJ_G5_5W_4;
        gtt[0][5].detJ = detJ_G6_4_4;
    }

    else if (com.model == M3MSci12) {
        gtt[0][0].g = g12_G1_p_111;
        gtt[0][1].g = g12_G2_p_111;
        gtt[0][2].g = g12_G3_p_111;
        gtt[0][3].g = g12_G4_111;
        gtt[0][4].g = g12_G5_111;
        gtt[0][5].g = g12_G6_111;
        gtt[0][6].g = g12_G1_z_111;
        gtt[0][7].g = g12_G1_m_111;
        gtt[0][8].g = g12_G2_m_111;
        gtt[0][9].g = g12_G3_m_111;

        gtt[13][0].g = g12_G1_p_222;
        gtt[13][1].g = g12_G2_p_222;
        gtt[13][2].g = g12_G3_p_222;
        gtt[13][3].g = g12_G4_222;
        gtt[13][4].g = g12_G5_222;
        gtt[13][5].g = g12_G6_222;
        gtt[13][6].g = g12_G1_z_222;
        gtt[13][7].g = g12_G1_m_222;
        gtt[13][8].g = g12_G2_m_222;
        gtt[13][9].g = g12_G3_m_222;

        gtt[1][1].g = g12_G2_p_112;
        gtt[1][2].g = g12_G3_p_112;
        gtt[1][3].g = g12_G4_112;
        gtt[1][4].g = g12_G5_112;
        gtt[1][5].g = g12_G6_112;
        gtt[1][6].g = g12_G1_z_112;
        gtt[1][7].g = g12_G1_m_112;
        gtt[1][8].g = g12_G2_m_112;
        gtt[1][9].g = g12_G3_m_112;

        gtt[12][1].g = g12_G2_p_221;
        gtt[12][2].g = g12_G3_p_221;
        gtt[12][3].g = g12_G4_221;
        gtt[12][4].g = g12_G5_221;
        gtt[12][5].g = g12_G6_221;
        gtt[12][6].g = g12_G1_z_221;
        gtt[12][7].g = g12_G1_m_221;
        gtt[12][8].g = g12_G2_m_221;
        gtt[12][9].g = g12_G3_m_221;

        gtt[2][2].g = g12_G3_p_113;
        gtt[2][4].g = g12_G5_113;
        gtt[2][5].g = g12_G6_113;
        gtt[2][9].g = g12_G3_m_113;

        gtt[14][2].g = g12_G3_p_223;
        gtt[14][4].g = g12_G5_223;
        gtt[14][5].g = g12_G6_223;
        gtt[14][9].g = g12_G3_m_223;

        gtt[5][4].g = g12_G5_123;
        gtt[5][5].g = g12_G6_123;
        gtt[5][9].g = g12_G3_m_123;

        gtt[8][2].g = g12_G3_133;
        gtt[8][5].g = g12_G6_133;

        gtt[26][0].g = g12_G1_333;
        gtt[26][2].g = g12_G3_333;
        gtt[26][5].g = g12_G6_333;

        gtt[ 0][0].bTot = bTot_G1_1_1;
        gtt[13][0].bTot = bTot_G1_2_2;
        gtt[26][0].bTot = bTot_G1_3_3;

        gtt[ 0][1].bTot = bTot_G2_1_5;
        gtt[13][1].bTot = bTot_G2_2_5;
        gtt[ 1][1].bTot = bTot_G2_1_5;
        gtt[12][1].bTot = bTot_G2_2_5;

        gtt[ 0][2].bTot = bTot_G3_1_4;
        gtt[13][2].bTot = bTot_G3_2_4;
        gtt[ 1][2].bTot = bTot_G3_1_4;
        gtt[12][2].bTot = bTot_G3_2_4;
        gtt[ 2][2].bTot = bTot_G3_1_4;
        gtt[14][2].bTot = bTot_G3_2_4;
        gtt[ 8][2].bTot = bTot_G3_3_4;
        gtt[26][2].bTot = bTot_G3_3_4;

        gtt[ 0][3].bTot = bTot_G4_5_5;
        gtt[13][3].bTot = bTot_G4_5_5;
        gtt[ 1][3].bTot = bTot_G4_5_5;
        gtt[12][3].bTot = bTot_G4_5_5;

        gtt[ 0][4].bTot = bTot_G5_5_4;
        gtt[13][4].bTot = bTot_G5_5_4;
        gtt[ 1][4].bTot = bTot_G5_5_4;
        gtt[12][4].bTot = bTot_G5_5_4;
        gtt[ 2][4].bTot = bTot_G5_5_4;
        gtt[14][4].bTot = bTot_G5_5_4;
        gtt[ 5][4].bTot = bTot_G5_5_4;

        gtt[ 0][5].bTot = bTot_G6_4_4;
        gtt[13][5].bTot = bTot_G6_4_4;
        gtt[ 1][5].bTot = bTot_G6_4_4;
        gtt[12][5].bTot = bTot_G6_4_4;
        gtt[ 2][5].bTot = bTot_G6_4_4;
        gtt[14][5].bTot = bTot_G6_4_4;
        gtt[ 5][5].bTot = bTot_G6_4_4;
        gtt[ 8][5].bTot = bTot_G6_4_4;
        gtt[26][5].bTot = bTot_G6_4_4;

        gtt[ 0][6].bTot = bTot_G1_1_XY;
        gtt[13][6].bTot = bTot_G1_2_XY;
        gtt[ 1][6].bTot = bTot_G1_1_XY;
        gtt[12][6].bTot = bTot_G1_2_XY;

        gtt[ 0][7].bTot = bTot_G1_XY_XY;
        gtt[13][7].bTot = bTot_G1_XY_XY;
        gtt[ 1][7].bTot = bTot_G1_XY_XY;
        gtt[12][7].bTot = bTot_G1_XY_XY;

        gtt[ 0][8].bTot = bTot_G2_XY_5;
        gtt[13][8].bTot = bTot_G2_XY_5;
        gtt[ 1][8].bTot = bTot_G2_XY_5;
        gtt[12][8].bTot = bTot_G2_XY_5;

        gtt[ 0][9].bTot = bTot_G3_XY_4;
        gtt[13][9].bTot = bTot_G3_XY_4;
        gtt[ 1][9].bTot = bTot_G3_XY_4;
        gtt[12][9].bTot = bTot_G3_XY_4;
        gtt[ 2][9].bTot = bTot_G3_XY_4;
        gtt[14][9].bTot = bTot_G3_XY_4;
        gtt[ 5][9].bTot = bTot_G3_XY_4;

        gtt[ 0][0].detJ = detJ_G1_1_1;
        gtt[13][0].detJ = detJ_G1_2_2;
        gtt[26][0].detJ = detJ_G1_3_3;

        gtt[ 0][1].detJ = detJ_G2_1_5;
        gtt[13][1].detJ = detJ_G2_2_5;
        gtt[ 1][1].detJ = detJ_G2_1_5;
        gtt[12][1].detJ = detJ_G2_2_5;

        gtt[ 0][2].detJ = detJ_G3_1_4;
        gtt[13][2].detJ = detJ_G3_2_4;
        gtt[ 1][2].detJ = detJ_G3_1_4;
        gtt[12][2].detJ = detJ_G3_2_4;
        gtt[ 2][2].detJ = detJ_G3_1_4;
        gtt[14][2].detJ = detJ_G3_2_4;
        gtt[ 8][2].detJ = detJ_G3_3_4;
        gtt[26][2].detJ = detJ_G3_3_4;

        gtt[ 0][3].detJ = detJ_G4_5_5;
        gtt[13][3].detJ = detJ_G4_5_5;
        gtt[ 1][3].detJ = detJ_G4_5_5;
        gtt[12][3].detJ = detJ_G4_5_5;

        gtt[ 0][4].detJ = detJ_G5_5_4;
        gtt[13][4].detJ = detJ_G5_5_4;
        gtt[ 1][4].detJ = detJ_G5_5_4;
        gtt[12][4].detJ = detJ_G5_5_4;
        gtt[ 2][4].detJ = detJ_G5_5_4;
        gtt[14][4].detJ = detJ_G5_5_4;
        gtt[ 5][4].detJ = detJ_G5_5_4;

        gtt[ 0][5].detJ = detJ_G6_4_4;
        gtt[13][5].detJ = detJ_G6_4_4;
        gtt[ 1][5].detJ = detJ_G6_4_4;
        gtt[12][5].detJ = detJ_G6_4_4;
        gtt[ 2][5].detJ = detJ_G6_4_4;
        gtt[14][5].detJ = detJ_G6_4_4;
        gtt[ 5][5].detJ = detJ_G6_4_4;
        gtt[ 8][5].detJ = detJ_G6_4_4;
        gtt[26][5].detJ = detJ_G6_4_4;

        gtt[ 0][6].detJ = detJ_G1_1_XY;
        gtt[13][6].detJ = detJ_G1_2_XY;
        gtt[ 1][6].detJ = detJ_G1_1_XY;
        gtt[12][6].detJ = detJ_G1_2_XY;

        gtt[ 0][7].detJ = detJ_G1_XY_XY;
        gtt[13][7].detJ = detJ_G1_XY_XY;
        gtt[ 1][7].detJ = detJ_G1_XY_XY;
        gtt[12][7].detJ = detJ_G1_XY_XY;

        gtt[ 0][8].detJ = detJ_G2_XY_5;
        gtt[13][8].detJ = detJ_G2_XY_5;
        gtt[ 1][8].detJ = detJ_G2_XY_5;
        gtt[12][8].detJ = detJ_G2_XY_5;

        gtt[ 0][9].detJ = detJ_G3_XY_4;
        gtt[13][9].detJ = detJ_G3_XY_4;
        gtt[ 1][9].detJ = detJ_G3_XY_4;
        gtt[12][9].detJ = detJ_G3_XY_4;
        gtt[ 2][9].detJ = detJ_G3_XY_4;
        gtt[14][9].detJ = detJ_G3_XY_4;
        gtt[ 5][9].detJ = detJ_G3_XY_4;
    }

    else if (com.model == M3MSci13) {
        gtt[0][0].g = g13_G1_p_111;
        gtt[0][1].g = g13_G2_p_111;
        gtt[0][2].g = g13_G3_p_111;
        gtt[0][3].g = g13_G4_111;
        gtt[0][4].g = g13_G5_111;
        gtt[0][5].g = g13_G6_111;
        gtt[0][6].g = g13_G1_z_111;
        gtt[0][7].g = g13_G1_m_111;
        gtt[0][8].g = g13_G2_m_111;
        gtt[0][9].g = g13_G3_m_111;

        gtt[13][0].g = g13_G1_222;
        gtt[13][1].g = g13_G2_222;
        gtt[13][2].g = g13_G3_222;
        gtt[13][3].g = g13_G4_222;
        gtt[13][4].g = g13_G5_222;
        gtt[13][5].g = g13_G6_222;

        gtt[1][1].g = g13_G2_p_112;
        gtt[1][2].g = g13_G3_p_112;
        gtt[1][3].g = g13_G4_112;
        gtt[1][4].g = g13_G5_112;
        gtt[1][5].g = g13_G6_112;
        gtt[1][8].g = g13_G2_m_112;
        gtt[1][9].g = g13_G3_m_112;

        gtt[12][1].g = g13_G2_221;
        gtt[12][2].g = g13_G3_221;
        gtt[12][3].g = g13_G4_221;
        gtt[12][4].g = g13_G5_221;
        gtt[12][5].g = g13_G6_221;

        gtt[2][1].g = g13_G2_p_113;
        gtt[2][2].g = g13_G3_p_113;
        gtt[2][3].g = g13_G4_113;
        gtt[2][4].g = g13_G5_113;
        gtt[2][5].g = g13_G6_113;
        gtt[2][6].g = g13_G1_z_113;
        gtt[2][7].g = g13_G1_m_113;
        gtt[2][8].g = g13_G2_m_113;
        gtt[2][9].g = g13_G3_m_113;

        gtt[14][1].g = g13_G2_223;
        gtt[14][2].g = g13_G3_223;
        gtt[14][3].g = g13_G4_223;
        gtt[14][4].g = g13_G5_223;
        gtt[14][5].g = g13_G6_223;

        gtt[5][3].g = g13_G4_123;
        gtt[5][4].g = g13_G5_123;
        gtt[5][5].g = g13_G6_123;
        gtt[5][8].g = g13_G2_m_123;
        gtt[5][9].g = g13_G3_m_123;

        gtt[8][1].g = g13_G2_p_133;
        gtt[8][2].g = g13_G3_p_133;
        gtt[8][3].g = g13_G4_133;
        gtt[8][4].g = g13_G5_133;
        gtt[8][5].g = g13_G6_133;
        gtt[8][6].g = g13_G1_z_133;
        gtt[8][7].g = g13_G1_m_133;
        gtt[8][8].g = g13_G2_m_133;
        gtt[8][9].g = g13_G3_m_133;

        gtt[17][1].g = g13_G2_p_233;
        gtt[17][2].g = g13_G3_p_233;
        gtt[17][3].g = g13_G4_233;
        gtt[17][4].g = g13_G5_233;
        gtt[17][5].g = g13_G6_233;
        gtt[17][8].g = g13_G2_m_233;
        gtt[17][9].g = g13_G3_m_233;

        gtt[26][0].g = g13_G1_p_333;
        gtt[26][1].g = g13_G2_p_333;
        gtt[26][2].g = g13_G3_p_333;
        gtt[26][3].g = g13_G4_333;
        gtt[26][4].g = g13_G5_333;
        gtt[26][5].g = g13_G6_333;
        gtt[26][6].g = g13_G1_z_333;
        gtt[26][7].g = g13_G1_m_333;
        gtt[26][8].g = g13_G2_m_333;
        gtt[26][9].g = g13_G3_m_333;

        gtt[ 0][0].bTot = bTot_G1_1_1;
        gtt[13][0].bTot = bTot_G1_2_2;
        gtt[26][0].bTot = bTot_G1_3_3;

        gtt[ 0][1].bTot = bTot_G2_1_5Z;
        gtt[13][1].bTot = bTot_G2_2_5;
        gtt[ 1][1].bTot = bTot_G2_1_5;
        gtt[12][1].bTot = bTot_G2_2_5;
        gtt[ 2][1].bTot = bTot_G2_1_5Z;
        gtt[14][1].bTot = bTot_G2_2_5;
        gtt[ 8][1].bTot = bTot_G2_3_5Z;
        gtt[17][1].bTot = bTot_G2_3_5;
        gtt[26][1].bTot = bTot_G2_3_5Z;

        gtt[ 0][2].bTot = bTot_G3_1_4;
        gtt[13][2].bTot = bTot_G3_2_4;
        gtt[ 1][2].bTot = bTot_G3_1_4;
        gtt[12][2].bTot = bTot_G3_2_4;
        gtt[ 2][2].bTot = bTot_G3_1_4;
        gtt[14][2].bTot = bTot_G3_2_4;
        gtt[ 8][2].bTot = bTot_G3_3_4;
        gtt[17][2].bTot = bTot_G3_3_4;
        gtt[26][2].bTot = bTot_G3_3_4;

        gtt[ 0][3].bTot = bTot_G4_5Z_5Z;
        gtt[13][3].bTot = bTot_G4_5_5;
        gtt[ 1][3].bTot = bTot_G4_5_5;
        gtt[12][3].bTot = bTot_G4_5_5;
        gtt[ 2][3].bTot = bTot_G4_5Z_5Z;
        gtt[14][3].bTot = bTot_G4_5_5;
        gtt[ 5][3].bTot = bTot_G4_5_5;
        gtt[ 8][3].bTot = bTot_G4_5Z_5Z;
        gtt[17][3].bTot = bTot_G4_5_5;
        gtt[26][3].bTot = bTot_G4_5Z_5Z;

        gtt[ 0][4].bTot = bTot_G5_5Z_4;
        gtt[13][4].bTot = bTot_G5_5_4;
        gtt[ 1][4].bTot = bTot_G5_5Z_4;
        gtt[12][4].bTot = bTot_G5_5_4;
        gtt[ 2][4].bTot = bTot_G5_5Z_4;
        gtt[14][4].bTot = bTot_G5_5_4;
        gtt[ 5][4].bTot = bTot_G5_5Z_4;
        gtt[ 8][4].bTot = bTot_G5_5Z_4;
        gtt[17][4].bTot = bTot_G5_5Z_4;
        gtt[26][4].bTot = bTot_G5_5Z_4;

        gtt[ 0][5].bTot = bTot_G6_4_4;
        gtt[13][5].bTot = bTot_G6_4_4;
        gtt[ 1][5].bTot = bTot_G6_4_4;
        gtt[12][5].bTot = bTot_G6_4_4;
        gtt[ 2][5].bTot = bTot_G6_4_4;
        gtt[14][5].bTot = bTot_G6_4_4;
        gtt[ 5][5].bTot = bTot_G6_4_4;
        gtt[ 8][5].bTot = bTot_G6_4_4;
        gtt[17][5].bTot = bTot_G6_4_4;
        gtt[26][5].bTot = bTot_G6_4_4;

        gtt[ 0][6].bTot = bTot_G1_1_XZ;
        gtt[ 2][6].bTot = bTot_G1_1_XZ;
        gtt[ 8][6].bTot = bTot_G1_3_XZ;
        gtt[26][6].bTot = bTot_G1_3_XZ;

        gtt[ 0][7].bTot = bTot_G1_XZ_XZ;
        gtt[ 2][7].bTot = bTot_G1_XZ_XZ;
        gtt[ 8][7].bTot = bTot_G1_XZ_XZ;
        gtt[26][7].bTot = bTot_G1_XZ_XZ;

        gtt[ 0][8].bTot = bTot_G2_XZ_5Z;
        gtt[ 1][8].bTot = bTot_G2_X_5;
        gtt[ 2][8].bTot = bTot_G2_XZ_5Z;
        gtt[ 5][8].bTot = bTot_G2_X_5;
        gtt[ 8][8].bTot = bTot_G2_XZ_5Z;
        gtt[17][8].bTot = bTot_G2_X_5;
        gtt[26][8].bTot = bTot_G2_XZ_5Z;

        gtt[ 0][9].bTot = bTot_G3_XZ_4;
        gtt[ 1][9].bTot = bTot_G3_XZ_4;
        gtt[ 2][9].bTot = bTot_G3_XZ_4;
        gtt[ 5][9].bTot = bTot_G3_XZ_4;
        gtt[ 8][9].bTot = bTot_G3_XZ_4;
        gtt[17][9].bTot = bTot_G3_XZ_4;
        gtt[26][9].bTot = bTot_G3_XZ_4;

        gtt[ 0][0].detJ = detJ_G1_1_1;
        gtt[13][0].detJ = detJ_G1_2_2;
        gtt[26][0].detJ = detJ_G1_3_3;

        gtt[ 0][1].detJ = detJ_G2_1_5Z;
        gtt[13][1].detJ = detJ_G2_2_5;
        gtt[ 1][1].detJ = detJ_G2_1_5;
        gtt[12][1].detJ = detJ_G2_2_5;
        gtt[ 2][1].detJ = detJ_G2_1_5Z;
        gtt[14][1].detJ = detJ_G2_2_5;
        gtt[ 8][1].detJ = detJ_G2_3_5Z;
        gtt[17][1].detJ = detJ_G2_3_5;
        gtt[26][1].detJ = detJ_G2_3_5Z;

        gtt[ 0][2].detJ = detJ_G3_1_4;
        gtt[13][2].detJ = detJ_G3_2_4;
        gtt[ 1][2].detJ = detJ_G3_1_4;
        gtt[12][2].detJ = detJ_G3_2_4;
        gtt[ 2][2].detJ = detJ_G3_1_4;
        gtt[14][2].detJ = detJ_G3_2_4;
        gtt[ 8][2].detJ = detJ_G3_3_4;
        gtt[17][2].detJ = detJ_G3_3_4;
        gtt[26][2].detJ = detJ_G3_3_4;

        gtt[ 0][3].detJ = detJ_G4_5Z_5Z;
        gtt[13][3].detJ = detJ_G4_5_5;
        gtt[ 1][3].detJ = detJ_G4_5_5;
        gtt[12][3].detJ = detJ_G4_5_5;
        gtt[ 2][3].detJ = detJ_G4_5Z_5Z;
        gtt[14][3].detJ = detJ_G4_5_5;
        gtt[ 5][3].detJ = detJ_G4_5_5;
        gtt[ 8][3].detJ = detJ_G4_5Z_5Z;
        gtt[17][3].detJ = detJ_G4_5_5;
        gtt[26][3].detJ = detJ_G4_5Z_5Z;

        gtt[ 0][4].detJ = detJ_G5_5Z_4;
        gtt[13][4].detJ = detJ_G5_5_4;
        gtt[ 1][4].detJ = detJ_G5_5Z_4;
        gtt[12][4].detJ = detJ_G5_5_4;
        gtt[ 2][4].detJ = detJ_G5_5Z_4;
        gtt[14][4].detJ = detJ_G5_5_4;
        gtt[ 5][4].detJ = detJ_G5_5Z_4;
        gtt[ 8][4].detJ = detJ_G5_5Z_4;
        gtt[17][4].detJ = detJ_G5_5Z_4;
        gtt[26][4].detJ = detJ_G5_5Z_4;

        gtt[ 0][5].detJ = detJ_G6_4_4;
        gtt[13][5].detJ = detJ_G6_4_4;
        gtt[ 1][5].detJ = detJ_G6_4_4;
        gtt[12][5].detJ = detJ_G6_4_4;
        gtt[ 2][5].detJ = detJ_G6_4_4;
        gtt[14][5].detJ = detJ_G6_4_4;
        gtt[ 5][5].detJ = detJ_G6_4_4;
        gtt[ 8][5].detJ = detJ_G6_4_4;
        gtt[17][5].detJ = detJ_G6_4_4;
        gtt[26][5].detJ = detJ_G6_4_4;

        gtt[ 0][6].detJ = detJ_G1_1_XZ;
        gtt[ 2][6].detJ = detJ_G1_1_XZ;
        gtt[ 8][6].detJ = detJ_G1_3_XZ;
        gtt[26][6].detJ = detJ_G1_3_XZ;

        gtt[ 0][7].detJ = detJ_G1_XZ_XZ;
        gtt[ 2][7].detJ = detJ_G1_XZ_XZ;
        gtt[ 8][7].detJ = detJ_G1_XZ_XZ;
        gtt[26][7].detJ = detJ_G1_XZ_XZ;

        gtt[ 0][8].detJ = detJ_G2_XZ_5Z;
        gtt[ 1][8].detJ = detJ_G2_X_5;
        gtt[ 2][8].detJ = detJ_G2_XZ_5Z;
        gtt[ 5][8].detJ = detJ_G2_X_5;
        gtt[ 8][8].detJ = detJ_G2_XZ_5Z;
        gtt[17][8].detJ = detJ_G2_X_5;
        gtt[26][8].detJ = detJ_G2_XZ_5Z;

        gtt[ 0][9].detJ = detJ_G3_XZ_4;
        gtt[ 1][9].detJ = detJ_G3_XZ_4;
        gtt[ 2][9].detJ = detJ_G3_XZ_4;
        gtt[ 5][9].detJ = detJ_G3_XZ_4;
        gtt[ 8][9].detJ = detJ_G3_XZ_4;
        gtt[17][9].detJ = detJ_G3_XZ_4;
        gtt[26][9].detJ = detJ_G3_XZ_4;
    }

    else if (com.model == M3MSci23) {
        gtt[13][0].g = g23_G1_p_222;
        gtt[13][1].g = g23_G2_p_222;
        gtt[13][2].g = g23_G3_p_222;
        gtt[13][3].g = g23_G4_222;
        gtt[13][4].g = g23_G5_222;
        gtt[13][5].g = g23_G6_222;
        gtt[13][6].g = g23_G1_z_222;
        gtt[13][7].g = g23_G1_m_222;
        gtt[13][8].g = g23_G2_m_222;
        gtt[13][9].g = g23_G3_m_222;

        gtt[0][0].g = g23_G1_111;
        gtt[0][1].g = g23_G2_111;
        gtt[0][2].g = g23_G3_111;
        gtt[0][3].g = g23_G4_111;
        gtt[0][4].g = g23_G5_111;
        gtt[0][5].g = g23_G6_111;

        gtt[12][1].g = g23_G2_p_221;
        gtt[12][2].g = g23_G3_p_221;
        gtt[12][3].g = g23_G4_221;
        gtt[12][4].g = g23_G5_221;
        gtt[12][5].g = g23_G6_221;
        gtt[12][8].g = g23_G2_m_221;
        gtt[12][9].g = g23_G3_m_221;

        gtt[1][1].g = g23_G2_112;
        gtt[1][2].g = g23_G3_112;
        gtt[1][3].g = g23_G4_112;
        gtt[1][4].g = g23_G5_112;
        gtt[1][5].g = g23_G6_112;

        gtt[14][1].g = g23_G2_p_223;
        gtt[14][2].g = g23_G3_p_223;
        gtt[14][3].g = g23_G4_223;
        gtt[14][4].g = g23_G5_223;
        gtt[14][5].g = g23_G6_223;
        gtt[14][6].g = g23_G1_z_223;
        gtt[14][7].g = g23_G1_m_223;
        gtt[14][8].g = g23_G2_m_223;
        gtt[14][9].g = g23_G3_m_223;

        gtt[2][1].g = g23_G2_113;
        gtt[2][2].g = g23_G3_113;
        gtt[2][3].g = g23_G4_113;
        gtt[2][4].g = g23_G5_113;
        gtt[2][5].g = g23_G6_113;

        gtt[5][3].g = g23_G4_123;
        gtt[5][4].g = g23_G5_123;
        gtt[5][5].g = g23_G6_123;
        gtt[5][8].g = g23_G2_m_123;
        gtt[5][9].g = g23_G3_m_123;

        gtt[17][1].g = g23_G2_p_233;
        gtt[17][2].g = g23_G3_p_233;
        gtt[17][3].g = g23_G4_233;
        gtt[17][4].g = g23_G5_233;
        gtt[17][5].g = g23_G6_233;
        gtt[17][6].g = g23_G1_z_233;
        gtt[17][7].g = g23_G1_m_233;
        gtt[17][8].g = g23_G2_m_233;
        gtt[17][9].g = g23_G3_m_233;

        gtt[8][1].g = g23_G2_p_133;
        gtt[8][2].g = g23_G3_p_133;
        gtt[8][3].g = g23_G4_133;
        gtt[8][4].g = g23_G5_133;
        gtt[8][5].g = g23_G6_133;
        gtt[8][8].g = g23_G2_m_133;
        gtt[8][9].g = g23_G3_m_133;

        gtt[26][0].g = g23_G1_p_333;
        gtt[26][1].g = g23_G2_p_333;
        gtt[26][2].g = g23_G3_p_333;
        gtt[26][3].g = g23_G4_333;
        gtt[26][4].g = g23_G5_333;
        gtt[26][5].g = g23_G6_333;
        gtt[26][6].g = g23_G1_z_333;
        gtt[26][7].g = g23_G1_m_333;
        gtt[26][8].g = g23_G2_m_333;
        gtt[26][9].g = g23_G3_m_333;

        gtt[13][0].bTot = bTot_G1_2_2;
        gtt[ 0][0].bTot = bTot_G1_1_1;
        gtt[26][0].bTot = bTot_G1_3_3;

        gtt[13][1].bTot = bTot_G2_2_5Z;
        gtt[ 0][1].bTot = bTot_G2_1_5;
        gtt[12][1].bTot = bTot_G2_2_5;
        gtt[ 1][1].bTot = bTot_G2_1_5;
        gtt[14][1].bTot = bTot_G2_2_5Z;
        gtt[ 2][1].bTot = bTot_G2_1_5;
        gtt[17][1].bTot = bTot_G2_3_5Z;
        gtt[ 8][1].bTot = bTot_G2_3_5;
        gtt[26][1].bTot = bTot_G2_3_5Z;

        gtt[13][2].bTot = bTot_G3_2_4;
        gtt[ 0][2].bTot = bTot_G3_1_4;
        gtt[12][2].bTot = bTot_G3_2_4;
        gtt[ 1][2].bTot = bTot_G3_1_4;
        gtt[14][2].bTot = bTot_G3_2_4;
        gtt[ 2][2].bTot = bTot_G3_1_4;
        gtt[17][2].bTot = bTot_G3_3_4;
        gtt[ 8][2].bTot = bTot_G3_3_4;
        gtt[26][2].bTot = bTot_G3_3_4;

        gtt[13][3].bTot = bTot_G4_5Z_5Z;
        gtt[ 0][3].bTot = bTot_G4_5_5;
        gtt[12][3].bTot = bTot_G4_5_5;
        gtt[ 1][3].bTot = bTot_G4_5_5;
        gtt[14][3].bTot = bTot_G4_5Z_5Z;
        gtt[ 2][3].bTot = bTot_G4_5_5;
        gtt[ 5][3].bTot = bTot_G4_5_5;
        gtt[17][3].bTot = bTot_G4_5Z_5Z;
        gtt[ 8][3].bTot = bTot_G4_5_5;
        gtt[26][3].bTot = bTot_G4_5Z_5Z;

        gtt[13][4].bTot = bTot_G5_5Z_4;
        gtt[ 0][4].bTot = bTot_G5_5_4;
        gtt[12][4].bTot = bTot_G5_5Z_4;
        gtt[ 1][4].bTot = bTot_G5_5_4;
        gtt[14][4].bTot = bTot_G5_5Z_4;
        gtt[ 2][4].bTot = bTot_G5_5_4;
        gtt[ 5][4].bTot = bTot_G5_5Z_4;
        gtt[17][4].bTot = bTot_G5_5Z_4;
        gtt[ 8][4].bTot = bTot_G5_5Z_4;
        gtt[26][4].bTot = bTot_G5_5Z_4;

        gtt[13][5].bTot = bTot_G6_4_4;
        gtt[ 0][5].bTot = bTot_G6_4_4;
        gtt[12][5].bTot = bTot_G6_4_4;
        gtt[ 1][5].bTot = bTot_G6_4_4;
        gtt[14][5].bTot = bTot_G6_4_4;
        gtt[ 2][5].bTot = bTot_G6_4_4;
        gtt[ 5][5].bTot = bTot_G6_4_4;
        gtt[17][5].bTot = bTot_G6_4_4;
        gtt[ 8][5].bTot = bTot_G6_4_4;
        gtt[26][5].bTot = bTot_G6_4_4;

        gtt[13][6].bTot = bTot_G1_2_YZ;
        gtt[14][6].bTot = bTot_G1_2_YZ;
        gtt[17][6].bTot = bTot_G1_3_YZ;
        gtt[26][6].bTot = bTot_G1_3_YZ;

        gtt[13][7].bTot = bTot_G1_YZ_YZ;
        gtt[14][7].bTot = bTot_G1_YZ_YZ;
        gtt[17][7].bTot = bTot_G1_YZ_YZ;
        gtt[26][7].bTot = bTot_G1_YZ_YZ;

        gtt[13][8].bTot = bTot_G2_YZ_5Z;
        gtt[12][8].bTot = bTot_G2_Y_5;
        gtt[14][8].bTot = bTot_G2_YZ_5Z;
        gtt[ 5][8].bTot = bTot_G2_Y_5;
        gtt[17][8].bTot = bTot_G2_YZ_5Z;
        gtt[ 8][8].bTot = bTot_G2_Y_5;
        gtt[26][8].bTot = bTot_G2_YZ_5Z;

        gtt[13][9].bTot = bTot_G3_YZ_4;
        gtt[12][9].bTot = bTot_G3_YZ_4;
        gtt[14][9].bTot = bTot_G3_YZ_4;
        gtt[ 5][9].bTot = bTot_G3_YZ_4;
        gtt[17][9].bTot = bTot_G3_YZ_4;
        gtt[ 8][9].bTot = bTot_G3_YZ_4;
        gtt[26][9].bTot = bTot_G3_YZ_4;

        gtt[13][0].detJ = detJ_G1_2_2;
        gtt[ 0][0].detJ = detJ_G1_1_1;
        gtt[26][0].detJ = detJ_G1_3_3;

        gtt[13][1].detJ = detJ_G2_2_5Z;
        gtt[ 0][1].detJ = detJ_G2_1_5;
        gtt[12][1].detJ = detJ_G2_2_5;
        gtt[ 1][1].detJ = detJ_G2_1_5;
        gtt[14][1].detJ = detJ_G2_2_5Z;
        gtt[ 2][1].detJ = detJ_G2_1_5;
        gtt[17][1].detJ = detJ_G2_3_5Z;
        gtt[ 8][1].detJ = detJ_G2_3_5;
        gtt[26][1].detJ = detJ_G2_3_5Z;

        gtt[13][2].detJ = detJ_G3_2_4;
        gtt[ 0][2].detJ = detJ_G3_1_4;
        gtt[12][2].detJ = detJ_G3_2_4;
        gtt[ 1][2].detJ = detJ_G3_1_4;
        gtt[14][2].detJ = detJ_G3_2_4;
        gtt[ 2][2].detJ = detJ_G3_1_4;
        gtt[17][2].detJ = detJ_G3_3_4;
        gtt[ 8][2].detJ = detJ_G3_3_4;
        gtt[26][2].detJ = detJ_G3_3_4;

        gtt[13][3].detJ = detJ_G4_5Z_5Z;
        gtt[ 0][3].detJ = detJ_G4_5_5;
        gtt[12][3].detJ = detJ_G4_5_5;
        gtt[ 1][3].detJ = detJ_G4_5_5;
        gtt[14][3].detJ = detJ_G4_5Z_5Z;
        gtt[ 2][3].detJ = detJ_G4_5_5;
        gtt[ 5][3].detJ = detJ_G4_5_5;
        gtt[17][3].detJ = detJ_G4_5Z_5Z;
        gtt[ 8][3].detJ = detJ_G4_5_5;
        gtt[26][3].detJ = detJ_G4_5Z_5Z;

        gtt[13][4].detJ = detJ_G5_5Z_4;
        gtt[ 0][4].detJ = detJ_G5_5_4;
        gtt[12][4].detJ = detJ_G5_5Z_4;
        gtt[ 1][4].detJ = detJ_G5_5_4;
        gtt[14][4].detJ = detJ_G5_5Z_4;
        gtt[ 2][4].detJ = detJ_G5_5_4;
        gtt[ 5][4].detJ = detJ_G5_5Z_4;
        gtt[17][4].detJ = detJ_G5_5Z_4;
        gtt[ 8][4].detJ = detJ_G5_5Z_4;
        gtt[26][4].detJ = detJ_G5_5Z_4;

        gtt[13][5].detJ = detJ_G6_4_4;
        gtt[ 0][5].detJ = detJ_G6_4_4;
        gtt[12][5].detJ = detJ_G6_4_4;
        gtt[ 1][5].detJ = detJ_G6_4_4;
        gtt[14][5].detJ = detJ_G6_4_4;
        gtt[ 2][5].detJ = detJ_G6_4_4;
        gtt[ 5][5].detJ = detJ_G6_4_4;
        gtt[17][5].detJ = detJ_G6_4_4;
        gtt[ 8][5].detJ = detJ_G6_4_4;
        gtt[26][5].detJ = detJ_G6_4_4;

        gtt[13][6].detJ = detJ_G1_2_YZ;
        gtt[14][6].detJ = detJ_G1_2_YZ;
        gtt[17][6].detJ = detJ_G1_3_YZ;
        gtt[26][6].detJ = detJ_G1_3_YZ;

        gtt[13][7].detJ = detJ_G1_YZ_YZ;
        gtt[14][7].detJ = detJ_G1_YZ_YZ;
        gtt[17][7].detJ = detJ_G1_YZ_YZ;
        gtt[26][7].detJ = detJ_G1_YZ_YZ;

        gtt[13][8].detJ = detJ_G2_YZ_5Z;
        gtt[12][8].detJ = detJ_G2_Y_5;
        gtt[14][8].detJ = detJ_G2_YZ_5Z;
        gtt[ 5][8].detJ = detJ_G2_Y_5;
        gtt[17][8].detJ = detJ_G2_YZ_5Z;
        gtt[ 8][8].detJ = detJ_G2_Y_5;
        gtt[26][8].detJ = detJ_G2_YZ_5Z;

        gtt[13][9].detJ = detJ_G3_YZ_4;
        gtt[12][9].detJ = detJ_G3_YZ_4;
        gtt[14][9].detJ = detJ_G3_YZ_4;
        gtt[ 5][9].detJ = detJ_G3_YZ_4;
        gtt[17][9].detJ = detJ_G3_YZ_4;
        gtt[ 8][9].detJ = detJ_G3_YZ_4;
        gtt[26][9].detJ = detJ_G3_YZ_4;
    }

    else {
        // initial states 111/222
        // integration limits
        // exponent
        gtt[0][0].T = gtt[13][0].T = T_G1_111;
        gtt[0][1].T = gtt[13][1].T = T_G2_111;
        gtt[0][2].T = gtt[13][2].T = T_G3_111;
        gtt[0][3].T = gtt[13][3].T = T_G4_111;
        gtt[0][4].T = gtt[13][4].T = T_G5_111;
        gtt[0][5].T = gtt[13][5].T = T_G6_111;
        // rate/jacobi factor
        gtt[ 0][0].RJ = gtt[ 0][3].RJ = RJ_special;
        gtt[13][0].RJ = gtt[13][3].RJ = RJ_special;
        gtt[ 0][1].RJ = gtt[ 0][2].RJ = gtt[ 0][4].RJ = gtt[ 0][5].RJ = RJ_normal;
        gtt[13][1].RJ = gtt[13][2].RJ = gtt[13][4].RJ = gtt[13][5].RJ = RJ_normal;

        if (com.model == M0) {
            gtt[0][0].bTot = bTot_G1_1_1;
            gtt[0][1].bTot = bTot_G2_1_5;
            gtt[0][2].bTot = bTot_G3_1_4;

            gtt[0][0].detJ = detJ_G1_1_1;
            gtt[0][1].detJ = detJ_G2_1_5;
            gtt[0][2].detJ = detJ_G3_1_4;

            gtt[13][0].bTot = bTot_G1_2_2;
            gtt[13][1].bTot = bTot_G2_2_5;
            gtt[13][2].bTot = bTot_G3_2_4;

            gtt[13][0].detJ = detJ_G1_2_2;
            gtt[13][1].detJ = detJ_G2_2_5;
            gtt[13][2].detJ = detJ_G3_2_4;
        }
        else if(com.model == M2SIM3s) {
            // gene tree probability
            gtt[0][0].f = gtt[13][0].f = f_G1_111;
            gtt[0][1].f = gtt[13][1].f = f_G2_111;
            gtt[0][2].f = gtt[13][2].f = f_G3_111;
            gtt[0][3].f = gtt[13][3].f = f_G4_111;
            gtt[0][4].f = gtt[13][4].f = f_G5_111;
            gtt[0][5].f = gtt[13][5].f = f_G6_111;

            gtt[0][0].bTot = bTot_G1_12_12;
            gtt[0][1].bTot = bTot_G2_12_5;
            gtt[0][2].bTot = bTot_G3_12_4;

            gtt[0][0].detJ = detJ_G1_12_12;
            gtt[0][1].detJ = detJ_G2_12_5;
            gtt[0][2].detJ = detJ_G3_12_4;

            gtt[13][0].bTot = bTot_G1_12_12;
            gtt[13][1].bTot = bTot_G2_12_5;
            gtt[13][2].bTot = bTot_G3_12_4;

            gtt[13][0].detJ = detJ_G1_12_12;
            gtt[13][1].detJ = detJ_G2_12_5;
            gtt[13][2].detJ = detJ_G3_12_4;
        }

        gtt[0][3].bTot = bTot_G4_5_5;
        gtt[0][4].bTot = bTot_G5_5_4;
        gtt[0][5].bTot = bTot_G6_4_4;

        gtt[0][3].detJ = detJ_G4_5_5;
        gtt[0][4].detJ = detJ_G5_5_4;
        gtt[0][5].detJ = detJ_G6_4_4;

        gtt[13][3].bTot = bTot_G4_5_5;
        gtt[13][4].bTot = bTot_G5_5_4;
        gtt[13][5].bTot = bTot_G6_4_4;

        gtt[13][3].detJ = detJ_G4_5_5;
        gtt[13][4].detJ = detJ_G5_5_4;
        gtt[13][5].detJ = detJ_G6_4_4;

        /* ------ */

        // initial states 112/122
        // exponent
        gtt[1][1].T = gtt[12][1].T = T_G2_112;
        gtt[1][2].T = gtt[12][2].T = T_G3_112;
        gtt[1][3].T = gtt[12][3].T = T_G4_112;
        gtt[1][4].T = gtt[12][4].T = T_G5_112;
        gtt[1][5].T = gtt[12][5].T = T_G6_112;
        // rate/jacobi factor
        gtt[ 1][3].RJ = gtt[12][3].RJ = RJ_special;
        gtt[ 1][1].RJ = gtt[ 1][2].RJ = gtt[ 1][4].RJ = gtt[ 1][5].RJ = RJ_normal;
        gtt[12][1].RJ = gtt[12][2].RJ = gtt[12][4].RJ = gtt[12][5].RJ = RJ_normal;

        if (com.model == M0) {
            gtt[1][1].bTot = bTot_G2_1_5;
            gtt[1][2].bTot = bTot_G3_1_4;

            gtt[1][1].detJ = detJ_G2_1_5;
            gtt[1][2].detJ = detJ_G3_1_4;

            gtt[12][1].bTot = bTot_G2_2_5;
            gtt[12][2].bTot = bTot_G3_2_4;

            gtt[12][1].detJ = detJ_G2_2_5;
            gtt[12][2].detJ = detJ_G3_2_4;
        }
        else if(com.model == M2SIM3s) {
            // gene tree probability
            gtt[1][0].f = gtt[12][0].f = f_G1_112;
            gtt[1][1].f = gtt[12][1].f = f_G2_112;
            gtt[1][2].f = gtt[12][2].f = f_G3_112;
            gtt[1][3].f = gtt[12][3].f = f_G4_112;
            gtt[1][4].f = gtt[12][4].f = f_G5_112;
            gtt[1][5].f = gtt[12][5].f = f_G6_112;

            gtt[1][0].bTot = bTot_G1_12_12;
            gtt[1][1].bTot = bTot_G2_12_5;
            gtt[1][2].bTot = bTot_G3_12_4;

            gtt[1][0].detJ = detJ_G1_12_12;
            gtt[1][1].detJ = detJ_G2_12_5;
            gtt[1][2].detJ = detJ_G3_12_4;

            gtt[12][0].bTot = bTot_G1_12_12;
            gtt[12][1].bTot = bTot_G2_12_5;
            gtt[12][2].bTot = bTot_G3_12_4;

            gtt[12][0].detJ = detJ_G1_12_12;
            gtt[12][1].detJ = detJ_G2_12_5;
            gtt[12][2].detJ = detJ_G3_12_4;
        }

        gtt[1][3].bTot = bTot_G4_5_5;
        gtt[1][4].bTot = bTot_G5_5_4;
        gtt[1][5].bTot = bTot_G6_4_4;

        gtt[1][3].detJ = detJ_G4_5_5;
        gtt[1][4].detJ = detJ_G5_5_4;
        gtt[1][5].detJ = detJ_G6_4_4;

        gtt[12][3].bTot = bTot_G4_5_5;
        gtt[12][4].bTot = bTot_G5_5_4;
        gtt[12][5].bTot = bTot_G6_4_4;

        gtt[12][3].detJ = detJ_G4_5_5;
        gtt[12][4].detJ = detJ_G5_5_4;
        gtt[12][5].detJ = detJ_G6_4_4;

        /* ------ */

        // initial states 113/223/123
        if (com.model == M0) {
            gtt[2][2].bTot = bTot_G3_1_4;
            gtt[2][2].detJ = detJ_G3_1_4;

            gtt[14][2].bTot = bTot_G3_2_4;
            gtt[14][2].detJ = detJ_G3_2_4;
        }
        else if (com.model == M2SIM3s) {
            // gene tree probability
            gtt[2][2].f = gtt[14][2].f = gtt[5][2].f = f_G3_113;
            gtt[2][4].f = gtt[14][4].f = gtt[5][4].f = f_G5_113;
            gtt[2][5].f = gtt[14][5].f = gtt[5][5].f = f_G6_113;

            gtt[2][2].bTot = bTot_G3_12_4;
            gtt[2][2].detJ = detJ_G3_12_4;

            gtt[14][2].bTot = bTot_G3_12_4;
            gtt[14][2].detJ = detJ_G3_12_4;

            gtt[5][2].bTot = bTot_G3_12_4;
            gtt[5][2].detJ = detJ_G3_12_4;
        }

        // exponent
        gtt[2][2].T = gtt[14][2].T = T_G3_113;
        gtt[2][4].T = gtt[14][4].T = T_G5_113;
        gtt[2][5].T = gtt[14][5].T = T_G6_113;
        gtt[5][4].T = T_G5_123; gtt[5][5].T = T_G6_123;
        // rate/jacobi factor
        gtt[2][2].RJ = gtt[2][4].RJ = gtt[2][5].RJ = gtt[14][2].RJ = gtt[14][4].RJ = gtt[14][5].RJ = gtt[5][4].RJ = gtt[5][5].RJ = RJ_normal;

        gtt[2][4].bTot = bTot_G5_5_4;
        gtt[2][5].bTot = bTot_G6_4_4;

        gtt[2][4].detJ = detJ_G5_5_4;
        gtt[2][5].detJ = detJ_G6_4_4;

        gtt[14][4].bTot = bTot_G5_5_4;
        gtt[14][5].bTot = bTot_G6_4_4;

        gtt[14][4].detJ = detJ_G5_5_4;
        gtt[14][5].detJ = detJ_G6_4_4;

        gtt[5][4].bTot = bTot_G5_5_4;
        gtt[5][5].bTot = bTot_G6_4_4;

        gtt[5][4].detJ = detJ_G5_5_4;
        gtt[5][5].detJ = detJ_G6_4_4;

        /* ------ */

        // initial states 133/233
        // exponent
        gtt[8][2].T = T_G35_133; gtt[8][5].T = T_G6_133;
        // rate/jacobi factor
        gtt[8][2].RJ = gtt[8][5].RJ = RJ_normal;

        gtt[8][2].bTot = bTot_G3_3_4;
        gtt[8][5].bTot = bTot_G6_4_4;

        gtt[8][2].detJ = detJ_G3_3_4;
        gtt[8][5].detJ = detJ_G6_4_4;

        // initial state 333
        // exponent
        gtt[26][0].T = T_G124_333; gtt[26][2].T = T_G35_333; gtt[26][5].T = T_G6_333;
        // rate/jacobi factor
        gtt[26][0].RJ = RJ_special;
        gtt[26][2].RJ = gtt[26][5].RJ = RJ_normal;

        gtt[26][0].bTot = bTot_G1_3_3;
        gtt[26][2].bTot = bTot_G3_3_4;
        gtt[26][5].bTot = bTot_G6_4_4;

        gtt[26][0].detJ = detJ_G1_3_3;
        gtt[26][2].detJ = detJ_G3_3_4;
        gtt[26][5].detJ = detJ_G6_4_4;
    }
}
