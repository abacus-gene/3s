/*
*  3s.h
*  3s
*
*  Created by Daniel Dalquen on 07.02.14.
*  Copyright (c) 2014 UCL. All rights reserved.
*/

#ifndef _s__s_h
#define _s__s_h

#include "paml.h"
#include <xmmintrin.h>
#include <signal.h>

#ifdef _MSC_VER
#define inline __inline
#endif

#define CONCAT_(x, y) x##y
#define CONCAT(x, y) CONCAT_(x, y)
#define STATIC_ASSERT_(e, x) typedef struct{unsigned int x:1-2*(int)!(e);}x
#define STATIC_ASSERT(e) STATIC_ASSERT_(e, CONCAT(assertion_failed_at_line_, __LINE__))
#define STATIC_ASSERT_MSG(e, m) STATIC_ASSERT_(e, CONCAT(CONCAT(assertion_failed_at_line_, __LINE__), CONCAT(__, m)))
#define LENGTH_OF(x) (sizeof(x)/sizeof(*(x)))

#define NSPECIES              3  /* max # of species */
#define NS                    3  /* max # of sequences per locus */
#define NGENE                 1  /* required by ReadSeq, but not really in use */
#define LSPNAME              50  /* max # characters in a sequence name */
#define NCODE                 4
#define IDSEP                '^'
#define NMODELS               8
#define NEXTMODELS            4
#define MAXPARAMETERS        15
#define NALLSTATES           36  // 27 + 9
#define NSTATES              27
#define NSTATES2SEQ           9
#define NINITIALSTATES       10
#define NINITIALSTATES2SEQ    6
#define MAXGTREETYPES        10  //  6 + 4 for M3
#define MAXGTREETYPES2SEQ     4  //  3 + 1 for M3
#define MAXGTREES            73
#define MAXGTREES2SEQ        18
#define NBRANCH         (NS*2-2) /* max # of branches */
#define NNODE           (NS*2-1) /* max # of nodes */
#define MAXNSONS              2  /* max # of sons per node */
#define MAXSPTREES            6

#ifndef MULTIPLIER
#define MULTIPLIER 1
#endif

#ifndef LBOUND
#define LBOUND 1.0E-5
#endif


struct BTEntry {
    int nGtrees, config;
    double (* RJ)(double, double), (* T)(double, double, double);
    void (* f)(double, double, int, double *, double *, double *, double *, double[3]);
    void (* g)(double, double, double *, double[3]);
    void (* helper)(double, double, double *, double *, double *);
    void (* density)(int, double *, double *, double *, double[3]);
    void (* t0t1)(double, double, double[2]);
    double lim[2];
    void (* b)(double, double, double, double[2]);
    void (* bTot)(double[2], double[2], double[2]);
    double (* detJ)(double, double);
};

struct CommonInfo {
    unsigned char *z[3];
    char *spname[3], outf[512], seqf[512], ratef[512], ctlf[512], Imapf[512], treef[512], fix_locusrate;
    int model, extModel, ncode, cleandata, seed, npoints, ncatBeta, UseMedianBeta, getSE;
    int usedata, verbose, aroundMLEM0, nthreads, fixto0[14], fix[3], asymmetric[NMODELS];
    int nGtree, ngene, seqtype, ns, ls, posG[1+1], lgene[1], *pose, npatt;
    int readpattern, simmodel, runmodels[NEXTMODELS], modelMap[NEXTMODELS];
    int paraNamesMap[MAXPARAMETERS], paraMap[MAXPARAMETERS];
    double *fpatt, kappa, alpha, rho, rgene[1], pi[4], piG[1][4], initials[NMODELS][MAXPARAMETERS];
    double *pDclass, *tau1beta, *bp0124[MAXGTREES+1], *wwprior[MAXGTREES+1];
    struct BTEntry ** GtreeTab;
    double * space; // free space to avoid many allocs
    int np, ntime, clock;
    double *conP;
}  com ;

struct TREEB {
    int  nbranch, nnode, root, branches[NBRANCH][2];
}  tree;

struct TREEN {
    int father, nson, sons[MAXNSONS], ibranch;
    double branch, age, *conP, label;
    char *nodeStr;
}  nodes[NNODE];

struct SPECIESTREE {
    int nbranch, nnode, root, nspecies;
    signed char pptable[NSPECIES * 2 - 1][NSPECIES * 2 - 1];
    int speciestree, sptree, nsptree;
    struct TREESPN {
        char name[LSPNAME * 2];
        int father, nson, sons[2];
        double age, theta;
    } nodes[2 * NSPECIES - 1];
}  stree;

struct DATA { /* locus-specific sequence or gene tree information */
    int ndata;
    int *Nij[MAXSPTREES], *chain, *initState[MAXSPTREES];
    int iStateCnt[MAXSPTREES][NALLSTATES], twoSeqLoci;
    double *lnLmax[MAXSPTREES], *locusrate;
    double *Bij;
    int *topology[MAXSPTREES];
    int *GtreeType;
}  data;

struct MLE {
    double para[MAXPARAMETERS];
    int np;
    double lnL;
}  MLEM0[MAXSPTREES];

int GetOptions (char *ctlf);
int ModelMap(int extModel);
//int ReadSiteCounts(char *datafile);
int ReadSpeciesTree(FILE* fctl, char *curline);
int ReadSeqData(FILE*fout, char seqfile[], char ratefile[], int cleandata);
int ReadTreeData(FILE*fout, char treefile[]);
int Initialize3s(double space[]);
double lfun(double x[], int np);
void setupGtreeTab();
void p0124Fromb0b1 (double p[5], double b[2]);
int Models0123(FILE *fout, FILE *frub, FILE *frst, double space[]);
int RunModel(FILE *fout, FILE *frub, FILE *frst, double space[]);
//int Simulation(FILE *fout, FILE *frub, double space[]);
int GenerateQ1SIM3S(double Q[], int nStates, double theta1, double theta2, double theta3, double M12, double M21);
int GenerateQ5(double Q[], double theta1, double theta2, double theta3, double M12, double M21, double M13, double M31, double M23, double M32);
int GenerateQ6(double Q[], double theta1, double theta2, double theta3, double M12, double M21, double M13, double M31, double M23, double M32);
int GenerateQ7(double Q[], double theta5, double theta3, double M53, double M35);
int GenerateQ8(double Q[], double theta5, double theta3, double M53, double M35);
void EigenSort(double d[], double U[], int n);

enum {ESeqData = 1, ETreeData};
enum {M0 = 0, M1DiscreteBeta, M2IM, M3MSci, M2SIM3s = 2, M2Pro, M2ProMax, M3MSci12, M3MSci13, M3MSci23} MODELS;
enum {G1c, G1b, G1a, G2c, G2b, G2a, G3c, G3b, G3a, G4c, G4b, G4a, G5c, G5b, G5a, G6c, G6b, G6a} GTREES;
enum {C1 = 8, C2 = 21, C3 = 4, C4, C5 = 28, C5_ex = 9, C6 = 7, C6_ex = 4, C7 = 9, C7_ex = 1, C8 = 4, C8_ex = 1} CHAIN;
enum {s111_C1, s112_C1, s122_C1, s222_C1, s11_C1, s12_C1, s22_C1, sA_C1} STATESC1;
enum {s111_C2, s112_C2, s121_C2, s122_C2, s211_C2, s212_C2, s221_C2, s222_C2, s1a1_C2, s1b1_C2, s1c1_C2, s1a2_C2, s1b2_C2, s1c2_C2, s12a_C2, s12b_C2, s12c_C2, s2a2_C2, s2b2_C2, s2c2_C2, sA_C2} STATESC2;
enum {s11, s12, s13, s21, s22, s23, s31, s32, s33} STATES2;

// Chain 5: { 111, 112, 113, 121, 122, 123, 131, 132, 133, 211, 212, 213, 221, 222, 223, 231, 232, 233, 311, 312, 313, 321, 322, 323, 331, 332, 333, a }
// C5_ex:   { 555, 553, 535, 355, 335, 353, 533, 333, tmp }
// Chain 6: { 1k1, 2k1, 2k2, 3k1, 3k2, 3k3, a }
// C6_ex:   { 5k5, 3k5, 3k3, tmp }
// Chain 7: { 555, 553, 535, 355, 335, 353, 533, 333, a }
// C7_ex:   { 444 }
// Chain 8: { 5k5, 3k5, 3k3, a }
// C8_ex:   { 4k4 }

#endif
