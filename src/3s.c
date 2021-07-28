/* 3s.c

 Ziheng Yang, 2002, 2009, 2011

 This is an ML program to estimate theta4, theta5, tau0, and tau1 for the 3-species problem,
 using numerical integration to calculate the likelihood.

 cl -O2 3s.c tools.c lfun3s.c
 cc -o 3s -O3 3s.c tools.c lfun3s.c -lm

 If GNU Scientific Library (GSL) is installed:
 cc -o 3s -O3 -DUSE_GSL -I/usr/local/include -L/usr/local/lib 3s.c tools.c lfun3s.c -lm -lgsl -lgslcblas

 If OpenMP is installed:
 cc -o 3s -O3 -DUSE_GSL -I/usr/local/include -L/usr/local/lib -fopenmp 3s.c tools.c lfun3s.c -lm -lgsl -lgslcblas

 3s
 3s <ctlfile>

 npoints can be 4, 8, 16, 32, 64, 128, ..., 1024.  The default is 32.
 */

#include "3s.h"

#if defined(USE_GSL) && defined(TEST_MULTIMIN)
#include "multimin.h"
#endif

#ifdef DEBUG_MEMORY_LEAK
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

extern double Small_Diff;
extern int noisy, NFunCall;

struct CommonInfo com;
//struct BTEntry * GtreeTab[27] = {0};

int LASTROUND;

double para[MAXPARAMETERS];  /* theta4 theta5 tau0 tau1 theta1 theta2 theta3 M12 M21 qbeta */
//double avgb[2] = {0,0}, varb[2] = {0,0};
//int debug = 0;
//#define DEBUG
#define DBGLOCUS 2

char * ModelStr[NEXTMODELS] = {"M0", "DiscreteBeta", "Isolation-with-Migration", "Introgression"};
char * paranames[MAXPARAMETERS] = {"theta4", "theta5", "tau0", "tau1", "theta1", "theta2", "theta3"};
//char *GtreeStr[18] = {"Gk", "G1b", "G1a", "G2c", "G2b", "G2a", "G3c", "G3b", "G3a", "G4c", "G4b", "G4a", "G5c", "G5b", "G5a", "G6c", "G6b", "G6a"};
char * GtreeStr[35] = {"111G1(3)", "111G2(3)", "111G3(3)", "111G4(3)", "111G5(3)", "111G6(3)", "222G1(3)", "222G2(3)", "222G3(3)", "222G4(3)", "222G5(3)", "222G6(3)", "112G2(1)", "112G3(1)", "112G4(3)", "112G5(3)", "112G6(3)", "221G2(1)", "221G3(1)", "221G4(3)", "221G5(3)", "221G6(3)", "113G3(1)", "113G5(1)", "113G6(3)", "223G3(1)", "223G5(1)", "223G6(3)", "123G5(1)", "123G6(3)", "133G3&5(1)", "133G6(3)", "333G1&2&4(3)", "333G3&5(3)", "333G6(3)"};
char * stateStr[NALLSTATES] = {"111", "112", "113", "121", "122", "123", "131", "132", "133", "211", "212", "213", "221", "222", "223", "231", "232", "233", "311", "312", "313", "321", "322", "323", "331", "332", "333", "11", "12", "13", "21", "22", "23", "31", "32", "33"};
char * spStr[MAXSPTREES] = { "((1, 2), 3)", "((1, 3), 2)", "((2, 3), 1)", "((2, 1), 3)", "((3, 1), 2)", "((3, 2), 1)" };
//                            0  1  2   3   4  5   6   7  8   9  10  11  12 13 14  15  16 17  18  19  20  21  22  23  24  25 26
const int initStateMap[27] = {0, 0, 0, -1, -1, 1, -1, -1, 0, -1, -1, -1,  1, 1, 2, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 2};
const int initStates[NINITIALSTATES] = { 0, 13, 1, 12, 2, 14, 5, 8, 17, 26 }; // { 111, 222, 112, 221, 113, 223, 123, 133, 233, 333 }
const int initStates2seq[NINITIALSTATES2SEQ] = { 0, 2, 1, 3, 4, 5 }; // { 11, 22, 12, 13, 23, 33 }
const int initStatesRaw2seq[NINITIALSTATES2SEQ] = { 27, 31, 28, 29, 32, 35 }; // { 11, 22, 12, 13, 23, 33 }
const int GtreeMapM0[35] = { 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1,12,12,12,12,12, 2, 2, 2,14,14,14, 5, 5, 8, 8,26,26,26};
const int GtreeMapM2[38] = { 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0};
const int GtOffsetM0[35] = { 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 42, 44, 45, 52, 54, 55, 64, 65, 72, 75, 90, 92, 95 };
const int GtOffsetM2[38] = { 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34, 35, 42, 44, 45, 52, 54, 55, 62, 64, 65, 72, 75, 90, 92, 95 };
const int GtOffsetM2Pro[60] = { 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34, 35, 40, 41, 42, 43, 44, 45, 50, 51, 52, 53, 54, 55, 60, 61, 62, 63, 64, 65, 70, 71, 72, 73, 74, 75, 80, 81, 82, 83, 84, 85, 90, 91, 92, 93, 94, 95 };
const int GtOffsetM2ProMax[60] = { 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34, 35, 40, 41, 42, 43, 44, 45, 50, 51, 52, 53, 54, 55, 60, 61, 62, 63, 64, 65, 70, 71, 72, 73, 74, 75, 80, 81, 82, 83, 84, 85, 90, 91, 92, 93, 94, 95 };
const int GtOffsetM3MSci12[54] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 42, 44, 45, 49, 52, 54, 55, 59, 64, 65, 69, 72, 75, 90, 92, 95 };
const int GtOffsetM3MSci13[73] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 28, 29, 31, 32, 33, 34, 35, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 63, 64, 65, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 85, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99 };
const int GtOffsetM3MSci23[73] = { 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 38, 39, 41, 42, 43, 44, 45, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 68, 69, 71, 72, 73, 74, 75, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99 };
const int spMap[MAXSPTREES][3] = { { 1, 2, 3 },{ 1, 3, 2 },{ 2, 3, 1 },{ 2, 1, 3 },{ 3, 1, 2 },{ 3, 2, 1 } };
const int spPermute[MAXSPTREES][3] = { { 1, 2, 3 },{ 1, 3, 2 },{ 3, 2, 1 },{ 1, 3, 2 },{ 3, 2, 1 },{ 1, 3, 2 } };
const double chi2CV_5pct[14] = { 0, 3.84146, 5.99146, 7.81473, 9.48773, 11.0705, 12.5916, 14.0671, 15.5073, 16.919, 18.307, 19.6751, 21.0261, 22.362 };

STATIC_ASSERT(MAXGTREES >= LENGTH_OF(GtOffsetM0));
STATIC_ASSERT(MAXGTREES >= LENGTH_OF(GtOffsetM2));
STATIC_ASSERT(MAXGTREES >= LENGTH_OF(GtOffsetM2Pro));
STATIC_ASSERT(MAXGTREES >= LENGTH_OF(GtOffsetM2ProMax));
STATIC_ASSERT(MAXGTREES >= LENGTH_OF(GtOffsetM3MSci12));
STATIC_ASSERT(MAXGTREES >= LENGTH_OF(GtOffsetM3MSci13));
STATIC_ASSERT(MAXGTREES >= LENGTH_OF(GtOffsetM3MSci23));

#define REALSEQUENCE
#define NODESTRUCTURE
#include "treesub.c"

FILE *fout, *frub, *frst, *fpGk;

#ifdef CONTOUR
double lbound1, ubound1, inc1, lbound2, ubound2, inc2;
#endif

int main (int argc, char* argv[])
{
    char VerStr[32] = "Version 3.0, Aug 2015";
    double* space;

    if (argc > 2 && !strcmp(argv[argc - 1], "--stdout-no-buf"))
        setvbuf(stdout, NULL, _IONBF, 0);

    printf("3s (%s)\n", VerStr);

    space = (double *) malloc(10*C5*C5*sizeof(double));
    if(space == NULL) error2("oom space");
    com.space = space+1500;

    // this is to raise a kernel exception on float exceptions
    //_mm_setcsr( _MM_MASK_MASK &~ (_MM_MASK_OVERFLOW|_MM_MASK_INVALID|_MM_MASK_DIV_ZERO));

#ifdef CONTOUR
    lbound1 = atof(argv[2]);
    ubound1 = atof(argv[3]);
    inc1 = atof(argv[4]);
    lbound2 = atof(argv[5]);
    ubound2 = atof(argv[6]);
    inc2 = atof(argv[7]);
#endif

    strcpy(com.ctlf, "3s.ctl");
    if(argc>1) strcpy(com.ctlf, argv[1]);
    starttimer();
    GetOptions (com.ctlf);

    if(com.seed<=0) com.seed = abs((2*(int)time(NULL)+1));
    SetSeed(com.seed, 0);

    fout=gfopen(com.outf, "w");
    frst=gfopen("rst", "w");
    frub=(FILE*)gfopen("rub","w");
    fprintf(fout, "3s (%s)\n", VerStr);
    /*
     ReadSiteCounts(datafile);
     */
#if(1)
    if (com.usedata == ESeqData)
        ReadSeqData((com.verbose) ? fout : NULL, com.seqf, com.ratef, com.cleandata);
    else
        ReadTreeData((com.verbose) ? fout : NULL, com.treef);

    noisy = 3;
    Models0123(fout, frub, frst,/* x,*/ space);

    if (com.usedata == ESeqData) {
        free(data.Nij[0]);
        free(data.initState[0]);
        free(data.lnLmax[0]);
    }
    else {
        free(data.Bij);
        free(data.topology[0]);
        free(data.GtreeType);
        free(data.initState[0]);
    }
#else
    Simulation(fout, frub, space);
#endif

    free(space);
    fclose(frst);
    fclose(frub);
    fclose(fout);

#ifdef DEBUG_MEMORY_LEAK
    _CrtDumpMemoryLeaks();
#endif

    return 0;
}


int GetOptions (char *ctlf)
{
    int iopt,i,j,offset, lline=4096;
    char line[4096],*pline, opt[32], *comment="*#"; //, *seqerrstr="0EF";
    char *optstr[] = {"seed", "nloci", "outfile", "seqfile", "ratefile",
        "cleandata", "npoints", "getSE", "Small_Diff", "simmodel", "models", "Imapfile", "treefile",
        "speciestree", "species&tree", "usedata", "verbose", "initialvalues", "nthreads",
        "est_M12", "est_M21", "est_M13", "est_M31", "est_M23", "est_M32", "est_M53", "est_M35",
        "est_phi12", "est_phi21", "est_phi13", "est_phi31", "est_phi23", "est_phi32",
        "est_thetaX", "est_thetaY", "est_thetaZ"};
    int nopt = LENGTH_OF(optstr);
    double t=1;
    int m[NEXTMODELS] = {0, 0, 0, 0}, runm[NEXTMODELS] = {0, 0, 0, 0}, nm = 0;
    FILE  *fctl=gfopen (ctlf, "r");
    enum { M12 = 0, M21, M13, M31, M23, M32, M53, M35, Phi12, Phi21, Phi13, Phi31, Phi23, Phi32, ThetaX = 0, ThetaY, ThetaZ };

    stree.speciestree = 0;
    stree.nspecies = 0;
    com.ncode = 4;
    com.npoints = 16;
    com.ncatBeta = 5;
    com.UseMedianBeta = 0;   /* 1 to use the median */
    com.usedata = ESeqData;
    com.verbose = 1;
    com.aroundMLEM0 = 1;
    com.nthreads = 1;
    com.cleandata = 1;
    com.fix_locusrate = 0;
    com.simmodel = 0; /* use asymmetric model by default */
    com.runmodels[M0] = com.runmodels[M1DiscreteBeta] = com.runmodels[M2IM] = com.runmodels[M3MSci] = 0;
    strcpy(com.Imapf, ""); /* by default, use */
    memset(com.initials, 0, NMODELS*MAXPARAMETERS*sizeof(double));
    memset(com.fixto0, 0, LENGTH_OF(com.fixto0)*sizeof(int));
    memset(com.fix, 0, LENGTH_OF(com.fix)*sizeof(int));
    memset(com.asymmetric, 0, LENGTH_OF(com.asymmetric)*sizeof(int));

    if (fctl) {
        if (noisy) printf ("\nReading options from %s..\n", ctlf);
        for (;;) {
            if(fgets(line, lline, fctl) == NULL) break;
            if(line[0]=='/' && line[1]=='/')
                break;
            for (i=0,t=0,pline=line; i<lline&&line[i]; i++)
                if (isalnum(line[i]))  { t=1; break; }
                else if (strchr(comment,line[i])) break;
            if (t==0) continue;
            sscanf (line, "%s%*s%lf", opt, &t);
            if ((pline=strstr(line, "="))==NULL)
                continue;

            for (iopt=0; iopt<nopt; iopt++) {
                if (strncmp(opt, optstr[iopt], 20)==0)  {
                    if (noisy>=9)
                        printf ("\n%3d %15s | %-20s %6.2f", iopt+1,optstr[iopt],opt,t);
                    switch (iopt) {
                        case ( 0): com.seed=(int)t;                    break;
                        case ( 1): data.ndata=(int)t;
                            break;
                        case ( 2): sscanf(pline+1, "%s", com.outf);    break;
                        case ( 3): sscanf(pline+1, "%s", com.seqf);    break;
                        case ( 4): sscanf(pline+1, "%s", com.ratef);   com.fix_locusrate = 1; break;
                        case ( 5):
                            com.cleandata=(int)t;
                            if(com.cleandata!=1) error2("use cleandata = 1");
                            break;
                        case ( 6):
                            sscanf(pline+1, "%d%d%d", &com.npoints, &com.ncatBeta, &com.UseMedianBeta);
                            break;
                        case ( 7): com.getSE=(int)t;                   break;
                        case ( 8): Small_Diff=t;                       break;
                        case ( 9): com.simmodel = (int)t;              break;
                        case (10): nm = sscanf(pline+1, "%d %d %d %d", &m[0], &m[1], &m[2], &m[3]); break;
                        case (11): sscanf(pline+1, "%s", com.Imapf);   break;
                        case (12): sscanf(pline+1, "%s", com.treef);   break;
                        case (13):
                            stree.speciestree = (int)t;
                            if (stree.speciestree != 0 && stree.speciestree != 1) error2("speciestree = 0 or 1");
                            break;
                        case (14):
                            stree.nspecies = (int)t;
                            if (stree.nspecies != NSPECIES) error2("3s only deals with 3 species");
                            ReadSpeciesTree(fctl, pline+1);
                            break;
                        case (15):
                            com.usedata = (int)t;
                            if (com.usedata != ESeqData && com.usedata != ETreeData) error2("usedata = 1 or 2");
                            break;
                        case (16):
                            com.verbose = (int)t;
                            if (com.verbose != 0 && com.verbose != 1) error2("verbose = 0 or 1");
                            break;
                        case (17):
                            com.aroundMLEM0 = (int)t;
                            if (com.aroundMLEM0 != 0 && com.aroundMLEM0 != 1) error2("initialvalues = 0 or 1");
                            break;
                        case (18):
                            com.nthreads = (int)t;
#ifdef _OPENMP
                            if (com.nthreads < 1 && com.nthreads != -1) error2("nthreads >= 1 or = -1");
#else
                            if (com.nthreads != 1 && com.nthreads != -1)
                                fprintf(stderr, "\nWarning: nthreads is ignored. To enable parallelization, please compile the program with OpenMP.\n");
                            com.nthreads = 1;
#endif
                            break;
                        case (19): com.fixto0[M12] = !(int)t;          break;
                        case (20): com.fixto0[M21] = !(int)t;          break;
                        case (21): com.fixto0[M13] = !(int)t;          break;
                        case (22): com.fixto0[M31] = !(int)t;          break;
                        case (23): com.fixto0[M23] = !(int)t;          break;
                        case (24): com.fixto0[M32] = !(int)t;          break;
                        case (25): com.fixto0[M53] = !(int)t;          break;
                        case (26): com.fixto0[M35] = !(int)t;          break;
                        case (27): com.fixto0[Phi12] = !(int)t;        break;
                        case (28): com.fixto0[Phi21] = !(int)t;        break;
                        case (29): com.fixto0[Phi13] = !(int)t;        break;
                        case (30): com.fixto0[Phi31] = !(int)t;        break;
                        case (31): com.fixto0[Phi23] = !(int)t;        break;
                        case (32): com.fixto0[Phi32] = !(int)t;        break;
                        case (33): com.fix[ThetaX] = !(int)t;          break;
                        case (34): com.fix[ThetaY] = !(int)t;          break;
                        case (35): com.fix[ThetaZ] = !(int)t;          break;
                    }
                    break;
                }
            }

            if (iopt==nopt)
            { fprintf(stderr, "\noption %s in %s\n", opt, ctlf);  exit (-1); }
        }

        fclose(fctl);

        if (stree.nspecies != NSPECIES)
            error2("\noption species&tree: species tree is required\n");

        if (nm > 0) {
            for (i = 0; i < nm; i++) {
                if (m[i] >= NEXTMODELS) {
                    error2("\noption models: unknown model\n");
                }
                runm[m[i]] = 1;
            }
            for (i = 0; i < NEXTMODELS; i++) {
                com.runmodels[i] = runm[i];
            }
        }

        if (!com.runmodels[M0])
            error2("\noption models: model M0 is required\n");

        for (i = 0; i < NEXTMODELS; i++) {
            if (com.runmodels[i])
                com.modelMap[i] = ModelMap(i);
        }

        com.asymmetric[M2SIM3s] = (!com.simmodel && com.fixto0[M12] != com.fixto0[M21]);
        com.asymmetric[M2Pro] = (com.fixto0[M12] != com.fixto0[M21] || com.fixto0[M13] != com.fixto0[M23] || com.fixto0[M31] != com.fixto0[M32]);
        com.asymmetric[M2ProMax] = com.asymmetric[M2Pro];
        com.asymmetric[M3MSci12] = (com.fixto0[Phi12] != com.fixto0[Phi21]);
        com.asymmetric[M3MSci13] = (!com.fixto0[Phi13] || !com.fixto0[Phi31]);
        com.asymmetric[M3MSci23] = (!com.fixto0[Phi23] || !com.fixto0[Phi32]);

        if (stree.speciestree)
            if (com.runmodels[M2IM] && com.asymmetric[com.modelMap[M2IM]] || com.runmodels[M3MSci] && com.asymmetric[com.modelMap[M3MSci]])
                stree.nsptree = 6;
            else
                stree.nsptree = 3;
        else
            stree.nsptree = 1;
    }
    else
        if (noisy) error2("\nno ctl file..");

    fctl = fopen("in.3s", "r");
    if (fctl) {
        for (i=0; i<NEXTMODELS; i++) {
            while (fgets(line, lline, fctl)) {
                for (j=0, t=0; j<lline && line[j]; j++) {
                    if (!isspace(line[j])) { t = 1; break; }
                }
                if (t && !strchr(comment, line[j])) break;
            }
            if (!com.runmodels[i]) {
                continue;
            }
            for (j=0, pline=line; j<MAXPARAMETERS; j++, pline+=offset) {
                if (sscanf(pline, "%lf%n", &com.initials[com.modelMap[i]][j], &offset) != 1)
                    break;
            }
        }
    }

    return(0);
}

int ModelMap(int extModel)
{
    enum { M12 = 0, M21, M13, M31, M23, M32, M53, M35, Phi12, Phi21, Phi13, Phi31, Phi23, Phi32 };
    int intModel;

    switch (extModel)
    {
    case M2IM:
        if(!com.fixto0[M53] || !com.fixto0[M35])
            intModel = M2ProMax;
        else if (!com.fixto0[M13] || !com.fixto0[M31] || !com.fixto0[M23] || !com.fixto0[M32])
            intModel = M2Pro;
        else
            intModel = M2SIM3s;
        break;
    case M3MSci:
        if (com.fixto0[Phi13] && com.fixto0[Phi31] && com.fixto0[Phi23] && com.fixto0[Phi32])
            intModel = M3MSci12;
        else if ((!com.fixto0[Phi13] || !com.fixto0[Phi31]) && com.fixto0[Phi12] && com.fixto0[Phi21] && com.fixto0[Phi23] && com.fixto0[Phi32])
            intModel = M3MSci13;
        else if ((!com.fixto0[Phi23] || !com.fixto0[Phi32]) && com.fixto0[Phi12] && com.fixto0[Phi21] && com.fixto0[Phi13] && com.fixto0[Phi31])
            intModel = M3MSci23;
        else
            error2("model not implemented yet!");
        break;
    default:
        intModel = extModel;
    }

    return intModel;
}

int DownSptreeSetSpnames(int inode, int SetSpNames)
{
   /* This traverse down the species tree to set stree.nodes[].name.
   */
   int k, ison;

   if (inode < stree.nspecies)
      error2("should not be here?");

   //if (SetSpNames && stree.nspecies > NPopLongNames)
   //   sprintf(stree.nodes[inode].name, "%d", inode + 1);

   for (k = 0; k < stree.nodes[inode].nson; k++) {
      ison = stree.nodes[inode].sons[k];
      if (stree.nodes[ison].nson)
         DownSptreeSetSpnames(ison, SetSpNames);
      if (SetSpNames/* && stree.nspecies <= NPopLongNames*/) {
         if (strlen(stree.nodes[inode].name) + strlen(stree.nodes[ison].name) > 2 * LSPNAME - 1)
            error2("we are in trouble.  Increase LSPNAME?");
         strcat(stree.nodes[inode].name, stree.nodes[ison].name);
      }
   }
   return(0);
}

int SetupPopPopTable(int PrintTable)
{
   int i, j, s = stree.nspecies, root = stree.root;

   for (i = 0; i < 2 * s - 1; i++) for (j = 0; j < 2 * s - 1; j++)
      stree.pptable[i][j] = (char)0;
   for (i = 0; i < 2 * s - 1; i++) stree.pptable[i][i] = stree.pptable[i][root] = (char)1;
   for (i = 0; i < 2 * s - 1; i++) {
      for (j = i; j != root; j = stree.nodes[j].father)
         stree.pptable[i][j] = (char)1;
   }

   if (PrintTable) {
      printf("\npop by pop table showing node numbers in species tree\n\n%22s", " ");
      for (i = 0; i < 2 * s - 1; i++)
         printf(" %2d", i + 1);
      FPN(F0);
      for (i = 0; i < 2 * s - 1; i++, FPN(F0)) {
         printf("species %2d %-10s ", i + 1, stree.nodes[i].name);
         for (j = 0; j < 2 * s - 1; j++)
            printf(" %2d", (int)stree.pptable[i][j]);
         if (i < stree.nspecies && stree.nodes[i].age)
            printf("\n\a  <-- age>0??\n");
//#ifdef SIMULATION
//         printf("  tau =%7.4f theta =%7.4f  ", stree.nodes[i].age, stree.nodes[i].theta);
//         if ((i >= s || stree.nseqsp[i] > 1) && stree.nodes[i].theta <= 0)
//            printf("  this theta must be > 0!!");
//#endif
      }
   }
   return(0);
}

int ReadSpeciesTree(FILE* fctl, char *curline)
{
    int i, ifields[1000];

    splitline(curline, ifields);
    for (i = 0; i < stree.nspecies; i++)
        if (sscanf(curline + ifields[i + 1], "%s", stree.nodes[i].name) != 1)
            error2("species name?");

    printf("\n%d species: ", stree.nspecies);
    for (i = 0; i < stree.nspecies; i++)
        printf(" %s", stree.nodes[i].name);
    putchar('\n');

    printf("((%s, %s), %s);", stree.nodes[0].name, stree.nodes[1].name, stree.nodes[2].name);

    stree.nnode = 2 * stree.nspecies - 1;
    stree.nbranch = 2 * stree.nspecies - 2;
    stree.root = stree.nspecies;

    stree.nodes[0].father = 4;
    stree.nodes[0].nson = 0;

    stree.nodes[1].father = 4;
    stree.nodes[1].nson = 0;

    stree.nodes[2].father = 3;
    stree.nodes[2].nson = 0;

    stree.nodes[3].father = -1;
    stree.nodes[3].nson = 2;
    stree.nodes[3].sons[0] = 4;
    stree.nodes[3].sons[1] = 2;

    stree.nodes[4].father = 3;
    stree.nodes[4].nson = 2;
    stree.nodes[4].sons[0] = 0;
    stree.nodes[4].sons[1] = 1;

    DownSptreeSetSpnames(stree.root, 1);
    SetupPopPopTable(1);

    return(0);
}

int ReadSeqData(FILE*fout, char seqfile[], char ratefile[], int cleandata)
{
    /* Read sequences at each locus and count sites.
     All sites with ambiguities are deleted right now.  cleandata is ignored.
     */
    FILE *fin = gfopen(seqfile,"r"), *fImap = NULL;

    int i,j, k, h, *n, im[3], tmp[3], initState, sptree, *Imap, nind = 0;
    double mr, mNij[5]={0};
    unsigned char *pz[3];
    char newOrder[3], *pline, **Inames, curname[100], line[10000];
    int maxnind = 30, lname = 100, s = stree.nspecies;

    if(com.Imapf[0] != '\0') {
        printf("\nReading Individual-Species map (Imap) from %s\n", com.Imapf);
        Imap = (int*)malloc(maxnind * sizeof(int));
        if (Imap == NULL) error2("oom Imap");
        Inames = (char**)malloc(maxnind * sizeof(char*));
        if (Inames == NULL) error2("oom Inames");
        Inames[0] = (char*)malloc(maxnind * lname * sizeof(char));
        if (Inames[0] == NULL) error2("oom Inames[0]");
        memset(Inames[0], 0, maxnind * lname * sizeof(char));
        fImap = gfopen(com.Imapf, "r");

        for(i=0; ; i++) {
            if(i == maxnind) {
                Imap = (int*)realloc(Imap, 2 * maxnind * sizeof(int));
                if (Imap == NULL) error2("oom Imap");
                Inames = (char**)realloc(Inames, 2 * maxnind * sizeof(char*));
                if (Inames == NULL) error2("oom Inames");
                Inames[0] = (char*)realloc(Inames[0], 2 * maxnind * lname * sizeof(char));
                if (Inames[0] == NULL) error2("oom Inames[0]");
                memset(Inames[0] + maxnind * lname, 0, maxnind * lname * sizeof(char));
                maxnind *= 2;
            }

            Inames[i] = Inames[0] + i * lname;
            if(fscanf(fImap, "%s%s", Inames[i], line) != 2) break;
            if(strstr(Inames[i], "//")) break;

            for (j = 0; j < i; j++)
                if (strcmp(Inames[j], Inames[i]) == 0) {
                    fprintf(stderr, "Duplicate individual label %s.", Inames[i]);
                    error2("Please fix the Imap file.");
                }

            for (j = 0; j < s; j++)
               if (strcmp(line, stree.nodes[j].name) == 0) {  /* ind i is from pop j. */
                  Imap[i] = j + 1;
                  break;
               }
            if (j == s) {
                Imap[i] = 0;
            }
        }

        printf("Individual -> Species map:\n");
        nind=i;
        for(i=0; i<nind; i++)
            if(Imap[i])
                printf("%10s -> %s\n", Inames[i], stree.nodes[Imap[i] - 1].name);

        fclose(fImap);
    }

    printf("\nReading sequence data..  %d loci\n", data.ndata);
    if((data.Nij[0]=(int*)malloc(stree.nsptree*data.ndata*5*sizeof(int)))==NULL) error2("oom");
    memset(data.Nij[0], 0, stree.nsptree*data.ndata*5*sizeof(int));
    if((data.initState[0]=(int*)malloc(stree.nsptree*data.ndata*sizeof(int)))==NULL) error2("oom");
    memset(data.initState[0], 0, stree.nsptree*data.ndata*sizeof(int));
    memset(data.iStateCnt, 0, MAXSPTREES*NALLSTATES*sizeof(int));

    if((data.lnLmax[0]=(double*)malloc(data.ndata*(stree.nsptree +com.fix_locusrate)*sizeof(double)))==NULL)
        error2("oom lnLmax");
    if(com.fix_locusrate) data.locusrate = data.lnLmax[0] + stree.nsptree * data.ndata;

    for (sptree = 1; sptree < stree.nsptree; sptree++) {
        data.Nij[sptree] = data.Nij[0] + sptree * data.ndata * 5;
        data.initState[sptree] = data.initState[0] + sptree * data.ndata;
        data.lnLmax[sptree] = data.lnLmax[0] + sptree * data.ndata;
    }

    for(i=0; i<data.ndata; i++) {
        if(fout) fprintf(fout, "\n\n*** Locus %d ***\n", i+1);
        ReadSeq (NULL, fin, cleandata, -1);

        if (com.ns != 3 && com.ns != 2)
            error2("3s only deals with 2 or 3 sequences at each locus");

        PatternWeightJC69like (fout);

        // process sequence names
        for(j = 0; j < com.ns; j++) {
            pline = strchr(com.spname[j], IDSEP);
            if(pline == NULL) error2("sequences must be tagged by population ID");
            sscanf(pline+1, "%s", curname);
            if (com.Imapf[0] == '\0') {
                for (k = 0; k < s; k++) {
                    if (strcmp(curname, stree.nodes[k].name) == 0) {
                        break;
                    }
                }
                if(k==s) {
                    fprintf(stderr, "Individual label %s not found in the control file.", curname);
                    error2("Please fix the control file or use Imap file.");
                }
                else {
                    im[j] = k + 1;
                }
            } else {
                for (k = 0; k < nind; k++) {
                    if (strcmp(curname, Inames[k]) == 0) {
                        break;
                    }
                }
                if(k==nind) {
                    fprintf(stderr, "Individual label %s not recognised.", curname);
                    error2("Please fix the Imap file.");
                }
                else if(!Imap[k]) {
                    fprintf(stderr, "Individual label %s mapped to a species that is not found in the control file.", curname);
                    error2("Please fix the Imap file.");
                }
                else {
                    im[j] = Imap[k];
                }
            }
        }

        for (sptree = 0; sptree < stree.nsptree; sptree++) {
            n = data.Nij[sptree] + i * 5;

            for (j = 0; j < com.ns; j++)
                im[j] = spPermute[sptree][im[j] - 1];

            // re-ordering sequences and determining initial state
            if (com.ns == 3) { // 3 sequences
                for(j=0; j<3; j++) {
                    pz[j] = com.z[j];
                    tmp[j] = im[j];
                }

                if(im[0] < im[1]) {
                    if(im[0] < im[2]) {
                        newOrder[0] = 0;
                        if(im[1] <= im[2]) {
                            newOrder[1] = 1;
                            newOrder[2] = 2;
                        } else {
                            newOrder[1] = 2;
                            newOrder[2] = 1;
                        }
                    } else {
                        newOrder[0] = 1;
                        newOrder[1] = 2;
                        newOrder[2] = 0;
                    }
                } else {
                    if(im[1] < im[2]) {
                        newOrder[1] = 0;
                        if(im[0] < im[2]) {
                            newOrder[0] = 1;
                            newOrder[2] = 2;
                        } else {
                            newOrder[0] = 2;
                            newOrder[2] = 1;
                        }
                    } else {
                        newOrder[0] = 2;
                        newOrder[1] = 1;
                        newOrder[2] = 0;
                    }
                }

                if(newOrder[0]!=newOrder[1] && newOrder[0]!=newOrder[2] && newOrder[1]!=newOrder[2]) {
                    for(j=0; j<3; j++) {
                        com.z[(int)newOrder[j]] = pz[j];
                        im[(int)newOrder[j]] = tmp[j];
                    }
                }

                // special case: 122 needs to be ordered 221
                if ((im[0] == 1) && (im[1] == 2) && (im[2] == 2)) {
                    im[0] = 2; im[2] = 1;
                    pz[0] = com.z[0]; com.z[0] = com.z[2]; com.z[2] = pz[0];
                }

                initState = (im[0]-1)*9 + (im[1]-1)*3 + (im[2]-1); // determine initial state

                for(h=0; h<com.npatt; h++) {
                    if(com.z[0][h]==com.z[1][h] && com.z[0][h]==com.z[2][h])
                        n[0] = (int)com.fpatt[h];
                    else if(com.z[0][h]==com.z[1][h] && com.z[0][h]!=com.z[2][h])
                        n[1] = (int)com.fpatt[h];
                    else if(com.z[0][h]!=com.z[1][h] && com.z[1][h]==com.z[2][h])
                        n[2] = (int)com.fpatt[h];
                    else if(com.z[0][h]==com.z[2][h] && com.z[0][h]!=com.z[1][h])
                        n[3] = (int)com.fpatt[h];
                    else
                        n[4] = (int)com.fpatt[h];
                }
                if (sptree == 0) printf("%5d: %5d%5d%5d%5d%5d%9d\r", i+1, n[0],n[1],n[2],n[3],n[4], n[0]+n[1]+n[2]+n[3]+n[4]);
            } else { // 2 sequences
                if (im[0] > im[1]) {
                    pz[0] = com.z[0];
                    tmp[0] = im[0];
                    com.z[0] = com.z[1];
                    im[0] = im[1];
                    com.z[1] = pz[0];
                    im[1] = tmp[0];
                }
                initState = NSTATES + (im[0]-1)*3 + (im[1]-1);

                for (h=0;  h<com.npatt; h++) {
                    if (com.z[0][h] == com.z[1][h]) {
                        n[0] = (int)com.fpatt[h];
                    } else {
                        n[1] = (int)com.fpatt[h];
                    }
                }
                if (sptree == 0) printf("%5d: %5d%5d%5d\r", i+1, n[0], n[1], n[0]+n[1]);
            }

            if(sptree == 0 && noisy>=3) printf("\n");
            data.initState[sptree][i] = initState;
            data.iStateCnt[sptree][initState]++;
        }
    }

    free(com.pose);
    free(com.fpatt);
    for(i=0; i<NS; i++) {
        free(com.spname[i]);
        free(com.z[i]);
    }
    fclose(fin);

    if (com.Imapf[0] != '\0') {
        free(Inames[0]);
        free(Inames);
        free(Imap);
    }

    data.twoSeqLoci = data.iStateCnt[0][27] + data.iStateCnt[0][28] + data.iStateCnt[0][29] + data.iStateCnt[0][31] + data.iStateCnt[0][32] + data.iStateCnt[0][35];
    
    printf("\n\ninit state\t #loci\n");
    for (i=0; i<NALLSTATES; i++) {
        if (data.iStateCnt[0][i] > 0) {
            printf("    %s   \t%6d\n", stateStr[i], data.iStateCnt[0][i]);
        }
    }

    if(com.fix_locusrate) {
        if((fin = gfopen(ratefile, "r")) == NULL)
            error2("ratefile open error");
        for(i=0,mr=0; i<data.ndata; i++) {
            if(fscanf(fin, "%lf", &data.locusrate[i]) != 1)
                error2("rate file..");
            mr = (mr*i + data.locusrate[i])/(i+1.0);
        }
        fclose(fin);
        for(i=0; i<data.ndata; i++)  data.locusrate[i] /= mr;
        printf("\nRelative rates for %d loci, scaled to have mean 1, will be used as constants\n", data.ndata);
        if(fout) {
            fprintf(fout, "\n\nRelative rates for %d loci, scaled to have mean 1, will be used as constants\n", data.ndata);
            fprintf(fout, "theta's & tau's are defined using the average rate\n");
        }
    }

    return(0);
}

int ReadTreeData(FILE*fout, char treefile[])
{
    FILE *fin = gfopen(treefile,"r"), *fImap = NULL;

    int i,j, k, im[3], tmp[3], initState, sptree, *Imap, nind = 0;
    double *b;
    char newOrder[3], *pline, **Inames, curname[100], line[10000];
    int maxnind = 30, lname = 100, s = stree.nspecies;
    int haslength, haslabel, intlSon, outgroup;

    if(com.Imapf[0] != '\0') {
        printf("\nReading Individual-Species map (Imap) from %s\n", com.Imapf);
        Imap = (int*)malloc(maxnind * sizeof(int));
        if (Imap == NULL) error2("oom Imap");
        Inames = (char**)malloc(maxnind * sizeof(char*));
        if (Inames == NULL) error2("oom Inames");
        Inames[0] = (char*)malloc(maxnind * lname * sizeof(char));
        if (Inames[0] == NULL) error2("oom Inames[0]");
        memset(Inames[0], 0, maxnind * lname * sizeof(char));
        fImap = gfopen(com.Imapf, "r");

        for(i=0; ; i++) {
            if(i == maxnind) {
                Imap = (int*)realloc(Imap, 2 * maxnind * sizeof(int));
                if (Imap == NULL) error2("oom Imap");
                Inames = (char**)realloc(Inames, 2 * maxnind * sizeof(char*));
                if (Inames == NULL) error2("oom Inames");
                Inames[0] = (char*)realloc(Inames[0], 2 * maxnind * lname * sizeof(char));
                if (Inames[0] == NULL) error2("oom Inames[0]");
                memset(Inames[0] + maxnind * lname, 0, maxnind * lname * sizeof(char));
                maxnind *= 2;
            }

            Inames[i] = Inames[0] + i * lname;
            if(fscanf(fImap, "%s%s", Inames[i], line) != 2) break;
            if(strstr(Inames[i], "//")) break;

            for (j = 0; j < i; j++)
                if (strcmp(Inames[j], Inames[i]) == 0) {
                    fprintf(stderr, "Duplicate individual label %s.", Inames[i]);
                    error2("Please fix the Imap file.");
                }

            for (j = 0; j < s; j++)
               if (strcmp(line, stree.nodes[j].name) == 0) {  /* ind i is from pop j. */
                  Imap[i] = j + 1;
                  break;
               }
            if (j == s) {
                Imap[i] = 0;
            }
        }

        printf("Individual -> Species map:\n");
        nind=i;
        for(i=0; i<nind; i++)
            if(Imap[i])
                printf("%10s -> %s\n", Inames[i], stree.nodes[Imap[i] - 1].name);

        fclose(fImap);
    }

    printf("\nReading tree data..  %d trees\n", data.ndata);
    if((data.Bij=(double*)malloc(data.ndata*2*sizeof(double)))==NULL) error2("oom");
    memset(data.Bij, 0, data.ndata*2*sizeof(double));
    if((data.topology[0]=(int*)malloc(stree.nsptree*data.ndata*sizeof(int)))==NULL) error2("oom");
    memset(data.topology[0], 0, stree.nsptree*data.ndata*sizeof(int));
    if((data.GtreeType=(int*)malloc(data.ndata*sizeof(int)))==NULL) error2("oom");
    memset(data.GtreeType, 0, data.ndata*sizeof(int));
    if((data.initState[0]=(int*)malloc(stree.nsptree*data.ndata*sizeof(int)))==NULL) error2("oom");
    memset(data.initState[0], 0, stree.nsptree*data.ndata*sizeof(int));
    memset(data.iStateCnt, 0, MAXSPTREES*NALLSTATES*sizeof(int));

    for (sptree = 1; sptree < stree.nsptree; sptree++) {
        data.topology[sptree] = data.topology[0] + sptree * data.ndata;
        data.initState[sptree] = data.initState[0] + sptree * data.ndata;
    }

    com.ns = NS;
    for(j=0; j<NS; j++) {
        if(com.spname[j]) free(com.spname[j]);
        com.spname[j] = (char*)malloc((LSPNAME+1)*sizeof(char));
        for(i=0; i<LSPNAME+1; i++) com.spname[j][i]=0;
    }

    for(i=0,b=data.Bij; i<data.ndata; i++,b+=2) {
        if(fout) fprintf(fout, "\n\n*** Tree %d ***\n", i+1);
        if(ReadTreeN(fin, &haslength, &haslabel, 1, 1) == -1)
            error2("tree error: EOF");

        intlSon = (nodes[3].father == -1) ? 4 : 3;
        for (j = 0; j < com.ns; j++)
            if (nodes[j].father != intlSon) break;
        outgroup = j;
        b[0] = nodes[intlSon].branch;
        b[1] = nodes[nodes[intlSon].sons[0]].branch;

        if (fabs(b[0] + b[1] - nodes[outgroup].branch) > 1e-6 || fabs(b[1] - nodes[nodes[intlSon].sons[1]].branch) > 1e-6)
            error2("not a labeled history");

        // process sequence names
        for(j = 0; j < com.ns; j++) {
            pline = strchr(com.spname[j], IDSEP);
            if(pline == NULL) error2("nodes must be tagged by population ID");
            sscanf(pline+1, "%s", curname);
            if (com.Imapf[0] == '\0') {
                for (k = 0; k < s; k++) {
                    if (strcmp(curname, stree.nodes[k].name) == 0) {
                        break;
                    }
                }
                if(k==s) {
                    fprintf(stderr, "Individual label %s not found in the control file.", curname);
                    error2("Please fix the control file or use Imap file.");
                }
                else {
                    im[j] = k + 1;
                }
            } else {
                for (k = 0; k < nind; k++) {
                    if (strcmp(curname, Inames[k]) == 0) {
                        break;
                    }
                }
                if(k==nind) {
                    fprintf(stderr, "Individual label %s not recognised.", curname);
                    error2("Please fix the Imap file.");
                }
                else if(!Imap[k]) {
                    fprintf(stderr, "Individual label %s mapped to a species that is not found in the control file.", curname);
                    error2("Please fix the Imap file.");
                }
                else {
                    im[j] = Imap[k];
                }
            }
        }

        for (sptree = 0; sptree < stree.nsptree; sptree++) {
            for (j = 0; j < com.ns; j++)
                im[j] = spPermute[sptree][im[j] - 1];

            // re-ordering sequences and determining initial state
            if (com.ns == 3) { // 3 sequences
                for(j=0; j<3; j++) {
                    tmp[j] = im[j];
                }

                if(im[0] < im[1]) {
                    if(im[0] < im[2]) {
                        newOrder[0] = 0;
                        if(im[1] <= im[2]) {
                            newOrder[1] = 1;
                            newOrder[2] = 2;
                        } else {
                            newOrder[1] = 2;
                            newOrder[2] = 1;
                        }
                    } else {
                        newOrder[0] = 1;
                        newOrder[1] = 2;
                        newOrder[2] = 0;
                    }
                } else {
                    if(im[1] < im[2]) {
                        newOrder[1] = 0;
                        if(im[0] < im[2]) {
                            newOrder[0] = 1;
                            newOrder[2] = 2;
                        } else {
                            newOrder[0] = 2;
                            newOrder[2] = 1;
                        }
                    } else {
                        newOrder[0] = 2;
                        newOrder[1] = 1;
                        newOrder[2] = 0;
                    }
                }

                if(newOrder[0]!=newOrder[1] && newOrder[0]!=newOrder[2] && newOrder[1]!=newOrder[2]) {
                    for(j=0; j<3; j++) {
                        im[(int)newOrder[j]] = tmp[j];
                    }
                    outgroup = (int)newOrder[outgroup];
                }

                // special case: 122 needs to be ordered 221
                if ((im[0] == 1) && (im[1] == 2) && (im[2] == 2)) {
                    im[0] = 2; im[2] = 1;
                    outgroup = 2 - outgroup;
                }

                initState = (im[0]-1)*9 + (im[1]-1)*3 + (im[2]-1); // determine initial state

                data.topology[sptree][i] = 2 - outgroup;

                if (sptree == 0) printf("%5d: %18.12f %18.12f\r", i+1, b[0], b[1]);
            }

            if(sptree == 0 && noisy>=3) printf("\n");
            data.initState[sptree][i] = initState;
            data.iStateCnt[sptree][initState]++;
        }
    }

    for(i=0; i<NS; i++)
        free(com.spname[i]);
    fclose(fin);

    if (com.Imapf[0] != '\0') {
        free(Inames[0]);
        free(Inames);
        free(Imap);
    }

    printf("\n\ninit state\t #trees\n");
    for (i=0; i<NALLSTATES; i++) {
        if (data.iStateCnt[0][i] > 0) {
            printf("    %s   \t%6d\n", stateStr[i], data.iStateCnt[0][i]);
        }
    }

    return 0;
}

static int locus_save;
static double f0124_locus[5];

//double e1, e2, e3;
void p0124Fromb0b1 (double p[5], double b[2])
{
    /* This calculates p0,p1,p2,p3,p4 for the 5 site patterns for 3 species,
     given branch lengths b0 and b1.
     */
    double e1, e2, e3, psum = 0;
    int i;

    e1 = exp(-4./3*b[1]);
    e2 = exp(-8./3*(b[0]+b[1]));
    e3 = e1*e2;
    //#ifdef TESTA1DBG
    //    printf("e1: %20.18f e2: %20.18f e3: %20.18f ", e1, e2, e3);
    //#endif
    e1 = e1*e1;
    //#ifdef TESTA1DBG
    //    printf("e4: %20.18f\n", e1);
    //#endif
    p[0]      = (1 + 3*e1 +  6*e2 +  6*e3)/16;
    p[1]      = (3 + 9*e1 -  6*e2 -  6*e3)/16;
    p[2]=p[3] = (3 - 3*e1 +  6*e2 -  6*e3)/16;
    p[4]      = (6 - 6*e1 - 12*e2 + 12*e3)/16;

    for(i=0; i<5; i++) {
        if(p[i] < 0) {
            if (p[i] < -DBL_EPSILON) {
                printf("p[%d] is negative and large: %20.18f eps: %20.18f\n", i, p[i], -DBL_EPSILON);
                raise(SIGINT);
            } else {
                p[i] = DBL_EPSILON;
            }
        }
#ifdef TESTA1DBG
        psum += p[i];
#endif
    }
    //#ifdef TESTA1DBG
    //    printf("psum: %20.18f\n", psum);
    //#endif
}

double lnLb0b1 (double b[], int np)
{
    /* this calculates the lnL for the gene tree for branch lengths b0 and b1.
     */
    int *n = data.Nij[stree.sptree] + locus_save*5;
    double lnL=0, *f=f0124_locus, p[5];

    p0124Fromb0b1 (p, b);
    lnL = f[0]*log(p[0]);
    if(f[1]) lnL += f[1]*log(p[1]);
    if(f[2]) lnL += f[2]*log(p[2]);
    if(n[4]) lnL += f[4]*log(p[4]);
    return(-lnL);
}

double lnLb2seq (double b[], int np) {
    double lnL=0, *f = f0124_locus;
    lnL = f[0]*log(1.0/4.0+3.0/4.0*exp(-8.0*b[0]/3.0));
    if (f[1]) lnL += f[1]*log(3.0/4.0-3.0/4.0*exp(-8.0*b[0]/3.0));
    return -lnL;
}

int Initialize3s(double space[])
{
    int i, *n, n123max;
    double nt, lnL, *f=f0124_locus, b[2], bb[2][2]={{1e-4,1},{1e-4,1}}, e=1e-7;
    double din, dout; //, avgb[2] = {0,0};

    for(locus_save=0; locus_save<data.ndata; locus_save++)  {
        for (stree.sptree = 0; stree.sptree < stree.nsptree; stree.sptree++) {
            n = data.Nij[stree.sptree] + locus_save * 5;
            if (data.initState[stree.sptree][locus_save] < NSTATES) {
                for(i=0,nt=0; i<5; i++)
                    nt += n[i];

                /* this max may be too large and is not used. */
                for(i=0,data.lnLmax[stree.sptree][locus_save]=0; i<5; i++) {
                    if(n[i])
                        data.lnLmax[stree.sptree][locus_save] += n[i] * log(n[i]/(double)nt);
                }

                n123max = max2(n[1], n[2]);
                n123max = max2(n123max, n[3]);
                f[0] = n[0]                     / nt;
                f[1] = n123max                  / nt;
                f[2] = (n[1]+n[2]+n[3]-n123max) / nt;
                f[4] = n[4]                     / nt;
                din = f[2]+f[4]; dout=f[1]+f[4]+f[2]/2;
                b[1] = din/2;
                b[0] = (dout-din)/2;
    #ifdef DEBUG1
                if (locus_save == DBGLOCUS) {
                    printf("f: %8.6f %8.6f %8.6f %8.6f\tdin: %8.6f\tdout: %8.6f\tb: %8.6f %8.6f\n", f[0],f[1],f[2],f[4],din,dout,b[0],b[1]);
                }
    #endif
                ming2(NULL, &lnL, lnLb0b1, NULL, b, bb, space, e, 2);
    #ifdef DEBUG1
                if (locus_save == DBGLOCUS) {
                    printf("lmax[%d]: %8.6f\n", locus_save, -lnL);
                }
    #endif
                data.lnLmax[stree.sptree][locus_save] = -lnL*nt - 300;   /* log(10^300) = 690.8 */
            } else { // 2 sequence case
                // TODO: check this is correct
                nt = n[0] + n[1];
                f[0] = n[0] / nt;
                f[1] = n[1] / nt;
                b[0] = f[1];
                ming2(NULL, &lnL, lnLb2seq, NULL, b, bb, space, e, 1);
                data.lnLmax[stree.sptree][locus_save] = -lnL*nt - 300;
            }
        }
    }
    return(0);
}

int InitializeGtreeTab() {
    int i, j;
    struct BTEntry ** gtt = com.GtreeTab;
    // initialize GtreeTab
    if(gtt[0]) free(gtt[0]);
    gtt[0] = (struct BTEntry *) malloc(NINITIALSTATES*MAXGTREETYPES*sizeof(struct BTEntry)); // 111
    if(gtt[0] == NULL) error2("oom gtt[0]");
    memset(gtt[0], 0, NINITIALSTATES*MAXGTREETYPES*sizeof(struct BTEntry));

    gtt[13] = gtt[0]+MAXGTREETYPES;   // 222
    gtt[1] = gtt[13]+MAXGTREETYPES; // 112
    gtt[12] = gtt[1]+MAXGTREETYPES; // 221
    gtt[2] = gtt[12]+MAXGTREETYPES; // 113
    gtt[14] = gtt[2]+MAXGTREETYPES; // 223
    gtt[5] = gtt[14]+MAXGTREETYPES; // 123
    gtt[8] = gtt[5]+MAXGTREETYPES;  // 133/233
    gtt[17] = gtt[8]+MAXGTREETYPES;  // 233
    gtt[26] = gtt[17]+MAXGTREETYPES; // 333
    
    if (com.model == M2Pro || com.model == M2ProMax) {
        for (j = 0; j < NINITIALSTATES; j++)
            for (i = 0; i < 6; i++)
                gtt[initStates[j]][i].nGtrees = 3;
    }
    else if (com.model == M3MSci13 || com.model == M3MSci23) {
        for (j = 0; j < NINITIALSTATES; j++)
            for (i = 3; i < 6; i++)
                gtt[initStates[j]][i].nGtrees = 3;

        for (i = 0; i < 3; i++) {
            gtt[ 0][i].nGtrees = 3;
            gtt[13][i].nGtrees = 3;
            gtt[26][i].nGtrees = 3;
        }

        for (i = 1; i < 3; i++) {
            gtt[ 1][i].nGtrees = 1;
            gtt[12][i].nGtrees = 1;
            gtt[ 2][i].nGtrees = 1;
            gtt[14][i].nGtrees = 1;
            gtt[ 8][i].nGtrees = 1;
            gtt[17][i].nGtrees = 1;

            gtt[ 1][i].config = 0;
            gtt[12][i].config = 0;
            gtt[ 2][i].config = 0;
            gtt[14][i].config = 0;
            gtt[ 8][i].config = 2;
            gtt[17][i].config = 2;
        }

        if (com.model == M3MSci13) {
            for (i = 6; i < 10; i++) {
                gtt[ 0][i].nGtrees = 3;
                gtt[ 2][i].nGtrees = 3;
                gtt[ 8][i].nGtrees = 3;
                gtt[26][i].nGtrees = 3;
            }

            gtt[ 2][6].nGtrees = 1;
            gtt[ 8][6].nGtrees = 1;

            gtt[ 2][6].config = 0;
            gtt[ 8][6].config = 2;

            for (i = 8; i < 10; i++) {
                gtt[ 1][i].nGtrees = 1;
                gtt[ 5][i].nGtrees = 1;
                gtt[17][i].nGtrees = 1;

                gtt[ 1][i].config = 0;
                gtt[ 5][i].config = 1;
                gtt[17][i].config = 2;
            }
        }
        else {
            for (i = 6; i < 10; i++) {
                gtt[13][i].nGtrees = 3;
                gtt[14][i].nGtrees = 3;
                gtt[17][i].nGtrees = 3;
                gtt[26][i].nGtrees = 3;
            }

            gtt[14][6].nGtrees = 1;
            gtt[17][6].nGtrees = 1;

            gtt[14][6].config = 0;
            gtt[17][6].config = 2;

            for (i = 8; i < 10; i++) {
                gtt[12][i].nGtrees = 1;
                gtt[ 5][i].nGtrees = 1;
                gtt[ 8][i].nGtrees = 1;

                gtt[12][i].config = 0;
                gtt[ 5][i].config = 2;
                gtt[ 8][i].config = 2;
            }
        }
    }
    else {
        // initial states 111/222/112/221
        for (i = 0; i < 6; i++) {
            gtt[ 0][i].nGtrees = 3;
            gtt[13][i].nGtrees = 3;
            gtt[ 1][i].nGtrees = 3;
            gtt[12][i].nGtrees = 3;
        }

        // initial states 112/122
        if (com.model == M0 || com.model == M1DiscreteBeta || com.model == M3MSci12) {
            gtt[ 1][0].nGtrees = 0;
            gtt[ 1][1].nGtrees = 1;
            gtt[ 1][2].nGtrees = 1;
            gtt[12][0].nGtrees = 0;
            gtt[12][1].nGtrees = 1;
            gtt[12][2].nGtrees = 1;

            gtt[ 1][1].config = 0;
            gtt[ 1][2].config = 0;
            gtt[12][1].config = 0;
            gtt[12][2].config = 0;
        }

        // initial states 111/222/112/221
        if (com.model == M3MSci12) {
            for (i = 6; i < 10; i++) {
                gtt[ 0][i].nGtrees = 3;
                gtt[13][i].nGtrees = 3;
                gtt[ 1][i].nGtrees = 3;
                gtt[12][i].nGtrees = 3;
            }

            gtt[ 1][6].nGtrees = 1;
            gtt[12][6].nGtrees = 1;

            gtt[ 1][6].config = 0;
            gtt[12][6].config = 0;
        }

        // initial states 113/223/123
        gtt[ 2][2].nGtrees = 1;
        gtt[ 2][4].nGtrees = 1;
        gtt[ 2][5].nGtrees = 3;
        gtt[14][2].nGtrees = 1;
        gtt[14][4].nGtrees = 1;
        gtt[14][5].nGtrees = 3;
        gtt[ 5][4].nGtrees = 1;
        gtt[ 5][5].nGtrees = 3;

        gtt[ 2][2].config = 0;
        gtt[ 2][4].config = 0;
        gtt[14][2].config = 0;
        gtt[14][4].config = 0;
        gtt[ 5][4].config = 0;

        if (com.model == M2SIM3s) {
            gtt[5][2].nGtrees = 1;
            gtt[5][2].config = 0;
        }
        else if (com.model == M3MSci12) {
            gtt[ 2][9].nGtrees = 1;
            gtt[14][9].nGtrees = 1;
            gtt[ 5][9].nGtrees = 1;

            gtt[ 2][9].config = 0;
            gtt[14][9].config = 0;
            gtt[ 5][9].config = 0;
        }

        // initial states 133/233/333
        gtt[8][2].nGtrees = 1;
        //if(com.model == M0)
        gtt[8][2].config = 2; // topology ((b,c), a)?
        gtt[8][5].nGtrees = gtt[26][0].nGtrees = gtt[26][2].nGtrees = gtt[26][5].nGtrees = 3;
    }

    setupGtreeTab();

    return 0;
}

int GetUVrootIM3s(double U[5*5], double V[5*5], double Root[5])
{
    /* This generates U, V, Root for Q for epoch E1.
     */
    double theta12=para[com.paraMap[4]], M12=para[com.paraMap[7]], c=2/theta12, m=4*M12/theta12;
    double a=sqrt(c*c + 16*m*m);
    //char *states[] = {"11", "12", "22", "1", "2"};

    U[0*5+0] = 1;  U[0*5+1] = -1;  U[0*5+2] = -1;  U[0*5+3] = 1;  U[0*5+4] = 1;
    U[1*5+0] = 1;  U[1*5+1] =  0;  U[1*5+2] =  0;  U[1*5+3] = (c-a)/(4*m);  U[1*5+4] = (c+a)/(4*m);
    U[2*5+0] = 1;  U[2*5+1] =  1;  U[2*5+2] =  1;  U[2*5+3] = 1;  U[2*5+4] = 1;
    U[3*5+0] = 1;  U[3*5+1] =  0;  U[3*5+2] = -1;  U[3*5+3] = 0;  U[3*5+4] = 0;
    U[4*5+0] = 1;  U[4*5+1] =  0;  U[4*5+2] =  1;  U[4*5+3] = 0;  U[4*5+4] = 0;

    V[0*5+0] = 0;          V[0*5+1] =  0;      V[0*5+2] = 0;          V[0*5+3] = 0.5;               V[0*5+4] =  0.5;
    V[1*5+0] = -0.5;       V[1*5+1] =  0;      V[1*5+2] = 0.5;        V[1*5+3] = 0.5;               V[1*5+4] = -0.5;
    V[2*5+0] = 0;          V[2*5+1] =  0;      V[2*5+2] = 0;          V[2*5+3] = -0.5;              V[2*5+4] =  0.5;
    V[3*5+0] = (1+c/a)/4;  V[3*5+1] = -2*m/a;  V[3*5+2] = (1+c/a)/4;  V[3*5+3] = -(c+a-4*m)/(4*a);  V[3*5+4] = -(c+a-4*m)/(4*a);
    V[4*5+0] = (1-c/a)/4;  V[4*5+1] =  2*m/a;  V[4*5+2] = (1-c/a)/4;  V[4*5+3] =  (c-a-4*m)/(4*a);  V[4*5+4] =  (c-a-4*m)/(4*a);

    Root[0] = 0;
    Root[1] = -(2*m+c);
    Root[2] = -2*m;
    Root[3] = -(4*m + c + a)/2;
    Root[4] = -(4*m + c - a)/2;
    return(0);

#if(0)
    zero(Q, s*s);
    Q[0*s+1] = 2*m1;     Q[0*s+3] = c1;
    Q[1*s+0] = m2;       Q[1*s+2] = m1;
    Q[2*s+1] = 2*m2;     Q[2*s+4] = c2;
    Q[3*s+4] = m1;
    Q[4*s+3] = m2;
    for(i=0; i<s; i++)
        Q[i*s+i] = -sum(Q+i*s, s);
    matout(F0, Q, s, s);

    if(debug==9) {
        printf("\nQ & P(%.5f)\n", t);
        matout2(F0, Q, s, s, 10, 6);
    }
    matexp(Q, t, s, (t>0.1 ? 31 : 7), space);
    matout(F0, Q, s, s);
#endif

    return(0);
}

int GetPMatIM3s(double Pt[5*5], double t, double U[5*5], double V[5*5], double Root[5])
{
    /* P(t) = U * exp(Root*t) U.
     */
    int i, j, k, n=5;
    double expt, uexpt, *pP;

    for (k=0,zero(Pt,n*n); k<n; k++)
        for (i=0,pP=Pt,expt=exp(t*Root[k]); i<n; i++)
            for (j=0,uexpt=U[i*n+k]*expt; j<n; j++)
                *pP++ += uexpt*V[k*n+j];

    return(0);
}


#if(0)
int GenerateQIM3s(double Q[])
{
    /* This is not used right now since we only need the Q for a 5-state chain.
     This generates Q1 and Q2 for epochs E1 and E2.
     */
    double tau0=0.05, tau1=0.025, q1=0.01, q2=0.02, q3=0.04, q4=0.04, q5=0.05, M1=0.01, M2=0.05;
    double m1=M1, m2=M2, c1=2/q1, c2=2/q2, c3=2/q3, c5=2/q5;
    int s1=17, s2=9, i, j;
    char *statesE1[] = {"111", "112", "113", "122", "123", "133", "222", "223",
        "233", "333", "11", "12", "13", "22", "23", "33", "1", "2", "3"};
    char *statesE2[] = {"333", "335", "355", "555", "33", "35", "55", "3", "5"};

    zero(Q1, s1);
    zero(Q2, s2);

    Q1[ 0*s1+ 1] = 3*m1;     Q1[ 0*s1+10] = 3*c1;
    Q1[ 1*s1+ 0] = m2;       Q1[ 1*s1+ 3] = 2*m1;    Q1[ 1*s1+11] = c1;
    Q1[ 2*s1+ 4] = 2*m1;     Q1[ 2*s1+12] = c1;
    Q1[ 3*s1+ 1] = 2*m1;     Q1[ 3*s1+ 6] = m1;      Q1[ 3*s1+11] = c2;
    Q1[ 4*s1+ 2] = m2;       Q1[ 4*s1+ 7] = m1;
    Q1[ 5*s1+ 8] = m1;       Q1[ 5*s1+12] = c3;
    Q1[ 6*s1+ 3] = 3*m2;     Q1[ 6*s1+13] = 3*c2;
    Q1[ 7*s1+ 4] = 2*m2;     Q1[ 7*s1+14] = c2;
    Q1[ 8*s1+ 5] = 2*m2;     Q1[ 8*s1+14] = c3;
    Q1[ 9*s1+15] = 3*c3;
    Q1[10*s1+11] = 2*m1;     Q1[10*s1+16] = c1;
    Q1[11*s1+10] = m2;       Q1[11*s1+13] = m1;
    Q1[12*s1+14] = m1;
    Q1[13*s1+11] = 2*m2;     Q1[13*s1+17] = c2;
    Q1[14*s1+12] = m2;
    Q1[15*s1+18] = c3;
    Q1[16*s1+17] = m1;
    Q1[17*s1+16] = m2;

    for(i=0; i<s1; i++)
        Q1[i*s1+i] = -sum(Q1+i*s1, s1);
    matout2(F0, Q1, s1, s1, 8, 3);

    Q2[0*s2+4] = 3*c3;
    Q2[1*s2+5] = c3;
    Q2[2*s2+5] = c5;
    Q2[3*s2+6] = 3*c5;
    Q2[4*s2+7] = c3;
    Q2[6*s2+8] = c5;

    for(i=0; i<s2; i++)
        Q2[i*s2+i] = -sum(Q2+i*s2, s2);
    matout2(F0, Q2, s2, s2, 8, 3);

    return(0);
}
#endif

/* returns Q matrix for chain 1 or 2 in Q, depending on the number of
 * states (nStates): chain 1: 21 x 21 matrix
 *                   chain 2: 8 x 8 matrix
 *                   chain 3: 10 x 10 matrix */
int GenerateQ1SIM3S(double Q[], int nStates, double theta1, double theta2, double theta3, double M12, double M21) {
    double c1 = 2/theta1, c2 = 2/theta2, w12 = 4*M12/theta2, w21 = 4*M21/theta1;
    int i, j;

    memset(Q, 0, nStates*nStates*sizeof(double));

    if(nStates == C2) {
        //chain 2
        // order of states: 111, 112, 121, 211, 122, 212, 221, 222, 11a, 11b, 11c
        //                   0    1    2    3    4    5    6    7    8    9    10
        //                  12a, 12b, 12c, 1a2, 1b2, 1c2, 22a, 22b, 22c, 1&2
        //                   11   12   13   14   15   16   17   18   19   20

        Q[1] = Q[2] = Q[3] = w21; Q[8] = Q[9] = Q[10] = c1;
        Q[nStates] = w12; Q[nStates+4] = Q[nStates+5] = w21; Q[nStates+13] = c1;
        Q[2*nStates] = w12; Q[2*nStates+4] = Q[2*nStates+6] = w21; Q[2*nStates+12] = c1;
        Q[4*nStates+1] = Q[4*nStates+2] = w12; Q[4*nStates+7] = w21; Q[4*nStates+14] = c2;
        Q[3*nStates] = w12; Q[3*nStates+5] = Q[3*nStates+6] = w21; Q[3*nStates+11] = c1;
        Q[5*nStates+1] = Q[5*nStates+3] = w12; Q[5*nStates+7] = w21; Q[5*nStates+15] = c2;
        Q[6*nStates+2] = Q[6*nStates+3] = w12; Q[6*nStates+7] = w21; Q[6*nStates+16] = c2;
        Q[7*nStates+4] = Q[7*nStates+5] = Q[7*nStates+6] = w12; Q[7*nStates+17] = Q[7*nStates+18] = Q[7*nStates+19] = c2;
        Q[8*nStates+11] = Q[8*nStates+14] = w21; Q[8*nStates+20] = c1;
        Q[9*nStates+12] = Q[9*nStates+15] = w21; Q[9*nStates+20] = c1;
        Q[10*nStates+13] = Q[10*nStates+16] = w21; Q[10*nStates+20] = c1;
        Q[11*nStates+8] = w12; Q[11*nStates+17] = w21;
        Q[12*nStates+9] = w12; Q[12*nStates+18] = w21;
        Q[13*nStates+10] = w12; Q[13*nStates+19] = w21;
        Q[14*nStates+8] = w12; Q[14*nStates+17] = w21;
        Q[15*nStates+9] = w12; Q[15*nStates+18] = w21;
        Q[16*nStates+10] = w12; Q[16*nStates+19] = w21;
        Q[17*nStates+11] = Q[17*nStates+14] = w12; Q[17*nStates+20] = c2;
        Q[18*nStates+12] = Q[18*nStates+15] = w12; Q[18*nStates+20] = c2;
        Q[19*nStates+13] = Q[19*nStates+16] = w12; Q[19*nStates+20] = c2;
    } else if(nStates == C1) {
        // chain 1
        // order of states: 111 112 122 222 11 12 22 1&2
        //                   0   1   2   3   4  5  6  7
        
        Q[1] = 3 * w21;
        Q[4] = 3 * c1;
        Q[nStates] = w12;
        Q[nStates+2] = 2 * w21;
        Q[nStates+5] = c1;
        Q[2*nStates+1] = 2 * w12;
        Q[2*nStates+3] = w21;
        Q[2*nStates+5] = c2;
        Q[3*nStates+2] = 3 * w12;
        Q[3*nStates+6] = 3 * c2;
        Q[4*nStates+5] = 2 * w21;
        Q[4*nStates+7] = c1;
        Q[5*nStates+4] = w12;
        Q[5*nStates+6] = w21;
        Q[6*nStates+5] = 2 * w12;
        Q[6*nStates+7] = c2;
        
//    } else if(nStates == 10) {
//        // chain 3
//        // order of states: 113, 123, 223, 133, 233, 333, 13, 23, 33, 3
//        //                   0    1    2    3    4    5    6   7   8  9
//
//        Q[1] = Q[12] = w21;
//        Q[10] = Q[21] = w12;
//        Q[6] = c1; Q[27] = c2;
//        Q[36] = Q[47] = Q[89] = 2/theta3;
//        Q[58] = 6/theta3;
    } else if(nStates == C3) {
        Q[1] = 2 * w21;
        Q[2*nStates+1] = 2 * w12;
        Q[3] = c1;
        Q[2*nStates+3] = c2;
        Q[nStates] = w12;
        Q[nStates+2] = w21;
    }

    for(i = 0 ; i < nStates; i++) {
        for(j = 0; j < nStates; j++) {
            if(i == j) continue;
            Q[i*nStates+i] -= Q[i*nStates+j];
        }
    }

    return 0;
}

typedef struct {
    int x;
    int y;
} MatrixIdx;

int GenerateQ5(double Q[], double theta1, double theta2, double theta3, double M12, double M21, double M13, double M31, double M23, double M32)
{
    static const MatrixIdx QIdxM12[] = { { 1, 0 },{ 3, 0 },{ 4, 1 },{ 4, 3 },{ 5, 2 },{ 7, 6 },{ 9, 0 },{ 10, 1 },{ 10, 9 },{ 11, 2 },{ 12, 3 },{ 12, 9 },{ 13, 4 },{ 13, 10 },{ 13, 12 },{ 14, 5 },{ 14, 11 },{ 15, 6 },{ 16, 7 },{ 16, 15 },{ 17, 8 },{ 19, 18 },{ 21, 18 },{ 22, 19 },{ 22, 21 },{ 23, 20 },{ 25, 24 } };
    static const MatrixIdx QIdxM13[] = { { 2, 0 },{ 5, 3 },{ 6, 0 },{ 7, 1 },{ 8, 2 },{ 8, 6 },{ 11, 9 },{ 14, 12 },{ 15, 9 },{ 16, 10 },{ 17, 11 },{ 17, 15 },{ 18, 0 },{ 19, 1 },{ 20, 2 },{ 20, 18 },{ 21, 3 },{ 22, 4 },{ 23, 5 },{ 23, 21 },{ 24, 6 },{ 24, 18 },{ 25, 7 },{ 25, 19 },{ 26, 8 },{ 26, 20 },{ 26, 24 } };
    static const MatrixIdx QIdxM23[] = { { 2, 1 },{ 5, 4 },{ 6, 3 },{ 7, 4 },{ 8, 5 },{ 8, 7 },{ 11, 10 },{ 14, 13 },{ 15, 12 },{ 16, 13 },{ 17, 14 },{ 17, 16 },{ 18, 9 },{ 19, 10 },{ 20, 11 },{ 20, 19 },{ 21, 12 },{ 22, 13 },{ 23, 14 },{ 23, 22 },{ 24, 15 },{ 24, 21 },{ 25, 16 },{ 25, 22 },{ 26, 17 },{ 26, 23 },{ 26, 25 } };
    STATIC_ASSERT(LENGTH_OF(QIdxM12) == LENGTH_OF(QIdxM13) && LENGTH_OF(QIdxM13) == LENGTH_OF(QIdxM23));

    double c1 = 2 / theta1;
    double c2 = 2 / theta2;
    double c3 = 2 / theta3;
    double w12 = 4 * M12 / theta2;
    double w21 = 4 * M21 / theta1;
    double w13 = 4 * M13 / theta3;
    double w31 = 4 * M31 / theta1;
    double w23 = 4 * M23 / theta3;
    double w32 = 4 * M32 / theta2;
    int i;

    memset(Q, 0, C5 * C5 * sizeof(double));

    for (i = 0; i < LENGTH_OF(QIdxM12); i++) {
        Q[C5 * QIdxM12[i].x + QIdxM12[i].y] = w12;
        Q[C5 * QIdxM12[i].x + QIdxM12[i].x] -= w12;
        Q[C5 * QIdxM12[i].y + QIdxM12[i].x] = w21;
        Q[C5 * QIdxM12[i].y + QIdxM12[i].y] -= w21;
        Q[C5 * QIdxM13[i].x + QIdxM13[i].y] = w13;
        Q[C5 * QIdxM13[i].x + QIdxM13[i].x] -= w13;
        Q[C5 * QIdxM13[i].y + QIdxM13[i].x] = w31;
        Q[C5 * QIdxM13[i].y + QIdxM13[i].y] -= w31;
        Q[C5 * QIdxM23[i].x + QIdxM23[i].y] = w23;
        Q[C5 * QIdxM23[i].x + QIdxM23[i].x] -= w23;
        Q[C5 * QIdxM23[i].y + QIdxM23[i].x] = w32;
        Q[C5 * QIdxM23[i].y + QIdxM23[i].y] -= w32;
    }

    Q[C5 *  0 + C5 - 1] = 3 * c1;
    Q[C5 * 13 + C5 - 1] = 3 * c2;
    Q[C5 * 26 + C5 - 1] = 3 * c3;

    Q[C5 * 1 + C5 - 1] = Q[C5 *  2 + C5 - 1] = Q[C5 *  3 + C5 - 1] = Q[C5 *  6 + C5 - 1] = Q[C5 *  9 + C5 - 1] = Q[C5 * 18 + C5 - 1] = c1;
    Q[C5 * 4 + C5 - 1] = Q[C5 * 10 + C5 - 1] = Q[C5 * 12 + C5 - 1] = Q[C5 * 14 + C5 - 1] = Q[C5 * 16 + C5 - 1] = Q[C5 * 22 + C5 - 1] = c2;
    Q[C5 * 8 + C5 - 1] = Q[C5 * 17 + C5 - 1] = Q[C5 * 20 + C5 - 1] = Q[C5 * 23 + C5 - 1] = Q[C5 * 24 + C5 - 1] = Q[C5 * 25 + C5 - 1] = c3;

    for (i = 0; i < C5; i++)
        Q[C5 * i + i] -= Q[C5 * i + C5 - 1];

    return 0;
}

int GenerateQ6(double Q[], double theta1, double theta2, double theta3, double M12, double M21, double M13, double M31, double M23, double M32)
{
    static const MatrixIdx QIdxM12[] = { { 1, 0 },{ 2, 1 },{ 4, 3 } };
    static const MatrixIdx QIdxM13[] = { { 3, 0 },{ 4, 1 },{ 5, 3 } };
    static const MatrixIdx QIdxM23[] = { { 3, 1 },{ 4, 2 },{ 5, 4 } };
    STATIC_ASSERT(LENGTH_OF(QIdxM12) == LENGTH_OF(QIdxM13) && LENGTH_OF(QIdxM13) == LENGTH_OF(QIdxM23));

    double c1 = 2 / theta1;
    double c2 = 2 / theta2;
    double c3 = 2 / theta3;
    double w12 = 4 * M12 / theta2;
    double w21 = 4 * M21 / theta1;
    double w13 = 4 * M13 / theta3;
    double w31 = 4 * M31 / theta1;
    double w23 = 4 * M23 / theta3;
    double w32 = 4 * M32 / theta2;
    int i;

    memset(Q, 0, C6 * C6 * sizeof(double));

    for (i = 0; i < LENGTH_OF(QIdxM12); i++) {
        Q[C6 * QIdxM12[i].x + QIdxM12[i].y] = w12;
        Q[C6 * QIdxM12[i].x + QIdxM12[i].x] -= w12;
        Q[C6 * QIdxM12[i].y + QIdxM12[i].x] = w21;
        Q[C6 * QIdxM12[i].y + QIdxM12[i].y] -= w21;
        Q[C6 * QIdxM13[i].x + QIdxM13[i].y] = w13;
        Q[C6 * QIdxM13[i].x + QIdxM13[i].x] -= w13;
        Q[C6 * QIdxM13[i].y + QIdxM13[i].x] = w31;
        Q[C6 * QIdxM13[i].y + QIdxM13[i].y] -= w31;
        Q[C6 * QIdxM23[i].x + QIdxM23[i].y] = w23;
        Q[C6 * QIdxM23[i].x + QIdxM23[i].x] -= w23;
        Q[C6 * QIdxM23[i].y + QIdxM23[i].x] = w32;
        Q[C6 * QIdxM23[i].y + QIdxM23[i].y] -= w32;
    }

    Q[C6 * 0 + 0] *= 2;
    Q[C6 * 0 + 1] *= 2;
    Q[C6 * 0 + 3] *= 2;
    Q[C6 * 2 + 1] *= 2;
    Q[C6 * 2 + 2] *= 2;
    Q[C6 * 2 + 4] *= 2;
    Q[C6 * 5 + 3] *= 2;
    Q[C6 * 5 + 4] *= 2;
    Q[C6 * 5 + 5] *= 2;

    Q[C6 * 0 + C6 - 1] = c1;
    Q[C6 * 2 + C6 - 1] = c2;
    Q[C6 * 5 + C6 - 1] = c3;

    Q[C6 * 0 + 0] -= c1;
    Q[C6 * 2 + 2] -= c2;
    Q[C6 * 5 + 5] -= c3;

    return 0;
}

int GenerateQ7(double Q[], double theta5, double theta3, double M53, double M35)
{
    static const MatrixIdx QIdxM53[] = { { 1, 0 },{ 2, 0 },{ 3, 0 },{ 4, 2 },{ 4, 3 },{ 5, 1 },{ 5, 3 },{ 6, 1 },{ 6, 2 },{ 7, 4 },{ 7, 5 },{ 7, 6 } };

    double c5 = 2 / theta5;
    double c3 = 2 / theta3;
    double w53 = 4 * M53 / theta3;
    double w35 = 4 * M35 / theta5;
    int i;

    memset(Q, 0, C7 * C7 * sizeof(double));

    for (i = 0; i < LENGTH_OF(QIdxM53); i++) {
        Q[C7 * QIdxM53[i].x + QIdxM53[i].y] = w53;
        Q[C7 * QIdxM53[i].x + QIdxM53[i].x] -= w53;
        Q[C7 * QIdxM53[i].y + QIdxM53[i].x] = w35;
        Q[C7 * QIdxM53[i].y + QIdxM53[i].y] -= w35;
    }

    Q[C7 * 0 + C7 - 1] = 3 * c5;
    Q[C7 * 7 + C7 - 1] = 3 * c3;

    Q[C7 * 1 + C7 - 1] = Q[C7 * 2 + C7 - 1] = Q[C7 * 3 + C7 - 1] = c5;
    Q[C7 * 4 + C7 - 1] = Q[C7 * 5 + C7 - 1] = Q[C7 * 6 + C7 - 1] = c3;

    for (i = 0; i < C7; i++)
        Q[C7 * i + i] -= Q[C7 * i + C7 - 1];

    return 0;
}

int GenerateQ8(double Q[], double theta5, double theta3, double M53, double M35)
{
    static const MatrixIdx QIdxM53[] = { { 1, 0 },{ 2, 1 } };

    double c5 = 2 / theta5;
    double c3 = 2 / theta3;
    double w53 = 4 * M53 / theta3;
    double w35 = 4 * M35 / theta5;
    int i;

    memset(Q, 0, C8 * C8 * sizeof(double));

    for (i = 0; i < LENGTH_OF(QIdxM53); i++) {
        Q[C8 * QIdxM53[i].x + QIdxM53[i].y] = w53;
        Q[C8 * QIdxM53[i].x + QIdxM53[i].x] -= w53;
        Q[C8 * QIdxM53[i].y + QIdxM53[i].x] = w35;
        Q[C8 * QIdxM53[i].y + QIdxM53[i].y] -= w35;
    }

    Q[C8 * 0 + 0] *= 2;
    Q[C8 * 0 + 1] *= 2;
    Q[C8 * 2 + 1] *= 2;
    Q[C8 * 2 + 2] *= 2;

    Q[C8 * 0 + C8 - 1] = c5;
    Q[C8 * 2 + C8 - 1] = c3;

    Q[C8 * 0 + 0] -= c5;
    Q[C8 * 2 + 2] -= c3;

    return 0;
}



//int Simulation (FILE *fout, FILE *frub, double space[])
//{
//    char timestr[96];
//    double theta4, theta5, tau0, tau1, mbeta, pbeta, qbeta, tau1beta[5];
//    double t[2], b[2], p[5], y, pG0, *dlnL;
//    double md12, md13, md23, d12, d13, d23, Ed12, Ed13, EtMRCA, mNij[5], x[5];
//#if(0)
//    double x0[5] = {0.04, 0.06, 0.06, 0.04, 2.0};
//#elif(0)
//    double x0[5] = {0.0035, 0.0060, 0.0066, 0.0041, 0};
//#elif(1)
//    double x0[5] = {0.005, 0.005, 0.006, 0.004, 0}; /* hominoid (BY08) */
//#elif(0)
//    double x0[5] = {0.01, 0.01, 0.02, 0.01, 0};    /* mangroves (Zhou et al. 2007) */
//#endif
//    int model=1, nii=1, nloci[]={1000, 100, 10}, ii, i,j, nr=200, ir, locus;
//
//    /*
//     printf("input model theta4 theta5 tau0 tau1 q? ");
//     scanf("%d%lf%lf%lf%lf%lf", &model, &x0[0], &x0[1], &x0[2], &x0[3], &x0[4]);
//     */
//
//    com.ls = 500;
//    noisy = 2;
//    if(com.fix_locusrate) error2("fix_locusrate in Simulation()?");
//    if((dlnL=(double*)malloc(nr*sizeof(double))) == NULL) error2("oom dlnL");
//    memset(dlnL, 0, nr*sizeof(double));
//    for(ii=0; ii<nii; ii++) {
//        data.ndata = nloci[ii];
//        printf("\n\nnloci = %d  length = %d\nParameters", data.ndata, com.ls);
//        matout(F0, x0, 1, 4+model);
//        fprintf(fout, "\n\nnloci = %d  length = %d\nParameters", data.ndata, com.ls);
//        matout(fout, x0, 1, 4+model);
//        fprintf(frst, "\n\nnloci = %d  length = %d  K = %d\nParameters", data.ndata, com.ls, com.npoints);
//        matout(frst, x0, 1, 4+model);
//        if((data.lnLmax=(double*)malloc(data.ndata*1*sizeof(double)))==NULL) error2("oom lnLmax");
//        if((data.Nij=(int*)malloc(data.ndata*5*sizeof(int)))==NULL) error2("oom");
//        theta4=x0[0]; theta5=x0[1]; tau0=x0[2]; tau1=x0[3]; qbeta=x0[4];
//        pG0 = 1 - exp(-2*(tau0-tau1)/theta5);
//        Ed12 = tau1 + theta5/2 + (1-pG0)*(theta4-theta5)/2;
//        Ed13 = tau0 + theta4/2;
//        EtMRCA = tau0 + theta4/2 * (1 + (1-pG0)*1./3);
//        printf("pG0 = %9.6f  E{tMRCA} = %9.6f\n", pG0, EtMRCA);
//
//        md12=0; md13=0; md23=0;
//        for(i=0; i<5; i++) mNij[i]=0;
//        for(ir=0; ir<nr; ir++) {
//            memset(data.Nij, 0, data.ndata*5*sizeof(int));
//            for(locus=0; locus<data.ndata; locus++) {
//                if(model==M1DiscreteBeta) {
//                    mbeta = x0[3]/x0[2];  pbeta = mbeta/(1-mbeta)*qbeta;
//                    if(0)    /* continuous beta */
//                        tau1 = tau0*rndbeta(pbeta, qbeta);
//                    else {   /* discrete beta */
//                        DiscreteBeta(p, tau1beta, pbeta, qbeta, com.ncatBeta, 0);
//                        tau1 = tau0*tau1beta[(int)(com.ncatBeta*rndu())];
//                    }
//                }
//                t[0] = rndexp(1.0);
//                t[1] = rndexp(1.0);
//                if(t[1]<2*(tau0-tau1)/theta5) {   /* G0 */
//                    b[0] = tau0 - tau1 - theta5*t[1]/2 + theta4*t[0]/2;
//                    b[1] = tau1 + theta5*t[1]/2;
//                    p0124Fromb0b1 (p, b);
//                }
//                else {                            /* G123 */
//                    t[1] = rndexp(1.0/3);
//                    b[0] = t[0]*theta4/2;
//                    b[1] = tau0 + t[1]*theta4/2;
//                    p0124Fromb0b1 (p, b);
//                    y = rndu();
//                    if(y<1.0/3)         { y=p[1]; p[1]=p[2]; p[2]=y; }
//                    else if (y<2.0/3)   { y=p[1]; p[1]=p[3]; p[3]=y; }
//                }
//
//                for(j=0; j<4; j++)  p[j+1] += p[j];
//                for(i=0; i<com.ls; i++) {
//                    for (j=0,y=rndu(); j<5-1; j++)
//                        if (y<p[j]) break;
//                    data.Nij[locus*5+j] ++;
//                }
//
//                d12 = (data.Nij[locus*5+2]+data.Nij[locus*5+3]+data.Nij[locus*5+4])/(double)com.ls;
//                d13 = (data.Nij[locus*5+1]+data.Nij[locus*5+2]+data.Nij[locus*5+4])/(double)com.ls;
//                d23 = (data.Nij[locus*5+1]+data.Nij[locus*5+3]+data.Nij[locus*5+4])/(double)com.ls;
//                d12 = -3/4.*log(1 - 4./3*d12);
//                d13 = -3/4.*log(1 - 4./3*d13);
//                d23 = -3/4.*log(1 - 4./3*d23);
//                md12 += d12/(2.0*nr*data.ndata);
//                md13 += d13/(2.0*nr*data.ndata);
//                md23 += d23/(2.0*nr*data.ndata);
//                for(i=0; i<5; i++)
//                    mNij[i] += (double)data.Nij[locus*5+i]/(data.ndata*com.ls);
//            }
//
//            printf("\nReplicate %3d\n", ir+1);
//            fprintf(fout, "\nReplicate %3d\n", ir+1);
//            fprintf(frub, "\nReplicate %3d\n", ir+1);
//
//            printf("\nmean Nij: %8.4f %8.4f %8.4f %8.4f %8.4f\n", mNij[0],mNij[1],mNij[2],mNij[3],mNij[4]);
//            for(i=0; i<4; i++)
//                x[i] = x0[i]*MULTIPLIER*(0.8+0.4*rndu());
//            x[4] = x0[4]*(0.8+0.4*rndu());
//            x[3] = x0[3]/x0[2];
//            dlnL[ir] = Models0123(fout, frub, frst, x, space);
//
//            printf("%3d/%3d %9.3f %ss\n", ir+1, nr, dlnL[ir], printtime(timestr));
//            for(i=0; i<5; i++) mNij[i] = 0;
//        }
//
//        printf("\nd12 %9.5f = %9.5f d13 d23: %9.5f = %9.5f = %9.5f\n", Ed12, md12, Ed13, md13, md23);
//
//        fprintf(fout, "\n\nList of DlnL\n");
//        for(i=0; i<nr; i++) fprintf(fout, "%9.5f\n", dlnL[i]);
//        free(data.Nij);  free(data.lnLmax);
//    }
//    printf("\nTime used: %s\n", printtime(timestr));
//    return 0;
//}



int GetParaMap() {
    int np = 0;
    int* iStateCnt = data.iStateCnt[stree.sptree];

    int s11 = iStateCnt[ 0] + iStateCnt[ 1] + iStateCnt[ 2] + iStateCnt[27]; // 111, 112, 113, 11
    int s22 = iStateCnt[12] + iStateCnt[13] + iStateCnt[14] + iStateCnt[31]; // 221, 222, 223, 22
    int s33 = iStateCnt[ 8] + iStateCnt[17] + iStateCnt[26] + iStateCnt[35]; // 133, 233, 333, 33

    int s12 = iStateCnt[ 1] + iStateCnt[12] + iStateCnt[5] + iStateCnt[28]; // 112, 221, 123, 12
    int s13 = iStateCnt[ 2] + iStateCnt[ 8] + iStateCnt[5] + iStateCnt[29]; // 113, 133, 123, 13
    int s23 = iStateCnt[14] + iStateCnt[17] + iStateCnt[5] + iStateCnt[32]; // 223, 233, 123, 23

    int s1 = iStateCnt[12] + iStateCnt[ 8] + iStateCnt[5] + iStateCnt[28] + iStateCnt[29]; // 221, 133, 123, 12, 13
    int s2 = iStateCnt[ 1] + iStateCnt[17] + iStateCnt[5] + iStateCnt[28] + iStateCnt[32]; // 112, 233, 123, 12, 23
    int s3 = iStateCnt[ 2] + iStateCnt[14] + iStateCnt[5] + iStateCnt[29] + iStateCnt[32]; // 113, 223, 123, 13, 23

    int s11_1 = iStateCnt[ 0] + iStateCnt[27]; // 111, 11
    int s22_2 = iStateCnt[13] + iStateCnt[31]; // 222, 22
    int s33_3 = iStateCnt[26] + iStateCnt[35]; // 333, 33

    int s12_1 = iStateCnt[ 1] + iStateCnt[28]; // 112, 12
    int s12_2 = iStateCnt[12] + iStateCnt[28]; // 221, 12
    int s12_3 = iStateCnt[ 5] + iStateCnt[28]; // 123, 12

    int s13_1 = iStateCnt[ 2] + iStateCnt[29]; // 113, 13
    int s13_2 = iStateCnt[ 5] + iStateCnt[29]; // 123, 13
    int s13_3 = iStateCnt[ 8] + iStateCnt[29]; // 133, 13

    int s23_1 = iStateCnt[ 5] + iStateCnt[32]; // 123, 23
    int s23_2 = iStateCnt[14] + iStateCnt[32]; // 223, 23
    int s23_3 = iStateCnt[17] + iStateCnt[32]; // 233, 23

    int s3k = iStateCnt[29] + iStateCnt[32]; // 13, 23
    int s2k = iStateCnt[28] + iStateCnt[32]; // 12, 23
    int s1k = iStateCnt[28] + iStateCnt[29]; // 12, 13

    int ss12 = iStateCnt[28]; // 12
    int ss13 = iStateCnt[29]; // 13
    int ss23 = iStateCnt[32]; // 23

    enum { M12 = 0, M21, M13, M31, M23, M32, M53, M35, Phi12, Phi21, Phi13, Phi31, Phi23, Phi32, ThetaX = 0, ThetaY, ThetaZ };

    memset(com.paraMap, -1, MAXPARAMETERS*sizeof(int));
    memset(com.paraNamesMap, -1, MAXPARAMETERS*sizeof(int));
    
    // theta4 is always estimable
    com.paraNamesMap[np] = 0;
    com.paraMap[0] = np++;
    // theta5 is estimable if there are >=2 sequences from species 1/2
    if (s11 + s22 + s12_3
        || com.model == M3MSci13 && !com.fixto0[Phi13]
        || com.model == M3MSci23 && !com.fixto0[Phi23]
        || com.model == M2Pro && !(com.fixto0[M13] && com.fixto0[M23])
        || com.model == M2ProMax && !(com.fixto0[M13] && com.fixto0[M23] && com.fixto0[M53])) {
        com.paraNamesMap[np] = 1;
        com.paraMap[1] = np++;
    }
    // tau0 is always estimable
    com.paraNamesMap[np] = 2;
    com.paraMap[2] = np++;
    // tau1 is estimable if there are >=2 sequences from species 1/2
    if (s11 + s22 + s12_3
        || com.model == M3MSci13 && !com.fixto0[Phi13]
        || com.model == M3MSci23 && !com.fixto0[Phi23]
        || com.model == M2Pro && !(com.fixto0[M13] && com.fixto0[M23])
        || com.model == M2ProMax && !(com.fixto0[M13] && com.fixto0[M23])) {
        com.paraNamesMap[np] = 3;
        com.paraMap[3] = np++;
    }
    if (com.simmodel && (s11 + s22 || (com.model == M2SIM3s && !com.fixto0[M12] && s12_3))) {
        // theta1&2 is estimable if there are >= 2 sequences from species 1 / 2
        com.paraNamesMap[np] = 4;
        com.paraMap[5] = com.paraMap[4] = np++;
    } else if (!com.simmodel) {
        // theta1 is estimable if there are >= 2 sequences from species 1
        if (s11 || (com.model == M2SIM3s && !com.fixto0[M12] && s22 + s12_3)
            || (com.model == M3MSci12 && com.fix[ThetaX] && !com.fixto0[Phi12] && s22 + s12_3)
            || (com.model == M3MSci13 && com.fix[ThetaX] && !com.fixto0[Phi13] && s33 + s13_2)
            || (com.model == M2Pro || com.model == M2ProMax) && (
                   !com.fixto0[M12] && s22 + s12_3
                || !com.fixto0[M13] && s33 + s13_2
                || !com.fixto0[M32] && !com.fixto0[M13] && s22 + s2k
                || !com.fixto0[M23] && !com.fixto0[M12] && s33 + s3k
                || !com.fixto0[M12] && !com.fixto0[M13] && ss23)) {
            com.paraNamesMap[np] = 4;
            com.paraMap[4] = np++;
        }
        // theta2 is estimable if there are >= 2 sequences from species 2
        if (s22 || (com.model == M2SIM3s && !com.fixto0[M21] && s11 + s12_3)
            || (com.model == M3MSci12 && com.fix[ThetaY] && !com.fixto0[Phi21] && s11 + s12_3)
            || (com.model == M3MSci23 && com.fix[ThetaY] && !com.fixto0[Phi23] && s33 + s23_1)
            || (com.model == M2Pro || com.model == M2ProMax) && (
                   !com.fixto0[M21] && s11 + s12_3
                || !com.fixto0[M23] && s33 + s23_1
                || !com.fixto0[M31] && !com.fixto0[M23] && s11 + s1k
                || !com.fixto0[M13] && !com.fixto0[M21] && s33 + s3k
                || !com.fixto0[M21] && !com.fixto0[M23] && ss13)) {
            com.paraNamesMap[np] = 5;
            com.paraMap[5] = np++;
        }
    }
    // theta3 is estimable if there are >= 2 sequences from species 3
    if (s33
        || (com.model == M3MSci13 && com.fix[ThetaZ] && !com.fixto0[Phi31] && s11 + s13_2)
        || (com.model == M3MSci23 && com.fix[ThetaZ] && !com.fixto0[Phi32] && s22 + s23_1)
        || com.model == M2ProMax && !com.fixto0[M35]
        || (com.model == M2Pro || com.model == M2ProMax) && (
               !com.fixto0[M31] && s11 + s13_2
            || !com.fixto0[M32] && s22 + s23_1
            || !com.fixto0[M21] && !com.fixto0[M32] && s11 + s1k
            || !com.fixto0[M12] && !com.fixto0[M31] && s22 + s2k
            || !com.fixto0[M31] && !com.fixto0[M32] && ss12)) {
        com.paraNamesMap[np] = 6;
        com.paraMap[6] = np++;
    }
    // beta parameter is always estimable under M1
    if (com.model == M1DiscreteBeta) {
        com.paraNamesMap[np] = 7;
        com.paraMap[7] = np++;
    // M1&2 is estimable if there are >=2 sequences of species 1/2
    } else if (com.model == M2SIM3s && s11 + s22 + s12_3) {
        if (com.simmodel && !com.fixto0[M12]) {
            com.paraNamesMap[np] = 7;
            com.paraMap[8] = com.paraMap[7] = np++;
        }
        // M21 is estimable if model is not symmetric
        else if (!com.simmodel) {
            if (!com.fixto0[M12] && (s22 + s12 || !com.fixto0[M21] && s11)) {
                com.paraNamesMap[np] = 7;
                com.paraMap[7] = np++;
            }
            if (!com.fixto0[M21] && (s11 + s12 || !com.fixto0[M12] && s22)) {
                com.paraNamesMap[np] = 8;
                com.paraMap[8] = np++;
            }
        }
    // T and p are estimable if there are >= 2 sequences from species 1
    } else if (com.model == M3MSci12) {
        if (!(com.fixto0[Phi12] && com.fixto0[Phi21]) && s12_3
            || !(com.fix[ThetaX] && com.fixto0[Phi21]) && s11
            || !(com.fix[ThetaY] && com.fixto0[Phi12]) && s22) {
            com.paraNamesMap[np] = 7;
            com.paraMap[7] = np++;
        }
        if (!com.fix[ThetaX] && (s11 || !com.fixto0[Phi12] && s22 + s12_3)) {
            com.paraNamesMap[np] = 8;
            com.paraMap[8] = np++;
        }
        if (!com.fix[ThetaY] && (s22 || !com.fixto0[Phi21] && s11 + s12_3)) {
            com.paraNamesMap[np] = 9;
            com.paraMap[9] = np++;
        }
        if (!com.fixto0[Phi12] && s22 + s12) {
            com.paraNamesMap[np] = 10;
            com.paraMap[10] = np++;
        }
        if (!com.fixto0[Phi21] && s11 + s12) {
            com.paraNamesMap[np] = 11;
            com.paraMap[11] = np++;
        }
    } else if (com.model == M3MSci13) {
        if (!(com.fixto0[Phi13] && com.fixto0[Phi31]) && s13_2
            || !(com.fix[ThetaX] && com.fixto0[Phi31]) && s11
            || !(com.fix[ThetaZ] && com.fixto0[Phi13]) && s33) {
            com.paraNamesMap[np] = 7;
            com.paraMap[7] = np++;
        }
        if (!com.fix[ThetaX] && (s11 || !com.fixto0[Phi13] && s33 + s13_2)) {
            com.paraNamesMap[np] = 8;
            com.paraMap[8] = np++;
        }
        if (!com.fix[ThetaZ] && (s33 || !com.fixto0[Phi31] && s11 + s13_2)) {
            com.paraNamesMap[np] = 9;
            com.paraMap[9] = np++;
        }
        if (!com.fixto0[Phi13] && s33 + s13) {
            com.paraNamesMap[np] = 10;
            com.paraMap[10] = np++;
        }
        if (!com.fixto0[Phi31] && s11 + s13) {
            com.paraNamesMap[np] = 11;
            com.paraMap[11] = np++;
        }
    } else if (com.model == M3MSci23) {
        if (!(com.fixto0[Phi23] && com.fixto0[Phi32]) && s23_1
            || !(com.fix[ThetaY] && com.fixto0[Phi32]) && s22
            || !(com.fix[ThetaZ] && com.fixto0[Phi23]) && s33) {
            com.paraNamesMap[np] = 7;
            com.paraMap[7] = np++;
        }
        if (!com.fix[ThetaY] && (s22 || !com.fixto0[Phi23] && s33 + s23_1)) {
            com.paraNamesMap[np] = 8;
            com.paraMap[8] = np++;
        }
        if (!com.fix[ThetaZ] && (s33 || !com.fixto0[Phi32] && s22 + s23_1)) {
            com.paraNamesMap[np] = 9;
            com.paraMap[9] = np++;
        }
        if (!com.fixto0[Phi23] && s33 + s23) {
            com.paraNamesMap[np] = 10;
            com.paraMap[10] = np++;
        }
        if (!com.fixto0[Phi32] && s22 + s23) {
            com.paraNamesMap[np] = 11;
            com.paraMap[11] = np++;
        }
    } else if (com.model == M2Pro || com.model == M2ProMax) {
        if (!com.fixto0[M12] && (
               s22 + s12
            || !com.fixto0[M21] && s11
            || !com.fixto0[M23] && s33 + s3
            || !com.fixto0[M13] && s23_3
            || !com.fixto0[M31] && !com.fixto0[M23] && s11_1
            || !com.fixto0[M13] && !com.fixto0[M21] && s33_3 + s13_3)) {
            com.paraNamesMap[np] = 7;
            com.paraMap[7] = np++;
        }
        if (!com.fixto0[M21] && (
               s11 + s12
            || !com.fixto0[M12] && s22
            || !com.fixto0[M13] && s33 + s3
            || !com.fixto0[M23] && s13_3
            || !com.fixto0[M32] && !com.fixto0[M13] && s22_2
            || !com.fixto0[M23] && !com.fixto0[M12] && s33_3 + s23_3)) {
            com.paraNamesMap[np] = 8;
            com.paraMap[8] = np++;
        }
        if (!com.fixto0[M13] && (
               s33 + s13
            || !com.fixto0[M31] && s11
            || !com.fixto0[M32] && s22 + s2
            || !com.fixto0[M12] && s23_2
            || !com.fixto0[M21] && !com.fixto0[M32] && s11_1
            || !com.fixto0[M12] && !com.fixto0[M31] && s22_2 + s12_2)) {
            com.paraNamesMap[np] = 9;
            com.paraMap[9] = np++;
        }
        if (!com.fixto0[M31] && (
               s11 + s13
            || !com.fixto0[M13] && s33
            || !com.fixto0[M12] && s22 + s2
            || !com.fixto0[M32] && s12_2
            || !com.fixto0[M23] && !com.fixto0[M12] && s33_3
            || !com.fixto0[M32] && !com.fixto0[M13] && s22_2 + s23_2)) {
            com.paraNamesMap[np] = 10;
            com.paraMap[10] = np++;
        }
        if (!com.fixto0[M23] && (
               s33 + s23
            || !com.fixto0[M32] && s22
            || !com.fixto0[M31] && s11 + s1
            || !com.fixto0[M21] && s13_1
            || !com.fixto0[M12] && !com.fixto0[M31] && s22_2
            || !com.fixto0[M21] && !com.fixto0[M32] && s11_1 + s12_1)) {
            com.paraNamesMap[np] = 11;
            com.paraMap[11] = np++;
        }
        if (!com.fixto0[M32] && (
               s22 + s23
            || !com.fixto0[M23] && s33
            || !com.fixto0[M21] && s11 + s1
            || !com.fixto0[M31] && s12_1
            || !com.fixto0[M13] && !com.fixto0[M21] && s33_3
            || !com.fixto0[M31] && !com.fixto0[M23] && s11_1 + s13_1)) {
            com.paraNamesMap[np] = 12;
            com.paraMap[12] = np++;
        }

        if (com.model == M2ProMax) {
            if (!com.fixto0[M53] && (
                   s33 + s3
                || !com.fixto0[M35] && s11_1 + s22_2 + s12
                || !com.fixto0[M31] && s11_1 + s12
                || !com.fixto0[M32] && s22_2 + s12
                || !com.fixto0[M21] && !com.fixto0[M32] && s11_1
                || !com.fixto0[M12] && !com.fixto0[M31] && s22_2)) {
                com.paraNamesMap[np] = 13;
                com.paraMap[13] = np++;
            }
            if (!com.fixto0[M35] && (
                   s11 + s22 + s12_3 + s13_3 + s23_3
                || !(com.fixto0[M13] && com.fixto0[M23] && com.fixto0[M53]) && s33_3)) {
                com.paraNamesMap[np] = 14;
                com.paraMap[14] = np++;
            }
        }
    }

    return np;
}

int GetInitials (int np, double x[], double xb[][2])
{
    int i, np0;
    double thetaU=1.99*MULTIPLIER, /*tmp,*/ MU=9.99;//0.15;  /* MU = 0.125 should be fine */

    for(i=0; i<np; i++)  { xb[i][0]=LBOUND*MULTIPLIER;  xb[i][1]=thetaU; }
    if (com.paraMap[3] != -1) {
        xb[com.paraMap[3]][1] = 0.999;  /* xtau */
    }

    if (com.model == M0) {
        for (i = 0; i<np; i++) {
            if (com.initials[com.model][com.paraNamesMap[i]] != 0) {
                x[i] = com.initials[com.model][com.paraNamesMap[i]];

                if (x[i] < 0) {
                    fprintf(stderr, "\nError: invalid initial value for %s in model %d.\n", paranames[com.paraNamesMap[i]], com.model);
                    exit(-1);
                }
            }
            else
                x[i] = (0.0010+rndu()/100)*MULTIPLIER;
        }

        if (com.paraMap[3] != -1) {
            if (com.initials[com.model][3] != 0) {
                x[com.paraMap[3]] = com.initials[com.model][3] / x[com.paraMap[2]];

                if (x[com.paraMap[3]] < 0 || x[com.paraMap[3]] > 1) {
                    fprintf(stderr, "\nError: invalid initial value for %s/%s in model %d.\n", paranames[3], paranames[2], com.model);
                    exit(-1);
                }
            }
            else
                x[com.paraMap[3]] = 0.4 + 0.5*rndu();
        }
    }

    else {
        for (i = 0, np0 = 0; i < 7; i++)
            if (com.paraMap[i] != -1) np0++;

        for (i = 0; i<np0; i++) {
            if (com.initials[com.model][com.paraNamesMap[i]] != 0) {
                x[i] = com.initials[com.model][com.paraNamesMap[i]];

                if (x[i] < 0) {
                    fprintf(stderr, "\nError: invalid initial value for %s in model %d.\n", paranames[com.paraNamesMap[i]], com.model);
                    exit(-1);
                }
            }
            else {
                if (com.aroundMLEM0 && MLEM0[stree.sptree].para[com.paraNamesMap[i]] != 0)
                    x[i] = MLEM0[stree.sptree].para[com.paraNamesMap[i]] * (0.95 + 0.001*rndu()*MULTIPLIER);
                else
                    x[i] = (0.0010 + rndu() / 100)*MULTIPLIER;

                if (x[i] <= xb[i][0])
                    x[i] = xb[i][0] * (1 + 0.001*rndu());
                else if (x[i] >= xb[i][1])
                    x[i] = xb[i][1] * (1 - 0.001*rndu());
            }
        }

        if (com.paraMap[3] != -1) {
            if (com.initials[com.model][3] != 0) {
                x[com.paraMap[3]] = com.initials[com.model][3] / x[com.paraMap[2]];

                if (x[com.paraMap[3]] < 0 || x[com.paraMap[3]] > 1) {
                    fprintf(stderr, "\nError: invalid initial value for %s/%s in model %d.\n", paranames[3], paranames[2], com.model);
                    exit(-1);
                }
            }
            else {
                if (com.aroundMLEM0 && MLEM0[stree.sptree].para[3] != 0)
                    x[com.paraMap[3]] = MLEM0[stree.sptree].para[3] / MLEM0[stree.sptree].para[2] * (0.95 + 0.001*rndu()*MULTIPLIER);
                else
                    x[com.paraMap[3]] = 0.4 + 0.5*rndu();

                if (x[com.paraMap[3]] <= xb[com.paraMap[3]][0])
                    x[com.paraMap[3]] = xb[com.paraMap[3]][0] * (1 + 0.001*rndu());
                else if (x[com.paraMap[3]] >= xb[com.paraMap[3]][1])
                    x[com.paraMap[3]] = xb[com.paraMap[3]][1] * (1 - 0.001*rndu());
            }
        }

        for (i = np0; i < np; i++) {
            if (com.initials[com.model][com.paraNamesMap[i]] != 0) {
                x[i] = com.initials[com.model][com.paraNamesMap[i]];

                if (x[i] < 0) {
                    fprintf(stderr, "\nError: invalid initial value for %s in model %d.\n", paranames[com.paraNamesMap[i]], com.model);
                    exit(-1);
                }
            }
            else {
                if(com.model==M1DiscreteBeta) {
                    x[i] = 1 + 5*rndu();                     /* qbeta */
                }
                else if (com.model == M2SIM3s || com.model == M2Pro || com.model == M2ProMax) {
#ifdef FIXM12
                    x[i] = 0.0001;
#else
                    x[i] = 0.01 + 0.1*rndu();                  /* M12 */ //zzz T/p
#endif
                }
                else if (com.model == M3MSci12 || com.model == M3MSci13 || com.model == M3MSci23) {
                    if (com.paraNamesMap[i] == 7 || com.paraNamesMap[i] == 10 || com.paraNamesMap[i] == 11)
                        x[i] = 0.4 + 0.5*rndu();
                    else
                        x[i] = (0.0010 + rndu() / 100)*MULTIPLIER;
                }
            }

            if (com.model==M1DiscreteBeta) {
                xb[i][0] = 0.1;                          /* q_beta */
                xb[i][1] = 499;                          /* q_beta */
            }
            else if (com.model == M2SIM3s || com.model == M2Pro || com.model == M2ProMax) {   //zzz
                xb[i][0] = LBOUND;            /* M12 */
                xb[i][1] = MU;
            }
            else if (com.model == M3MSci12 || com.model == M3MSci13 || com.model == M3MSci23) {//zzz
                if (com.paraNamesMap[i] == 7 || com.paraNamesMap[i] == 10 || com.paraNamesMap[i] == 11) {
                    xb[i][0] = LBOUND;            /* xT&p */
                    xb[i][1] = 0.999; //xT has upper limit 1
                }
                else {
                    xb[i][0] = LBOUND*MULTIPLIER;
                    xb[i][1] = thetaU;
                }
            }
        }

        if (com.paraMap[7] != -1 && (com.model == M3MSci12 || com.model == M3MSci13 || com.model == M3MSci23)) { //zzz,if tau1 is not estimable, so is T; if T is estimable, so is tau1
            if (com.initials[com.model][7] != 0) {
                x[com.paraMap[7]] = com.initials[com.model][7] / (x[com.paraMap[3]] * x[com.paraMap[2]]); //convert to xT

                if (x[com.paraMap[7]] < 0 || x[com.paraMap[7]] > 1) {
                    fprintf(stderr, "\nError: invalid initial value for %s/%s in model %d.\n", paranames[7], paranames[3], com.model);
                    exit(-1);
                }
            }
        }
    }

    if(noisy) {
        printf("\nInitials & bounds\n");
        for (i=0; i<np; i++) {
            if(com.paraNamesMap[i] == 3)
                printf(" %4s/%s", paranames[3], paranames[2]);
            else if(com.paraNamesMap[i] == 7 && (com.model == M3MSci12 || com.model == M3MSci13 || com.model == M3MSci23))
                printf(" %4s/%s", paranames[7], paranames[3]);
            else
                printf(" %9s", paranames[com.paraNamesMap[i]]);
        }
        printf("\n");
//        if(com.model==M0)             printf("theta4    theta5      tau0     xtau1  theta1&2    theta3\n");
//        if(com.model==M1DiscreteBeta) printf("theta4    theta5      tau0     xtau1  theta1&2    theta3     qbeta\n");
//        if(com.model==M2SIM3s)        printf("theta4    theta5      tau0     xtau1  theta1&2    theta3      M1&2\n");
#ifdef TESTA1DBG
        FOR(i,np) printf(" %20.18f", x[i]); FPN(F0);
        FOR(i,np) printf(" %20.18f", xb[i][0]);  FPN(F0);
        FOR(i,np) printf(" %20.18f", xb[i][1]);  FPN(F0);
#else
        FOR(i,np) printf(" %9.6f", x[i]); FPN(F0);
        FOR(i,np) printf(" %9.5f", xb[i][0]);  FPN(F0);
        FOR(i,np) printf(" %9.5f", xb[i][1]);  FPN(F0);
#endif
    }

    return(0);
}

#if defined(USE_GSL) && defined(TEST_MULTIMIN)

int gradientB(int n, double x[], double f0, double g[],
    double(*fun)(double x[], int n), double space[], int xmark[]);

void lfun_f(const size_t n, const double *x, void *params, double *f) {
    *f = lfun((double*)x, n);
}

void lfun_df(const size_t n, const double *x, void *params, double *g) {
    int i;
    double *xmin = (double*)params;
    double *xmax = xmin + n;
    int *xmark = (int*)(xmax + 2 * n);
    double *space = xmax + 3 * n;

    for (i = 0; i<n; i++) { xmark[i] = 0; }
    for (i = 0; i<n; i++) {
        if (x[i] <= xmin[i]) { xmark[i] = -1; continue; }
        if (x[i] >= xmax[i]) { xmark[i] = 1; continue; }
    }

    gradientB(n, (double*)x, lfun((double*)x, n), g, lfun, space, xmark);
}

void lfun_fdf(const size_t n, const double *x, void *params, double *f, double *g) {
    int i;
    double *xmin = (double*)params;
    double *xmax = xmin + n;
    int *xmark = (int*)(xmax + 2 * n);
    double *space = xmax + 3 * n;

    for (i = 0; i<n; i++) { xmark[i] = 0; }
    for (i = 0; i<n; i++) {
        if (x[i] <= xmin[i]) { xmark[i] = -1; continue; }
        if (x[i] >= xmax[i]) { xmark[i] = 1; continue; }
    }

    *f = lfun((double*)x, n);
    gradientB(n, (double*)x, *f, g, lfun, space, xmark);
}

int minGSL(FILE *fout, double *f,
        void(*fun)(const size_t, const double*, void*, double*),
        void(*dfun)(const size_t, const double*, void*, double*),
        void(*fdfun)(const size_t, const double*, void*, double*, double*),
        double x[], double xb[][2], double space[], double e, int n) {
   int i;
   int maxround = 10000;
   double *xmin = space;
   double *xmax = xmin + n;
   unsigned *type = (unsigned*)(xmax + n);
   struct multimin_params optparams = { 1e-4, 1e-2, maxround, e, 0, 5, (unsigned)noisy };

   for (i = 0; i < n; i++) {
       type[i] = 3;
       xmin[i] = xb[i][0];
       xmax[i] = xb[i][1];
   }

   multimin(n, x, f, type, xmin, xmax, fun, dfun, fdfun, space, optparams);

   return 0;
}

#endif

int Models0123 (FILE *fout, FILE *frub, FILE *frst, double space[])
{
    int i, s, noisy0=noisy, K=com.npoints, nTrees = MAXGTREES, nTrees2seq = MAXGTREES2SEQ;

#ifdef DEBUG_GTREE_PROB
    if (com.usedata == ETreeData) error2("DEBUG_GTREE_PROB usedata = 1");
#endif

    if (com.usedata == ESeqData) {
        com.pDclass = (double*)malloc((com.ncatBeta*data.ndata+com.ncatBeta)*sizeof(double));
        if(com.pDclass==NULL) error2("oom Models01231");
        com.tau1beta = com.pDclass + com.ncatBeta*data.ndata;
        s = (com.fix_locusrate ? K*K*2 : K*K*5);  /* 2 for b0 & b1; 5 for p0124 */
        com.bp0124[0] = (double*)malloc((nTrees*s+nTrees2seq*K)*sizeof(double));
        com.wwprior[0] = (double*)malloc((nTrees*3*K*K+nTrees2seq*K)*sizeof(double));
        if(com.bp0124[0]==NULL || com.wwprior[0]==NULL)
            error2("oom Models01232");
        for(i=1; i<nTrees+1; i++) {
            com.bp0124[i] = com.bp0124[i-1] + s;
            com.wwprior[i] = com.wwprior[i-1] + 3*K*K;
        }
    }

    com.GtreeTab = (struct BTEntry **)malloc(NSTATES*sizeof(struct BTEntry *));
    if(com.GtreeTab == NULL) error2("oom Models01233");
    memset(com.GtreeTab, 0, NSTATES*sizeof(struct BTEntry *));

    noisy = 0;
    if (com.usedata == ESeqData)
        Initialize3s(space);
    noisy = noisy0;

#ifdef M0DEBUG
    for(com.model=0; com.model<1; com.model++) // fix model to M0
#elif defined(M1DEBUG)
        for (com.model=1; com.model < 2; com.model++)
#elif defined(M2DEBUG)
        for(com.model=2; com.model<3; com.model++) // fix model to M2SIM3s
#else
    for (com.extModel = 0; com.extModel < NEXTMODELS; com.extModel++)
#endif
    {
        if (!com.runmodels[com.extModel])
            continue;
        com.model = com.modelMap[com.extModel];
        RunModel(fout, frub, frst, space);
    }

    if (com.usedata == ESeqData) {
        free(com.pDclass);
        free(com.bp0124[0]);
        free(com.wwprior[0]);
    }
    free(com.GtreeTab[0]);
    free(com.GtreeTab);

    return 0;
}

int RunModel (FILE *fout, FILE *frub, FILE *frst, double space[])
{
    int np, np0, i, j, noisy0=noisy, sptree, nsptree, bestsptree, bestmodel, len, fw;
    char timestr[96];
    //char * paranames[7] = {"theta4", "theta5", "tau0", "tau1", "theta1&2", "theta3"};
    double *var, lnL, lnL0=0, e=1e-8, M12=0.0, M21=0.0, tmp;
    double xb[MAXPARAMETERS][2];
    double x[MAXPARAMETERS] = { 1,1,1,1,1,1,1 };
    int nps[MAXSPTREES];
    double lnLs[MAXSPTREES];
    int LRT[MAXSPTREES];
    char sptreeStr[MAXSPTREES][sizeof(stree.nodes[0].name) * NSPECIES + 8];
    char buf[512];
    int* iStateCnt;
    static int bestsptreeM0, fw0;

    if (noisy) printf("\n\n*** Model %d (%s) ***\n", com.extModel, ModelStr[com.extModel]);

    if (com.usedata == ETreeData && com.model == M1DiscreteBeta) {
        if(noisy) printf("\nUse gene trees as data. Skipping M1!\n");
        return 0;
    }

    if(fout) {
        fprintf(fout, "\n\n*** Model %d (%s) ***\n", com.extModel, ModelStr[com.extModel]);
        fprintf(frub, "\n\n*** Model %d (%s) ***\n", com.extModel, ModelStr[com.extModel]);
    }

    if(com.model==M0) {
        com.nGtree = LENGTH_OF(GtOffsetM0);
    } else if(com.model==M1DiscreteBeta) {
        com.nGtree = LENGTH_OF(GtOffsetM0);
        paranames[ 7] = "qbeta";
        paranames[ 8] = 0;
    } else if(com.model==M2SIM3s) {
        com.nGtree = LENGTH_OF(GtOffsetM2);
        paranames[ 7] = "M12";
        paranames[ 8] = "M21";
        paranames[ 9] = 0;
    } else if(com.model==M2Pro) {
        com.nGtree = LENGTH_OF(GtOffsetM2Pro);
        paranames[ 7] = "M12";
        paranames[ 8] = "M21";
        paranames[ 9] = "M13";
        paranames[10] = "M31";
        paranames[11] = "M23";
        paranames[12] = "M32";
        paranames[13] = 0;
    } else if(com.model == M2ProMax) {
        com.nGtree = LENGTH_OF(GtOffsetM2ProMax);
        paranames[ 7] = "M12";
        paranames[ 8] = "M21";
        paranames[ 9] = "M13";
        paranames[10] = "M31";
        paranames[11] = "M23";
        paranames[12] = "M32";
        paranames[13] = "M53";
        paranames[14] = "M35";
    } else if(com.model==M3MSci12) {
        com.nGtree = LENGTH_OF(GtOffsetM3MSci12);
        paranames[ 7] = "T";
        paranames[ 8] = "thetaX";
        paranames[ 9] = "thetaY";
        paranames[10] = "phi12";
        paranames[11] = "phi21";
        paranames[12] = 0;
    } else if(com.model==M3MSci13) {
        com.nGtree = LENGTH_OF(GtOffsetM3MSci13);
        paranames[ 7] = "T";
        paranames[ 8] = "thetaX";
        paranames[ 9] = "thetaZ";
        paranames[10] = "phi13";
        paranames[11] = "phi31";
        paranames[12] = 0;
    } else if(com.model==M3MSci23) {
        com.nGtree = LENGTH_OF(GtOffsetM3MSci23);
        paranames[ 7] = "T";
        paranames[ 8] = "thetaY";
        paranames[ 9] = "thetaZ";
        paranames[10] = "phi23";
        paranames[11] = "phi32";
        paranames[12] = 0;
    }

    if (com.simmodel) {
        paranames[4] = "theta1&2";
        paranames[7] = "M1&2";
    }

    InitializeGtreeTab();

    if (stree.speciestree)
        if (com.asymmetric[com.model])
            nsptree = 6;
        else
            nsptree = 3;
    else
        nsptree = 1;

    for (sptree = 0; sptree < nsptree; sptree++)
        len = snprintf(sptreeStr[sptree], sizeof(sptreeStr[sptree]), "((%s, %s), %s)", stree.nodes[spMap[sptree][0] - 1].name, stree.nodes[spMap[sptree][1] - 1].name, stree.nodes[spMap[sptree][2] - 1].name);

    for (stree.sptree = 0; stree.sptree < nsptree; stree.sptree++) {
        if (stree.speciestree) {
            if (noisy) printf("\nSpecies tree %d: %s\n", stree.sptree + 1, sptreeStr[stree.sptree]);
            if (fout) {
                fprintf(fout, "\nSpecies tree %d: %s\n", stree.sptree + 1, sptreeStr[stree.sptree]);
                fprintf(frub, "\nSpecies tree %d: %s\n", stree.sptree + 1, sptreeStr[stree.sptree]);
            }
        }

        np = GetParaMap();

        if (com.model == M1DiscreteBeta && com.paraMap[3] == -1) {
            if(noisy) printf("\ntau1 not estimable. Skipping M1!\n");
            return 0;
        }

#ifdef DEBUG_GTREE_PROB
        printf("\nCalculate the gene tree probabilities at fixed parameters read from 'in.3s':\n");
        FOR(i,MAXPARAMETERS) if(paranames[i]) printf(" %9s", paranames[i]); else break; FPN(F0);
        FOR(i,MAXPARAMETERS) if(paranames[i]) printf(" %9.6f", com.initials[com.model][i]); else break; FPN(F0);
        for(i=0; i<MAXPARAMETERS; i++)
            if(com.paraMap[i] != -1)
                x[com.paraMap[i]] = com.initials[com.model][i];
        NFunCall = 0;
        LASTROUND = 2;
        lnL = lfun(x,np);
        if (stree.speciestree)
            printf("Results are written to 'pGk%d-%d.txt'\n", com.extModel, stree.sptree + 1);
        else
            printf("Results are written to 'pGk%d.txt'\n", com.extModel);
        continue;
#endif

#ifdef DEBUG_LIKELIHOOD
        printf("\nCalculate the log likelihood for the data at fixed parameters read from 'in.3s':\n");
        FOR(i, MAXPARAMETERS) if (paranames[i]) printf(" %9s", paranames[i]); else break; FPN(F0);
        FOR(i, MAXPARAMETERS) if (paranames[i]) printf(" %9.6f", com.initials[com.model][i]); else break; FPN(F0);
        for (i = 0; i<MAXPARAMETERS; i++)
            if (com.paraMap[i] != -1)
                x[com.paraMap[i]] = com.initials[com.model][i];
        NFunCall = 0;
        LASTROUND = 1;
        lnL = lfun(x, np);
        printf("lnL = %18.12f\n", -lnL);
        continue;
#endif

#if defined(DEBUG_GTREE_PROB) && defined(DEBUG_LIKELIHOOD)
#error DEBUG_GTREE_PROB and DEBUG_LIKELIHOOD
#endif

        LASTROUND = 0;

        GetInitials (np, x, xb);

                /*
                 printf("input initials? ");
                 for(i=0; i<np; i++)
                 scanf("%lf", &x[i]);
                 */
        var = space + np;

#if(0)
                printf("\nTesting lfun at fixed parameter values using ChenLi data of 53 loci\n");
                x[0]=0.30620; x[1]=0.09868; x[2]=0.62814; x[3]=0.82706; x[4]=1.1; x[5]=2.2;
                /* x[0]=0.35895; x[1]=0.42946; x[2]=0.66026; x[3]=0.65449; x[4]=2; */
                printf("  0.30620  0.09868  0.62814  0.82706  lnL = %12.6f (K=inf)\n", -3099.411263);

                for(com.npoints=32; com.npoints<=64; com.npoints*=2) {
                    for(x[4]=0.5; x[4]<1000; x[4] *= 2) {
                        for(i=0; i<np; i++) printf(" %8.5f", x[i]);
                        lnL = lfun(x, np);
                        printf("  lnL = %12.6f (%2d) %s\n", -lnL, com.npoints, printtime(timestr));
                        if(com.model==M0 || com.model==M2SIM3s) break;
                    }
                    break;
                }
                exit(0);
#elif(0)
                printf("\nContour for the hominoid data (model %s)\n", ModelStr[com.model]);
                LASTROUND = 1;
                {
                    int ii, jj, nii=22, njj=22;
                    double x0[]={0.361874, 0.369049, 0.658947, 0.459227, 2.601699, 12.508765};
                    double theta12Set[] = {0.5, 0.7, 0.8, 1, 1.5, 2, 2.5, 2.60, 2.7, 2.8,
                        3, 3.2, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10};
                    double Mset[]       = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 2,
                        4, 5, 7, 8, 10, 12, 15, 16, 18, 20, 22, 25};

                    xtoy(x0, x, np);
                    for(ii=0; ii<nii; ii++) {
                        x[4] = theta12Set[ii];
                        for(jj=0; jj<njj; jj++) {
                            x[5] = Mset[jj];
                            lnL = lfun(x, np);
                            printf("%.6f\t%.6f\t%.4f\n", x[4], x[5], -lnL);
                        }
                    }
                    exit(0);
                }
#endif

#ifdef TESTA1
                printf("Test A1: stability of quadrature: using K=%d points\n", com.npoints);
                x[0] = 20; x[1] = 15; x[2] = 10; x[3] = 0.5 ; x[4] = 12; x[5] = 30; x[6] = 0;
#endif

#ifdef DEBUG1
                printf("testing floating point problem\n");
                //        x[0] = 0.1; x[1] = 0.4322; x[2] = 0.1678; x[3] = 0.5906; x[4] = 0.4487; x[5] = 0.9638;
                //                x[0] = 0.001; x[1] = 0.9019; x[2] = 12.6675; x[3] = 0.5358; x[4] = 0.8317; x[5] = 64.2162;
                x[0] = 1.00016; x[1] = 2.10741; x[2] = 8.63080; x[3] = 0.593506; x[4] = 1.002878; x[5] = 9.8984;
#endif
                // x[0] = 20; x[1] = 15; x[2] = 10; x[3] = 0.5; x[4] = 12; x[5] = 30; x[6] = 0.0001;
        noisy=3;

#ifdef CONTOUR
                //for(x[1] = 1e-7; x[1] < 1e-4; x[1] += (x[1] < 1e-5 ? 4.95e-7 : 1.125e-6)) {
                //    for(x[3] = 0.85; x[3] <= 0.999; x[3] += (x[3] < 0.95 ? 2e-3 : 9.8e-4)) {
                //for(x[1] = 0.015; x[1] <= 0.025; x[1] += 1e-4) {
                //for(x[1] = 0.01; x[1] < 0.015; x[1] += 1e-4) {
                //for(x[1] = 0.005; x[1] < 0.01; x[1] += 1e-4) {
                //for(x[1] = 0.0001; x[1] < 0.005; x[1] += 1e-4) {
                //    for(x[3] = 0.4; x[3] <= 0.9999; x[3] += 5.999e-3) {
                for(x[1] = lbound1; x[1] <= ubound1; x[1] += inc1) {
                    for(x[3] = lbound2; x[3] <= ubound2; x[3] += inc2) {
                        lnL = lfun(x, np);
                        printf("%.8f %.8f %.8f\n", x[1], x[3], -lnL);
                    }
                }
                exit(0);
#endif

        lnL = lfun(x, np);
        if(noisy) printf("\nlnL0 = %12.6f\n", -lnL);
                //exit(0);

#if defined (TESTA1) || defined (DEBUG1)
                exit(0);
#endif

//                if (com.model == M2SIM3s) {
//                    exit(0);
//                }
#ifdef M2DEBUG
                exit(0);
#endif
        NFunCall=0;
        if(com.getSE <= 1) {
#if defined(USE_GSL) && defined(TEST_MULTIMIN)
            if(noisy) printf("\n");
            minGSL(frub, &lnL, lfun_f, lfun_df, lfun_fdf, x, xb, space, e, np);
#else
            ming2(frub, &lnL, lfun, NULL, x, xb, space, e, np);
            fprintf(frub, "\n");
#endif
        }

        LASTROUND = 2;

        if (com.paraMap[3] != -1) {
            x[com.paraMap[3]] *= x[com.paraMap[2]];     /* xtau1 -> tau1 */

            if (com.paraMap[7] != -1 && (com.model == M3MSci12 || com.model == M3MSci13 || com.model == M3MSci23))
                x[com.paraMap[7]] *= x[com.paraMap[3]]; // T from xT
        }

        if (com.model == M0) {
            for (i = stree.sptree; i < stree.nsptree; i += 3) {
                memset(MLEM0[i].para, 0, MAXPARAMETERS * sizeof(double));

                for (j = 0; j < np; j++)
                    MLEM0[i].para[com.paraNamesMap[j]] = x[j];

                MLEM0[i].np = np;
                MLEM0[i].lnL = lnL;
            }
        }
        else {
            np0 = MLEM0[stree.sptree].np;
            lnL0 = MLEM0[stree.sptree].lnL;

            if (np == np0)
                LRT[stree.sptree] = -1;
            else
                LRT[stree.sptree] = (int)(2 * (lnL0 - lnL) > chi2CV_5pct[np - np0]);
        }

        nps[stree.sptree] = np;
        lnLs[stree.sptree] = lnL;

        if(noisy) {
            printf("\nlnL  = %12.6f  (%5d lfun calls)\n", -lnL, NFunCall);
            if (com.model != M0) {
                printf("2DlnL = %+12.6f\n", 2*(lnL0-lnL));
                if (com.model == M2SIM3s && !com.simmodel || com.model == M2Pro || com.model == M2ProMax || com.model == M3MSci12 || com.model == M3MSci13 || com.model == M3MSci23) {
                    if (LRT[stree.sptree] == -1)
                        printf("    Data does not contain efficient information.\n");
                    else
                        printf("    The test is%s significant under model %d (%s) at 5%% significance level.\n", ((LRT[stree.sptree])? "" : " not"), com.extModel, ModelStr[com.extModel]);
                }
            }
            printf("MLEs\n");
            for (i=0; i<np; i++) {
                printf(" %9s", paranames[com.paraNamesMap[i]]);
            }
            printf("\n");
            for(i=0; i<np; i++)    printf(" %9.6f", x[i]);
            printf("\n");
        }
        if(fout) {
            fprintf(fout, "\nlnL  = %12.6f\n", -lnL);
            if (com.model != M0) {
                fprintf(fout, "2DlnL = %+12.6f\n", 2*(lnL0-lnL));
                if (com.model == M2SIM3s && !com.simmodel || com.model == M2Pro || com.model == M2ProMax || com.model == M3MSci12 || com.model == M3MSci13 || com.model == M3MSci23) {
                    if (LRT[stree.sptree] == -1)
                        fprintf(fout, "    Data does not contain efficient information.\n");
                    else
                        fprintf(fout, "    The test is%s significant under model %d (%s) at 5%% significance level.\n", ((LRT[stree.sptree])? "" : " not"), com.extModel, ModelStr[com.extModel]);
                }
            }
            fprintf(fout, "MLEs\n");
            for (i=0; i<np; i++) {
                fprintf(fout, " %9s", paranames[com.paraNamesMap[i]]);
            }
            fprintf(fout, "\n");
            for(i=0; i<np; i++)    fprintf(fout, " %9.6f", x[i]);
            fprintf(fout, "\n");
            fflush(fout);
        }

        if (com.usedata == ESeqData)
            // call lfun to compute gene tree posteriors
            lfun(x,np);

        LASTROUND = 1;

        if(fout && com.getSE) {
            Hessian (np, x, lnL, space, var, lfun, var+2*np*np);
            matinv(var, np, np, var+np*np);

            if(noisy) {
                printf("SEs:\n");
                for(i=0;i<np;i++) printf(" %9.6f",(var[i*np+i]>0.?sqrt(var[i*np+i]):-1));
                printf("\n");
            }
            fprintf(fout, "SEs:\n");
            for(i=0;i<np;i++) fprintf(fout, " %9.6f", (var[i*np+i]>0. ? sqrt(var[i*np+i]) : -1));
            fprintf(fout, "\n\nCorrelation matrix\n");
            for(i=0;i<np;i++) for(j=0;j<i;j++)
                var[j*np+i] = var[i*np+j] =
                (var[i*np+i]>0 && var[j*np+j]>0 ? var[i*np+j]/sqrt(var[i*np+i]*var[j*np+j]) : -9);
            for(i=0;i<np;i++) var[i*np+i]=1;
            matout2(fout, var, np, np, 10, 5);
            fflush(fout);
        }
        if(com.model==M1DiscreteBeta) {
            fprintf(fout, "\ntau1 from the discrete beta (ncat = %d), using %s\ntau1: ", com.ncatBeta, (com.UseMedianBeta?"median":"mean"));
            for(i=0; i<com.ncatBeta; i++) fprintf(fout, " %9.6f", com.tau1beta[i]);
            fprintf(fout, "\nfreq: ");
            for(i=0; i<com.ncatBeta; i++) fprintf(fout, " %9.6f", 1.0/com.ncatBeta);
            fprintf(fout, "\n");
            for(i=0; i<np; i++) fprintf(frst, " %9.6f", x[i]);
            fprintf(frst, " %9.6f\n", 2*(lnL0-lnL));
        }

        iStateCnt = data.iStateCnt[stree.sptree];

        if(com.model==M2SIM3s && !com.fixto0[0] && (com.simmodel || !com.fixto0[1]) /* enum { M12 = 0, M21 = 1 }; */
            && (iStateCnt[5] + iStateCnt[28])
            && !(iStateCnt[0] + iStateCnt[13] + iStateCnt[1] + iStateCnt[12] + iStateCnt[2] + iStateCnt[14] + iStateCnt[27] + iStateCnt[31])) {
            for(i=0; i<np; i++) fprintf(frst, "\t%.6f", x[i]);
            if(noisy) printf("\nThe other local peak is at\n");
            fprintf(fout, "\nThe other local peak is at\n");
            M12 = x[com.paraMap[7]]/MULTIPLIER;
            x[com.paraMap[4]] = x[com.paraMap[4]]/(8*M12);  /*  theta/(8M)  */
            x[com.paraMap[7]] = MULTIPLIER/(64*M12);    /*  1/(64M)  */
            if (!com.simmodel) {
                M21 = x[com.paraMap[8]]/MULTIPLIER;
                x[com.paraMap[5]] = x[com.paraMap[5]]/(8*M21);  /*  theta/(8M)  */
                x[com.paraMap[8]] = MULTIPLIER/(64*M21);    /*  1/(64M)  */
            }
            lnL = lfun(x, np);
            if(noisy) {
                for(i=0; i<np; i++) printf(" %9.6f", x[i]); FPN(F0);
                printf(" %12.6f\n", -lnL);
            }
            for(i=0; i<np; i++) fprintf(fout, " %9.6f", x[i]); FPN(fout);
            fprintf(fout, " %12.6f\n", -lnL);
            fprintf(frst, "\t%.6f\n", 2*(lnL0-lnL));
        }

        if(com.model== M3MSci12 && !com.fixto0[8] && !com.fixto0[9] && !com.fix[0] && !com.fix[1] /* enum { Phi12 = 8, Phi21 = 9, ThetaX = 0, ThetaY = 1 }; */
            && (iStateCnt[0] + iStateCnt[13] + iStateCnt[1] + iStateCnt[12] + iStateCnt[2] + iStateCnt[14] + iStateCnt[5] + iStateCnt[27] + iStateCnt[31] + iStateCnt[28])) {
            for(i=0; i<np; i++) fprintf(frst, "\t%.6f", x[i]);
            if(noisy) printf("\nThe other local peak is at\n");
            fprintf(fout, "\nThe other local peak is at\n");
            tmp = x[com.paraMap[8]];
            x[com.paraMap[8]] = x[com.paraMap[9]];
            x[com.paraMap[9]] = tmp;
            x[com.paraMap[10]] = 1 - x[com.paraMap[10]];
            x[com.paraMap[11]] = 1 - x[com.paraMap[11]];
            lnL = lfun(x, np);
            if(noisy) {
                for(i=0; i<np; i++) printf(" %9.6f", x[i]); FPN(F0);
                printf(" %12.6f\n", -lnL);
            }
            for(i=0; i<np; i++) fprintf(fout, " %9.6f", x[i]); FPN(fout);
            fprintf(fout, " %12.6f\n", -lnL);
            fprintf(frst, "\t%.6f\n", 2*(lnL0-lnL));
        }

        fflush(frst);
    }

#if !defined(DEBUG_GTREE_PROB) && !defined(DEBUG_LIKELIHOOD)
    if (stree.speciestree) {
        for (bestsptree = 0, sptree = 1; sptree < nsptree; sptree++) {
            if (lnLs[bestsptree] + nps[bestsptree] > lnLs[sptree] + nps[sptree])
                bestsptree = sptree;
        }

        if (com.model == M0) {
            bestsptreeM0 = bestsptree;
        }
        else {
            if (bestsptreeM0 == bestsptree % 3)
                bestmodel = (LRT[bestsptree] > 0) ? com.extModel : M0;
            else if (LRT[bestsptreeM0] > 0 && LRT[bestsptree] > 0)
                bestmodel = com.extModel;
            else if (LRT[bestsptreeM0] <= 0 && LRT[bestsptree] <= 0)
                bestmodel = M0;
            else
                bestmodel = (MLEM0[bestsptreeM0].lnL + MLEM0[bestsptreeM0].np > lnLs[bestsptree] + nps[bestsptree]) ? com.extModel : M0;

            if (bestmodel == M0)
                bestsptree = bestsptreeM0;
        }

        if (noisy || fout) {
            fw = 0;
            for (sptree = 0; sptree < nsptree; sptree++) {
                i = snprintf(buf, sizeof(buf), "%12.6f", -lnLs[sptree]);
                if (fw < i) fw = i;
            }
            if (com.model == M0) fw0 = fw;
        }

        if (noisy) {
            printf("\nSummary:\n");
            if (com.model == M0) {
                printf("    %-*s    %*s  %2s\n", len + 1, "species tree", fw, "lnL", "np");
                for (sptree = 0; sptree < nsptree; sptree++)
                    printf("  %d: %s    %*.6f  %2d\n", sptree + 1, sptreeStr[sptree], fw, -lnLs[sptree], nps[sptree]);
            }
            else {
                printf("    %-*s    %*s  %3s    %*s  %2s\n", len + 1, "species tree", fw0, "lnL0", "np0", fw, "lnL", "np");
                for (sptree = 0; sptree < nsptree; sptree++)
                    printf("  %d: %s    %*.6f  %3d    %*.6f  %2d\n", sptree + 1, sptreeStr[sptree], fw0, -MLEM0[sptree].lnL, MLEM0[sptree].np, fw, -lnLs[sptree], nps[sptree]);
            }

            printf("\nBest species tree:\n");
            printf("  %d: %s", bestsptree + 1, sptreeStr[bestsptree]);
            if (com.model != M0)
                printf(" under model %d (%s)", bestmodel, ModelStr[bestmodel]);
            printf("\n");
        }

        if (fout) {
            fprintf(fout, "\nSummary:\n");
            if (com.model == M0) {
                fprintf(fout, "    %-*s    %*s  %2s\n", len + 1, "species tree", fw, "lnL", "np");
                for (sptree = 0; sptree < nsptree; sptree++)
                    fprintf(fout, "  %d: %s    %*.6f  %2d\n", sptree + 1, sptreeStr[sptree], fw, -lnLs[sptree], nps[sptree]);
            }
            else {
                fprintf(fout, "    %-*s    %*s  %3s    %*s  %2s\n", len + 1, "species tree", fw0, "lnL0", "np0", fw, "lnL", "np");
                for (sptree = 0; sptree < nsptree; sptree++)
                    fprintf(fout, "  %d: %s    %*.6f  %3d    %*.6f  %2d\n", sptree + 1, sptreeStr[sptree], fw0, -MLEM0[sptree].lnL, MLEM0[sptree].np, fw, -lnLs[sptree], nps[sptree]);
            }

            fprintf(fout, "\nBest species tree:\n");
            fprintf(fout, "  %d: %s", bestsptree + 1, sptreeStr[bestsptree]);
            if (com.model != M0)
                fprintf(fout, " under model %d (%s)", bestmodel, ModelStr[bestmodel]);
            fprintf(fout, "\n");

            fflush(fout);
        }
    }
#endif

    printf("\nTime used: %s\n", printtime(timestr));

    return 0;
}

/* adapted from routine hqr from the Numerical Recipes book
 * finds all eigenvalues of an upper Hessenberg matrix (stored in a).
 * on return, wr contains the eigenvalues and the columns of matrix U
 * contain the corresponding eigenvectors.
 *
 * We assume that all eigenvalues will be real, so we return an error
 * if complex eigenvalues are encountered.
 */
int EigenHessenbQRImplicit(double *a, int n, double wr[], double *U) {
    int nn, na, m, l, k, j, its, i, mmin;
    double z, y, x, w, v, u, t, s, r, q, p, anorm;

    // compute matrix norm for possible use in locating single small
    // subdiagonal element
    anorm = 0.0;
    for(i = 0; i < n; i++)
        for(j = ((i-1) > 0 ? (i-1) : 0); j < n; j++)
            anorm += fabs(a[n*i+j]);
    nn = n-1;
    t = 0.0; // gets changed only by an exceptional shift
    while(nn >= 0) { // begin search for next eigenvalue
        its = 0;
        do {
            // begin iteration: look for single small subdiagonal element
            for(l = nn; l >= 1; l--) {
                s = fabs(a[n*(l-1)+(l-1)]) + fabs(a[n*l+l]);
                if (s == 0.0)
                    s = anorm;
                if((double)(fabs(a[n*l+(l-1)]) + s) == s) {
                    a[n*l+(l-1)] = 0.0;
                    break;
                }
            }
            x = a[n*nn+nn];
            if(l == nn) { // one root found...
                wr[nn] = a[n*nn+nn] = x+t;
                nn--;
            } else {
                y = a[n*(nn-1)+(nn-1)];
                w = a[n*nn+(nn-1)] * a[n*(nn-1)+nn];
                if(l == (nn-1) ) { // two roots found...
                    p = 0.5 * (y-x);
                    q = p * p + w;
                    z = sqrt(fabs(q));
                    x += t;
                    a[n*nn+nn] = x;
                    a[n*(nn-1)+(nn-1)] = y + t;
                    if(q >= 0.0) { // ...a real pair
                        z = p + (p >= 0.0 ? fabs(z) : -fabs(z));
                        wr[nn-1] = wr[nn] = x + z;
                        if(z)
                            wr[nn] = x - w / z;
                        x = a[n*nn+(nn-1)];
                        s = fabs(x) + fabs(z);
                        p = x / s;
                        q = z / s;
                        r = sqrt(p*p + q*q);
                        p /= r;
                        q /= r;
                        for(j = nn-1; j < n; j++) { // row modification
                            z = a[n*(nn-1)+j];
                            a[n*(nn-1)+j] = q * z + p * a[n*nn+j];
                            a[n*nn+j] = q * a[n*nn+j] - p * z;
                        }
                        for(i = 0; i <= nn; i++) { // column modification
                            z = a[n*i+(nn-1)];
                            a[n*i+(nn-1)] = q * z + p * a[n*i+nn];
                            a[n*i+nn] = q * a[n*i+nn] - p * z;
                        }
                        for(i = 0; i < n; i++) { // accumulate transformations
                            z = U[n*i+(nn-1)];
                            U[n*i+(nn-1)] = q * z + p * U[n*i+nn];
                            U[n*i+nn] = q * U[n*i+nn] - p * z;
                        }

                    } else { // ...a complex pair
                        printf("matrix has complex eigenvalues - aborting!\n");
                        return -1;
                    }
                    nn -= 2;
                } else { // no roots found. continue iteration.
                    if(its == 30) {
                        printf("too many iterations in hqr\n");
                        return -1;
                    }
                    if(its == 10 || its == 20) { // form exceptional shift
                        t += x;
                        for(i = 0; i <= nn; i++)
                            a[n*i+i] -= x;
                        s = fabs(a[n*nn+(nn-1)]) + fabs(a[n*(nn-1)+(nn-2)]);
                        y = x = 0.75 * s;
                        w = -0.4375 * s * s;
                    }
                    ++its;
                    for(m = (nn-2); m >= l; m--) { // form shift and then look for 2 consecutive small sub-diagonal elements
                        z = a[n*m+m];
                        r = x - z;
                        s = y - z;
                        p = (r * s - w)/ a[n*(m+1)+m] + a[n*m+(m+1)]; // equation 11.6.23
                        q = a[n*(m+1)+(m+1)] - z - r - s;
                        r = a[n*(m+2)+(m+1)];
                        s = fabs(p) + fabs(q) + fabs(r); // scale to prevent overflow or underflow
                        p /= s;
                        q /= s;
                        r /= s;
                        if(m == 0)
                            break;
                        u = fabs(a[n*m+(m-1)]) * (fabs(q) + fabs(r));
                        v = fabs(p) * (fabs(a[n*(m-1)+(m-1)]) + fabs(z) + fabs(a[n*(m+1)+(m+1)]));
                        if((double)(u+v) == v) // equation 11.6.26
                            break;
                    }
                    for(i = m+2; i <= nn; i++) {
                        a[n*i+(i-2)] = 0.0;
                        if(i != (m+2))
                            a[n*i+(i-3)] = 0.0;
                    }
                    for(k = m; k <= nn-1; k++) { // double QR step in rows 1 to nn and columns m to nn
                        if(k != m) {
                            p = a[n*k+(k-1)]; // begin setup of householder vector
                            q = a[n*(k+1)+(k-1)];
                            r = 0.0;
                            if(k != (nn-1))
                                r = a[n*(k+2)+(k-1)];
                            if((x = fabs(p) + fabs(q) + fabs(r)) != 0.0) { // scale to prevent overflow or underflow
                                p /= x;
                                q /= x;
                                r /= x;
                            }
                        }
                        s = sqrt(p*p + q*q + r*r);
                        if((s = (p >= 0 ? fabs(s) : -fabs(s))) != 0.0) {
                            if(k == m) {
                                if(l != m)
                                    a[n*k+(k-1)] = -a[n*k+(k-1)];
                            } else {
                                a[n*k+(k-1)] = -s * x;
                            }
                            p += s; // equations 11.6.24
                            x = p / s;
                            y = q / s;
                            z = r / s;
                            q /= p;
                            r /= p;
                            for(j = k; j < n; j++) { // row modification
                                p = a[n*k+j] + q * a[n*(k+1)+j];
                                if(k != (nn-1)) {
                                    p += r * a[n*(k+2)+j];
                                    a[n*(k+2)+j] -= p * z;
                                }
                                a[n*(k+1)+j] -= p * y;
                                a[n*k+j]  -= p * x;
                            }
                            mmin = nn < k+3 ? nn : k+3;
                            for(i = 0; i <= mmin; i++) { // column modification
                                p = x * a[n*i+k] + y * a[n*i+(k+1)];
                                if(k != (nn-1)) {
                                    p += z * a[n*i+(k+2)];
                                    a[n*i+(k+2)] -= p * r;
                                }
                                a[n*i+(k+1)] -= p * q;
                                a[n*i+k] -= p;
                            }
                            for(i = 0; i < n; i++) { // accumulate transformations
                                p = x * U[n*i+k] + y * U[n*i+(k+1)];
                                if(k != nn-1) {
                                    p += z * U[n*i+(k+2)];
                                    U[n*i+(k+2)] -= p * r;
                                }
                                U[n*i+(k+1)] -= p * q;
                                U[n*i+k] -= p;
                            }
                        }
                    }
                }
            }
        } while(l < nn-1);
    }
    if(anorm != 0.0) {
        for(nn = n-1; nn >= 0; nn--) {
            p = wr[nn];
            na = nn - 1;
            m = nn;
            a[n*nn+nn] = 1.0;
            for(i = nn-1; i >= 0; i--) {
                w = a[n*i+i] - p;
                r = 0;
                for(j = m; j <= nn; j++)
                    r += a[n*i+j] * a[n*j+nn];
                m = i;

                t = w;
                if(t == 0.0)
                    t = DBL_EPSILON * anorm;
                a[n*i+nn] = -r/t;
                t = fabs(a[n*i+nn]); // overflow control
                if(DBL_EPSILON * t * t > 1)
                    for(j = i; j <= nn; j++)
                        a[n*j+nn] /= t;
            }
        }
        for(j = n-1; j >= 0; j--)
            for(i = 0; i < n; i++) {
                z = 0;
                for(k = 0; k <= j; k++)
                    z += U[n*i+k] * a[n*k+j];
                U[n*i+j] = z;
            }
    }
    return 0;
}

void MatBalancing(double *a, double *sc, int n) {
    int i, j;
    int last = 0;
    double c, f, g, r, s;
    const double sqrdx = 2.0*2, rdx = 2.0;

    while(!last) {
        last = 1;
        for(i = 0; i < n; i++) { // calculate row and column norms
            r = c = 0;
            for(j = 0; j < n; j++) {
                if(j != i) {
                    c += fabs(a[n*j+i]);
                    r += fabs(a[n*i+j]);
                }
            }
            if(c && r) {
                g = r / rdx;
                f = 1;
                s = c + r;
                // find integer power of machine radix taht comes closest to
                // balancing the matrix
                while(c < g) {
                    f *= rdx;
                    c *= sqrdx;
                }
                g = r * rdx;
                while(c > g) {
                    f /= rdx;
                    c /= sqrdx;
                }
                if((c + r) / f < 0.95 * s) {
                    last = 0;
                    g = 1 / f;
                    sc[i] *= f;
                    // apply similarity transformation
                    for(j = 0; j < n; j++)
                        a[n*i+j] *= g;
                    for(j = 0; j < n; j++)
                        a[n*j+i] *= f;
                }
            }
        }
    }
}

void EigenUnbalance(double *U, double * sc, int n) {
    int i, j;
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            U[n*i+j] *= sc[i];
        }
    }
}

void HessenbBacktransform(double *a, double *U, int *p, int n) {
    int i, j, ip, k;
    for(ip = n - 2; ip > 0; ip--) {
        for(k = ip+1; k < n; k++) {
            U[n*k+ip] = a[n*k+(ip-1)];
        }
        i = p[ip];
        if(i != ip) {
            for(j = ip; j < n; j++) {
                U[n*ip+j] = U[n*i+j];
                U[n*i+j] = 0;
            }
            U[n*i+ip] = 1;
        }
    }
}

void HessenbTransform(double *a, double * U, double * space, int n) {

    int i, j, m;
    double x, y;

    double * sc = space;
    int * p = (int *)(sc+n);

    MatBalancing(a, sc, n);

    for(m = 1; m < n-1; m++) {
        x = 0;
        i = m;
        for(j = m; j < n; j++) { // find the pivot
            if(fabs(a[n*j+(m-1)]) > fabs(x)) {
                x = a[n*j+(m-1)];
                i = j;
            }
        }
        p[m] = i;
        if(i != m) {// interchange rows and columns
            for(j = m-1; j < n; j++) {
                y = a[n*m+j];
                a[n*m+j] = a[n*i+j];
                a[n*i+j] = y;
            }
            for(j = 0; j < n; j++) {
                y = a[n*j+i];
                a[n*j+i] = a[n*j+m];
                a[n*j+m] = y;
            }
        }
        if(x) { // carry out the elimination
            for(i = m + 1; i < n; i++) {
                if((y = a[n*i+(m-1)]) != 0) {
                    y /= x;
                    a[n*i+(m-1)] = y;
                    for(j = m; j < n; j++)
                        a[n*i+j] -= y * a[n*m+j];
                    for(j = 0; j < n; j++)
                        a[n*j+m] += y * a[n*j+i];
                }
            }
        }
    }
    // set lower left elements to 0
    for(i = 2; i < n; i++) {
        for(j = 0; j < i - 1; j++) {
            a[n*i+j] = 0;
        }
    }

    HessenbBacktransform(a, U, p, n);
}

int eigenRealGen(double A[], int n, double root[], double work[]) {
    int status = 0, i;

    double * U = work;
    double * space = U+n*n;

    for(i = 0; i < n; i++) {
        U[n*i+i] = 1;
        space[i] = 1;
    }

    HessenbTransform(A, U, space, n);
    status = EigenHessenbQRImplicit(A, n, root, U);
    EigenUnbalance(U, space, n);
    EigenSort(root, U, n);
    memcpy(A, U, n*n);

    return(status);
}
