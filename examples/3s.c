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

extern double Small_Diff;
extern int noisy, NFunCall;

struct CommonInfo com;
//struct BTEntry * GtreeTab[27] = {0};

int LASTROUND;

double para[9];  /* theta4 theta5 tau0 tau1 theta1 theta2 theta3 M12 M21 qbeta */
//double avgb[2] = {0,0}, varb[2] = {0,0};
//int debug = 0;
//#define DEBUG
#define DBGLOCUS 2

char *ModelStr[3] = {"M0", "DiscreteBeta", "SIM3s"};
//char *GtreeStr[18] = {"Gk", "G1b", "G1a", "G2c", "G2b", "G2a", "G3c", "G3b", "G3a", "G4c", "G4b", "G4a", "G5c", "G5b", "G5a", "G6c", "G6b", "G6a"};
char * GtreeStr[35] = {"111G1(3)", "111G2(3)", "111G3(3)", "111G4(3)", "111G5(3)", "111G6(3)", "222G1(3)", "222G2(3)", "222G3(3)", "222G4(3)", "222G5(3)", "222G6(3)", "112G2(1)", "112G3(1)", "112G4(3)", "112G5(3)", "112G6(3)", "221G2(1)", "221G3(1)", "221G4(3)", "221G5(3)", "221G6(3)", "113G3(1)", "113G5(1)", "113G6(3)", "223G3(1)", "223G5(1)", "223G6(3)", "123G5(1)", "123G6(3)", "133G3&5(1)", "133G6(3)", "333G1&2&4(3)", "333G3&5(3)", "333G6(3)"};
char * stateStr[36] = {"111", "112", "113", "121", "122", "123", "131", "132", "133", "211", "212", "213", "221", "222", "223", "231", "232", "233", "311", "312", "313", "321", "322", "323", "331", "332", "333", "11", "12", "13", "21", "22", "23", "31", "32", "33"};
//                            0  1  2   3   4  5   6   7  8   9  10  11  12 13 14  15  16 17  18  19  20  21  22  23  24  25 26
const int initStateMap[27] = {0, 0, 0, -1, -1, 1, -1, -1, 0, -1, -1, -1,  1, 1, 2, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 2};
const int GtreeMapM0[35] = { 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1,12,12,12,12,12, 2, 2, 2,14,14,14, 5, 5, 8, 8,26,26,26};
const int GtreeMapM2[38] = { 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0};
const int GtOffsetM0[35] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,13,14,15,16,17,19,20,21,22,23,26,28,29,32,34,35,40,41,44,47,48,50,53};
const int GtOffsetM2[38] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,26,28,29,32,34,35,38,40,41,44,47,48,50,53};
char * paranames[9] = {"theta4", "theta5", "tau0", "tau1", "theta1", "theta2", "theta3"};

#define REALSEQUENCE
#include "treesub.c"

FILE *fout, *frub, *frst, *fpGk;

#ifdef CONTOUR
double lbound1, ubound1, inc1, lbound2, ubound2, inc2;
#endif

int main (int argc, char* argv[])
{
    char VerStr[32] = "Version 3.0, Aug 2015";
    double x[9]={1,1,1,1,1,1}, * space;

    space = (double *) malloc(14*C2*C2*sizeof(double));
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
    printf("3s (%s)\n", VerStr);
    fprintf(fout, "3s (%s)\n", VerStr);
    /*
     ReadSiteCounts(datafile);
     */
#if(1)
    ReadSeqData(fout, com.seqf, com.ratef, com.cleandata);

    noisy = 3;
    Models0123(fout, frub, frst, x, space);
    free(com.Nij);
#else
    Simulation(fout, frub, space);
#endif

    free(space);
    fclose(frst);
    fclose(frub);
    fclose(fout);
    return 0;
}


int GetOptions (char *ctlf)
{
    int iopt,i, nopt=12, lline=4096;
    char line[4096],*pline, opt[32], *comment="*#"; //, *seqerrstr="0EF";
    char *optstr[] = {"seed", "nloci", "outfile", "seqfile", "ratefile",
        "cleandata", "npoints", "getSE", "Small_Diff", "simmodel", "models", "Imapfile"};
    double t=1;
    int m[3] = {0, 0, 0}, runm[3] = {0, 0, 0}, nm = 0;
    FILE  *fctl=gfopen (ctlf, "r");

    com.ncode = 4;
    com.npoints = 16;
    com.ncatBeta = 5;
    com.UseMedianBeta = 0;   /* 1 to use the median */
    com.cleandata = 1;
    com.fix_locusrate = 0;
    com.simmodel = 0; /* use asymmetric model by default */
    com.runmodels[0] = com.runmodels[1] = com.runmodels[2] = 1; /* run all models by default */
    strcpy(com.Imapf, ""); /* by default, use */
    memset(com.initials, 0, 192);
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
                if (strncmp(opt, optstr[iopt], 8)==0)  {
                    if (noisy>=9)
                        printf ("\n%3d %15s | %-20s %6.2f", iopt+1,optstr[iopt],opt,t);
                    switch (iopt) {
                        case ( 0): com.seed=(int)t;                    break;
                        case ( 1): com.ndata=(int)t;
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
                        case (10): nm = sscanf(pline+1, "%d %d %d", &m[0], &m[1], &m[2]); break;
                        case (11): sscanf(pline+1, "%s", com.Imapf);   break;
                    }
                    break;
                }
            }

            if (iopt==nopt)
            { printf ("\noption %s in %s\n", opt, ctlf);  exit (-1); }
        }

        fclose(fctl);
        
        if (nm > 0) {
            for (i = 0; i < nm; i++) {
                if (m[i] > 2) {
                    error2("\noption models: unknown model\n");
                }
                runm[m[i]] = 1;
            }
            for (i = 0; i < 3; i++) {
                com.runmodels[i] = runm[i];
            }
        }
    }
    else
        if (noisy) error2("\nno ctl file..");

    fctl = fopen("in.3s", "r");
    if (fctl) {
        for (i=0; i<3; i++) {
            if (!fgets(line, lline, fctl)) {
                break;
            }
            sscanf(line, "%lf%lf%lf%lf%lf%lf%lf%lf%lf", &com.initials[i][0], &com.initials[i][1], &com.initials[i][2], &com.initials[i][3], &com.initials[i][4], &com.initials[i][5], &com.initials[i][6], &com.initials[i][7], &com.initials[i][8]);
        }
    }

    return(0);
}

int ReadSeqData(FILE*fout, char seqfile[], char ratefile[], int cleandata)
{
    /* Read sequences at each locus and count sites.
     All sites with ambiguities are deleted right now.  cleandata is ignored.
     */
    FILE *fin = gfopen(seqfile,"r"), *fImap = NULL;

    int i,j, k, h, *n, im[3], tmp[3], maxID, minID, initState, Imap[9], nind = 0;
    double mr, mNij[5]={0};
    unsigned char *pz[3];
    char newOrder[3], *pline, Inames[10][32], curname[32];
    
    if(com.Imapf[0] != '\0') {
        printf("\nReading Individual-Species map (Imap) from %s\n", com.Imapf);

        fImap = gfopen(com.Imapf, "r");
        for(i=0; i<9; i++) { // we allow at most three different labels per species
            if(fscanf(fImap, "%s %d", Inames[i], &Imap[i]) != 2) break;
            if(strstr(Inames[i], "//")) break;

            if(Imap[i] < 1 || Imap[i] > 3) {
                printf("\nspecies %d in map file is out of place (only species 1, 2 and 3 allowed).\n", Imap[i]);
                exit(-1);
            }
        }
        printf("Individual -> Species map: ");
        nind=i;
        for(i=0; i<nind; i++)
            printf(" %d", Imap[i]);
        fputc('\n', F0);
    }
    
    printf("\nReading sequence data..  %d loci\n", com.ndata);
    if((com.Nij=(int*)malloc(com.ndata*5*sizeof(int)))==NULL) error2("oom");
    memset(com.Nij, 0, com.ndata*5*sizeof(int));
    if((com.initState=(int*)malloc(com.ndata*sizeof(int)))==NULL) error2("oom");
    memset(com.initState, 0, com.ndata*sizeof(int));
    if((com.chain=(int*)malloc(com.ndata*sizeof(int)))==NULL) error2("oom");
    memset(com.chain, 0, com.ndata*sizeof(int));
    memset(com.iStateCnt, 0, 27*sizeof(int));

    if((com.lnLmax=(double*)malloc(com.ndata*(1+com.fix_locusrate)*sizeof(double)))==NULL)
        error2("oom lnLmax");
    if(com.fix_locusrate) com.locusrate = com.lnLmax + com.ndata;
    
    for(i=0,n=com.Nij; i<com.ndata; i++,n+=5) {
        fprintf(fout, "\n\n*** Locus %d ***\n", i+1);
        ReadSeq (NULL, fin, cleandata, -1);

        PatternWeightJC69like (fout);

        // process sequence names
        minID = 4; maxID = 0;
        for(j = 0; j < com.ns; j++) {
            pline = strchr(com.spname[j], IDSEP);
            if(pline == NULL) error2("sequences must be tagged by population ID");
            sscanf(pline+1, "%s", curname);
            if (com.Imapf[0] == '\0') {
                im[j] = atoi(curname);
                if (im[j] < 1 || im[j] > 3) {
                    printf("unknown individual ID: %d. Must be 1, 2 or 3 unless mapping file is used.\n", im[j]);
                    exit(-1);
                }
            } else {
                for (k = 0; k < nind; k++) {
                    if (strcmp(curname, Inames[k]) == 0) {
                        break;
                    }
                }
                if(k==nind) {
                    printf("Individual label %s not recognised.", curname);
                    error2("Please fix the Imap file.");
                }
                else {
                    im[j] = Imap[k];  /* sequence i is individual ind and species is. */
                }
            }
            if(minID > im[j]) minID = im[j];
            if(maxID < im[j]) maxID = im[j];
        }

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

            if(maxID < 3) { // chain 1 or 2
                if(maxID == minID)
                    com.chain[i] = C1;

                else
                    com.chain[i] = C2;
            } else {
                if(initState == 14 || initState < 6) // chain 3
                    com.chain[i] = C3;
                else // chain 4
                    com.chain[i] = C4;
            }

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
            //printf("%5d: %5d%5d%5d%5d%5d\n", i+1, n[0],n[1],n[2],n[3],n[4]);
            printf("%5d: %5d%5d%5d%5d%5d%9d\r", i+1, n[0],n[1],n[2],n[3],n[4], n[0]+n[1]+n[2]+n[3]+n[4]);
        } else { // 2 sequences
            if (im[0] > im[1]) {
                pz[0] = com.z[0];
                tmp[0] = im[0];
                com.z[0] = com.z[1];
                im[0] = im[1];
                com.z[1] = pz[0];
                im[1] = tmp[0];
            }
            initState = 27 + (im[0]-1)*3 + (im[1]-1);

            for (h=0;  h<com.npatt; h++) {
                if (com.z[0][h] == com.z[1][h]) {
                    n[0] = (int)com.fpatt[h];
                } else {
                    n[1] = (int)com.fpatt[h];
                }
            }
            printf("%5d: %5d%5d%5d\r", i+1, n[0], n[1], n[0]+n[1]);
        }

        if(noisy>=3) printf("\n");
        com.initState[i] = initState;
        com.iStateCnt[initState]++;


        //        for(j=0; j<5; j++)
        //            mNij[j] += (double)n[j]/(com.ndata*com.ls);
    }
    free(com.fpatt);
    for(i=0; i<com.ns; i++) {
        free(com.spname[i]);
        free(com.z[i]);
    }
    fclose(fin);

    com.twoSeqLoci = com.iStateCnt[27] + com.iStateCnt[28] + com.iStateCnt[29] + com.iStateCnt[31] + com.iStateCnt[32] + com.iStateCnt[35];
    
    printf("\n\ninit state\t #loci\n");
    for (i=0; i<36; i++) {
        if (com.iStateCnt[i] > 0) {
            printf("    %s   \t%6d\n", stateStr[i], com.iStateCnt[i]);
        }
    }

    //    printf("\n\nmean Nij: %8.4f %8.4f %8.4f %8.4f %8.4f\n", mNij[0],mNij[1],mNij[2],mNij[3],mNij[4]);

    if(com.fix_locusrate) {
        if((fin = gfopen(ratefile, "r")) == NULL)
            error2("ratefile open error");
        for(i=0,mr=0; i<com.ndata; i++) {
            if(fscanf(fin, "%lf", &com.locusrate[i]) != 1)
                error2("rate file..");
            mr = (mr*i + com.locusrate[i])/(i+1.0);
        }
        fclose(fin);
        for(i=0; i<com.ndata; i++)  com.locusrate[i] /= mr;
        printf("\nRelative rates for %d loci, scaled to have mean 1, will be used as constants\n", com.ndata);
        fprintf(fout, "\n\nRelative rates for %d loci, scaled to have mean 1, will be used as constants\n", com.ndata);
        fprintf(fout, "theta's & tau's are defined using the average rate\n");
    }
    return(0);
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
    int *n = com.Nij + locus_save*5;
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
    int i, *n=com.Nij, n123max;
    double nt, lnL, *f=f0124_locus, b[2], bb[2][2]={{1e-4,1},{1e-4,1}}, e=1e-7;
    double din, dout; //, avgb[2] = {0,0};

    for(locus_save=0; locus_save<com.ndata; locus_save++,n+=5)  {
        if (com.initState[locus_save] < 27) {
            for(i=0,nt=0; i<5; i++)
                nt += n[i];

            /* this max may be too large and is not used. */
            for(i=0,com.lnLmax[locus_save]=0; i<5; i++) {
                if(n[i])
                    com.lnLmax[locus_save] += n[i] * log(n[i]/(double)nt);
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
            com.lnLmax[locus_save] = -lnL*nt - 300;   /* log(10^300) = 690.8 */
        } else { // 2 sequence case
            // TODO: check this is correct
            nt = n[0] + n[1];
            f[0] = n[0] / nt;
            f[1] = n[1] / nt;
            b[0] = f[1];
            ming2(NULL, &lnL, lnLb2seq, NULL, b, bb, space, e, 1);
            com.lnLmax[locus_save] = -lnL*nt - 300;
        }
    }
    return(0);
}

int InitializeGtreeTab() {
    int i;
    struct BTEntry ** gtt = com.GtreeTab;
    // initialize GtreeTab
    if(gtt[0]) free(gtt[0]);
    gtt[0] = (struct BTEntry *) malloc(9*6*sizeof(struct BTEntry)); // 111
    memset(gtt[0], 0, 9*6*sizeof(struct BTEntry));

    gtt[13] = gtt[0]+6;   // 222
    gtt[1] = gtt[13]+6; // 112
    gtt[12] = gtt[1]+6; // 221
    gtt[2] = gtt[12]+6; // 113
    gtt[14] = gtt[2]+6; // 223
    gtt[5] = gtt[14]+6; // 123
    gtt[8] = gtt[5]+6;  // 133/233
    gtt[26] = gtt[8]+6; // 333
    
    // initial states 111/222/112/221
    for(i=0; i < 6; i++) {
        gtt[0][i].nGtrees = 3;
        gtt[13][i].nGtrees = 3;
        gtt[1][i].nGtrees = 3;
        gtt[12][i].nGtrees = 3;
    }

    // initial states 112/122
    if (com.model == M0 || com.model == M1DiscreteBeta) {
        gtt[1][0].nGtrees = gtt[12][0].nGtrees = 0;
        gtt[1][1].nGtrees = gtt[1][2].nGtrees = gtt[12][1].nGtrees = gtt[12][2].nGtrees = 1;
        gtt[1][1].config = gtt[1][2].config = gtt[12][1].config = gtt[12][2].config = 0;
    }

    // initial states 113/223/123
    gtt[2][2].nGtrees = gtt[2][4].nGtrees = gtt[14][2].nGtrees = gtt[14][4].nGtrees = gtt[5][4].nGtrees= 1;
    gtt[2][5].nGtrees = gtt[14][5].nGtrees = gtt[5][5].nGtrees = 3;
    if(com.model == M2SIM3s)
        gtt[5][2].nGtrees = 1;

    // initial states 133/233/333
    gtt[8][2].nGtrees = 1;
    //if(com.model == M0)
    gtt[8][2].config = 2; // topology ((b,c), a)?
    gtt[8][5].nGtrees = gtt[26][0].nGtrees = gtt[26][2].nGtrees = gtt[26][5].nGtrees = 3;

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



int Simulation (FILE *fout, FILE *frub, double space[])
{
    char timestr[96];
    double theta4, theta5, tau0, tau1, mbeta, pbeta, qbeta, tau1beta[5];
    double t[2], b[2], p[5], y, pG0, *dlnL;
    double md12, md13, md23, d12, d13, d23, Ed12, Ed13, EtMRCA, mNij[5], x[5];
#if(0)
    double x0[5] = {0.04, 0.06, 0.06, 0.04, 2.0};
#elif(0)
    double x0[5] = {0.0035, 0.0060, 0.0066, 0.0041, 0};
#elif(1)
    double x0[5] = {0.005, 0.005, 0.006, 0.004, 0}; /* hominoid (BY08) */
#elif(0)
    double x0[5] = {0.01, 0.01, 0.02, 0.01, 0};    /* mangroves (Zhou et al. 2007) */
#endif
    int model=1, nii=1, nloci[]={1000, 100, 10}, ii, i,j, nr=200, ir, locus;

    /*
     printf("input model theta4 theta5 tau0 tau1 q? ");
     scanf("%d%lf%lf%lf%lf%lf", &model, &x0[0], &x0[1], &x0[2], &x0[3], &x0[4]);
     */

    com.ls = 500;
    noisy = 2;
    if(com.fix_locusrate) error2("fix_locusrate in Simulation()?");
    if((dlnL=(double*)malloc(nr*sizeof(double))) == NULL) error2("oom dlnL");
    memset(dlnL, 0, nr*sizeof(double));
    for(ii=0; ii<nii; ii++) {
        com.ndata = nloci[ii];
        printf("\n\nnloci = %d  length = %d\nParameters", com.ndata, com.ls);
        matout(F0, x0, 1, 4+model);
        fprintf(fout, "\n\nnloci = %d  length = %d\nParameters", com.ndata, com.ls);
        matout(fout, x0, 1, 4+model);
        fprintf(frst, "\n\nnloci = %d  length = %d  K = %d\nParameters", com.ndata, com.ls, com.npoints);
        matout(frst, x0, 1, 4+model);
        if((com.lnLmax=(double*)malloc(com.ndata*1*sizeof(double)))==NULL) error2("oom lnLmax");
        if((com.Nij=(int*)malloc(com.ndata*5*sizeof(int)))==NULL) error2("oom");
        theta4=x0[0]; theta5=x0[1]; tau0=x0[2]; tau1=x0[3]; qbeta=x0[4];
        pG0 = 1 - exp(-2*(tau0-tau1)/theta5);
        Ed12 = tau1 + theta5/2 + (1-pG0)*(theta4-theta5)/2;
        Ed13 = tau0 + theta4/2;
        EtMRCA = tau0 + theta4/2 * (1 + (1-pG0)*1./3);
        printf("pG0 = %9.6f  E{tMRCA} = %9.6f\n", pG0, EtMRCA);

        md12=0; md13=0; md23=0;
        for(i=0; i<5; i++) mNij[i]=0;
        for(ir=0; ir<nr; ir++) {
            memset(com.Nij, 0, com.ndata*5*sizeof(int));
            for(locus=0; locus<com.ndata; locus++) {
                if(model==M1DiscreteBeta) {
                    mbeta = x0[3]/x0[2];  pbeta = mbeta/(1-mbeta)*qbeta;
                    if(0)    /* continuous beta */
                        tau1 = tau0*rndbeta(pbeta, qbeta);
                    else {   /* discrete beta */
                        DiscreteBeta(p, tau1beta, pbeta, qbeta, com.ncatBeta, 0);
                        tau1 = tau0*tau1beta[(int)(com.ncatBeta*rndu())];
                    }
                }
                t[0] = rndexp(1.0);
                t[1] = rndexp(1.0);
                if(t[1]<2*(tau0-tau1)/theta5) {   /* G0 */
                    b[0] = tau0 - tau1 - theta5*t[1]/2 + theta4*t[0]/2;
                    b[1] = tau1 + theta5*t[1]/2;
                    p0124Fromb0b1 (p, b);
                }
                else {                            /* G123 */
                    t[1] = rndexp(1.0/3);
                    b[0] = t[0]*theta4/2;
                    b[1] = tau0 + t[1]*theta4/2;
                    p0124Fromb0b1 (p, b);
                    y = rndu();
                    if(y<1.0/3)         { y=p[1]; p[1]=p[2]; p[2]=y; }
                    else if (y<2.0/3)   { y=p[1]; p[1]=p[3]; p[3]=y; }
                }

                for(j=0; j<4; j++)  p[j+1] += p[j];
                for(i=0; i<com.ls; i++) {
                    for (j=0,y=rndu(); j<5-1; j++)
                        if (y<p[j]) break;
                    com.Nij[locus*5+j] ++;
                }

                d12 = (com.Nij[locus*5+2]+com.Nij[locus*5+3]+com.Nij[locus*5+4])/(double)com.ls;
                d13 = (com.Nij[locus*5+1]+com.Nij[locus*5+2]+com.Nij[locus*5+4])/(double)com.ls;
                d23 = (com.Nij[locus*5+1]+com.Nij[locus*5+3]+com.Nij[locus*5+4])/(double)com.ls;
                d12 = -3/4.*log(1 - 4./3*d12);
                d13 = -3/4.*log(1 - 4./3*d13);
                d23 = -3/4.*log(1 - 4./3*d23);
                md12 += d12/(2.0*nr*com.ndata);
                md13 += d13/(2.0*nr*com.ndata);
                md23 += d23/(2.0*nr*com.ndata);
                for(i=0; i<5; i++)
                    mNij[i] += (double)com.Nij[locus*5+i]/(com.ndata*com.ls);
            }

            printf("\nReplicate %3d\n", ir+1);
            fprintf(fout, "\nReplicate %3d\n", ir+1);
            fprintf(frub, "\nReplicate %3d\n", ir+1);

            printf("\nmean Nij: %8.4f %8.4f %8.4f %8.4f %8.4f\n", mNij[0],mNij[1],mNij[2],mNij[3],mNij[4]);
            for(i=0; i<4; i++)
                x[i] = x0[i]*MULTIPLIER*(0.8+0.4*rndu());
            x[4] = x0[4]*(0.8+0.4*rndu());
            x[3] = x0[3]/x0[2];
            dlnL[ir] = Models0123(fout, frub, frst, x, space);

            printf("%3d/%3d %9.3f %ss\n", ir+1, nr, dlnL[ir], printtime(timestr));
            for(i=0; i<5; i++) mNij[i] = 0;
        }

        printf("\nd12 %9.5f = %9.5f d13 d23: %9.5f = %9.5f = %9.5f\n", Ed12, md12, Ed13, md13, md23);

        fprintf(fout, "\n\nList of DlnL\n");
        for(i=0; i<nr; i++) fprintf(fout, "%9.5f\n", dlnL[i]);
        free(com.Nij);  free(com.lnLmax);
    }
    printf("\nTime used: %s\n", printtime(timestr));
    return 0;
}

int GetParaMap() {
    int np = 0;
    // states with one 1 and 2 each
    int s12 = com.iStateCnt[5] + com.iStateCnt[28];
    // states with two or more 1
    int s11 = com.iStateCnt[0] + com.iStateCnt[1] + com.iStateCnt[2]
              + com.iStateCnt[27];
    // states with two or more 2
    int s22 = com.iStateCnt[12] + com.iStateCnt[13] + com.iStateCnt[14]
              + com.iStateCnt[31];
    // states with one 1/2 and 3 each
    int s13 = com.iStateCnt[29] + com.iStateCnt[32];
    // states with two or more 3
    int s3 = com.iStateCnt[8] + com.iStateCnt[17] + com.iStateCnt[26]
             + com.iStateCnt[35];

    memset(com.paraMap, -1, 9*sizeof(int));
    memset(com.paraNamesMap, -1, 9*sizeof(int));
    
    // theta4 is always estimable
    com.paraNamesMap[np] = 0;
    com.paraMap[0] = np++;
    // theta5 is estimable if there are >=2 sequences from species 1/2
    if (s12+s11+s22) {
        com.paraNamesMap[np] = 1;
        com.paraMap[1] = np++;
    }
    // tau0 is always estimable
    com.paraNamesMap[np] = 2;
    com.paraMap[2] = np++;
    // tau1 is estimable if there are >=2 sequences from species 1/2
    if (s12+s11+s22) {
        com.paraNamesMap[np] = 3;
        com.paraMap[3] = np++;
    }
    if (com.simmodel && (s11+s22 || (com.model == M2SIM3s && s12))) {
        // theta1&2 is estimable if there are >= 2 sequences from species 1 / 2
        com.paraNamesMap[np] = 4;
        com.paraMap[5] = com.paraMap[4] = np++;
    } else if (!com.simmodel) {
        // theta1 is estimable if there are >= 2 sequences from species 1
        if (s11 || (com.model == M2SIM3s && s22+s12)) {
            com.paraNamesMap[np] = 4;
            com.paraMap[4] = np++;
        }
        // theta2 is estimable if there are >= 2 sequences from species 2
        if (s22 || (com.model == M2SIM3s && s11+s12)) {
            com.paraNamesMap[np] = 5;
            com.paraMap[5] = np++;
        }
    }
    // theta3 is estimable if there are >= 2 sequences from species 3
    if (s3) {
        com.paraNamesMap[np] = 6;
        com.paraMap[6] = np++;
    }
    // beta parameter is always estimable under M1
    if (com.model == M1DiscreteBeta) {
        com.paraNamesMap[np] = 7;
        com.paraMap[7] = np++;
    // M1&2 is estimable if there are >=2 sequences of species 1/2
    } else if (com.model == M2SIM3s && s11 + s12 + s22) {
        com.paraNamesMap[np] = 7;
        com.paraMap[8] = com.paraMap[7] = np++;
        // M21 is estimable if model is not symmetric
        if (!com.simmodel) {
            com.paraNamesMap[np] = 8;
            com.paraMap[8] = np++;
        }
    }
    
    return np;
}


int GetInitials (int np, double x[], double xb[][2])
{
    int i, np0;
    double thetaU=1.99*MULTIPLIER, tmp, MU=9.99;//0.15;  /* MU = 0.125 should be fine */

    for(i=0; i<np; i++)  { xb[i][0]=LBOUND*MULTIPLIER;  xb[i][1]=thetaU; }
    if (com.paraMap[3] != -1) {
        xb[com.paraMap[3]][1] = 0.999;  /* xtau */
        tmp = x[com.paraMap[3]];
    }
    //    for(i=0; i<np; i++) x[i] = 0.1+rndu();
    if (com.model == M0) {
        for(i=0; i<np; i++){
            if (com.initials[com.model][i] != 0) {
                x[i] = com.initials[com.model][i];
            } else {
                x[i] = (0.0010+rndu()/100)*MULTIPLIER;
            }
        }
        if (com.paraMap[3] != -1) {
            if (com.initials[com.model][com.paraMap[3]] != 0) {    /*  xtau  */
                x[com.paraMap[3]] = com.initials[com.model][com.paraMap[3]] / x[com.paraMap[2]];
            } else {
                x[com.paraMap[3]] = 0.4 + 0.5*rndu();
            }
        }
    }

    else if(com.model==M2SIM3s || com.model == M1DiscreteBeta) {
        //        xb[4][0] = 0.001;  xb[4][1] = thetaU;       /* theta12 */
        if (com.paraMap[3] != -1) {
            x[com.paraMap[3]] = x[com.paraMap[3]] / x[com.paraMap[2]]; // convert to xtau1 ahead of changing parameter values
        }

        if (com.paraMap[7] != -1) {
            np0 = np-1;
            if (com.model == M2SIM3s && !com.simmodel) {
                np0 = np-2;
            }
        } else {
            np0 = np;
        }

        for (i=0; i<np0; i++) {
            if (com.initials[com.model][i] != 0) {
                x[i] = com.initials[com.model][i]; // use start parameter
            } else { // use value around previous estimate
                x[i] *= 0.95 + 0.001*rndu()*MULTIPLIER;
            }

            // check boundaries
            if (x[i] <= xb[i][0]) { // estimate at lower boundary
                x[i] = xb[i][0] * (1 + 0.001*rndu());
            } else if (x[i] >= xb[i][1]) { // estimate at upper boundary
                x[i] = xb[i][1] * (1 - 0.001*rndu());
            }
        }
        if (com.paraMap[3] != -1) {
            if (com.initials[com.model][com.paraMap[3]] != 0) {
                x[com.paraMap[3]] = com.initials[com.model][com.paraMap[3]] / x[com.paraMap[2]];
                //        } else if (tmp <= xb[3][0] || tmp >= xb[3][1]) {
                //            x[3] = 0.4 + 0.5*rndu();
                //        } else {
                //            x[3] = x[3]/x[2];
            
            }
        }
        if (com.paraMap[7] != -1) {
            for (i = 7; i < (com.model==M1DiscreteBeta || com.simmodel ? 8 : 9); i++) {
                if (com.initials[com.model][com.paraMap[i]] != 0) {
                    x[com.paraMap[i]] = com.initials[com.model][com.paraMap[i]];
                } else {
                    if(com.model==M1DiscreteBeta) {
                        x[com.paraMap[i]] = 1 + 5*rndu();                     /* qbeta */
                    } else {
#ifdef FIXM12
                        x[com.paraMap[i]] = 0.0001;
#else
                        x[com.paraMap[i]] = 0.01 + 0.1*rndu();                  /* M12 */
#endif
                    }
                }
                if (com.model==M1DiscreteBeta) {
                    xb[com.paraMap[i]][0] = 0.1;                          /* q_beta */
                    xb[com.paraMap[i]][1] = 499;                          /* q_beta */
                } else {
                    xb[com.paraMap[i]][0] = LBOUND;            /* M12 */
                    xb[com.paraMap[i]][1] = MU;
                }
            }
        }
    }
    
    // store initial values to determine which parameters could be estimated
    for (i=0; i<np; i++) {
            com.initials[com.model][i] = x[i];
    }
    
    //    x[0] = 1.00016; x[1] = 2.10741; x[2] = 3.63080; x[3] = 0.4; x[4] = 1.002878; x[5] = 1.8984;

    if(noisy) {
        printf("\nInitials & bounds\n");
        for (i=0; i<np; i++) {
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

double Models0123 (FILE *fout, FILE *frub, FILE *frst, double x[], double space[])
{
    int np, i,j, s, noisy0=noisy, K=com.npoints, nTrees = 38;
    char timestr[96];
    //char * paranames[7] = {"theta4", "theta5", "tau0", "tau1", "theta1&2", "theta3"};
    double *var, lnL, lnL0=0, e=1e-8, M12=0.0, M21=0.0;
    double xb[9][2];

    com.pDclass = (double*)malloc((com.ncatBeta*com.ndata+com.ncatBeta)*sizeof(double));
    if(com.pDclass==NULL) error2("oom Models01231");
    com.tau1beta = com.pDclass + com.ncatBeta*com.ndata;
    s = (com.fix_locusrate ? K*K*2 : K*K*5);  /* 2 for b0 & b1; 5 for p0124 */
    com.bp0124[0] = (double*)malloc((nTrees*s+12*K)*sizeof(double));
    com.wwprior[0] = (double*)malloc((nTrees*3*K*K+12*K)*sizeof(double));
    com.GtreeTab = (struct BTEntry **)malloc(27*sizeof(struct BTEntry *));
    memset(com.GtreeTab, 0, 27*sizeof(struct BTEntry *));
    for(i=1; i<nTrees+1; i++) {
        com.bp0124[i] = com.bp0124[i-1] + s;
        com.wwprior[i] = com.wwprior[i-1] + 3*K*K;
    }

    if(com.bp0124[0]==NULL || com.wwprior[0]==NULL)
        error2("oom Models01232");

    noisy = 0;
    Initialize3s(space);
    noisy = noisy0;
#ifdef M0DEBUG
    for(com.model=0; com.model<1; com.model++) // fix model to M0
#elif defined(M1DEBUG)
        for (com.model=1; com.model < 2; com.model++)
#elif defined(M2DEBUG)
        for(com.model=2; com.model<3; com.model++) // fix model to M2SIM3s
#else
            for(com.model=0; com.model<3; com.model++)
#endif
            {
                if (!com.runmodels[com.model]) {
                    continue;
                }
                
                np = GetParaMap();
                
                if (com.model == M1DiscreteBeta && com.paraMap[3] == -1) {
                    printf("tau1 not estimable. Skipping M1!\n");
                    continue;
                }

                if(com.model==M0) {
                    /* np=6; */
                    com.nGtree=35;
                } else if(com.model==M1DiscreteBeta) {
                    /* np=7; */
                    com.nGtree=35;
                    paranames[7] = "qbeta";
                } else if(com.model==M2SIM3s) {
                    /* np=7; */
                    com.nGtree=38;
                    paranames[7] = "M12";
                    paranames[8] = "M21";
                }
                
                if (com.simmodel) {
                    paranames[4] = "theta1&2";
                    paranames[7] = "M1&2";
                }

                InitializeGtreeTab();

                LASTROUND = 0;
                if(noisy) printf("\n\n*** Model %d (%s) ***\n", com.model, ModelStr[com.model]);
                if(fout) {
                    fprintf(fout, "\n\n*** Model %d (%s) ***\n", com.model, ModelStr[com.model]);
                    fprintf(frub, "\n\n*** Model %d (%s) ***\n", com.model, ModelStr[com.model]);
                }

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
                if(com.getSE <= 1)
                    ming2(frub, &lnL, lfun, NULL, x, xb, space, e, np);

                if(com.model==M0) lnL0 = lnL;
                LASTROUND = 2;
                
                if (com.paraMap[3] != -1) {
                    x[com.paraMap[3]] *= x[com.paraMap[2]];     /* xtau1 -> tau1 */
                }
                
                if(noisy) {
                    printf("\nlnL  = %12.6f  (%5d lfun calls)\n", -lnL, NFunCall);
                    if (com.model != M0) {
                        printf("2DlnL = %+12.6f\nMLEs\n    ", 2*(lnL0-lnL));
                    }
                    printf("MLEs\n");
                    for (i=0; i<np; i++) {
                        printf(" %9s", paranames[com.paraNamesMap[i]]);
                    }
                    printf("\n");
//                    if(com.model==M0) {        printf("MLEs\n    theta4    theta5      tau0      tau1  theta1&2    theta3\n");
//                    else {
//                        printf("2DlnL = %+12.6f\nMLEs\n    ", 2*(lnL0-lnL));
//                        if(com.model==M1DiscreteBeta)  printf("theta4    theta5      tau0      tau1  theta1&2    theta3     qbeta\n");
//                        if(com.model==M2SIM3s)         printf("theta4    theta5      tau0      tau1  theta1&2    theta3      M1&2\n");
//                    }
                    for(i=0; i<np; i++)    printf(" %9.6f", x[i]);
                }
                if(fout) {
                    fprintf (fout, "\nlnL  = %12.6f\n", -lnL);
                    if (com.model != M0) {
                        fprintf(fout, "2DlnL = %+12.6f\nMLEs\n    ", 2*(lnL0-lnL));
                    }
                    fprintf(fout, "MLEs\n");
                    for (i=0; i<np; i++) {
                        fprintf(fout, " %9s", paranames[com.paraNamesMap[i]]);
                    }
                    fprintf(fout, "\n");
//                    if(com.model==M0)        fprintf(fout, "MLEs\n    theta4    theta5      tau0      tau1  theta1&2    theta3\n");
//                    else {
//                        fprintf(fout, "2DlnL = %+12.6f\nMLEs\n    ", 2*(lnL0-lnL));
//                        if(com.model==M1DiscreteBeta)  fprintf(fout, "theta4    theta5      tau0      tau1  theta1&2    theta3     qbeta\n");
//                        if(com.model==M2SIM3s)         fprintf(fout, "theta4    theta5      tau0      tau1  theta1&2    theta3      M1&2\n");
//                    }
                    for(i=0; i<np; i++)    fprintf(fout, " %9.6f", x[i]);
                    fprintf(fout, "\n");
                    fflush(fout);
                }

                // call lfun to compute gene tree posteriors
                lfun(x,np);

                LASTROUND = 1;

                if(fout && com.getSE) {
                    Hessian (np, x, lnL, space, var, lfun, var+2*np*np);
                    matinv(var, np, np, var+np*np);

                    printf("\nSEs:\n");
                    for(i=0;i<np;i++) printf(" %9.6f",(var[i*np+i]>0.?sqrt(var[i*np+i]):-1));
                    printf("\n");
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

                if(com.model==M2SIM3s && (com.iStateCnt[5]+com.iStateCnt[28]) && !(com.iStateCnt[0]+com.iStateCnt[13]+com.iStateCnt[1]+com.iStateCnt[12]+com.iStateCnt[2]+com.iStateCnt[14]+com.iStateCnt[27]+com.iStateCnt[31])) {
                    for(i=0; i<np; i++) fprintf(frst, "\t%.6f", x[i]);
                    printf("\nThe other local peak is at\n");
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
                    for(i=0; i<np; i++) printf(" %9.6f", x[i]); FPN(F0);
                    printf(" %12.6f\n", -lnL);
                    for(i=0; i<np; i++) fprintf(fout, " %9.6f", x[i]); FPN(fout);
                    fprintf(fout, " %12.6f\n", -lnL);
                    fprintf(frst, "\t%.6f\t%.6f\t%.6f\n", 2*(lnL0-lnL), x[4], x[6]);
                }
                fflush(frst);

                printf("\nTime used: %s\n", printtime(timestr));
            }  /* for(com.model) */

    free(com.pDclass);
    free(com.bp0124[0]);
    free(com.wwprior[0]);
    return(2*(lnL0-lnL));
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
