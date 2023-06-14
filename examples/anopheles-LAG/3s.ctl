      seed = -1
   outfile = out
   seqfile = seqdata.txt     * sequence alignment file
   treefile = 3s.tre         * tree file
   Imapfile = Imap.txt       * map of sequences to species (necessary if

      nloci = 13196
    npoints = 16 5 1 * use 8, 16 or 32
      getSE = 0
 Small_Diff = 1e-8

    usedata = 1              * 1: sequence  2: tree
     models = 0 2            * models to use, 0, 1, 2, 3
  migration = 3
              1 2
              2 1
              5 3
