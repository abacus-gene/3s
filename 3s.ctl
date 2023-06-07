               seed = -1

            outfile = out
            seqfile = seq.txt          * sequence alignment file
 *         treefile = 3s.tre           * tree file
           Imapfile = Imap.txt         * map of sequences to species
 *         ratefile = Rate.txt         * for variable rates among loci

              nloci = 53

            usedata = 1                * 1: sequence  2: tree
            verbose = 1                * whether to print the site patterns of all loci to outfile (0 or 1)
      initialvalues = 1                * 0: random  1: around MLEs from M0
           nthreads = 1                * positive integer or -1 (as many threads as possible)
            npoints = 16 5 1           * use 8, 16 or 32 
              getSE = 1
         Small_Diff = 0.5e-9

        speciestree = 0                * 0: species tree fixed  1: estimate species tree
       species&tree = 3  A B C         * the 3rd species is to be the outgroup

             models = 0 2 3            * models to use, 0, 1, 2 or 3
                                       * 0: MSC
                                       * 1: DiscreteBeta
                                       * 2: Isolation-with-Migration (MSC-M)
                                       * 3: Introgression (MSci)

           simmodel = 0                * whether to use the symmetric migration model of Zhu and Yang (2012) (0 or 1)
       GIM_2species = 0                * whether to use the generalised IM model for 2 species (0 or 1)

          migration = 3
                      1 2
                      2 1
                      5 3

      introgression = B 1 3

