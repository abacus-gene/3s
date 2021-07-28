            seed = -1
  
         outfile = out
         seqfile = seq.txt          * sequence alignment file
 *      treefile = 3s.tre           * tree file
        Imapfile = Imap.txt         * map of sequences to species
 *      ratefile = Rate.txt         * uncomment this line to run the variable-rates model
  
           nloci = 53
  
         npoints = 16 5 1           * use 8, 16 or 32 
           getSE = 1
      Small_Diff = 0.5e-9
  
          models = 0 2 3            * models to compute, 0, 1, 2 or 3
                                    * 0: M0
                                    * 1: DiscreteBeta
                                    * 2: Isolation-with-Migration
                                    * 3: Introgression
  
     speciestree = 0                * 0: species tree fixed  1: estimate species tree
    species&tree = 3  A B C         * the number of species should always be 3
                                    * the 3rd species is to be the outgroup
  
         usedata = 1                * 1: sequence  2: tree
         verbose = 1                * whether to print the site patterns of all loci to outfile
   initialvalues = 1                * 0: random  1: around MLEs from M0
        nthreads = 1                * positive integer or -1
                                    * -1 means let the program determine the number of threads
  
         est_M12 = 1                * 0: fix to zero  1: estimate
         est_M21 = 1
         est_M13 = 1
         est_M31 = 1
         est_M23 = 1
         est_M32 = 1
         est_M53 = 1
         est_M35 = 1
       est_phi12 = 1
       est_phi21 = 1
       est_phi13 = 0
       est_phi31 = 0
       est_phi23 = 0
       est_phi32 = 0
      est_thetaX = 1
      est_thetaY = 1
      est_thetaZ = 1
  