          seed = -1

       seqfile = ChenLi3s.txt     * sequence alignment file
      Imapfile = ChenLi3s.Imap.txt    * map of sequences to species (necessary if
                                      * sequences are not tagged with {1, 2, 3})
*      ratefile = ChenLi3s.Rate.txt   * uncomment this line to run the variable-rates model
       outfile = out.txt

         nloci = 53

       npoints = 16 5 1 * use 8, 16 or 32 
         getSE = 1
    Small_Diff = 0.5e-9
        models = 0 * M0: no gene flow
