# L858R
A public code repository for the paper submitted to PLOS Comp Bio in March 2023.
The TIF files containing fluorescent images of the cells are included at day 7 and day 14 (for instance, "at2-e2-b", "at2-e2-d" and "at2-e2-f" are the three files containing images of the second experiment on EGFR-mutant cells; "at2-t1-b", etc, are the equivalent without the mutation.)
To produce area quantifications, the ImageJ macro must be run on the user's computer. This produces csv files which can be loaded into Python and analysed quantitatively using the file "quant_analysis_for_plos.py". This encompasses all quantitative analysis described in the Results and Appendix sections, except for the quantitative analysis of deformation. Perimeter-to-area ratios were calculated from "perimeter-quant.py" from observations made in the csv file "measurements-t-et.csv", itself calculated by hand-drawing around organoids in Figure 1 using ImageJ.
The simulations discussed in the result section can be run in their entirety from the file "simulation_for_plos.py", and visualised using the file "visualiser.py".

