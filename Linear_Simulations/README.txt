####################
-------README-------
####################
Written by Paul Platzer in 2019. contact: paul.platzer@imt-atlantique.fr

Description:
  This file explains how to use "Fig1.py", "Fig2.py", "Fig4.py", "Fig5_data.py",
  "Fig5_plot_k0Lf*.py" and "Linear_Init_Integr.py" in order to produce the figures
  of the scientific article "Wave group focusing in the ocean: estimations using
  crest velocities and a Gaussian linear model".

Note:
  'width_choice' corresponds to $k_0L_f$ in the notations of the article.

----------------------------------------------------------------------------------

-------------
'Fig1.py'
-This script can be executed independently.
-It runs very fast on a laptop.
-This script uses approximately 7Go of RAM.
-By setting the value of 'plot_time' one can produce the different plots
of the first figure of the article. Three runs are needed to produce the
whole figure.

-------------
'Fig2.py'
-This script can be executed independently.
-It runs very fast on a laptop.
-It plots the second figure of the article.

-------------
'Fig4.py'
-This script must be run in the folder containing 'Linear_Init_Integr.py'.
-It takes only a few minutes to run on a laptop.
-The values of 'width_choice' and 'random_phase' must be set inside the code
'Linear_Init_Integr.py' before running 'Fig4.py'. Thus three independent runs
are needed to reproduce the whole 4th figure of the article.

-------------
'Fig5_data.py'
-This script must be run in the folder containing 'Linear_Init_Integr.py'.
-It takes only a few minutes to run on a laptop.
-It produces the data to be plotted in the 5th figure of the article.
-The values of 'width_choice' and 'random_phase' must be set in the code
'Linear_Init_Integr.py' before running 'Fig5.py'.
-The data is stored to a folder named 'Fig5_data/' which must be created
inside the folder from which the file is executed.
-'Fig5.py' must be run a large number of times in order to generate enough
data to reproduce the fifth figure of the article. For this reason, a simple
shell script called 'Fig5_data_run.sh' was created to run 'Fig5_data.py'
a given number of times.

--------------
'Fig5_plot_k0Lf2'; 'Fig5_plot_k0Lf5'; 'Fig5_plot_k0Lf10'
-Each code loads the files that were previously generated using 'Fig5_data.py'
with related values of 'width_choice'. To plot the data that was actually used
to produce the fifth figure of the article, just comment/uncomment
two lines in the '## Call all the files' subsection (around line 70) of those
python scripts.
-It runs very fast on a laptop.
-For each value of 'width_choice', the three corresponding plots for the fifth
figure of the article are drawn.
