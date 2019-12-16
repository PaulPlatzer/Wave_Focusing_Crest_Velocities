####################
-------README-------
####################
Written by Paul Platzer in 2019. contact: paul.platzer@imt-atlantique.fr

Description:
  This file explains how to use "Fig6_data.py", "Fig6_plot.py", "Fig7_data.py",
  "Fig7_plot", "Fig8_data.py", "Fig8_plot.py", "NonLin_Init_Integr.py", "Gauss.py"
  "RK4.py" and "NLSE.py" in order to produce the figures   of the scientific article
  "Wave group focusing in the ocean: estimations using crest velocities and a
  Gaussian linear model".

Note:
  'amp_choice' is proportional (but not equal) to $k_0A_f^{L}$ in the notations
  of the article.

----------------------------------------------------------------------------------

-------------
'Fig6_data.py'
-This script must be run in the folder containing 'NonLin_Init_Integr.py'.
-It takes only a few minutes to run on a laptop.
-This script uses up to approximately 11Go of RAM.
-The value of 'amp_choice' must be set inside the code
'NonLin_Init_Integr.py' before running 'Fig6_data.py'. Thus 6 independent runs
are needed to generate all the data for the 6th figure of the article.

-------------
'Fig6_plot.py'
-This script can be executed independently.
-It runs very fast on a laptop.
-It loads the 6 out files generated with "Fig6_data.py" and plots the sixth
figure of the article.

-------------
'Fig7_data.py'
-This script must be run in the folder containing 'NonLin_Init_Integr.py'.
-It takes only a few minutes to run on a laptop.
-The value of 'amp_choice' must be set inside the code
'NonLin_Init_Integr.py' before running 'Fig7_data.py'. Thus 6 independent runs
are needed to generate all the data for the 7th figure of the article.

-------------
'Fig7_plot.py'
-This script can be executed independently.
-It runs very fast on a laptop.
-It loads the 6 out files generated with "Fig7_data.py" and plots the seventh
figure of the article.

-------------
'Fig8_data.py'
-This script must be run in the folder containing 'NonLin_Init_Integr.py'.
-This script uses up to approximately 11Go of RAM.
-It can take a few tens of hours to run on a laptop, depending on the value of
"N8" that is set (line 210). The eigth figure of the article was generated
using N8=176.
-The value of 'amp_choice=1.1' must be set inside the code
'NonLin_Init_Integr.py' before running 'Fig8_data.py'.
-This code generates only one out file.

-------------
'Fig8_plot.py'
-This script can be executed independently.
-It runs very fast on a laptop.
-It loads the out file generated with "Fig8_data.py" and plots the eigth
figure of the article.
