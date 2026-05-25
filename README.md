# Masters-Thesis
My MSc thesis on designing and simulating an ion guide system to connect two linear Paul traps at IKS KU Leuven. The first linear Paul trap named STRIPE was made to trap and cool incoming ions from a 10 kV ISOL beam using Doppler laser cooling, while the second linear Paul trap named BICEPS is used as a high precision laser spectroscopy trap.

The repository includes .gem files of the potential array structures used in SIMION 8.1 as well as the accompanying lua code for each workbench. Finally, some of the data analysis code is also provided.

DETAILS:

-> STRIPE-IG.gem: The first potential array geometry used in SIMION to generate the PA# file. Contains the last four STRIPE electrode segments with the first half of the ion guide (extended by 10 mm)

-> STRIPE-IG.lua: Contains the user program for the STRIPE-IG iob file. Used to perform STRIPE extraction simulations utilizing the ion guide.
Some different modes are present depending on the application.

-> STRIPE-IG-ES6-7.fly2 & STRIPE-IG-ES6.fly2: contain the fly2 files, which are used to initialize the ions in the STRIPE-IG.iob. They contain the ions trapped between ES6-7 and at ES6 corresponding to the trapping potentials 11.8 8.8 8.8 11.8 and 9.4 8.8 8.8 11.8 for the last 4 STRIPE electrode segments.

-> IG-BICEPS.gem: The second potential array geometry used in SIMION to generate the PA# file. Contains the second half of the ion guide (extended by 10 mm) and the entire BICEPS trap with three electrode segments.

-> IG-BICEPS.lua: The user program to go along with the IG-BICEPS iob file. This user program is used to fly ions saved from the previous potential array (STRIPE-IG) along the second half of the IG and into the BICEPS trap. However, no direct trapping code is added.

-> IG-BICEPS_injection.lua: The user program to go along with the IG-BICEPS iob file. This user program is used to inject and trap ions in BICEPS, here too the fly2 file can be made by using the STRIPE-IG.lua user program in the STRIPE-IG iob file.
