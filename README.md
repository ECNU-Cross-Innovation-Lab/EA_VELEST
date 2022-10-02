# EA_VELEST

Article name: Three-Dimensional Velocity Structure and Earthquake Relocation of the 2022 Maerkang, Sichuan Earthquake Swarm.
This repository shows the code in part 3.1 of the article,which names 3.1VELEST relocation and one dimensional velocity optimization based on EA(Evolutionary Algorithm).

In `EA_VELEST.py`, Function `re_velest` is the application of the VELEST method and the search process of RMS residuals. Function `mk_syn_model` is the process of using EA to iterate the parameters of the one-dimensional velocity model, in which function `re_velest` is nested to implement the VELEST method and iterate the RMS residual。
