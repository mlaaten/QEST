QEST code for inverting a 1-D frequency dependent scattering and intrinsic attenuation model.

mc_lay.f90 needs to be compiled first with

``python -m numpy.f2py -c mc_lay.f90 -m mc_lay``

config.json contains all configuration parameters and needs to be adapted

mc_attdepth_noniso.py is the inversion script, each model tested is saved to the log-file

the first value of the log-file is the root mean square, the first array are the log10(g_star) values, the second array the log10(b) and the last array the log10(epsilon) values of the different layers
