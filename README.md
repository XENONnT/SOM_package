# SOM_package

*** NOTE ***

Train.py is still under development. Do not use these functions!!!

This package is in development but for now it has the basic tools we need to do the following:
- Make data into datacubes that can be processed and which can construct an SOM out of it. (Data_to_NeuroScope_format)
- Recall the resulting datacube for analysis (SOM_recall file)
- Make plots (Plotting)
- Recall data using an existing datacube (SOM_recall)

In the future I intend to add functionality to train an SOM with a variaety of choices and improving/expanding upon the plotting functions which are currently available.

This package is evolving/diverging for its current form, I will write here some of the implicit assumptions that I will be making as I build this code, these things might need to be revised and changed in the future:

- I am assuming a lot of memory is available and creating very large arrays, this should help reduce some overhead but might not be worth it since it could be to demanding for most systems to run. Need to run some tests to get a feeling of how much overhead this will cause. Maybe having a low memory more will be a good idea?