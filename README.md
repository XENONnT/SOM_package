# SOM_package

The package allows users to train a Self-Organizing Map (SOM) and use it for data analysis.

This package is in development but for now it has the basic tools we need to do the following:
- Training a SOM in a user flexible manner allowing them to utliize many different different versions of SOM and allow for a lot of flexibility and costumization. As such we will try to implement the functions for the SOM in a modular style.
- You should be able to use a weight cube trained in any system, import it and use our recall functions as well as other function that will be useful for the data analysis.
- Useful vizualisation methods: Most of the job involved in training an SOM involves visualizing the data in many different ways and deciding where to draw the cluster boundaries, as such we will include many different visualization functions.
- We need a way to draw the cluster boundaries in an easy manner, specially so when training big SOMs so we will try to include tools to quickly draw cluster boundaries and generate images onto which we can draw cluster boundaries.

In the future I intend to add functionality to train an SOM with a variaety of choices and improving/expanding upon the plotting functions which are currently available.

This package is evolving/diverging for its current form, I will write here some of the implicit assumptions that I will be making as I build this code, these things might need to be revised and changed in the future:

- I am assuming a lot of memory is available and creating very large arrays, this should help reduce some overhead but might not be worth it since it could be to demanding for most systems to run. Need to run some tests to get a feeling of how much overhead this will cause. Maybe having a low memory more will be a good idea?

Finally note that the cSOM function is not fully opperational as of yet so I would advise against using it.

If you see/find any bugs feel free to either report the issue or submit a pull request!