<!-- SPHINX-START -->
# SOM_package

The package allows users to train a Self-Organizing Map (SOM) and use it for data analysis.

To install this package:

```
git clone https://github.com/RiceAstroparticleLab/SOM_package.git
cd SOM_package
pip install -e .
```

This package is in development but for now, it has the basic tools we need to do the following:
- Training a SOM in a flexible manner allows users to utilize many different versions of SOM and allows for a lot of flexibility and customization. As such we will try to implement the functions for the SOM in a modular style.
- You should be able to use a weight cube trained in any system, import it, and use our recall functions as well as other functions that will be useful for the data analysis.
- Useful visualization methods: Most of the jobs involved in training an SOM involves visualizing the data in many different ways and deciding where to draw the cluster boundaries, as such we will include many different visualization functions.
- We need a way to draw the cluster boundaries easily, especially when training big SOMs so we will try to include tools to quickly draw cluster boundaries and generate images onto which we can draw cluster boundaries.

To see the current documentation, please visit: https://scisom.readthedocs.io/en/latest/modules.html

In the future, I intend to add functionality to train a SOM with a variety of choices and improve/expand upon the plotting functions that are currently available.

This package is evolving/diverging for its current form, I will write here some of the implicit assumptions that I will be making as I build this code, these things might need to be revised and changed in the future:

- I am assuming a lot of memory is available and creating very large arrays, this should help reduce some overhead but might not be worth it since it could be to demanding for most systems to run. Need to run some tests to get a feeling of how much overhead this will cause. Maybe having a low memory more will be a good idea?

Finally note that the cSOM function is not fully opperational as of yet so I would advise against using it.

If you see/find any bugs feel free to either report the issue or submit a pull request!
