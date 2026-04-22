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

Adding small test here

## Credit

* This part can go to remap*
The remap module was mimicked by a module of the same name in NeuroScope. This NueroScope remap module has much more sofisticated capabilities, the implementation here is a rudamentary version of this. For more information on NeuroScope please contact [Erzsébet Merényi](https://profiles.rice.edu/faculty/erzsebet-merenyi): 713-348-3595 | [erzsebet@rice.edu](mailto:erzsebet@rice.edu?subject=Subject&body=Text)


Some of the sciSOM functions were mimicked (in a paired-down way) after the same-name modules in NeuroScope. We point this out in the in-line documentation of the respective functions.

NeuroScope is an algorithm development and data analysis environment comprising modules for SOM learning and related capabilities to evaluate the correctness of learning, topology preservation, and visualization and other techniques, for “precision extraction” of clusters from learned SOMs. Relevant to Dark Matter search, the capabilities include preferential detection of (unknown) rare clusters. NeuroScope also contains broader capabilities such as SOM-based precise multi-class supervised classification [1-6]. This environment has been developed and maintained by Prof. Erzsébet Merényi and her group with years of support from NASA and other government and non-government agencies. The capabilities have extensively been used in scientific research projects with demonstrated consistently accurate and reliable results including strong discovery potential from high-dimensional, large data with complex structure [7-9] A subset of NeuroScope capabilities relevant for an astronomy project for clustering hyperspectral ALMA cubes to map regions of different kinematic behavior in astrophysical objects is portrayed in [10-11].

[1] E. Mer\ ́enyi, K. Tasdemir and L. Zhang, Learning highly structured manifolds: Harnessing the power of SOMs, in Similarity-Based Clustering: Recent Developments and Biomedical Applications, M. Biehl, B. Hammer, M. Verleysen and T. Villmann, eds., (Berlin, Heidelberg), pp. 138–168, Springer Berlin Heidelberg (2009), DOI

[2]K. Ta\ ̧sdemir and E. Mer´enyi, Exploiting data topology in visualization and clustering of self-organizing maps, Neural Netw. 20 (2009) 549–562.

[3] K. Ta\ ̧sdemir and E. Mer\ ́enyi, A validity index for prototype based clustering of data sets with complex structures, IEEE Transactions on Systems, Man and Cybernetics, Part B 41 (2011) 1039.

[4] L. Zhang and E. Mer\ ́enyi, Weighted differential topographic function: A refinement of
the topographic function, in Proceedings of the 14th European Symposium on Artificial
Neural Networks (ESANN’2006), (Bruges, Belgium), pp. 13–18, April, 2006.

[5] K. Ta ̧sdemir and E. Mer ́enyi, SOM-based topology visualization for interactive analysis of high-dimensional large datasets, Machine Learning Reports 05 (2012) .

[6] E. Mer\ ́enyi, A. Jain and T. Villmann, Explicit magnification control of self-organizing maps for forbidden data, IEEE Transactions on Neural Networks 18 (2007) 786.

[7] E. Mer\ ́enyi, J. Taylor and A. Isella, Deep data: discovery and visualization application to hyperspectral alma imagery, Proceedings of the International Astronomical Union 12 (2016) 281–290.

[8] E. Mer\ ́enyi and J. Taylor, SOM-empowered graph segmentation for fast automatic clustering of large and complex data, in 2017 12th International Workshop on Self-Organizing Maps and Learning Vector Quantization, Clustering and Data Visualization (WSOM), pp. 1–9, 2017, DOI.

[9] E. Mer ́enyi, R.B. Singer and J.S. Miller, Mapping of spectral variations on the surface of mars from high spectral resolution telescopic images, Icarus 124 (1996) 280–295.

[10] E. Mer ́enyi, et al. NeuroScope: Neural machine intelligence tools for discovery and interpretation in complex ALMA data, Final Report ALMA Study 358232, Rice University (June, 2019).

[11] E. Mer ́enyi, A. Isella and J. Taylor, “NeuroScope for ALMA data.” Rice University,
2019. https://neuroscope.blogs.rice.edu/.

## Disclaimer

Prof. Merényi was not consulted on the implementation of sciSOM functions that intend to mimic NeuroScope functionalities of the same name, nor did she have opportunity to inspect proof of faithfulness to the same-name module in NeuroScope or correctness of the corresponding sciSOM code. Therefore, Dr. Merényi and the NeuroScope group take no responsibility for the likeness and the correctness of the functions implemented to mimic (partial) NeuroScope capabilities in sciSOM.