# SHREC 2021: Retrieval of Cultural Heritage Objects

This is the official repository of the paper "SHREC 2021: Retrieval of Cultural Heritage Objects" by Sipiran, Lazo, Lopez, Jimenez, et al., which will be published in the Computers & Graphics journal.

This paper describes the benchmark and results of a shape retrieval contest, where the dataset is composed of cultural heritage objects.

## Setup 
We use Ubuntu 18.04 with Python 3.6. Follow the next instructions to establish the environment.

1. Create an environment
~~~
python3 -m venv shrec2021 --system-site-packages
source shrec2021/bin/activate
~~~

2. Install dependencies
~~~
pip install -r requirements.txt
~~~

## The Track
3D shape analysis has a historical symbiosis with the cultural heritage domain. The list of contributions of computer graphics and graphics technology in cultural heritage applications is long, and it includes 3D digitization, reconstruction, virtual restoration, immersive interfaces, and so on. Most of these techniques rely on having a suitable 3D representation of archaeological objects, making it possible to process and analyze these objects with computational methods effectively. Nevertheless, some contributions of shape analysis and geometry processing in CH applications have been limited by the relatively small number of items we can use in experimentation.

This track presents our initiative to promote the research of 3D shape analysis methods in CH domains. Our dataset contains digitized versions of archaeological objects from pre-Columbian cultures in Peru, with varied geometry and artistic styles. The real artifacts are in the Josefina Ramos de Cox (JRC) museum in Lima, Perú. The models were scanned as part of a research project funded by the National Agency for Science and Technology in Peru (CONCYTEC) and in conjunction with the Pontifical Catholic University of Peru, which oversees the administration of the mentioned museum.

In this track, we present two retrieval challenges considering two aspects: the shape and the culture. Regarding the shape, archaeologists in the JRC museum classified the scanned objects by shape using specific taxonomies for Peruvian pre-Colombian artifacts. Regarding culture, the JRC museum keeps records of the pre-Colombian cultures to which the artifact belongs. We collected all this metadata for the scanned models, which serves as input for our retrieval tasks.

The proposed challenges have different degrees of complexity. Retrieval by shape is probably the more affordable challenge given that there exist suitable methods in the 3D shape retrieval literature to deal with geometric characterization. Nevertheless, there are cases where the distinction between objects in different classes is barely perceived. On the other hand, retrieval by culture is a more difficult challenge. Models from the same culture can have varied shapes, and probably the most distinguishable characteristic is the combination of geometry and painting style.

## The Challenges

The dataset consists of 3D scanned models from cultural heritage objects captured in the Josefina Ramos de Cox museum in Lima, Perú. The technology used to acquire the 3D models was a structured-light desktop scanner, which produces high-resolution 3D textured models. We applied a post-processing step to normalize the position by translating the objects' center to the origin of 3D space. We also change the orientation manually, such that shapes are oriented up in the Y-axis. We keep the original scale of models because the scale can be a distinctive feature that differentiates objects. Finally, we simplify the triangular mesh of each shape to have nearly 40,000 triangle faces.

**Note**:It is worth to mention that our dataset's objects could contain scanning defects and could be non-manifold.

*   Retrieval by Shape: The dataset consists of 938 objects classified into eight categories: jar, pitcher, bowl, figurine, basin, pot, plate, and vase. We split the dataset into a collection set (70% of the dataset) and a query set (30% of the dataset). The collection set contains 661 objects, and the query set has 277 objects. The class with the highest number of models is bowl (with 221 models in total), and the class with the lowest number of models is vase (with 34 objects in total). Next figure shows examples of models in each class.

<center><img src="https://ivan-sipiran.com/images/shape.png" alt="Test"
	height="200" /> </center>

*   Retrieval by Culture: The dataset consists of 637 objects classified into six categories: Chancay, Lurin, Maranga, Nazca, Pando, Supe. We split the dataset into a collection set (70%of the dataset) and a test set (30% of the dataset). The collection set contains 448 objects, and the test set has 189 objects. The class with the highest number of models is Lurin (with 455 shapes in total), and the class with the lowest number of models is Nazca (with seven shapes in total). Next figure shows examples of objects in each class. The procedure for the competition is the same than in the challenge described above.

<center><img src="https://ivan-sipiran.com/images/culture.png" alt="Test"
	height="200" /> </center>

## Downloads
The data for the challenges can be found in the following links. Each archive contains the collection dataset, the testing dataset, and the classification file for the collection dataset.

*   [Challenge 1](https://drive.google.com/file/d/1E38j-iopOMMzpaRwCRwDKXZpzRptrMga/view?usp=sharing): Retrieval by shape (~2GB)
*   [Challenge 2](https://drive.google.com/file/d/1rxmMABISRWcNqWNWwKnzH6njdtdcwg0v/view?usp=sharing): Retrieval by culture (~1GB)

In this repository we also share the groundtruth (CLA files) to run experiments on our dataset.

# Evaluation tool
We share the evaluation code to run experiments with our dataset. In our evaluation, we use the following metrics:

*   **Mean Average Precision (MAP):** Given a query, its average precision is the average of all precision values computed in each relevant object in the retrieved list. Given several queries, the mean average precision is the mean of average precision of each query.
*   **Nearest Neighbor (NN):** Given a query, it is the precision at the first object of the retrieved list.
*   **First Tier (FT):** Given a query, it is the precision when C objects have been retrieved, where C is the number of relevant objects to the query.
*   **Second Tier (ST):** Given a query, it is the precision when 2*C objects have been retrieved, where C is the number of relevant objects to the query.
*   **Normalized Discounted Cumulative Gain (NDCG)**
*   **Precision-Recall curves**
*   **Confusion Matrix**

The input for the evaluation is a distance matrix of size Nq x Nc, where Nq is the number of objects in the query set and Nc is the number of objects in the collection set. All the evaluation metrics are computed on a single query, and subsequently, we calculate the final measure as the average of the individual measures. Given a query object, we create a ranked list of objects in the collection set according to the query’s increasing distance. Precision is the ratio of retrieved relevant objects concerning a retrieved ranked list. Similarly, recall is the ratio of retrieved relevant objects concerning the complete list of relevant objects.

## Running the evaluation tool
The Python program "experiments.py" allows us to run the evaluation code. This program can be used to create all the figures of the paper. Let's see some examples on how to use this tool:

* Generate overall metrics (precision-recall plot, confusion matrices, text and Latex file), best run per method. The metrics are averaged with respect to the objects.

~~~
> python experiments.py --inputFolder=<input> --outputFolder=<output> --target=datasetShape.cla --query=queryShape.cla --granularity=all
~~~

* Generate overall metrics (precision-recall plot, confusion matrices, text and Latex file), best run per method. The metrics are averaged with respect to the classes.

~~~
> python experiments.py --inputFolder=<input> --outputFolder=<output> --target=datasetShape.cla --query=queryShape.cla --granularity=avg_class
~~~

* Generate metrics per class(precision-recall plot, confusion matrices, text and Latex file), best run per method. The parameter *idGranularity* is the index of the analyzed class.

~~~
> python experiments.py --inputFolder=<input> --outputFolder=<output> --target=datasetShape.cla --query=queryShape.cla --granularity=class --idGranularity=0
~~~

*   To compute the metrics for every run, the command is:

~~~
> python experiments.py --inputFolder=<input> --outputFolder=<output> --target=datasetShape.cla --query=queryShape.cla --granularity=all --methodByMethod
~~~

### Format of input matrices
The name of the text files with the distance matrices should be in the following form: <name_method>.<name_run>.txt
These files should be located in the input folder.

## Citation
If you find this work useful, please cite us:

    @article{SIPIRAN20211,
        title = {SHREC 2021: Retrieval of cultural heritage objects},
        journal = {Computers & Graphics},
        volume = {100},
        pages = {1-20},
        year = {2021},
        issn = {0097-8493},
        doi = {https://doi.org/10.1016/j.cag.2021.07.010},
        url = {https://www.sciencedirect.com/science/article/pii/S0097849321001412},
        author = {Ivan Sipiran and Patrick Lazo and Cristian Lopez and Milagritos Jimenez and Nihar Bagewadi and Benjamin Bustos and Hieu Dao and Shankar Gangisetty and Martin Hanik and Ngoc-Phuong Ho-Thi and Mike Holenderski and Dmitri Jarnikov and Arniel Labrada and Stefan Lengauer and Roxane Licandro and Dinh-Huan Nguyen and Thang-Long Nguyen-Ho and Luis A. {Perez Rey} and Bang-Dang Pham and Minh-Khoi Pham and Reinhold Preiner and Tobias Schreck and Quoc-Huy Trinh and Loek Tonnaer and Christoph {von Tycowicz} and The-Anh Vu-Le},
        keywords = {Benchmarking, 3D model retrieval, Cultural heritage}}