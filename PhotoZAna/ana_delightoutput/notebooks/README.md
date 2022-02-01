# README.md

Info on notebooks used to analyse Delight-RAIL

- author : Sylvie Dagoret-Campagne
- affiliation : IJCLab/IN2P3/CNRS
- creation date : May 9th 2021
- last update : May 9th 2021

## Data

- For SED and filters analysis input directories **tmp/delight_indata** or  **tmpsim/delight_indata** are required

- For Delight analysis output directories **tmp/delightdata** or  **tmpsim/delight_data** are required

### DC2 data

		tree tmp
		tmp
		├── delight_data
		│   ├── galaxies-fluxredshifts.txt
		│   ├── galaxies-fluxredshifts2.txt
		│   ├── galaxies-gpparams.txt
		│   ├── galaxies-redshiftmetrics-cww.txt
		│   ├── galaxies-redshiftpdfs-cww.txt
		│   ├── galaxies-redshiftpdfs_1.txt
		├── delight_indata
		│   ├── BROWN_SEDs
		│   ├── CWW_SEDs
		│   └── FILTERS
		├── parametersTest.cfg

### Internal Mock data

		tree tmpsim
		tmpsim
		├── delight_data
		│   ├── galaxies-fluxredshifts.txt
		│   ├── galaxies-fluxredshifts2.txt
		│   ├── galaxies-gpparams.txt
		│   ├── galaxies-redshiftmetrics-cww.txt
		│   ├── galaxies-redshiftpdfs-cww.txt
		│   ├── galaxies-redshiftpdfs_1.txt
		├── delight_indata
		│   ├── BROWN_SEDs
		│   ├── CWW_SEDs
		│   └── FILTERS
		├── parametersTest.cfg

### Configuration files

- To access data it is necessary to access to **tmp/parametersTest.cfg** or **tmpsim/parametersTest.cfg**


## notebooks

### Filters

- Analyse the filters and their fits with Gaussian Mixtures by Delight


		Filters/
		├── delight_checkfitfilters.ipynb
		└── delight_filters.ipynb



### SED

- Analyse the SED used by Delight


		SED
		├── delight_fluxredshiftsSEDModel.ipynb
		├── delight_sed.ipynb
		├── delight_sed_and_filters.ipynb



### analysis_input_rail


- When accessing directly directly hdf5 files, no cut is applied


		analysis_input_rail/
		├── Dumphdf5InputFiles.ipynb               : Check directly the content of HDF5 files
		├── Dumphdf5InputFileswithRAIL.ipynb       : Check directly the content of HDF5 using RAIL **utils.py**
		├── README.md

		
		├── calibrateTemplateMixturePriors.ipynb
		├── calibrateTemplatePriors.ipynb
		├── compareFluxRedshiftDataModel.ipynb

		├── no_groupname_test.hdf5
		├── test_dc2_training_9816.hdf5
		├── test_dc2_validation_9816.hdf5
		├── training_100gal.hdf5
		└── validation_10gal.hdf5

		├── tmp -> ../tmp
		├── tmpsim -> ../tmpsim







