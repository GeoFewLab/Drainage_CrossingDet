
Place samples in their own directory inside ./big_geo_data directory. For convenience, sample sub directory should be named "Sample800"

If images are not normalized by default, uncomment the "normalize_images()" function in ./big_geo_data/data_preperation.py

Run the requirement.txt to install all necessay libraries for FRCNN implementation

To Run:
	1. configure your training parameters in config.py
	2. go to ./big_geo_data and run data_preperation.py
	3. go back to root and run engine.py to train the model
	4. run validate.py

Results may be viewed in ./validation_qualitative and ./validation_results directories.

For comparison with spatial analysis traditional method of culvert detection.
Run the analyze_val in the spatial analysis folder.

Feel free to contact me with any questions.

Dataset is at availabe at https://doi.org/10.6084/m9.figshare.25909453.v1


