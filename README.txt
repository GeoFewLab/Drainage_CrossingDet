
Place samples in their own directory inside ./big_geo_data directory. For convenience, sample sub directory should be named "Sample800"

If images are not normalized by default, uncomment the "normalize_images()" function in ./big_geo_data/data_preperation.py

Refer to the linked tutorial for specific data input/compatibility requirements for the faster rcnn model

To Run:
	1. configure your training parameters in config.py
	2. go to ./big_geo_data and run data_preperation.py
	3. go back to root and run engine.py to train the model
	4. run validate.py

Results may be viewed in ./validation_qualitative and ./validation_results directories.

For comparison with spatial analysis traditional method of culvert detection.Run the analyze_val in the spatial analysis folder.

Feel free to contact me with any questions.



Initially model was built using the following tutorial:
https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/

