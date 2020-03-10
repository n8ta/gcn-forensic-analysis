# Running
1. Assuming you have procmon traces in the json format show in data/our_data/txt/*.txt. Run conversions.py. Your data is now picked
into `data/full.*`. 

2. Train the model. `python train.py`. Your trained model is now present in the `trained` directory.

3. Prepare prediction data. Using conversions.py to prepare another set of data to make predictions on. Change the output name from 'full'
to whatever you want. 

3. Predict on new data. Modify predict_and_graph.py changing the input_data variable to the name of the dataset you want to predict. (From step3)

4. predictions.pkl is created, this contians the predictions (rows = nodes) (columns = classes of data).   