# Use NUAA Photograph Imposter Database Dataset for face spoof Detection

## Steps
1. Download dataset from [here](http://parnec.nuaa.edu.cn/xtan/data/nuaaimposterdb.html)
2. Uncompress the dataset.
3. organize the photos into `train` folder with two subfolders `real` and `fake` 

4. train with tensorflow.hub Inception v3
```
python ~/ws/hub/examples/image_retraining/retrain.py \
	--image_dir train \
	--output_graph model/spoof_graph.pb \
	--output_labels model/spoof_labels.txt \
	--how_many_training_steps 4000
```
5. freeze the model with `strip_unused` and `quantization`
```
python ~/ws/tensorflow/tensorflow/python/tools/strip_unused.py \
	--input_graph=./model/spoof_graph.pb \
	--output_graph=./model/stripped_spoof_graph.pb \
	--input_node_names=Placeholder \
	--output_node_names=final_result \
	--input_binary=true
```
and
```
python ~/ws/tensorflow/tensorflow/tools/quantization/quantize_graph.py \
	--input=model/stripped_spoof_graph.pb \
	--output_node_names=final_result \
	--output=model/quantized_stripped_spoof_graph.pb \
	--mode=weights
```
6. deploy to the mobile tf_classify app and run by changing the following in `ClassifierActivity.java`
```java
private static final int INPUT_SIZE = 299; 
private static final int IMAGE_MEAN = 0; 
private static final float IMAGE_STD = 255; 
private static final String INPUT_NAME = "Placeholder"; 
private static final String OUTPUT_NAME = "final_result"; 
private static final String MODEL_FILE = "file:///android_asset/quantized_stripped_spoof_graph.pb"; 
private static final String LABEL_FILE = 
"file:///android_asset/spoof_labels.txt";  
```