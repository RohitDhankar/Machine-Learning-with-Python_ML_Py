
- [ADVERSARIAL_AutoEncoder]



- [Semi_Supervised_Learning_SGAN]

#

- [pytorch_torchvision_various_models](https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg)
> Trying Various Pre Trained - Torch Vision models for MNIST and CIFAR basic Transfer Learning - classification tasks 

#


- [1-detectron2-DensePose-CSE-Continuous_Surface_Embeddings](https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/DENSEPOSE_CSE.md#animal-cse-models)
- [1-detectron2-DensePose] 
- [TODO--> VAE_variational_autoencoder]()
- [Number_Plate_license-plate-detection](https://paperswithcode.com/task/license-plate-detection)

#

> Ludwig -- Ludwig is a low-code framework for building custom AI models like LLMs and other deep neural networks.
- [Ludwig.ai](https://ludwig.ai/latest/)

#
- [Vehicle_ID_Through_Traffic_VehicleRear](https://github.com/icarofua/vehicle-rear)
> AUTHORS --Ícaro Oliveira de Oliveira, Rayson Laroca, David Menotti, Keiko Veronica Ono Fonseca, Rodrigo Minetto
Vehicle-Rear: A New Dataset to Explore Feature Fusion For Vehicle Identification Using Convolutional Neural Networks

> two-stream Convolutional Neural Network (CNN) that simultaneously uses two of the most distinctive and persistent features available: the vehicle’s appearance and its license plate. 

#

- [AppleMobile_TuriCreate_SupportVectorMachine_Classifier](https://apple.github.io/turicreate/docs/api/generated/turicreate.svm_classifier.create.html#turicreate.svm_classifier.create)
> Apple Mobile -- SVM with TuriCreate 
Create a SVMClassifier to predict the class of a binary target variable based on a model of which side of a hyperplane the example falls on. In addition to standard numeric and categorical types, features can also be extracted automatically from list- or dictionary-type SFrame columns.
Zhang et al. - Modified Logistic Regression: An Approximation to SVM and its Applications in Large-Scale Text Categorization (ICML 2003)
```python
>>> data =  turicreate.SFrame('https://static.turi.com/datasets/regression/houses.csv')
>>> data['is_expensive'] = data['price'] > 30000
>>> model = turicreate.svm_classifier.create(data, 'is_expensive')
```

#

- [ONNX_Runtime](https://onnxruntime.ai/)
- https://onnxruntime.ai/index.html#getStartedTable
- Train in Python but deploy into a C#/C++/Java app [Deploy_ONNX](https://onnxruntime.ai/docs/)
> Get a model. This can be trained from any framework that supports export/conversion to ONNX format. 
See the tutorials for some of the popular frameworks/libraries.
- https://onnxruntime.ai/docs/api/python/tutorial.html
- [Android_App_ONNX_Runtime](https://onnxruntime.ai/docs/tutorials/on-device-training/android-app.html)

 
```python
import onnxruntime as ort

# Load the model and create InferenceSession
model_path = "path/to/your/onnx/model"
session = ort.InferenceSession(model_path)

# Load and preprocess the input image inputTensor
...

# Run inference
outputs = session.run(None, {"input": inputTensor})
print(outputs)
```
#
```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type)
with open("logreg_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```

#

- [Android_Studio_TensorFlowLite](https://www.tensorflow.org/lite/android/quickstart)
- https://stackoverflow.com/questions/49193985/fastest-way-to-run-recurrent-neural-network-inference-on-mobile-device
- [TensorRt](https://github.com/NVIDIA/TensorRT)
- 

