# Classifier using AWS SageMaker


Project GOAL:
Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. 

![workflow](https://user-images.githubusercontent.com/61661948/210153657-b2e88bd2-b3b2-46d5-9b8f-1559c085e07f.png)

As an ML Engineer, we need to track and coordinate the flow of data (which could be images, models, metrics etc) through these different steps. The goal of this project is not to train an accurate model, but to set up an infrastructure that enables other developers to train such models.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.

## Dataset
The provided dataset is the dogbreed classification dataset.
The project is designed to be dataset independent. All the implementation was done with flexible data loaders. The required file structure is a folder with images splitted in to train test val.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning

The training Jobs on AWS:

![trainingJobsAWS](https://user-images.githubusercontent.com/61661948/210153341-56965935-eec9-42a8-a35b-49a8791d0d72.png)

The two hyperparameters tuned were the learning_rate and the batch_size. As the purpose of this is to replicate this training job over and over with new input of new data understanding the right batch size and learning_rate is curcial because it allows us a proper full use of the GPU that gets charged by the second.
```
hyperparameter_ranges = {
    "learning_rate": ContinuousParameter(0.001, 0.1),
    "batch_size": CategoricalParameter([32, 64, 128, 256, 512]),
}
```

![hpologs](https://user-images.githubusercontent.com/61661948/210153320-74681071-458b-47a6-8e3f-2278bd168c5a.png)
The final hyper-parameters of the best model were:

```
{'_tuning_objective_metric': '"average test loss"',
 'batch_size': '"128"',
 'learning_rate': '0.03087708635920261',
 'sagemaker_container_log_level': '20',
 'sagemaker_estimator_class_name': '"PyTorch"',
 'sagemaker_estimator_module': '"sagemaker.pytorch.estimator"',
 'sagemaker_job_name': '"pytorch-training-2022-12-24-17-08-50-491"',
 'sagemaker_program': '"hpo.py"',
 'sagemaker_region': '"us-east-1"',
 'sagemaker_submit_directory': '"s3://sagemaker-us-east-1-447805070819/pytorch-training-2022-12-24-17-08-50-491/source/sourcedir.tar.gz"'}
```



## Debugging and Profiling
In order to debug and Profile the model I implemented some hooks with the following rules to better understand if the model was overfitting and allow for the best GPU usage

```
rules = [
    Rule.sagemaker(rule_configs.loss_not_decreasing()),
    ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()),
    ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
    Rule.sagemaker(rule_configs.overfit()),
    Rule.sagemaker(rule_configs.overtraining()),
]

profiler_config = ProfilerConfig(
    system_monitor_interval_millis=500, framework_profile_params=FrameworkProfile(num_steps=10)
)
debugger_config = DebuggerHookConfig(
    hook_parameters={"train.save_interval": "100", "eval.save_interval": "10"}
)
```

Then the hooks were implemented on the training script. For example this hook is inside the train method.
```
hook = get_hook(create_if_not_exists=True)
    if hook:
        hook.set_mode(modes.TRAIN)
        
    loss_fn = criterion
    
    if hook:
        hook.register_loss(criterion)
```

### Results
The profiling help me understand that perhaps the number of data loaders and workers could be optimized to a greater number achieving a better usage of the GPU. Furthermore we could also pre-fetch the images for quicker loading time.


## Model Deployment
The endpoint was deployed with a Image Serializer or JPEG format and with a JSON deserializer 

```
role = sagemaker.get_execution_role()
jpeg_serializer = sagemaker.serializers.IdentitySerializer("image/jpeg")
json_deserializer = sagemaker.deserializers.JSONDeserializer()

class ImgPredictor(Predictor):
    def __init__( self, endpoint_name, sagemaker_session):
        super( ImgPredictor, self).__init__(
            endpoint_name,
            sagemaker_session = sagemaker_session,
            serializer = jpeg_serializer,
            deserializer = json_deserializer
        )
        
pytorch_model = PyTorchModel( model_data = model_data_artifacts,
                            role = role,
                             entry_point= inference_path,
                             py_version = "py36",
                             framework_version = "1.6",
                            predictor_cls = ImgPredictor
                            )

predictor = pytorch_model.deploy( endpoint_name=endpoint_name,initial_instance_count = 1, 
                                 instance_type = instance_type)

```

And to run inference on this deployment endpoint we do:
```
filename = "./Australian_cattle_dog_00728.jpg"

with open(filename , "rb") as f:
    payload = f.read()
    response = predictor.predict(payload, initial_args={"ContentType": "image/jpeg"})


```
![endpoint](https://user-images.githubusercontent.com/61661948/210153577-f585449d-f036-49df-87a5-f26c704eaec1.png)


