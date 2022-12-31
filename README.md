# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

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
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
