# sagemaker-stuff

#### Demos
- https://www.youtube.com/watch?v=R0vC31OXt-g
- https://www.youtube.com/watch?v=ym7NEYEx9x4

#### Amazon SageMaker features

- Amazon SageMaker Studio: An integrated machine learning environment where you can build, train, deploy, and analyze your models all in the same application.
- Amazon SageMaker Ground Truth: High-quality training datasets by using workers along with machine learning to create labeled datasets.
- Amazon Augmented AI: Human-in-the-loop reviews
- Amazon SageMaker Studio Notebooks: The next generation of Amazon SageMaker notebooks that include SSO integration, fast start-up times, and single-click sharing.
- Preprocessing: Analyze and pre-process data, tackle feature engineering, and evaluate models.
- Amazon SageMaker Experiments: Experiment management and tracking. You can use the tracked data to reconstruct an experiment, incrementally build on experiments conducted by peers, and trace model lineage for compliance and audit verifications.
- Amazon SageMaker Debugger: Inspect training parameters and data throughout the training process. Automatically detect and alert users to commonly occurring errors such as parameter values getting too large or small.
- Amazon SageMaker Autopilot: Users without machine learning knowledge can quickly build classification and regression models.
- Reinforcement Learning: Maximize the long-term reward that an agent receives as a result of its actions.
- Batch Transform: Preprocess datasets, run inference when you don't need a persistent endpoint, and associate input records with inferences to assist the interpretation of results.
- Amazon SageMaker Model Monitor: Monitor and analyze models in production (endpoints) to detect data drift and deviations in model quality.
- Amazon SageMaker Neo: Train machine learning models once, then run anywhere in the cloud and at the edge.
- Amazon SageMaker Elastic Inference: Speed up the throughput and decrease the latency of getting real-time inferences.

#### Model management
- This demo shows how you can use SageMaker Experiment Management Python SDK to organize, track, compare, and evaluate your machine learning (ML) model training experiments https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-experiments/mnist-handwritten-digits-classification-experiment.ipynb
- Track and evaluate SageMaker experiments using SageMaker Studio (in preview) https://docs.aws.amazon.com/sagemaker/latest/dg/experiments-mnist.html
- Performance: Instance types https://aws.amazon.com/sagemaker/pricing/instance-types/ (ml.p3dn.24xlarge provides up to 96 vCPUs, 8xV100 GPUs, 768 RAM, and 256 GB GPU RAM) and Inference acceleration using Elastic Inference up to 4 FP-32 TFLOPS
- Automate model development: Amazon SageMaker Autopilot with SageMaker Studio (in preview) simplifies the machine learning experience by helping you explore your data and try different algorithms. It also automatically trains and tunes models on your behalf, to help you find the best algorithm. https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-automate-model-development-create-experiment.html
- Auto pilot without SageMaker studio: This notebook, as a first glimpse, will use the AWS SDKs to simply create and deploy a machine learning model https://github.com/awslabs/amazon-sagemaker-examples/blob/master/autopilot/sagemaker_autopilot_direct_marketing.ipynb

#### Monitoring
- Amazon SageMaker Debugger is a new capability of Amazon SageMaker that allows debugging machine learning models. It lets you go beyond just looking at scalars like losses and accuracies during training and gives you full visibility into all the tensors 'flowing through the graph' during training. SageMaker Debugger helps you to monitor your training in near real time using rules and would provide you alerts, once it has detected an inconsistency in the training flow.
https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-debugger/mnist_tensor_plot/mnist-tensor-plot.ipynb
- Amazon SageMaker model monitor: continuously monitors the quality of Amazon SageMaker machine learning models in production. It enables developers to set alerts for when there are deviations in the model quality, such as data drift.
https://aws.amazon.com/blogs/aws/amazon-sagemaker-model-monitor-fully-managed-automatic-monitoring-for-your-machine-learning-models/
  - capture data
  - create a baseline
  - schedule monitoring jobs
  - interpret results

#### Retraining and re-usability
- In order to retrain models in Amazon ML, you would need to create a new model based on your new training data https://docs.aws.amazon.com/machine-learning/latest/dg/retraining-models-on-new-data.html
- With incremental training, you can use the artifacts from an existing model and use an expanded dataset to train a new model. You can use AWS Console and SageMaker API to retrain: https://docs.aws.amazon.com/sagemaker/latest/dg/incremental-training.html
- Notebook with code to perform incremental training. In this example, we will show you how to train an object detector by re-using a model you previously trained in the SageMaker. With this model re-using ability, you can save the training time when you update the model with new data or improving the model quality with the same data: https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/object_detection_pascalvoc_coco/object_detection_incremental_training.ipynb

#### Pricing
Building, training, and deploying ML models is billed by the second, with no minimum fees and no upfront commitments. Pricing within Amazon SageMaker is broken down by on-demand ML instances, ML storage, and fees for data processing in hosting instances. There are 4 factors
  - building
  - processing
  - model training
  - model deployment

If you use SageMaker for model training and deployment using Amazon P3 ML instances. There are multiple instance types that you can use as shows here https://aws.amazon.com/sagemaker/pricing/ but if you prefer using P3 instances then:
- http://d1.awsstatic.com/Amazon_EC2_P3_Infographic.pdf 
- https://aws.amazon.com/ec2/instance-types/p3/
- Amazon EC2 P3 instances deliver high performance compute in the cloud with up to 8 NVIDIA® V100 Tensor Core GPUs and up to 100 Gbps of networking throughput for machine learning and HPC applications. These instances deliver up to one petaflop of mixed-precision performance per instance to significantly accelerate machine learning and high performance computing applications. Amazon EC2 P3 instances have been proven to reduce machine learning training times from days to minutes, as well as increase the number of simulations completed for high performance computing by 3-4x.

Training example: If you are using one instance of ml.p3.8xlarge to train your data and you run it for 100 hours in a month with 1 TB of SageMaker storage, and 10 TB of S3 storage 

| Category | Type | Usage | Unit cost | Estimated cost |
| ----------- | ----------- | ----------- | ----------- |
| Compute | ml.p3.8xlarge | 100 hours | $17.136 | $17.136*100 = $1714 |
| Storage | S3 | 10 TB per month | $0.023 per GB | $235 |
| Storage | SageMaker local storage | 1 TB per month | $0.169 per GB-month | $140 |

Total for this example: $2089

Deployment example (deploying your model to run inferences at scale): After training, you deployed the model on SageMaker that runs all the time and performs real-time inference

| Category | Type | Usage | Unit cost | Estimated Cost |
| ----------- | ----------- | ----------- | ----------- |
| Compute | ml.c5.9xlarge | 724 hours | $2.661 | $2.661*724 = $1915 |
| Storage | S3 | 10 TB per month | $0.023 per GB | $235 |
| Storage | SageMaker local storage | 1 TB per month | $0.169 per GB-month | $160 |
| Data processing | SageMaker | 1 TB per month | $0.016 per GB | $160 |

Total for this example: $2430

Cost effective training at scale using Spot instances: Managed Spot Training with Amazon SageMaker lets you train your machine learning models using Amazon EC2 Spot instances, while reducing the cost of training your models by up to 90%.

Other pricing examples: https://aws.amazon.com/sagemaker/pricing/

#### Scalability 
- Multi-model endpoints: Amazon SageMaker Multi-Model Endpoints provides a scalable and cost effective way to deploy large numbers of custom machine learning models. SageMaker Multi-Model endpoints enable you to deploy multiple models with a single click on a single endpoint and serve them using a single serving container.
  - Use case: Increasingly companies are training machine learning models based on individual user data. For example, a music streaming service will train custom models based on each listener’s music history to personalize music recommendations or a taxi service will train custom models based on each city’s traffic patterns to predict rider wait times. Building custom ML models for each use case leads to higher inference accuracy, but the cost of deploying and managing models increases significantly. These challenges become more pronounced when not all models are accessed at the same rate but still need to be available at all times.
- Endpoint auto-scaling: Amazon SageMaker supports automatic scaling for production variants. Automatic scaling dynamically adjusts the number of instances provisioned for a production variant in response to changes in your workload. https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html

#### Development and CI/CD
- Using Step Functions: https://aws.amazon.com/blogs/machine-learning/automated-and-continuous-deployment-of-amazon-sagemaker-models-with-aws-step-functions/
- PFA for ways to perform CI/CD
- To run an arbitrary script-based program in a Docker container using the Amazon SageMaker Containers, build a Docker container with an Amazon SageMaker notebook instance https://docs.aws.amazon.com/sagemaker/latest/dg/build-container-to-train-script-get-started.html
- To run inferences https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html
- Custom TensorFlow python code: You can use Amazon SageMaker to train and deploy a model using custom TensorFlow code. The Amazon SageMaker Python SDK TensorFlow estimators and models and the Amazon SageMaker open-source TensorFlow containers make writing a TensorFlow script and running it in Amazon SageMaker easier
- Use the following frameworks with SageMaker: https://docs.aws.amazon.com/sagemaker/latest/dg/frameworks.html
