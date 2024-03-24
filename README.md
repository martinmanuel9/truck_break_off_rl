# Truck Break Off Example 
This source code provides an example for developing model operations within the MLOps capability 
It is important to note that based on the information provided and the use cases developed this may 
serve as an example and point of departure to enable MLOps in your organization. 

## Assumptions 
- Data operations and access to information has been completed by data engineering team
  - Distributed compute (ECR) configured and enabled
  - SLAs established with data science team
- standard repositories within Github have been connected to AWS CodeStar connection
- Secrets are stored and shared within Github account
- IAM and security has been configured and completed within AWS 
- S3 bucket service has been set up for model registry 
- Lambda layer has been developed and connected 
- AWS Cloud formation
  - Established templates to orchestrate environments and layers within AWS (yml files)
    - Includes all AWS services (Lambda Layers, Github repo connections, secrets)