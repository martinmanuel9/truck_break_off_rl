

# MLOps Recommendations Full Environment Set-Up
1. Set up IAM Center instance name
2. Ensure that you have aws cli installed
3. Create a connection between Github and AWS
   - create a github personal access token 
   - connect it using AWS connections via Developer tools (https://console.aws.amazon.com/codesuite/settings/connections)
   - Select "Install new app"
   - Follow steps and connect to Github
   - Make sure that the connection status is "available"
4. In AWS Secrets Manger create a new secret 
   - Store new secret: call it github_secret
   - To securely store your GitHub personal access token
5. Generate a IAM user for Github actions
   - Create user
   - generate an access key
6. Enable github Secrets
   - create repository for to manage source code as well as SageMaker
   - Create a new repository secret to include AWS IAM 
   - Select repository
   - select Secrets 
   - select actions
7. Setting Github Environment
   - In github navigate settings and select environment
   - create new environment and call it "Production"
   - In configuration of Production environment select "Required reviewers" and search for reviewers
8. Lambda function for deployment
   - within source code complete the following: 
      ```bash
      cd lambda_functions/lambda_github_workflow_trigger
      zip lambda-github-workflow-trigger.zip lambda_function.py
      ```
    - Upload the lambda-github-workflow-trigger.zip to an S3 bucket 
      ```bash
      aws s3 cp lambda-github-workflow-trigger.zip s3://your-bucket/
      ```
    - set up a python virtual env: 
      ```bash
        mkdir lambda_layer
        cd lambda_layer
        python3 -m venv .env
        source .env/bin/activate
        pip install pygithub
        deactivate
      ```
      - generate zip file
      ```bash
        mv .env/lib/python3.9/site-packages/ python
        zip -r layer.zip python
      ```

      - publish lambda
      ```bash
      aws lambda publish-layer-version \
        --layer-name python311-github-arm64 \
        --description "Python3.11 pygithub" \
        --license-info "MIT" \
        --zip-file fileb://layer.zip \
        --compatible-runtimes python3.11 \
        --compatible-architectures "arm64"
      ```

  9. Create a portfolio 
   - Create portfolio under administration under service catalog
   - Name it "SageMaker Organization Templates"
   - Create a product and call it "build-deploy-github"
   - you may need to update line 83 to update the S3 bucket name:
    ```
      GitHubWorkflowTriggerLambda:
          Type: 'AWS::Lambda::Function'
          Properties:
            Description: To trigger the GitHub Workflow
            Handler: lambda_function.lambda_handler
            Runtime: python3.11
            FunctionName: !Sub sagemaker-${SageMakerProjectId}-github-trigger
            Timeout: 900
            Role: !GetAtt GitHubWorkflowTriggerLambdaExecutionRole.Arn
            Code:
              S3Bucket: martymdlregistry
              S3Key: lambda-github-workflow-trigger.zip
            Layers:
              - !Sub arn:aws:lambda:${AWS::Region}:${AWS::AccountId}:layer:python39-github-arm64:1
            Architectures:
    ```

    - uplodad new product
    - version title = 1.0
    - choose review
    - On the Tags tab, add the following tag to the product:
      ```bash
        Key =sagemaker:studio-visibility
        Value = true
      ```
     - back in the portfolio create a constraint:
       - Select IAM role
       - select "AmazonSageMakerServiceCatalogProductsLaunchRole"
       - Choose add access
  10. SageMaker Set up
