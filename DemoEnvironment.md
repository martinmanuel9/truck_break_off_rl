1. Create IAM User
2. Provide Admin Access
3. Create access key and download from AWS the .csv with this information
3. AWS CLI enable and configure 
```bash
aws configure
```
4. Add access key information
5. Within and or directory (VS Code or IDE) run terminal 
6. Create a new python environment 
```bash
conda create -p myenv python=3.11
```
7. Activate envrionment
```bash
conda activate myenv/
```
8. Create a requirements.txt file for package 
```txt
sagemaker
scikit-learn
pandas
numpy
tensorflow
ipykernel
random
sklearn
```
9. Run the following installation within 'myenv'
```bash
pip install -r requirements.txt
```
10. In AWS create an S3 Bucket
11. 
