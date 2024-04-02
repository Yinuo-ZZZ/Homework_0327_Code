import json
import boto3
import numpy as np
from io import BytesIO
import joblib

s3 = boto3.client('s3')
bucket_name = 'EdgeFaultDetectio-bucket' #create S3 Bucket EdgeFaultDetectio-bucket on server
model_key = 'EdgeFaultDetection/model.joblib'

def load_model_from_s3():
    response = s3.get_object(Bucket=bucket_name, Key=model_key)
    model_str = response['Body'].read()
    model = joblib.load(BytesIO(model_str))
    return model

model = load_model_from_s3()

def lambda_handler(event, context):
    # Parse input data
    data = json.loads(event['body'])
    # data (after prepocessing)
    predictions = model.predict(data)  # predict

    # 返回预测结果
    return {
        'statusCode': 200,
        'body': json.dumps({'predictions': predictions.tolist()})
    }
