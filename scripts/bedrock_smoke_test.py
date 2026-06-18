import os

import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv


def main():
    load_dotenv()

    # The region and model ID are configurable so we can switch models/regions
    # without changing the source code.
    region = os.getenv("AWS_REGION", "eu-north-1")
    model_id = os.getenv("BEDROCK_MODEL_ID", "eu.anthropic.claude-haiku-4-5-20251001-v1:0")

    # bedrock-runtime is the AWS service client used for model inference.
    client = boto3.client("bedrock-runtime", region_name=region)

    # Converse is Bedrock's unified chat API. It avoids provider-specific
    # request formats and matches the request style shown in the AWS Playground.
    response = client.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [{"text": "Reply with exactly one sentence: Bedrock is working for Cody."}],
            }
        ],
        inferenceConfig={"maxTokens": 100, "temperature": 0.2},
    )

    # Extract and display the model response along with usage and performance metrics.
    answer = response["output"]["message"]["content"][0]["text"]

    usage = response.get("usage", {})
    metrics = response.get("metrics", {})

    print("\n=== Cody AI Agent Test ===")
    print(f"Model: {model_id}")
    print(f"Region: {region}")
    print()
    print("Response:")
    print(answer)
    print()
    print("Metrics:")
    print(f"Latency      : {metrics.get('latencyMs')} ms")
    print(f"Input Tokens : {usage.get('inputTokens')}")
    print(f"Output Tokens: {usage.get('outputTokens')}")
    print(f"Total Tokens : {usage.get('totalTokens')}")

if __name__ == "__main__":
    try:
        main()
    except NoCredentialsError:
        print(
            "AWS credentials were not found. Configure an AWS SSO profile or run this "
            "script from an AWS environment with an attached IAM role."
        )
        raise SystemExit(1)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        print(f"AWS API error ({error_code}): \n{e.response['Error']['Message']}")
        raise SystemExit(1)
