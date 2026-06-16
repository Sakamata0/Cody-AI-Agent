import json
import os
import boto3
from botocore.exceptions import NoCredentialsError


def load_dotenv():
    # Load local environment variables without adding an external dependency.
    # This lets us keep the Bedrock API key in `.env` during local development.
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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

    # Extract and print the generated assistant message from the Bedrock response.
    print(response["output"]["message"]["content"][0]["text"])


if __name__ == "__main__":
    try:
        main()
    except NoCredentialsError:
        print(
            "AWS credentials were not found. Configure an AWS SSO profile or run this "
            "script from an AWS environment with an attached IAM role."
        )
        raise SystemExit(1)
