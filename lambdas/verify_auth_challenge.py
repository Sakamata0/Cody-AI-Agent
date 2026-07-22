"""
Verify Auth Challenge Lambda — Cognito Custom Auth Flow.

Compares the user's submitted code against the expected code
stored in privateChallengeParameters.
"""


def lambda_handler(event, context):
    expected_code = event["request"]["privateChallengeParameters"].get("code", "")
    user_code = event["request"]["challengeAnswer"]

    if user_code == expected_code:
        event["response"]["answerCorrect"] = True
    else:
        event["response"]["answerCorrect"] = False

    return event
