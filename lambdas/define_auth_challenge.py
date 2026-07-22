"""
Define Auth Challenge Lambda — Cognito Custom Auth Flow.

This Lambda decides what authentication challenge to present.
- If no previous challenge: issue a CUSTOM_CHALLENGE (OTP)
- If previous challenge answered correctly: auth succeeds
- If previous challenge answered wrong: auth fails
"""


def lambda_handler(event, context):
    session = event["request"]["session"]

    if len(session) == 0:
        # No challenges yet — issue OTP challenge
        event["response"]["issueTokens"] = False
        event["response"]["failAuthentication"] = False
        event["response"]["challengeName"] = "CUSTOM_CHALLENGE"
    elif session[-1]["challengeResult"] is True:
        # Last challenge answered correctly — issue tokens
        event["response"]["issueTokens"] = True
        event["response"]["failAuthentication"] = False
    else:
        # Challenge answered wrong — fail
        event["response"]["issueTokens"] = False
        event["response"]["failAuthentication"] = True

    return event
