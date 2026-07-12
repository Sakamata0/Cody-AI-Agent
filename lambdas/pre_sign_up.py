"""
Pre Sign-Up Lambda — auto-confirms users and verifies email.

This is required for the passwordless OTP flow to work.
Without this, new users stay in UNCONFIRMED state and can't authenticate.

Trigger: Cognito User Pool → Pre sign-up
"""


def lambda_handler(event, context):
    # Auto-confirm the user so they can immediately use custom auth
    event['response']['autoConfirmUser'] = True

    # Auto-verify email (since we verify via OTP anyway)
    event['response']['autoVerifyEmail'] = True

    return event
