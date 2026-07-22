"""
Create Auth Challenge Lambda — Cognito Custom Auth Flow.

Generates a 6-digit OTP code and sends it to the user's email via AWS SES.
The code is stored in privateChallengeParameters for verification.
"""

import random
import boto3

ses = boto3.client("ses", region_name="eu-north-1")

# Change this to your verified sender email/domain
SENDER_EMAIL = "noreply@cody.formaa.studio"


def lambda_handler(event, context):
    # Generate 6-digit code
    code = str(random.randint(100000, 999999))

    # Get user email
    email = event["request"]["userAttributes"].get("email", "")

    if not email:
        # Fallback: use the username (which might be the email)
        email = event["userName"]

    # Send email via SES
    try:
        ses.send_email(
            Source=SENDER_EMAIL,
            Destination={"ToAddresses": [email]},
            Message={
                "Subject": {
                    "Data": "Your Cody sign-in code",
                    "Charset": "UTF-8",
                },
                "Body": {
                    "Html": {
                        "Data": f"""
                        <div style="font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 520px; margin: 0 auto; padding: 40px 20px; background: #ffffff;">
                            <div style="text-align: center; margin-bottom: 32px;">
                                <table cellpadding="0" cellspacing="0" border="0" style="margin: 0 auto;">
                                    <tr>
                                        <td style="vertical-align: middle; padding-right: 10px;">
                                            <img src="https://res.cloudinary.com/dfml64rbi/image/upload/v1783515821/cody_qnflxy.png" alt="Cody" width="36" height="36" style="border-radius: 10px; display: block;" />
                                        </td>
                                        <td style="vertical-align: middle;">
                                            <span style="color: #1a1a1a; font-size: 20px; font-weight: 600; font-family: 'Bricolage Grotesque', sans-serif;">Cody</span>
                                        </td>
                                    </tr>
                                </table>
                            </div>
                            
                            <div style="background: #f9fafb; border-radius: 12px; padding: 32px; text-align: center;">
                                <p style="color: #374151; font-size: 15px; margin: 0 0 8px 0;">A sign in attempt has been requested for your Cody account.</p>
                                <p style="color: #374151; font-size: 15px; margin: 0 0 24px 0;">Use the code below to complete your sign in:</p>
                                
                                <div style="display: inline-block; background: #111827; color: #ffffff; padding: 16px 36px; border-radius: 10px; font-size: 32px; font-weight: 700; letter-spacing: 8px; font-family: 'SF Mono', 'Fira Code', monospace;">
                                    {code}
                                </div>
                                
                                <p style="color: #6b7280; font-size: 13px; margin: 24px 0 0 0;">This code expires in 10 minutes.</p>
                            </div>
                            
                            <p style="color: #9ca3af; font-size: 13px; text-align: center; margin: 24px 0 0 0;">
                                If you didn't request this, you can safely ignore this email.
                            </p>
                            
                            <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 32px 0;" />
                            
                            <p style="color: #9ca3af; font-size: 11px; text-align: center; margin: 0;">
                                This code is private. Don't share it with anyone, including people claiming to be from Cody support.
                            </p>
                            <p style="color: #9ca3af; font-size: 11px; text-align: center; margin: 8px 0 0 0;">
                                &copy; 2026 SMARTOVATE LTD — Cody AI Agent
                            </p>
                        </div>
                        """,
                        "Charset": "UTF-8",
                    },
                },
            },
        )
    except Exception as e:
        print(f"Failed to send email to {email}: {e}")
        raise e

    # Store the code for verification (Cognito passes this to verify lambda)
    event["response"]["privateChallengeParameters"] = {"code": code}
    event["response"]["publicChallengeParameters"] = {"email": email}
    event["response"]["challengeMetadata"] = f"OTP-{code}"

    return event
