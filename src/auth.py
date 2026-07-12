"""
Auth router for the Cody AI Agent — OTP (passwordless) + Google OAuth.

Flow:
1. POST /auth/otp/request — Initiates custom auth, triggers Lambda to send OTP email
2. POST /auth/otp/verify — Verifies OTP code, returns JWT tokens
3. POST /auth/oauth/token — Exchanges Google OAuth code for tokens
4. POST /auth/refresh — Refreshes access token
5. POST /auth/logout — Signs out user

Environment variables required:
- COGNITO_USER_POOL_ID: Cognito User Pool ID
- COGNITO_CLIENT_ID: Cognito App Client ID
- COGNITO_REGION: AWS region
- COGNITO_DOMAIN: Cognito hosted UI domain (for OAuth)
"""

import base64
import json
import os

import boto3
import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.middleware import get_current_user

router = APIRouter()


# --- Pydantic Models ---

class OTPRequestBody(BaseModel):
    email: str


class OTPVerifyBody(BaseModel):
    email: str
    code: str
    session: str


class OAuthCodeRequest(BaseModel):
    code: str
    redirect_uri: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    id_token: str
    refresh_token: str | None = None
    user: "UserInfo"


class UserInfo(BaseModel):
    user_id: str
    email: str


class MessageResponse(BaseModel):
    message: str


class OTPRequestResponse(BaseModel):
    session: str
    message: str


# --- Helpers ---

def _get_cognito_client():
    region = os.environ.get("COGNITO_REGION", "eu-north-1")
    return boto3.client("cognito-idp", region_name=region)


def _get_client_id() -> str:
    client_id = os.environ.get("COGNITO_CLIENT_ID")
    if not client_id:
        raise RuntimeError("COGNITO_CLIENT_ID environment variable is required")
    return client_id


def _get_user_pool_id() -> str:
    pool_id = os.environ.get("COGNITO_USER_POOL_ID")
    if not pool_id:
        raise RuntimeError("COGNITO_USER_POOL_ID environment variable is required")
    return pool_id


def _decode_id_token_payload(id_token: str) -> dict:
    parts = id_token.split(".")
    if len(parts) != 3:
        return {}
    payload = parts[1]
    padding = 4 - len(payload) % 4
    if padding != 4:
        payload += "=" * padding
    try:
        decoded = base64.urlsafe_b64decode(payload)
        return json.loads(decoded)
    except Exception:
        return {}


def _ensure_user_exists(email: str):
    """Create user in Cognito if they don't exist (for first-time OTP login)."""
    cognito = _get_cognito_client()
    client_id = _get_client_id()

    try:
        # Try to sign up with a random password (user won't need it — OTP only)
        # The Pre Sign-Up Lambda auto-confirms the user
        import secrets
        temp_password = f"Tmp{secrets.token_urlsafe(16)}!1"
        cognito.sign_up(
            ClientId=client_id,
            Username=email,
            Password=temp_password,
            UserAttributes=[{"Name": "email", "Value": email}],
        )
        print(f"[AUTH] Created new user: {email}")
    except Exception as e:
        error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
        if error_code == "UsernameExistsException":
            pass  # User already exists, that's fine
        else:
            print(f"[AUTH] Error ensuring user exists: {error_code} - {e}")
            raise


# --- Endpoints ---

@router.post("/otp/request", response_model=OTPRequestResponse)
async def request_otp(body: OTPRequestBody) -> OTPRequestResponse:
    """
    Initiate OTP login. Creates user if needed, then starts custom auth flow.
    Cognito triggers the Create Auth Challenge Lambda which sends the email.
    Returns a session token needed for verification.
    """
    cognito = _get_cognito_client()
    client_id = _get_client_id()

    # Ensure user exists (auto-create for first-time users)
    _ensure_user_exists(body.email)

    # Initiate custom auth
    try:
        response = cognito.initiate_auth(
            ClientId=client_id,
            AuthFlow="CUSTOM_AUTH",
            AuthParameters={
                "USERNAME": body.email,
            },
        )
    except Exception as e:
        error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
        print(f"[AUTH] OTP request error: {error_code} - {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to send verification code. Please try again.",
        )

    session = response.get("Session", "")
    if not session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to initiate authentication.",
        )

    return OTPRequestResponse(
        session=session,
        message="Verification code sent to your email.",
    )


@router.post("/otp/verify", response_model=TokenResponse)
async def verify_otp(body: OTPVerifyBody) -> TokenResponse:
    """
    Verify OTP code. If correct, Cognito issues tokens.
    """
    cognito = _get_cognito_client()
    client_id = _get_client_id()

    try:
        response = cognito.respond_to_auth_challenge(
            ClientId=client_id,
            ChallengeName="CUSTOM_CHALLENGE",
            Session=body.session,
            ChallengeResponses={
                "USERNAME": body.email,
                "ANSWER": body.code,
            },
        )
    except Exception as e:
        error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
        error_msg = getattr(e, "response", {}).get("Error", {}).get("Message", str(e))
        print(f"[AUTH] OTP verify error: {error_code} - {error_msg}")

        if "Invalid session" in error_msg or error_code == "CodeMismatchException":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired code. Please request a new one.",
            )
        if error_code == "NotAuthorizedException":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect code. Please try again.",
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Verification failed. Please try again.",
        )

    # Check if we got tokens (auth complete)
    auth_result = response.get("AuthenticationResult")
    if not auth_result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect code. Please try again.",
        )

    access_token = auth_result.get("AccessToken", "")
    id_token = auth_result.get("IdToken", "")
    refresh_token = auth_result.get("RefreshToken")

    claims = _decode_id_token_payload(id_token)
    user_id = claims.get("sub", "")
    email = claims.get("email", body.email)

    return TokenResponse(
        access_token=access_token,
        id_token=id_token,
        refresh_token=refresh_token,
        user=UserInfo(user_id=user_id, email=email),
    )


@router.post("/oauth/token", response_model=TokenResponse)
async def exchange_oauth_code(request: OAuthCodeRequest) -> TokenResponse:
    """Exchange Google OAuth authorization code for tokens."""
    client_id = _get_client_id()
    client_secret = os.environ.get("COGNITO_CLIENT_SECRET", "")
    cognito_domain = os.environ.get("COGNITO_DOMAIN", "")

    if not cognito_domain:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OAuth not configured",
        )

    token_url = f"{cognito_domain}/oauth2/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "code": request.code,
        "redirect_uri": request.redirect_uri,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    if client_secret:
        credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        headers["Authorization"] = f"Basic {credentials}"

    async with httpx.AsyncClient() as client:
        response = await client.post(token_url, data=data, headers=headers)

    if response.status_code != 200:
        detail = response.json().get("error_description", "Token exchange failed")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)

    tokens = response.json()
    access_token = tokens.get("access_token", "")
    id_token = tokens.get("id_token", "")
    refresh_token = tokens.get("refresh_token")

    claims = _decode_id_token_payload(id_token)
    user_id = claims.get("sub", "")
    email = claims.get("email", "")

    return TokenResponse(
        access_token=access_token,
        id_token=id_token,
        refresh_token=refresh_token,
        user=UserInfo(user_id=user_id, email=email),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(request: RefreshRequest) -> TokenResponse:
    """Refresh access token using refresh token."""
    cognito = _get_cognito_client()
    client_id = _get_client_id()

    try:
        response = cognito.initiate_auth(
            ClientId=client_id,
            AuthFlow="REFRESH_TOKEN_AUTH",
            AuthParameters={"REFRESH_TOKEN": request.refresh_token},
        )
    except Exception as e:
        error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
        if error_code == "NotAuthorizedException":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired. Please sign in again.",
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable",
        )

    auth_result = response.get("AuthenticationResult", {})
    access_token = auth_result.get("AccessToken", "")
    id_token = auth_result.get("IdToken", "")

    claims = _decode_id_token_payload(id_token)
    user_id = claims.get("sub", "")
    email = claims.get("email", "")

    return TokenResponse(
        access_token=access_token,
        id_token=id_token,
        refresh_token=None,
        user=UserInfo(user_id=user_id, email=email),
    )


@router.post("/logout", response_model=MessageResponse)
async def logout(user_id: str = Depends(get_current_user)) -> MessageResponse:
    """Sign out user."""
    cognito = _get_cognito_client()
    user_pool_id = _get_user_pool_id()

    try:
        cognito.admin_user_global_sign_out(
            UserPoolId=user_pool_id,
            Username=user_id,
        )
    except Exception:
        pass

    return MessageResponse(message="Logged out successfully")
