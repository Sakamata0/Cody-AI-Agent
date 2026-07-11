"""
JWT verification middleware for FastAPI.

Validates Cognito Access Tokens using cached JWKS public keys.
Provides `get_current_user` dependency that extracts the user_id (sub claim)
from a valid Bearer token.

Environment variables required:
- COGNITO_USER_POOL_ID: Cognito User Pool ID (e.g., eu-north-1_XXXXXXXXX)
- COGNITO_CLIENT_ID: Cognito App Client ID
- COGNITO_REGION: AWS region for the Cognito User Pool (e.g., eu-north-1)
"""

import os
import time

import httpx
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)


class JWTVerifier:
    """
    Verifies Cognito JWTs using cached JWKS public keys.
    Caches keys for 1 hour to reduce network calls.
    """

    CACHE_TTL = 3600  # 1 hour in seconds

    def __init__(self, user_pool_id: str, region: str, client_id: str):
        self.jwks_url = (
            f"https://cognito-idp.{region}.amazonaws.com/"
            f"{user_pool_id}/.well-known/jwks.json"
        )
        self.issuer = (
            f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}"
        )
        self.client_id = client_id
        self._jwks_cache: dict | None = None
        self._cache_time: float = 0

    async def get_jwks(self) -> dict:
        """Fetch and cache JWKS from Cognito endpoint."""
        now = time.time()
        if self._jwks_cache and (now - self._cache_time) < self.CACHE_TTL:
            return self._jwks_cache

        async with httpx.AsyncClient() as client:
            response = await client.get(self.jwks_url)
            response.raise_for_status()
            self._jwks_cache = response.json()
            self._cache_time = now

        return self._jwks_cache

    async def verify_token(self, token: str) -> dict:
        """
        Verify JWT signature, expiry, issuer, and extract claims.

        Returns decoded claims dict on success.
        Raises HTTPException(401) on failure with appropriate detail messages.
        """
        # Decode the token header to get the key ID (kid)
        try:
            unverified_header = jwt.get_unverified_header(token)
        except jwt.exceptions.DecodeError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )

        kid = unverified_header.get("kid")
        if not kid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )

        # Fetch JWKS and find the matching key
        try:
            jwks = await self.get_jwks()
        except httpx.HTTPError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )

        key_data = None
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                key_data = key
                break

        if not key_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )

        # Build the public key from JWK
        try:
            public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key_data)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )

        # Decode and verify the token
        try:
            claims = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                issuer=self.issuer,
                options={
                    "verify_exp": True,
                    "verify_iss": True,
                    "verify_aud": False,
                    "verify_iat": False,
                    "require": ["sub", "exp", "iss"],
                },
            )
        except jwt.ExpiredSignatureError:
            print("[AUTH] Token expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
            )
        except jwt.InvalidIssuerError:
            print(f"[AUTH] Invalid issuer. Expected: {self.issuer}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
        except jwt.InvalidTokenError as e:
            print(f"[AUTH] Invalid token error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )

        # Ensure sub claim is present
        if "sub" not in claims:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )

        return claims


def _get_verifier() -> JWTVerifier:
    """
    Create and return a JWTVerifier instance configured from environment variables.

    Raises RuntimeError if required environment variables are not set.
    """
    user_pool_id = os.environ.get("COGNITO_USER_POOL_ID")
    client_id = os.environ.get("COGNITO_CLIENT_ID")
    region = os.environ.get("COGNITO_REGION")

    if not user_pool_id:
        raise RuntimeError("COGNITO_USER_POOL_ID environment variable is required")
    if not client_id:
        raise RuntimeError("COGNITO_CLIENT_ID environment variable is required")
    if not region:
        raise RuntimeError("COGNITO_REGION environment variable is required")

    return JWTVerifier(user_pool_id=user_pool_id, region=region, client_id=client_id)


# Module-level verifier instance (lazy initialization)
_verifier: JWTVerifier | None = None


def get_verifier() -> JWTVerifier:
    """Get or create the singleton JWTVerifier instance."""
    global _verifier
    if _verifier is None:
        _verifier = _get_verifier()
    return _verifier


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    """
    FastAPI dependency. Extracts and validates Bearer token.
    Returns user_id (Cognito sub claim).

    Raises 401 for:
    - Missing Authorization header
    - Expired tokens
    - Malformed/tampered tokens
    - Invalid signature
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    token = credentials.credentials
    print(f"[AUTH] Verifying token: {token[:20]}...")
    verifier = get_verifier()
    claims = await verifier.verify_token(token)
    print(f"[AUTH] Token valid, user: {claims['sub']}")
    return claims["sub"]
