"""
auth.py – Authentication routes for Brain Checker AI.
Handles: signup, login, logout, profile fetch, product selection.

Phase 3 fixes:
  - signup: full_name now correctly passed via options.data (Supabase standard)
  - logout: returns clear session instructions for frontend cleanup
  - select-product: also returns pdf_labels for the UI upload gate
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional
import datetime

from backend.config import supabase, supabase_admin, MONTHLY_QUESTION_LIMIT

router = APIRouter(prefix="/api/auth", tags=["auth"])
security = HTTPBearer()


# ─────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────

class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ProductSelectRequest(BaseModel):
    product_slug: str

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    product_id: Optional[int]
    product_slug: Optional[str]
    product_name: Optional[str]
    questions_used: int
    questions_remaining: int


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate JWT token and return Supabase user object."""
    token = credentials.credentials
    try:
        user = supabase.auth.get_user(token)
        if not user or not user.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token.")
        return user.user
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")


def reset_monthly_usage_if_needed(profile: dict) -> dict:
    """Reset question count if it's a new calendar month."""
    reset_date = profile.get("questions_reset")
    if isinstance(reset_date, str):
        reset_date = datetime.datetime.fromisoformat(reset_date.replace("Z", "+00:00"))

    now = datetime.datetime.now(datetime.timezone.utc)
    if reset_date and (now.year > reset_date.year or now.month > reset_date.month):
        supabase_admin.table("profiles").update({
            "questions_used": 0,
            "questions_reset": now.isoformat()
        }).eq("id", profile["id"]).execute()
        profile["questions_used"] = 0

    return profile


def build_user_response(user_id: str, email: str) -> UserResponse:
    """Fetch profile + product and return a unified UserResponse."""
    profile_res = supabase_admin.table("profiles").select(
        "*, products(id, slug, name)"
    ).eq("id", user_id).single().execute()

    profile = profile_res.data or {}
    if profile:
        profile["id"] = user_id          # ensure id present for reset helper
        profile = reset_monthly_usage_if_needed(profile)

    product = profile.get("products") or {}
    questions_used = profile.get("questions_used", 0)

    return UserResponse(
        id=user_id,
        email=email,
        full_name=profile.get("full_name"),
        product_id=product.get("id"),
        product_slug=product.get("slug"),
        product_name=product.get("name"),
        questions_used=questions_used,
        questions_remaining=max(0, MONTHLY_QUESTION_LIMIT - questions_used)
    )


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@router.post("/signup")
async def signup(req: SignupRequest):
    """
    Register a new user with email + password.

    Phase 3 fix: full_name is now passed through options.data so Supabase
    stores it in raw_user_meta_data, which the handle_new_user() trigger
    reads to populate the profiles table automatically.
    """
    try:
        res = supabase.auth.sign_up({
            "email": req.email,
            "password": req.password,
            "options": {
                "data": {"full_name": req.full_name}
            }
        })
        if not res.user:
            raise HTTPException(status_code=400, detail="Signup failed. Please try again.")

        return {
            "status": "success",
            "message": (
                "Account created! Please check your email to verify your account, "
                "then log in."
            ),
            "user_id": res.user.id
        }
    except Exception as e:
        err = str(e).lower()
        if "already registered" in err or "already exists" in err:
            raise HTTPException(
                status_code=409,
                detail="An account with this email already exists."
            )
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login")
async def login(req: LoginRequest):
    """
    Log in and return access token + full user profile.

    The access_token must be stored by the frontend (sessionStorage recommended)
    and sent as 'Authorization: Bearer <token>' on every subsequent request.
    """
    try:
        res = supabase.auth.sign_in_with_password({
            "email": req.email,
            "password": req.password
        })
        if not res.user or not res.session:
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        user_data = build_user_response(res.user.id, res.user.email)

        return {
            "status": "success",
            "access_token": res.session.access_token,
            "refresh_token": res.session.refresh_token,
            "user": user_data.dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        err = str(e).lower()
        if "invalid" in err or "credentials" in err:
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/logout")
async def logout(user=Depends(get_current_user)):
    """
    Log out the current user from Supabase Auth.

    IMPORTANT — frontend logout sequence:
      1. Call DELETE /api/session/{session_id}/chunks  (free database space)
      2. Call POST  /api/auth/logout                   (invalidate token)
      3. Clear sessionStorage on the client

    The chunk cleanup must happen BEFORE this call because after logout
    the token is invalidated and the delete endpoint would reject it.
    """
    try:
        supabase.auth.sign_out()
    except Exception:
        pass   # Even if sign_out fails, we return success so the frontend clears state

    return {
        "status": "success",
        "message": "Logged out successfully."
    }


@router.get("/me")
async def get_me(user=Depends(get_current_user)):
    """Return current user's profile and monthly usage stats."""
    user_data = build_user_response(user.id, user.email)
    return {"status": "success", "user": user_data.dict()}


@router.post("/select-product")
async def select_product(req: ProductSelectRequest, user=Depends(get_current_user)):
    """
    User selects their test/product after login.

    Creates a new session and returns everything the frontend needs
    to build the PDF upload gate and chat dashboard:
      - session_id
      - product.slug, name, required_pdfs, pdf_labels, cards

    Phase 3: pdf_labels is now included in the response so the upload
    gate can render the correct slot labels without a second API call.
    """
    product_res = supabase_admin.table("products").select("*").eq(
        "slug", req.product_slug
    ).single().execute()

    if not product_res.data:
        raise HTTPException(status_code=404, detail="Product not found.")

    product = product_res.data

    # Update profile with selected product
    supabase_admin.table("profiles").update({
        "product_id": product["id"],
        "updated_at": "now()"
    }).eq("id", user.id).execute()

    # Create a new session for this product selection
    session_res = supabase_admin.table("sessions").insert({
        "user_id": user.id,
        "product_id": product["id"]
    }).execute()

    session_id = session_res.data[0]["id"] if session_res.data else None

    return {
        "status": "success",
        "session_id": session_id,
        "product": {
            "id": product["id"],
            "slug": product["slug"],
            "name": product["name"],
            "required_pdfs": product["required_pdfs"],
            "pdf_labels": product["pdf_labels"],     # e.g. ["DMIT Report", "Recommendation Report"]
            "cards": product["cards"],               # e.g. ["understand_report", "career_roadmap"]
            "is_corporate": product["is_corporate"]
        }
    }


@router.get("/products")
async def list_products():
    """
    Return all available products.
    Public endpoint — no auth required.
    Used by the product selector screen.
    """
    res = supabase_admin.table("products").select(
        "id, slug, name, required_pdfs, pdf_labels, cards, is_corporate"
    ).order("id").execute()

    return {"status": "success", "products": res.data or []}


@router.get("/usage")
async def get_usage(user=Depends(get_current_user)):
    """Return current user's question usage for the current month."""
    profile_res = supabase_admin.table("profiles").select(
        "questions_used, questions_reset"
    ).eq("id", user.id).single().execute()

    profile = profile_res.data or {}
    profile["id"] = user.id
    profile = reset_monthly_usage_if_needed(profile)

    used = profile.get("questions_used", 0)
    remaining = max(0, MONTHLY_QUESTION_LIMIT - used)

    return {
        "status": "success",
        "questions_used": used,
        "questions_remaining": remaining,
        "monthly_limit": MONTHLY_QUESTION_LIMIT
    }
