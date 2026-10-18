import json

from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from PIL import UnidentifiedImageError

from . import analysis, ml
from .models import ACRE_IN_HA, Claim

MAX_BYTES = 5 * 1024 * 1024
UNITS = {"acre": ACRE_IN_HA, "ha": 1.0, "guntha": ACRE_IN_HA / 40}
UNIT_CHOICES = [("acre", "Acres"), ("ha", "Hectares"), ("guntha", "Guntha")]
CROPS = ["Onion", "Grapes", "Wheat", "Sugarcane", "Tomato", "Other"]


@login_required
def predict(request):
    result = error = None
    if request.method == "POST":
        upload = request.FILES.get("chip")
        if upload is None:
            error = "Choose an image first."
        elif upload.size > MAX_BYTES:
            error = "The image is larger than 5 MB."
        else:
            try:
                result = ml.predict_image(upload)
            except (UnidentifiedImageError, OSError):
                error = "That file is not a readable image."
            except FileNotFoundError as exc:
                error = str(exc)
    return render(request, "tasks/predict.html", {"result": result, "error": error})


def read_claim_form(post):
    try:
        value = float(post.get("claimed_value", ""))
    except ValueError:
        raise ValueError("Enter the claimed area as a number.") from None
    unit = post.get("claimed_unit", "acre")
    if unit not in UNITS:
        raise ValueError("Choose a unit for the claimed area.")
    if not 0 < value <= 100000:
        raise ValueError("The claimed area must be greater than 0.")
    try:
        geometry = json.loads(post.get("geometry", ""))
    except ValueError:
        raise ValueError("Draw the field boundary on the map first.") from None
    if not isinstance(geometry, dict) or geometry.get("type") != "Polygon" or not geometry.get("coordinates"):
        raise ValueError("Draw the field boundary on the map first.")
    if sum(len(ring) for ring in geometry["coordinates"]) > 2000:
        raise ValueError("The boundary has too many corners.")
    return value, unit, geometry


def areas(ha):
    return {"ha": ha, "acres": None if ha is None else ha / ACRE_IN_HA}


@login_required
def claims(request):
    error = None
    if request.method == "POST":
        try:
            value, unit, geometry = read_claim_form(request.POST)
            claimed_ha = value * UNITS[unit]
            result = analysis.analyse_geometry(geometry, claimed_ha)
            claim = Claim.objects.create(
                owner=request.user, farm_id=request.POST.get("farm_id", "").strip()[:40], farmer_name=request.POST.get("farmer_name", "").strip()[:120],
                crop_type=request.POST.get("crop_type", "").strip()[:60], claimed_area_ha=claimed_ha, entered_value=value, entered_unit=unit,
                geometry=geometry, result=result, priority=analysis.priority(result["flags"]))
            return redirect("claim_detail", claim_id=claim.id)
        except analysis.AnalysisUnavailable as exc:
            error = str(exc)
        except analysis.OutsideStudyArea as exc:
            error = str(exc)
        except ValueError as exc:
            error = str(exc)
    return render(request, "tasks/claims.html", {
        "claims": Claim.objects.filter(owner=request.user), "error": error, "form": request.POST if error else {},
        "crops": CROPS, "units": UNIT_CHOICES, "bounds": analysis.study_bounds()})


@login_required
def claim_detail(request, claim_id):
    claim = get_object_or_404(Claim, id=claim_id, owner=request.user)
    r = claim.result
    flags = [{**f, "help": analysis.FLAG_HELP.get(f["code"], "")} for f in r["flags"]]
    return render(request, "tasks/claim_detail.html", {
        "claim": claim, "r": r, "flags": flags, "claimed": areas(claim.claimed_area_ha), "detected": areas(r.get("cropland_area_ha")),
        "boundary": areas(r.get("polygon_area_ha")), "thresholds": analysis.thresholds(), "geometry": claim.geometry})


@login_required
def delete_claim(request, claim_id):
    claim = get_object_or_404(Claim, id=claim_id, owner=request.user)
    if request.method == "POST":
        claim.delete()
    return redirect("claims")