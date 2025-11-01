# ============================================================
# ✅ IMPORT LIBRARIES
# ============================================================
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import ipaddress
import joblib
import matplotlib.pyplot as plt
import io
import base64
import os
import gc
import requests
import math
from typing import List

# Optional: ipwhois for WHOIS lookups
try:
    from ipwhois import IPWhois
    IPWHOIS_AVAILABLE = True
except Exception:
    IPWHOIS_AVAILABLE = False

# ============================================================
# ✅ APP SETUP
# ============================================================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ============================================================
# ✅ LOAD MODEL, ENCODERS, AND TRAIN FEATURES
# ============================================================
model_path = "trained_model_RF.pkl"
encoder_path = "label_encoders.pkl"
xtrain_path = "X_train.csv"

if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Model file not found: trained_model_RF.pkl")
if not os.path.exists(encoder_path):
    raise FileNotFoundError("❌ Label encoders file not found: label_encoders.pkl")
if not os.path.exists(xtrain_path):
    raise FileNotFoundError("❌ Training features file not found: X_train.csv")

model = joblib.load(model_path)
label_encoders = joblib.load(encoder_path)
X_train = pd.read_csv(xtrain_path)
model_features = X_train.columns.tolist()

# Auto-detect location column
possible_city_cols = [c for c in X_train.columns if c.lower() in ["city", "location", "city_name", "place"]]
_train_city_series = X_train[possible_city_cols[0]] if possible_city_cols else None

# Numeric features for distance similarity
NUMERIC_FEATURES_FOR_DISTANCE = [
    c for c in ["avg_rtt", "min_rtt", "max_rtt", "rtt_range", "rtt_ratio",
                "octet_1", "octet_2", "octet_3", "octet_4"]
    if c in X_train.columns
]

_distance_mean = X_train[NUMERIC_FEATURES_FOR_DISTANCE].mean()
_distance_std = X_train[NUMERIC_FEATURES_FOR_DISTANCE].std().replace(0, 1.0)
_train_numeric_matrix = (X_train[NUMERIC_FEATURES_FOR_DISTANCE] - _distance_mean) / _distance_std

print("✅ Model, Encoders, and X_train loaded successfully!")
print(f"   WHOIS available: {IPWHOIS_AVAILABLE}")
print(f"   Numeric distance features: {NUMERIC_FEATURES_FOR_DISTANCE}")
print(f"   Location column found: {possible_city_cols}")

# ============================================================
# ✅ PUBLIC DNS LIST
# ============================================================
PUBLIC_DNS_IPS = {
    "8.8.8.8": "Google DNS",
    "8.8.4.4": "Google DNS (Secondary)",
    "1.1.1.1": "Cloudflare DNS",
    "1.0.0.1": "Cloudflare DNS (Secondary)",
    "9.9.9.9": "Quad9 DNS",
    "208.67.222.222": "OpenDNS (Cisco)",
    "208.67.220.220": "OpenDNS (Cisco)"
}

# ============================================================
# ✅ HELPER FUNCTIONS
# ============================================================
def is_private_ip(ip_str: str) -> bool:
    try:
        return ipaddress.ip_address(ip_str).is_private
    except:
        return False

def parse_ip(ip_str: str):
    try:
        parts = ip_str.split(".")
        if len(parts) != 4:
            return None
        return [int(p) for p in parts]
    except:
        return None

def extract_ip_features(ip_str: str):
    octets = parse_ip(ip_str)
    if not octets:
        return None

    first_octet = octets[0]
    ip_class = 'A' if first_octet < 128 else ('B' if first_octet < 192 else 'C')
    is_private = int(is_private_ip(ip_str))

    avg_rtt = X_train["avg_rtt"].mean() if "avg_rtt" in X_train.columns else 0.0
    min_rtt = X_train["min_rtt"].mean() if "min_rtt" in X_train.columns else 0.0
    max_rtt = X_train["max_rtt"].mean() if "max_rtt" in X_train.columns else 1.0
    rtt_range = X_train["rtt_range"].mean() if "rtt_range" in X_train.columns else (avg_rtt - min_rtt)
    rtt_ratio = X_train["rtt_ratio"].mean() if "rtt_ratio" in X_train.columns else (avg_rtt / max_rtt if max_rtt != 0 else 0.0)
    reverse_dns = X_train["reverse_dns"].mode()[0] if "reverse_dns" in X_train.columns else 0

    feature_dict = {
        "avg_rtt": avg_rtt,
        "min_rtt": min_rtt,
        "max_rtt": max_rtt,
        "rtt_range": rtt_range,
        "rtt_ratio": rtt_ratio,
        "octet_1": octets[0],
        "octet_2": octets[1],
        "octet_3": octets[2],
        "octet_4": octets[3],
        "is_private": is_private,
        "ip_class": ip_class,
        "reverse_dns": reverse_dns
    }

    df_input = pd.DataFrame([feature_dict])

    for col, le in label_encoders.items():
        if col in df_input.columns:
            try:
                val = df_input[col].iloc[0]
                if val not in le.classes_:
                    le.classes_ = np.append(le.classes_, val)
                df_input[col] = le.transform(df_input[col])
            except Exception:
                df_input[col] = 0

    for col in model_features:
        if col not in df_input.columns:
            df_input[col] = 0

    return df_input[model_features]

# ============================================================
# ✅ WHOIS, IPAPI, SIMILARITY
# ============================================================
def whois_lookup(ip: str):
    out = {"asn": "-", "org": "-", "city": "-", "country": "-"}
    if not IPWHOIS_AVAILABLE:
        return out
    try:
        obj = IPWhois(ip)
        r = obj.lookup_rdap(depth=1)
        out["asn"] = r.get("asn") or r.get("asn_registry") or "-"
        net = r.get("network") or {}
        out["org"] = net.get("name") or "-"
        out["country"] = net.get("country") or r.get("country") or "-"
        out["city"] = net.get("city") or "-"
    except Exception:
        pass
    return out

def ipapi_lookup(ip: str, timeout: float = 2.0):
    try:
        resp = requests.get(f"https://ipapi.co/{ip}/json/", timeout=timeout)
        if resp.status_code == 200:
            j = resp.json()
            city = j.get("city") or j.get("region") or "-"
            country = j.get("country_name") or j.get("country") or "-"
            return city, country
    except Exception:
        pass
    return "-", "-"

# ============================================================
# ✅ IMPROVED SIMILARITY COMPUTATION
# ============================================================
def compute_similarity(ip_features: pd.DataFrame, k: int = 5):
    try:
        if _train_city_series is None or _train_city_series.empty:
            return "-", "-"
        if ip_features is None or len(NUMERIC_FEATURES_FOR_DISTANCE) == 0:
            return "-", "-"

        vec = []
        for col in NUMERIC_FEATURES_FOR_DISTANCE:
            val = ip_features.iloc[0].get(col, _distance_mean[col])
            if pd.isna(val):
                val = _distance_mean[col]
            vec.append(float(val))

        vec = np.array(vec, dtype=float)
        vec_norm = (vec - _distance_mean.values) / _distance_std.values

        diffs = _train_numeric_matrix.values - vec_norm
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        k = min(k, len(dists))
        idx = np.argsort(dists)[:k]

        nearest = [(str(_train_city_series.iloc[i]), float(dists[i])) for i in idx]

        city_counts = {}
        for c, d in nearest:
            c = str(c)
            city_counts[c] = city_counts.get(c, 0) + 1

        top_cities = [f"{c} ({cnt})" for c, cnt in sorted(city_counts.items(), key=lambda x: x[1], reverse=True)]

        mean_dist = float(np.mean([d for _, d in nearest])) if k > 0 else None
        if mean_dist is not None and not np.isnan(mean_dist):
            denom = math.sqrt(len(NUMERIC_FEATURES_FOR_DISTANCE)) or 1.0
            sim_val = 1.0 / (1.0 + mean_dist / denom)
            sim_percent = f"{round(sim_val * 100, 2)}%"
        else:
            sim_percent = "-"

        return (", ".join(top_cities) if top_cities else "-"), sim_percent

    except Exception as e:
        print(f"⚠ Similarity computation failed: {e}")
        return "-", "-"

# ============================================================
# ✅ MAIN PREDICTION LOGIC
# ============================================================
def predict_ips(ip_list: List[str], k_neighbors: int = 5):
    results = []
    for ip in ip_list:
        feats = extract_ip_features(ip)
        if feats is None:
            results.append({
                "IP": ip,
                "Predicted_City": "-",
                "Confidence (%)": 0.0,
                "Error Bound (%)": 100.0,
                "WHOIS_ASN": "-",
                "WHOIS_City": "-",
                "WHOIS_Country": "-",
                "IPAPI_City": "-",
                "Nearest_Cities": "-",
                "Similarity_Score": "-",
                "Match (%)": "-"
            })
            continue

        if is_private_ip(ip):
            results.append({
                "IP": ip,
                "Predicted_City": "Private IP — Location cannot be found",
                "Confidence (%)": 0.0,
                "Error Bound (%)": 100.0,
                "WHOIS_ASN": "-",
                "WHOIS_City": "-",
                "WHOIS_Country": "-",
                "IPAPI_City": "-",
                "Nearest_Cities": "-",
                "Similarity_Score": "-",
                "Match (%)": "-"
            })
            continue

        if ip.strip() in PUBLIC_DNS_IPS:
            results.append({
                "IP": ip,
                "Predicted_City": f"{PUBLIC_DNS_IPS[ip.strip()]} — Global Service",
                "Confidence (%)": 0.0,
                "Error Bound (%)": 100.0,
                "WHOIS_ASN": "-",
                "WHOIS_City": "-",
                "WHOIS_Country": "-",
                "IPAPI_City": "-",
                "Nearest_Cities": "-",
                "Similarity_Score": "-",
                "Match (%)": "-"
            })
            continue

        try:
            pred = model.predict(feats)[0]
            conf = model.predict_proba(feats)[0].max() if hasattr(model, "predict_proba") else 1.0
        except Exception:
            pred, conf = "-", 0.0

        who = whois_lookup(ip)
        who_asn = who.get("asn", "-")
        who_city = who.get("city", "-")
        who_country = who.get("country", "-")
        ipapi_city, ipapi_country = ipapi_lookup(ip)

        nearest_cities, sim_score = compute_similarity(feats, k=k_neighbors)

        results.append({
            "IP": ip,
            "Predicted_City": pred or "-",
            "Confidence (%)": round(conf * 100, 2),
            "Error Bound (%)": round((1 - conf) * 100, 2),
            "WHOIS_ASN": who_asn,
            "WHOIS_City": who_city,
            "WHOIS_Country": who_country,
            "IPAPI_City": ipapi_city,
            "Nearest_Cities": nearest_cities,
            "Similarity_Score": sim_score,
            "Match (%)": sim_score  # display same similarity as percentage column
        })

    gc.collect()
    return results

# ============================================================
# ✅ CONFIDENCE PLOT
# ============================================================
def plot_confidence(results):
    if not results:
        return None
    ips = [r["IP"] for r in results]
    conf = [r["Confidence (%)"] for r in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ips, conf, color="skyblue")
    ax.set_xlabel("IP")
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Prediction Confidence per IP")
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    gc.collect()
    return img_base64

# ============================================================
# ✅ ROUTES
# ============================================================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "results": None, "plot": None, "error": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  ip_single: str = Form(""),
                  ip_multiple: str = Form(""),
                  csv_file: UploadFile = File(None),
                  k_neighbors: int = Form(3)):
    ip_list: List[str] = []

    if ip_single and ip_single.strip():
        ip_list.append(ip_single.strip())
    elif ip_multiple and ip_multiple.strip():
        ip_list.extend([ip.strip() for ip in ip_multiple.split(",") if ip.strip()])
    elif csv_file:
        try:
            df = pd.read_csv(csv_file.file)
            if "IP" in df.columns:
                ip_list.extend(df["IP"].astype(str).tolist())
        except Exception:
            return templates.TemplateResponse("home.html", {"request": request, "results": None, "plot": None, "error": "Invalid CSV file uploaded."})

    if len(ip_list) > 200:
        ip_list = ip_list[:200]

    if not ip_list:
        return templates.TemplateResponse("home.html", {"request": request, "results": None, "plot": None, "error": "Please enter or upload IPs."})

    try:
        results = predict_ips(ip_list, k_neighbors=k_neighbors)
        plot_img = plot_confidence(results)
    except Exception as e:
        return templates.TemplateResponse("home.html", {"request": request, "results": None, "plot": None, "error": f"Prediction failed: {str(e)}"})

    for row in results:
        for k in list(row.keys()):
            if row[k] is None or row[k] == "N/A":
                row[k] = "-"

    return templates.TemplateResponse("home.html", {"request": request, "results": results, "plot": plot_img, "error": None})
