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

print("✅ Model and encoders loaded successfully!")

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
    """Parse IP string and ensure each octet is between 0–255."""
    try:
        parts = ip_str.strip().split(".")
        if len(parts) != 4:
            return None
        octets = [int(p) for p in parts]
        if all(0 <= o <= 255 for o in octets):
            return octets
        else:
            return None
    except ValueError:
        return None


def extract_ip_features(ip_str: str):
    """Extract octet and default RTT features for model input."""
    octets = parse_ip(ip_str)
    if not octets:
        return None

    first_octet = octets[0]
    ip_class = 'A' if first_octet < 128 else ('B' if first_octet < 192 else 'C')
    is_private = int(is_private_ip(ip_str))

    avg_rtt = X_train["avg_rtt"].mean() if "avg_rtt" in X_train.columns else 0.0
    min_rtt = X_train["min_rtt"].mean() if "min_rtt" in X_train.columns else 0.0
    max_rtt = X_train["max_rtt"].mean() if "max_rtt" in X_train.columns else 1.0
    rtt_range = avg_rtt - min_rtt
    rtt_ratio = avg_rtt / max_rtt if max_rtt != 0 else 0.0
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
# ✅ WHOIS AND IPAPI LOOKUPS
# ============================================================
def whois_lookup(ip: str):
    out = {"asn": "-", "org": "-", "country": "-"}
    if not IPWHOIS_AVAILABLE:
        return out
    try:
        obj = IPWhois(ip)
        r = obj.lookup_rdap(depth=1)
        out["asn"] = r.get("asn") or "-"
        net = r.get("network") or {}
        out["org"] = net.get("name") or r.get("asn_description") or "-"
        out["country"] = net.get("country") or r.get("country") or "-"
    except Exception as e:
        print(f"WHOIS lookup failed for {ip}: {e}")
    return out


def ipapi_lookup(ip: str, timeout: float = 3.0):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(f"https://ipapi.co/{ip}/json/", headers=headers, timeout=timeout)
        if resp.status_code == 200:
            j = resp.json()
            city = j.get("city") or j.get("region") or "-"
            country = j.get("country_name") or j.get("country") or "-"
            return city, country
    except Exception as e:
        print(f"IPAPI lookup failed for {ip}: {e}")
    return "-", "-"


# ============================================================
# ✅ MAIN PREDICTION LOGIC
# ============================================================
def predict_ips(ip_list: List[str]):
    results = []
    for ip in ip_list:
        if not parse_ip(ip):
            results.append({
                "IP": ip,
                "Predicted_City": "❌ Invalid IP format (0–255 required)",
                "Confidence (%)": 0.0,
                "Error Bound (%)": 100.0,
                "WHOIS_ASN": "-",
                "WHOIS_Country": "-",
                "IPAPI_City": "-",
                "Match (%)": "-"
            })
            continue

        feats = extract_ip_features(ip)
        if feats is None:
            results.append({
                "IP": ip,
                "Predicted_City": "-",
                "Confidence (%)": 0.0,
                "Error Bound (%)": 100.0,
                "WHOIS_ASN": "-",
                "WHOIS_Country": "-",
                "IPAPI_City": "-",
                "Match (%)": "-"
            })
            continue

        if is_private_ip(ip):
            results.append({
                "IP": ip,
                "Predicted_City": "🔒 Private IP — Location not traceable",
                "Confidence (%)": 0.0,
                "Error Bound (%)": 100.0,
                "WHOIS_ASN": "-",
                "WHOIS_Country": "-",
                "IPAPI_City": "-",
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
                "WHOIS_Country": "-",
                "IPAPI_City": "-",
                "Match (%)": "-"
            })
            continue

        try:
            pred = model.predict(feats)[0]
            conf = model.predict_proba(feats)[0].max() if hasattr(model, "predict_proba") else 1.0
        except Exception:
            pred, conf = "-", 0.0

        who = whois_lookup(ip)
        ipapi_city, _ = ipapi_lookup(ip)

        results.append({
            "IP": ip,
            "Predicted_City": pred or "-",
            "Confidence (%)": round(conf * 100, 2),
            "Error Bound (%)": round((1 - conf) * 100, 2),
            "WHOIS_ASN": who.get("asn", "-"),
            "WHOIS_Country": who.get("country", "-"),
            "IPAPI_City": ipapi_city,
            "Match (%)": f"{round(conf * 100, 2)}%"
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
                  csv_file: UploadFile = File(None)):
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
        results = predict_ips(ip_list)
        plot_img = plot_confidence(results)
    except Exception as e:
        return templates.TemplateResponse("home.html", {"request": request, "results": None, "plot": None, "error": f"Prediction failed: {str(e)}"})

    for row in results:
        for k in list(row.keys()):
            if row[k] is None or row[k] == "N/A":
                row[k] = "-"

    return templates.TemplateResponse("home.html", {"request": request, "results": results, "plot": plot_img, "error": None})
