import io
import wave

import numpy as np
import pandas as pd
import streamlit as st


# =====================================================================
# LAYER 1: VOICE BIOMETRIC SIGNAL PROCESSING ENGINE
# =====================================================================
class VoiceBiometricEngine:
    """
    Measures F0, jitter and shimmer from a sustained-vowel WAV file.

    Pipeline:
      1. Parse the WAV properly (header, channels, bit depth, real sample rate).
      2. Low-pass filter (numpy only, no scipy needed).
      3. Frame-wise autocorrelation -> find voiced frames and median F0.
      4. Track individual glottal cycles (sub-sample peak picking).
      5. Jitter  = mean |T[i]-T[i-1]| / mean T     (cycle-to-cycle period change)
         Shimmer = mean |A[i]-A[i-1]| / mean A     (cycle-to-cycle amplitude change)
      6. Instability index scaled so that "healthy limit" voices land near 12
         (the baseline used by this app) and the alert threshold of 20 is
         crossed at roughly 1.7x the healthy limit.
    """

    F0_MIN = 70.0
    F0_MAX = 400.0
    MAX_SECONDS = 15.0
    # Commonly used "upper limit of normal" values for local jitter / shimmer
    JITTER_REF = 0.0104
    SHIMMER_REF = 0.0381
    BASELINE_SCORE = 12.0

    # ---------------------------------------------------------------- I/O
    @staticmethod
    def read_wav(audio_bytes: bytes):
        """Return (mono float32 signal in [-1, 1], sample_rate)."""
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                n_ch = wf.getnchannels()
                sw = wf.getsampwidth()
                sr = wf.getframerate()
                raw = wf.readframes(wf.getnframes())
        except (wave.Error, EOFError) as exc:
            raise ValueError(f"Could not read WAV file ({exc}). Use 8/16/24/32-bit PCM WAV.")

        if sw == 1:
            x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sw == 2:
            x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sw == 3:
            b = np.frombuffer(raw, dtype=np.uint8)
            b = b[: (len(b) // 3) * 3].reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            v = np.where(v >= (1 << 23), v - (1 << 24), v)
            x = v.astype(np.float32) / float(1 << 23)
        elif sw == 4:
            x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sw} bytes")

        if n_ch > 1:
            x = x[: (len(x) // n_ch) * n_ch].reshape(-1, n_ch).mean(axis=1)
        return x, int(sr)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _lowpass(x, sr, cutoff=900.0):
        cutoff = min(cutoff, 0.45 * sr)
        numtaps = max(int(sr / cutoff * 4), 31) | 1
        n = np.arange(numtaps) - (numtaps - 1) / 2
        h = np.sinc(2 * cutoff / sr * n) * np.hamming(numtaps)
        h /= h.sum()
        return np.convolve(x, h, mode="same")

    @staticmethod
    def _parabolic(y, i):
        """Sub-sample peak position and value around index i."""
        ym, y0, yp = y[i - 1], y[i], y[i + 1]
        denom = ym - 2.0 * y0 + yp
        delta = 0.0 if denom == 0 else float(np.clip(0.5 * (ym - yp) / denom, -0.5, 0.5))
        return i + delta, float(y0 - 0.25 * (ym - yp) * delta)

    def _voiced_frames(self, x, sr):
        frame = int(0.04 * sr)
        hop = max(int(0.01 * sr), 1)
        lag_min = int(sr / self.F0_MAX)
        lag_max = min(int(sr / self.F0_MIN), frame - 2)
        if lag_max - lag_min < 4 or len(x) < 2 * frame:
            return [], frame

        win = np.hanning(frame)
        nfft = 1 << int(np.ceil(np.log2(2 * frame)))
        wspec = np.fft.rfft(win, nfft)
        wacf = np.fft.irfft(wspec * np.conj(wspec))[:frame]
        wacf = wacf / wacf[0]

        rows = []
        for start in range(0, len(x) - frame, hop):
            seg = x[start:start + frame]
            seg = seg - seg.mean()
            rms = float(np.sqrt(np.mean(seg ** 2)))
            if rms < 1e-6:
                continue
            spec = np.fft.rfft(seg * win, nfft)
            acf = np.fft.irfft(spec * np.conj(spec))[:frame]
            if acf[0] <= 0:
                continue
            acf = (acf / acf[0]) / np.maximum(wacf, 1e-3)

            rng = acf[lag_min:lag_max + 1]
            is_peak = (rng[1:-1] > rng[:-2]) & (rng[1:-1] >= rng[2:])
            cand = np.where(is_peak)[0] + 1
            if len(cand) == 0:
                continue
            best = rng[cand].max()
            # smallest lag among strong peaks -> avoids octave-down errors
            k = lag_min + int(cand[rng[cand] >= 0.9 * best][0])
            lag, r = self._parabolic(acf, k)
            rows.append((start, sr / lag, float(r), rms))
        return rows, frame

    # ------------------------------------------------------------- public
    def analyze_uploaded_file(self, audio_bytes: bytes) -> dict:
        fail = lambda msg: {"valid": False, "message": msg}

        try:
            x, sr = self.read_wav(audio_bytes)
        except ValueError as exc:
            return fail(str(exc))

        x = x[: int(self.MAX_SECONDS * sr)]
        if len(x) < int(0.25 * sr):
            return fail("Recording is too short. Record at least 1 second of a sustained vowel (e.g. 'aaah').")
        x = x - float(np.mean(x))
        peak = float(np.max(np.abs(x)))
        if peak < 1e-4:
            return fail("The recording is silent.")
        x = x / peak
        y = self._lowpass(x, sr)

        rows, frame = self._voiced_frames(x, sr)
        if not rows:
            return fail("Could not analyse this recording (sample rate too low or file too short).")

        max_rms = max(r[3] for r in rows)
        voiced = [r for r in rows if r[2] >= 0.5 and r[3] >= 0.1 * max_rms]
        voiced_fraction = len(voiced) / len(rows)
        if len(voiced) < 8:
            return fail("No stable voiced sound detected. Use a clear, sustained vowel with little background noise.")

        T0 = sr / float(np.median([r[1] for r in voiced]))

        # Build mask of voiced samples
        mask = np.zeros(len(y), dtype=bool)
        for start, _, _, _ in voiced:
            mask[start:start + frame] = True
        edges = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
        run_starts = np.where(edges == 1)[0]
        run_ends = np.where(edges == -1)[0]
        amp_ref = float(np.percentile(np.abs(y[mask]), 95))

        # Track glottal cycles inside each voiced run
        segments = []
        for a, b in zip(run_starts, run_ends):
            hi = min(a + int(1.3 * T0) + 1, b)
            if hi - a < 5:
                continue
            idx = a + int(np.argmax(y[a:hi]))
            if idx <= a or idx >= b - 1 or y[idx] < 0.3 * amp_ref:
                continue
            t, amp = self._parabolic(y, idx)
            T = T0
            marks = [(t, amp)]
            while True:
                lo = int(round(t + T - 0.25 * T))
                hi = int(round(t + T + 0.25 * T)) + 1
                if hi + 1 >= b or lo < 1:
                    break
                j = lo + int(np.argmax(y[lo:hi]))
                if j == lo or j == hi - 1 or y[j] < 0.3 * amp_ref:
                    break
                t_new, a_new = self._parabolic(y, j)
                T = float(np.clip(0.7 * T + 0.3 * (t_new - t), 0.8 * T0, 1.2 * T0))
                marks.append((t_new, a_new))
                t = t_new
            if len(marks) >= 4:
                # Amplitude = peak-to-peak of the RAW signal around each cycle
                # (the low-passed copy would smooth away amplitude variation).
                times = np.array([m[0] for m in marks])
                half = 0.5 * float(np.median(np.diff(times)))
                amps = []
                for tt in times:
                    lo = max(int(round(tt - half)), 0)
                    hi = min(int(round(tt + half)) + 1, len(x))
                    amps.append(float(x[lo:hi].max() - x[lo:hi].min()))
                segments.append([(tm, am) for tm, am in zip(times, amps)])

        all_periods = np.concatenate([np.diff([m[0] for m in s]) for s in segments]) if segments else np.array([])
        if len(all_periods) < 10:
            return fail("Too few vocal cycles detected for a reliable measurement. Try a longer, steadier vowel.")

        med = float(np.median(all_periods))
        ok_all = []
        period_pairs = []
        amp_pairs = []
        amps_all = []
        for s in segments:
            times = np.array([m[0] for m in s])
            amps = np.array([m[1] for m in s])
            per = np.diff(times)
            ok = (per > 0.75 * med) & (per < 1.25 * med)
            ok_all.append(per[ok])
            for i in range(1, len(per)):
                if ok[i] and ok[i - 1]:
                    period_pairs.append(abs(per[i] - per[i - 1]))
            for i in range(1, len(amps)):
                amp_pairs.append(abs(amps[i] - amps[i - 1]))
            amps_all.append(amps)

        periods = np.concatenate(ok_all)
        if len(periods) < 10 or len(period_pairs) < 5:
            return fail("Voice was too irregular to track cycles. Try recording again.")

        mean_period = float(np.mean(periods)) / sr
        f0 = 1.0 / mean_period
        jitter = float(np.mean(period_pairs)) / float(np.mean(periods))
        shimmer = float(np.mean(amp_pairs)) / float(np.mean(np.concatenate(amps_all)))

        score = self.BASELINE_SCORE * (0.5 * jitter / self.JITTER_REF + 0.5 * shimmer / self.SHIMMER_REF)
        score = float(np.clip(score, 0.0, 100.0))
        husky = float(np.clip(shimmer * 4.0 + jitter * 10.0, 0.0, 1.0))

        return {
            "valid": True,
            "message": "OK",
            "jitter": jitter,
            "shimmer": shimmer,
            "instability_score": round(score, 2),
            "fundamental_pitch_hz": round(f0, 1),
            "husky_factor": round(husky, 2),
            "n_cycles": int(len(periods)),
            "voiced_fraction": round(voiced_fraction, 2),
            "duration_s": round(len(x) / sr, 2),
            "sample_rate": sr,
        }


@st.cache_data(show_spinner="Analysing voice sample...")
def analyze_voice(audio_bytes: bytes) -> dict:
    return VoiceBiometricEngine().analyze_uploaded_file(audio_bytes)


# =====================================================================
# LAYER 2: BASELINE GENERATOR FOR NEW PATIENTS
# =====================================================================
def create_initial_patient_history(seed_id: int) -> pd.DataFrame:
    """Generates a default 30-day baseline timeline for newly initialised profiles."""
    rng = np.random.default_rng(seed_id)
    days = 30
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq="D")

    return pd.DataFrame({
        "Date": dates,
        "Heart_Rate_BPM": rng.normal(74, 3, days).round(1),
        "Oxygen_Sat_Percentage": np.clip(rng.normal(98.0, 0.4, days), 85, 100).round(1),
        "Systolic_BP_mmHg": rng.normal(122, 4, days).round(1),
        "Voice_Instability_Index": np.clip(rng.normal(12.2, 1.5, days), 0, 100).round(2),
        "Fundamental_Pitch_Hz": rng.normal(135.0, 5.0, days).round(1),
    }).set_index("Date")


# =====================================================================
# LAYER 3: APP UI & MULTI-PATIENT STATE
# =====================================================================
st.set_page_config(page_title="RPMA Dashboard Panel", page_icon="🏥", layout="wide")

st.title("🏥 Remote Patient Monitoring Agent (RPMA)")
st.markdown("Category: **Remote Patient Monitoring (RPM)** | Systemic Endocrine & Neuro-Respiratory Tracking Framework.")
st.caption("Demonstration / screening tool only. Not a medical device and not a substitute for clinical diagnosis.")
st.markdown("---")

if "master_registry" not in st.session_state:
    st.session_state.master_registry = {
        "PT-01": {"name": "Sarah Jenkins", "age": 52, "condition": "Suspected Primary Hypothyroidism",
                  "history": create_initial_patient_history(42)},
        "PT-02": {"name": "David Miller", "age": 29, "condition": "Post-Viral Respiratory Evaluation",
                  "history": create_initial_patient_history(101)},
    }

# --- SIDEBAR: REGISTRATION & NAVIGATION ---
st.sidebar.header("📝 Register New Patient File")
with st.sidebar.form("registration_form", clear_on_submit=True):
    new_name = st.text_input("Full Patient Name:")
    new_age = st.number_input("Age (Years):", min_value=1, max_value=120, value=35)
    new_condition = st.selectbox("Suspected Target Diagnostics Condition:", [
        "Thyroid Fluid Accumulation (Hypothyroidism)",
        "Neuro-Vocal Degradation",
        "General Respiratory Screening",
    ])
    submit_registration = st.form_submit_button("Register Patient")

    if submit_registration and new_name:
        new_id = f"PT-{len(st.session_state.master_registry) + 1:02d}"
        st.session_state.master_registry[new_id] = {
            "name": new_name,
            "age": int(new_age),
            "condition": new_condition,
            "history": create_initial_patient_history(int(np.random.randint(1, 500))),
        }
        st.sidebar.success(f"Registered {new_name} as ID: {new_id}")

st.sidebar.markdown("---")
st.sidebar.header("📁 Electronic Health Records Selector")
patient_options = {pid: f"{pid} - {meta['name']}" for pid, meta in st.session_state.master_registry.items()}
selected_pid = st.sidebar.selectbox(
    "Active Patient Profile Selection:",
    options=list(patient_options.keys()),
    format_func=lambda x: patient_options[x],
)

active_patient = st.session_state.master_registry[selected_pid]
clinical_history = active_patient["history"]
latest_vitals = clinical_history.iloc[-1].copy()

# --- SECTION 1: PROFILE HEADER ---
col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
with col_meta1:
    st.markdown(f"👤 **Patient Name:**\n### {active_patient['name']}")
with col_meta2:
    st.markdown(f"🎂 **Chronological Age:**\n### {active_patient['age']} Years")
with col_meta3:
    st.markdown(f"🩺 **Monitored Condition Path:**\n`{active_patient['condition']}`")
with col_meta4:
    st.markdown(f"🆔 **Internal Record Key:**\n`{selected_pid}`")

st.markdown("---")

# --- SECTION 2: INPUTS & AUDIO ---
st.subheader("📥 Log New Vital Diagnostics Entry & Audio Sample")
form_col1, form_col2 = st.columns(2)

with form_col1:
    st.markdown("##### 📝 Input Vital Measurements")
    input_hr = st.number_input(
        "Heart Rate (BPM):", min_value=40, max_value=200,
        value=int(np.clip(latest_vitals["Heart_Rate_BPM"], 40, 200)), step=1, key=f"hr_in_{selected_pid}")
    input_bp = st.number_input(
        "Systolic Blood Pressure (mmHg):", min_value=70, max_value=220,
        value=int(np.clip(latest_vitals["Systolic_BP_mmHg"], 70, 220)), step=1, key=f"bp_in_{selected_pid}")
    input_spo2 = st.slider(
        "Oxygen Saturation Level - SpO2 (%):", min_value=80.0, max_value=100.0,
        value=float(np.clip(latest_vitals["Oxygen_Sat_Percentage"], 80.0, 100.0)), step=0.1,
        key=f"spo2_in_{selected_pid}")

audio_result = None
file_bytes = None
with form_col2:
    st.markdown("##### 🔊 Acoustic Analysis (Voice Record)")
    uploaded_file = st.file_uploader(
        "Upload a sustained vowel sound (.wav), e.g. 'aaaah' for 3-5 seconds", type=["wav"])
    if uploaded_file is not None:
        # getvalue() is safe to call repeatedly (read() would return b'' the 2nd time)
        file_bytes = uploaded_file.getvalue()
        st.audio(file_bytes, format="audio/wav")
        audio_result = analyze_voice(file_bytes)

        if audio_result["valid"]:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Instability Index", f"{audio_result['instability_score']:.2f}")
            m2.metric("Pitch F0", f"{audio_result['fundamental_pitch_hz']:.1f} Hz")
            m3.metric("Jitter", f"{audio_result['jitter'] * 100:.2f} %")
            m4.metric("Shimmer", f"{audio_result['shimmer'] * 100:.2f} %")
            st.caption(
                f"{audio_result['n_cycles']} vocal cycles analysed | "
                f"{audio_result['duration_s']} s @ {audio_result['sample_rate']} Hz | "
                f"voiced {audio_result['voiced_fraction'] * 100:.0f}%")
        else:
            st.error(f"Voice sample not usable: {audio_result['message']}")

if st.button("🚀 Commit Log Entry"):
    if audio_result is not None and audio_result["valid"]:
        voice_idx = float(audio_result["instability_score"])
        pitch = float(audio_result["fundamental_pitch_hz"])
    else:
        # No usable audio: store NaN instead of inventing a value.
        voice_idx = np.nan
        pitch = np.nan

    new_entry_row = pd.DataFrame([{
        "Heart_Rate_BPM": float(input_hr),
        "Oxygen_Sat_Percentage": float(input_spo2),
        "Systolic_BP_mmHg": float(input_bp),
        "Voice_Instability_Index": voice_idx,
        "Fundamental_Pitch_Hz": pitch,
    }], index=pd.DatetimeIndex([pd.Timestamp.now()], name="Date"))

    st.session_state.master_registry[selected_pid]["history"] = pd.concat([clinical_history, new_entry_row])
    st.session_state["flash"] = (
        f"Entry saved for {active_patient['name']}"
        + ("" if not np.isnan(voice_idx) else " (no voice sample - voice values left blank)"))
    st.rerun()

if "flash" in st.session_state:
    st.toast(st.session_state.pop("flash"), icon="💾")

st.markdown("---")

# =====================================================================
# LAYER 4: CLINICAL STATUS & RISK EVALUATION
# =====================================================================
st.subheader("📋 Automated Clinical Assessment & Health Status Report")

HR_MIN, HR_MAX = 60, 100
SPO2_MIN = 95.0
BP_SYS_MIN, BP_SYS_MAX = 90, 140
VOICE_INSTABILITY_THRESHOLD = 20.0

current_hr = float(latest_vitals["Heart_Rate_BPM"])
current_spo2 = float(latest_vitals["Oxygen_Sat_Percentage"])
current_bp = float(latest_vitals["Systolic_BP_mmHg"])

# Latest *real* voice measurement (skips entries that had no audio)
voice_series = clinical_history["Voice_Instability_Index"].dropna()
current_voice = float(voice_series.iloc[-1]) if len(voice_series) else 14.20
voice_is_stale = bool(pd.isna(latest_vitals["Voice_Instability_Index"]))

voice_status = "Normal" if current_voice <= VOICE_INSTABILITY_THRESHOLD else "Bad (Elevated)"

risk_flags = []
if not (HR_MIN <= current_hr <= HR_MAX):
    risk_flags.append(f"Abnormal Heart Rate: {current_hr} BPM (Normal range: {HR_MIN}-{HR_MAX} BPM)")
if current_spo2 < SPO2_MIN:
    risk_flags.append(f"Low Oxygen Saturation (SpO2): {current_spo2}% (Threshold: >={SPO2_MIN}%)")
if not (BP_SYS_MIN <= current_bp <= BP_SYS_MAX):
    risk_flags.append(f"Abnormal Systolic BP: {current_bp} mmHg (Normal range: {BP_SYS_MIN}-{BP_SYS_MAX} mmHg)")
if current_voice > VOICE_INSTABILITY_THRESHOLD:
    risk_flags.append(
        f"Elevated Voice Instability Index: {current_voice:.2f} (Threshold: <={VOICE_INSTABILITY_THRESHOLD})")

if risk_flags:
    st.error(f"⚠️ **Clinical Alert: Potential Health Issues Detected for {active_patient['name']}**")
    st.markdown("The system telemetry has flagged the following concerns:")
    for flag in risk_flags:
        st.markdown(f"- 🔴 {flag}")
    st.warning("💡 **Recommended Action:** Schedule a clinical follow-up or review trends below.")
else:
    st.success(f"✅ **Health Status: Stable ({active_patient['name']})**")
    st.markdown("All vital measurements and acoustic indices are within the configured reference ranges.")
    st.markdown(f"- **Heart Rate:** {current_hr} BPM (Stable)")
    st.markdown(f"- **Oxygen Saturation (SpO2):** {current_spo2}% (Optimal)")
    st.markdown(f"- **Systolic BP:** {current_bp} mmHg (Normal)")
    st.markdown(f"- **Voice Instability Index:** {current_voice:.2f} ({voice_status})")
    st.info("💡 **Recommended Action:** Continue standard remote monitoring schedule.")

if voice_is_stale:
    st.caption("ℹ️ The latest entry had no voice sample; the voice index shown is the most recent measured value.")

st.markdown("---")

# --- SECTION 5: TRENDS ---
st.subheader("📈 Historical Telemetry & Trends")
chart_tab1, chart_tab2 = st.tabs(["Vitals & Hemodynamics", "Acoustic & Voice Biomarkers"])

with chart_tab1:
    st.line_chart(clinical_history[["Heart_Rate_BPM", "Systolic_BP_mmHg", "Oxygen_Sat_Percentage"]])

with chart_tab2:
    st.line_chart(clinical_history[["Voice_Instability_Index", "Fundamental_Pitch_Hz"]])
