import io
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Workspace Paths & Directories
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
ANNOTATIONS_FILE = DATA_DIR / "annotations.json"
REVIEWS_FILE = DATA_DIR / "description_reviews.json"

# ---------------------------------------------------------------------------
# Data Persistence Helpers
# ---------------------------------------------------------------------------
def load_records():
    """Safely loads annotation records from disk."""
    if ANNOTATIONS_FILE.exists():
        try:
            content = ANNOTATIONS_FILE.read_text(encoding="utf-8").strip()
            if not content:
                return []
            return json.loads(content)
        except (json.JSONDecodeError, OSError):
            return []
    return []

def save_records(records):
    """Safely saves annotation records to disk."""
    try:
        ANNOTATIONS_FILE.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        return True
    except OSError:
        return False

def load_description_reviews():
    """Safely loads AI-description review records from disk."""
    if REVIEWS_FILE.exists():
        try:
            content = REVIEWS_FILE.read_text(encoding="utf-8").strip()
            if not content:
                return []
            return json.loads(content)
        except (json.JSONDecodeError, OSError):
            return []
    return []

def save_description_reviews(reviews):
    """Safely saves AI-description review records to disk."""
    try:
        REVIEWS_FILE.write_text(
            json.dumps(reviews, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        return True
    except OSError:
        return False

def reviews_to_dataframe(reviews):
    """Flattens nested description review records into a tabular DataFrame for export."""
    rows = []
    for r in reviews:
        review_id = r.get("id", "")
        video_name = r.get("video_name", "")
        created_at = r.get("created_at_utc", "")
        for idx, item in enumerate(r.get("claims", []), 1):
            rows.append({
                "review_id": review_id,
                "video_name": video_name,
                "claim_index": idx,
                "claim": item.get("claim", ""),
                "verdict": item.get("verdict", "Unreviewed"),
                "rationale": item.get("rationale", ""),
                "created_at_utc": created_at
            })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Video Processing & Frame Extraction Helpers
# ---------------------------------------------------------------------------
def get_cached_video_path(uploaded_file):
    """
    Caches uploaded video bytes to a temporary local file once per upload.
    Prevents repeated disk writes on Windows during frame scrubbing.
    """
    if uploaded_file is None:
        return None

    file_bytes = uploaded_file.getvalue()
    file_id = f"{uploaded_file.name}-{len(file_bytes)}"

    cached_path = st.session_state.get("cached_video_path")
    cached_id = st.session_state.get("cached_video_id")

    if cached_path and cached_id == file_id and Path(cached_path).exists():
        return Path(cached_path)

    # Clean up previous cached file if exists
    if cached_path:
        try:
            p = Path(cached_path)
            if p.exists():
                os.unlink(p)
        except OSError:
            pass

    suffix = Path(uploaded_file.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = Path(tmp.name)
    finally:
        tmp.close()

    st.session_state["cached_video_path"] = str(tmp_path)
    st.session_state["cached_video_id"] = file_id
    st.session_state["video_bytes"] = file_bytes
    return tmp_path

def video_metadata(video_source, suffix=".mp4"):
    """
    Extracts video duration, resolution, frame count, and FPS using OpenCV.
    Accepts either a Path/str filepath or raw bytes.
    Ensures safe file handle cleanup on Windows.
    """
    temp_created = False
    if isinstance(video_source, (str, Path)):
        video_path = str(video_source)
    elif isinstance(video_source, (bytes, bytearray)):
        if not video_source:
            return {"fps": 0.0, "frames": 0, "width": 0, "height": 0, "duration": 0.0, "is_valid": False, "error": "Empty video data"}
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(video_source)
            tmp.flush()
            video_path = tmp.name
            temp_created = True
        finally:
            tmp.close()
    else:
        return {"fps": 0.0, "frames": 0, "width": 0, "height": 0, "duration": 0.0, "is_valid": False, "error": "Invalid video source"}

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return {
                "fps": 0.0, "frames": 0, "width": 0, "height": 0, "duration": 0.0,
                "is_valid": False, "error": "OpenCV could not decode the video file. The codec may be unsupported or file corrupted."
            }
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (frames / fps) if fps > 0 else 0.0
        return {
            "fps": round(fps, 2),
            "frames": frames,
            "width": width,
            "height": height,
            "duration": round(duration, 2),
            "is_valid": True,
            "error": None
        }
    finally:
        cap.release()
        if temp_created:
            try:
                os.unlink(video_path)
            except OSError:
                pass

def extract_frame(video_source, seconds: float, suffix=".mp4"):
    """
    Extracts an RGB frame from video at the specified timestamp.
    Accepts either a Path/str filepath or raw bytes.
    Ensures safe handle release on Windows before returning.
    """
    temp_created = False
    if isinstance(video_source, (str, Path)):
        video_path = str(video_source)
    elif isinstance(video_source, (bytes, bytearray)):
        if not video_source:
            return None
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(video_source)
            tmp.flush()
            video_path = tmp.name
            temp_created = True
        finally:
            tmp.close()
    else:
        return None

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(seconds)) * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    finally:
        cap.release()
        if temp_created:
            try:
                os.unlink(video_path)
            except OSError:
                pass

# ---------------------------------------------------------------------------
# Natural Language Claim Parsing Helper
# ---------------------------------------------------------------------------
def parse_claims(text: str) -> list[str]:
    """
    Splits descriptive text into individual reviewable claims.
    Correctly handles numbered lists, bullet points, and multi-sentence paragraphs.
    Note: Splitting is an editorial aid only; human review against the clip is required.
    """
    if not text or not text.strip():
        return []

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    claims = []

    for line in lines:
        # Strip common list markers: "1. ", "1) ", "[1] ", "- ", "* ", "• "
        cleaned = re.sub(r"^(\[?\d+[\.\)\]]?\s*|[\-\*\u2022]\s*)", "", line).strip()
        if not cleaned:
            continue

        # Split on sentence terminals followed by whitespace and an uppercase character or quote
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", cleaned)
        for part in parts:
            part = part.strip()
            part = re.sub(r"^(\[?\d+[\.\)\]]?\s*|[\-\*\u2022]\s*)", "", part).strip()
            if len(part) > 2 and not re.match(r"^\d+[\.\)]?$", part):
                claims.append(part)

    return claims

# ---------------------------------------------------------------------------
# Streamlit App Execution
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="VideoLens AI | Video Data Annotation & QA",
        page_icon="🎬",
        layout="wide"
    )

    # Custom styling
    st.markdown("""
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {background: rgba(125,125,125,.08); padding: 14px; border-radius: 10px; border: 1px solid rgba(125,125,125,.15);}
    .reviewer-banner {background: rgba(30,144,255,.08); border-left: 4px solid #1e90ff; padding: 10px 14px; border-radius: 4px; margin-bottom: 1rem;}
    .claim-box {background: rgba(255,255,255,.03); border: 1px solid rgba(125,125,125,.2); border-radius: 8px; padding: 12px; margin-bottom: 12px;}
    </style>
    """, unsafe_allow_html=True)

    # Session State Initialization
    if "records" not in st.session_state:
        st.session_state.records = load_records()
    if "current_video_id" not in st.session_state:
        st.session_state.current_video_id = ""
    if "current_video_name" not in st.session_state:
        st.session_state.current_video_name = ""
    if "video_bytes" not in st.session_state:
        st.session_state.video_bytes = None
    if "evaluation_claims" not in st.session_state:
        st.session_state.evaluation_claims = []
    if "evaluation_video_name" not in st.session_state:
        st.session_state.evaluation_video_name = ""
    if "eval_batch_id" not in st.session_state:
        st.session_state.eval_batch_id = 1

    # App Header
    st.title("🎬 VideoLens AI")
    st.caption("Human-in-the-Loop Video Data Annotation & AI-Description Quality Evaluation")

    # Sidebar Navigation
    with st.sidebar:
        st.header("VideoLens Workspace")
        st.write("**Role Focus:** AI/ML Video Data Annotation")
        st.write(f"Saved Annotations: **{len(st.session_state.records)}**")
        reviews = load_description_reviews()
        st.write(f"AI Reviews Saved: **{len(reviews)}**")
        if st.session_state.current_video_name:
            st.info(f"Active Clip: **{st.session_state.current_video_name}**")
        st.divider()
        page = st.radio(
            "Navigate Modules",
            ["Annotate video", "Evaluate AI description", "Quality dashboard", "Export & guide"]
        )
        st.divider()
        st.caption("Portfolio prototype designed for the micro1 AI/ML Video Annotation role.")

    # -----------------------------------------------------------------------
    # Module 1: Video Annotation Studio
    # -----------------------------------------------------------------------
    if page == "Annotate video":
        st.header("1 · Video Annotation Studio")
        st.markdown(
            "Upload short video clips, inspect individual frame timestamps, and record structured annotations "
            "with categories, review statuses, and human evidence."
        )

        uploaded = st.file_uploader(
            "Upload a short video clip (MP4, MOV, AVI, MKV, WebM)",
            type=["mp4", "mov", "avi", "mkv", "webm"]
        )

        video_path = None
        if uploaded:
            video_path = get_cached_video_path(uploaded)
            video_id = f"{uploaded.name}-{len(uploaded.getvalue())}"
            st.session_state.current_video_id = video_id
            st.session_state.current_video_name = uploaded.name

            # Metadata computation
            meta = video_metadata(video_path)
            if not meta.get("is_valid"):
                st.error(f"Video Error: {meta.get('error', 'Unsupported or unreadable video file.')}")
                max_time = 0.0
            else:
                st.video(st.session_state.video_bytes)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Duration", f"{meta['duration']:.2f}s")
                c2.metric("Resolution", f"{meta['width']} × {meta['height']}")
                c3.metric("Total Frames", f"{meta['frames']:,}")
                c4.metric("Frame Rate", f"{meta['fps']:.2f} fps" if meta["fps"] > 0 else "Variable")
                max_time = max(0.0, float(meta["duration"]))

            # Frame Inspector
            if meta.get("is_valid") and max_time > 0:
                st.subheader("Timestamp Frame Inspector")
                frame_time = st.slider(
                    "Select timestamp (seconds)",
                    min_value=0.0,
                    max_value=float(max_time),
                    value=0.0,
                    step=0.1
                )
                frame = extract_frame(video_path, frame_time)
                if frame is not None:
                    st.image(frame, caption=f"Extracted Frame at {frame_time:.2f}s", use_container_width=True)
                else:
                    st.warning("Frame could not be extracted at this specific timestamp.")

                # Annotation Form
                st.subheader("Create Structured Annotation")
                with st.form("annotation_form", clear_on_submit=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        start = st.number_input(
                            "Start time (seconds)",
                            min_value=0.0,
                            max_value=float(max_time),
                            value=float(frame_time),
                            step=0.1
                        )
                        end = st.number_input(
                            "End time (seconds)",
                            min_value=0.0,
                            max_value=float(max_time),
                            value=float(frame_time),
                            step=0.1
                        )
                        category = st.selectbox(
                            "Label Category",
                            ["Action", "Object", "Person", "Scene", "Audio event", "Other"]
                        )
                    with col2:
                        label = st.text_input("Label / Description", placeholder="e.g., person opens cabinet door")
                        confidence = st.select_slider("Confidence Level", options=["Low", "Medium", "High"], value="High")
                        status = st.selectbox("Review Status", ["Needs review", "Reviewed"])
                    evidence = st.text_area(
                        "Reviewer Evidence / Rationale",
                        placeholder="Document strictly visible or audible evidence at this timestamp. Note visual ambiguity."
                    )
                    submitted = st.form_submit_button("Save Annotation", type="primary")

                    if submitted:
                        if not label.strip():
                            st.error("Validation Error: Please provide an annotation label before saving.")
                        elif end < start:
                            st.error("Validation Error: End time cannot be earlier than start time.")
                        else:
                            record = {
                                "id": f"ann-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
                                "video_id": video_id,
                                "video_name": uploaded.name,
                                "start_seconds": round(float(start), 3),
                                "end_seconds": round(float(end), 3),
                                "category": category,
                                "label": label.strip(),
                                "confidence": confidence,
                                "review_status": status,
                                "evidence": evidence.strip(),
                                "created_at_utc": datetime.now(timezone.utc).isoformat()
                            }
                            st.session_state.records.append(record)
                            save_records(st.session_state.records)
                            st.success("Annotation successfully saved.")
                            st.rerun()
        else:
            st.info("Upload a video clip above to begin timestamp inspection and structured annotation.")
            st.markdown(
                "> **Dataset Note:** For portfolio demonstrations, use clips 10–30s long showing specific actions "
                "(e.g., pouring water, walking, picking up objects). Ensure no confidential or private data is uploaded."
            )

        # Saved Annotations Table
        st.divider()
        st.subheader("Saved Annotations")
        if st.session_state.records:
            df = pd.DataFrame(st.session_state.records)
            display_cols = ["id", "video_name", "start_seconds", "end_seconds", "category", "label", "confidence", "review_status"]
            avail_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[avail_cols], use_container_width=True, hide_index=True)

            del_col1, del_col2 = st.columns([3, 1])
            with del_col1:
                ids = [r["id"] for r in st.session_state.records]
                selected_id = st.selectbox("Select annotation to delete", ["— select an annotation to remove —"] + ids)
            with del_col2:
                st.write("")
                st.write("")
                if selected_id != "— select an annotation to remove —" and st.button("Delete Annotation"):
                    st.session_state.records = [r for r in st.session_state.records if r["id"] != selected_id]
                    save_records(st.session_state.records)
                    st.success(f"Annotation {selected_id} deleted.")
                    st.rerun()
        else:
            st.caption("No annotations saved yet.")

    # -----------------------------------------------------------------------
    # Module 2: AI Description Evaluator
    # -----------------------------------------------------------------------
    elif page == "Evaluate AI description":
        st.header("2 · AI-Generated Description Evaluator")
        st.markdown("""
        <div class="reviewer-banner">
            <strong>Human Evaluation Protocol:</strong> This tool segments AI-generated video captions into discrete claims for human verification.
            Automated tools <em>do not</em> establish visual truth. The reviewer must verify each claim directly against video evidence.
        </div>
        """, unsafe_allow_html=True)

        # Video Reference Section
        if st.session_state.video_bytes:
            with st.expander("📺 Video Reference Player (Active Clip)", expanded=False):
                st.video(st.session_state.video_bytes)
                if st.session_state.get("cached_video_path"):
                    meta = video_metadata(st.session_state["cached_video_path"])
                    if meta.get("is_valid") and meta.get("duration", 0) > 0:
                        eval_time = st.slider("Inspect Reference Frame (seconds)", 0.0, float(meta["duration"]), 0.0, 0.2, key="eval_slider")
                        ref_frame = extract_frame(st.session_state["cached_video_path"], eval_time)
                        if ref_frame is not None:
                            st.image(ref_frame, caption=f"Reference Frame at {eval_time:.1f}s", width=420)

        # Checklist Generation Form
        with st.form("description_input_form"):
            default_name = st.session_state.current_video_name or st.session_state.current_video_id or "demo-clip-01"
            video_name = st.text_input("Associated Video Name / ID", value=default_name)
            description = st.text_area(
                "Paste AI-Generated Video Description / Caption",
                height=120,
                placeholder="e.g., A person enters the room wearing a black hoodie. They pick up a blue backpack from the desk and walk out the door."
            )
            create_checklist = st.form_submit_button("Generate Claim Review Checklist", type="primary")

        if create_checklist:
            if not description.strip():
                st.error("Please paste an AI-generated description first.")
            else:
                parsed = parse_claims(description)
                if not parsed:
                    st.warning("Could not extract any claims from the provided text. Ensure text contains valid sentences.")
                else:
                    st.session_state.eval_batch_id += 1
                    st.session_state.evaluation_description = description.strip()
                    st.session_state.evaluation_video_name = video_name.strip()
                    st.session_state.evaluation_claims = [
                        {"claim": c, "verdict": "Unreviewed", "rationale": ""}
                        for c in parsed
                    ]
                    st.success(f"Extracted {len(parsed)} claims for human evaluation.")

        # Interactive Claim Review Checklist
        if st.session_state.evaluation_claims:
            st.subheader(f"Claim-by-Claim Evaluation ({len(st.session_state.evaluation_claims)} Claims)")
            st.caption("Verdicts: **Supported** (visible/audible evidence), **Incorrect** (contradicts clip), **Uncertain** (ambiguous/unclear), **Missing Context** (partially true but omits key details).")

            batch_id = st.session_state.eval_batch_id
            verdict_options = ["Unreviewed", "Supported", "Incorrect", "Uncertain", "Missing Context"]

            updated_claims = []
            for i, item in enumerate(st.session_state.evaluation_claims):
                with st.container(border=True):
                    st.markdown(f"**Claim {i+1}:** {item['claim']}")
                    current_verdict = item.get("verdict", "Unreviewed")
                    current_idx = verdict_options.index(current_verdict) if current_verdict in verdict_options else 0

                    v_col, r_col = st.columns([2, 3])
                    with v_col:
                        verdict = st.radio(
                            "Verdict",
                            verdict_options,
                            index=current_idx,
                            key=f"eval_verdict_{batch_id}_{i}",
                            horizontal=False
                        )
                    with r_col:
                        rationale = st.text_area(
                            "Reviewer Rationale & Timestamp Evidence",
                            value=item.get("rationale", ""),
                            key=f"eval_rationale_{batch_id}_{i}",
                            placeholder="Cite specific timestamp or explain discrepancy/ambiguity...",
                            height=100
                        )
                    updated_claims.append({
                        "claim": item["claim"],
                        "verdict": verdict,
                        "rationale": rationale.strip()
                    })

            # Save Review Action
            c_save, c_clear = st.columns([2, 1])
            with c_save:
                if st.button("Save Description Review", type="primary", use_container_width=True):
                    reviewed_count = sum(1 for c in updated_claims if c["verdict"] != "Unreviewed")
                    if reviewed_count == 0:
                        st.warning("Please assign a verdict (Supported, Incorrect, Uncertain, or Missing Context) to at least one claim before saving.")
                    else:
                        review_record = {
                            "id": f"rev-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
                            "video_name": st.session_state.evaluation_video_name,
                            "original_description": st.session_state.get("evaluation_description", ""),
                            "total_claims": len(updated_claims),
                            "reviewed_claims": reviewed_count,
                            "claims": updated_claims,
                            "created_at_utc": datetime.now(timezone.utc).isoformat()
                        }
                        reviews = load_description_reviews()
                        reviews.append(review_record)
                        save_description_reviews(reviews)
                        st.session_state.evaluation_claims = updated_claims
                        st.success("Description review saved successfully to disk.")
                        st.download_button(
                            "Download This Review (JSON)",
                            json.dumps(review_record, indent=2, ensure_ascii=False),
                            file_name=f"review_{review_record['id']}.json",
                            mime="application/json"
                        )
            with c_clear:
                if st.button("Reset Current Checklist", use_container_width=True):
                    st.session_state.evaluation_claims = []
                    st.session_state.evaluation_description = ""
                    st.rerun()

        # Display Prior Reviews
        all_reviews = load_description_reviews()
        if all_reviews:
            st.divider()
            st.subheader(f"Previous Description Reviews ({len(all_reviews)})")
            for r in reversed(all_reviews[-5:]):
                with st.expander(f"Review {r.get('id')} — Video: {r.get('video_name', 'N/A')} ({r.get('reviewed_claims', 0)}/{r.get('total_claims', 0)} claims reviewed)"):
                    st.write(f"**Original Caption:** {r.get('original_description', 'N/A')}")
                    claim_rows = [
                        {"Claim": c.get("claim"), "Verdict": c.get("verdict"), "Rationale": c.get("rationale")}
                        for c in r.get("claims", [])
                    ]
                    st.dataframe(pd.DataFrame(claim_rows), use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Module 3: Quality Dashboard
    # -----------------------------------------------------------------------
    elif page == "Quality dashboard":
        st.header("3 · Annotation Quality & Review Dashboard")
        st.markdown(
            "Overview of dataset throughput, reviewer coverage, confidence distributions, "
            "and AI description verdict metrics."
        )

        st.subheader("Annotation Dataset Metrics")
        records = st.session_state.records
        if not records:
            st.info("No annotation records present. Create annotations in the Studio to view metrics.")
        else:
            df = pd.DataFrame(records)
            total_ann = len(df)
            reviewed_ann = int((df["review_status"] == "Reviewed").sum()) if "review_status" in df.columns else 0
            needs_review_ann = total_ann - reviewed_ann
            high_conf = int((df["confidence"] == "High").sum()) if "confidence" in df.columns else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Annotations", total_ann)
            m2.metric("Reviewed Coverage", f"{reviewed_ann / total_ann:.1%}", f"{reviewed_ann} reviewed")
            m3.metric("Review Queue", needs_review_ann)
            m4.metric("High Confidence", high_conf, f"{high_conf / total_ann:.1%}")

            c_left, c_right = st.columns(2)
            with c_left:
                st.write("**Labels by Category**")
                if "category" in df.columns:
                    cat_counts = df["category"].value_counts()
                    st.bar_chart(cat_counts)
            with c_right:
                st.write("**Confidence Breakdown**")
                if "confidence" in df.columns:
                    conf_counts = df["confidence"].value_counts()
                    st.bar_chart(conf_counts)

            st.write("**Items Requiring Quality Review**")
            if "review_status" in df.columns:
                queue = df[df["review_status"] != "Reviewed"]
                if not queue.empty:
                    st.dataframe(queue, use_container_width=True, hide_index=True)
                else:
                    st.success("All current annotations have been marked as Reviewed.")

        st.divider()
        st.subheader("AI Description Evaluation Metrics")
        all_reviews = load_description_reviews()
        if not all_reviews:
            st.info("No description evaluations saved yet. Review captions in the Evaluator to populate these metrics.")
        else:
            all_verdicts = [c.get("verdict", "Unreviewed") for r in all_reviews for c in r.get("claims", [])]
            total_claims = len(all_verdicts)
            evaluated_claims = sum(1 for v in all_verdicts if v != "Unreviewed")

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Descriptions Evaluated", len(all_reviews))
            e2.metric("Total Claims", total_claims)
            e3.metric("Claims Evaluated", evaluated_claims, f"{evaluated_claims / total_claims:.1%}" if total_claims else "0%")
            supported_count = sum(1 for v in all_verdicts if v == "Supported")
            e4.metric("Supported Claims", supported_count, f"{supported_count / evaluated_claims:.1%}" if evaluated_claims else "0%")

            st.write("**Verdict Distribution**")
            v_series = pd.Series(all_verdicts).value_counts().rename_axis("Verdict").reset_index(name="Count")
            st.dataframe(v_series, use_container_width=True, hide_index=True)

        st.caption(
            "Quality Assurance Note: Metrics reflect human annotation progress and verdict distribution. "
            "They do not represent automated model validation."
        )

    # -----------------------------------------------------------------------
    # Module 4: Export & Reviewer Guide
    # -----------------------------------------------------------------------
    elif page == "Export & guide":
        st.header("4 · Dataset Export & Reviewer Guidelines")
        st.markdown(
            "Export structured dataset artifacts in standard formats (JSON/CSV) for downstream AI/ML training, "
            "benchmarking, and quality assurance."
        )

        st.subheader("Download Annotation Data")
        records = st.session_state.records
        if records:
            col1, col2 = st.columns(2)
            with col1:
                json_str = json.dumps(records, indent=2, ensure_ascii=False)
                st.download_button(
                    "Download Annotations (JSON)",
                    json_str,
                    file_name="annotations.json",
                    mime="application/json",
                    use_container_width=True
                )
            with col2:
                csv_bytes = pd.DataFrame(records).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Annotations (CSV)",
                    csv_bytes,
                    file_name="annotations.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("No annotation records to export.")

        st.subheader("Download Description Evaluation Data")
        all_reviews = load_description_reviews()
        if all_reviews:
            col3, col4 = st.columns(2)
            with col3:
                reviews_json = json.dumps(all_reviews, indent=2, ensure_ascii=False)
                st.download_button(
                    "Download Description Reviews (JSON)",
                    reviews_json,
                    file_name="description_reviews.json",
                    mime="application/json",
                    use_container_width=True
                )
            with col4:
                df_reviews = reviews_to_dataframe(all_reviews)
                csv_reviews = df_reviews.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Claim-Level Reviews (CSV)",
                    csv_reviews,
                    file_name="description_claims_reviewed.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("No description review records to export.")

        st.divider()
        st.subheader("Video Data Annotation Guidelines (micro1 Standard)")
        st.markdown("""
        1. **Strict Observable Evidence:** Annotate solely what is physically visible or audible within the clip. Never speculate about off-screen context, human intent, or unobservable states.
        2. **Precise Boundary Timestamps:** Mark the tightest start and end bounds that encapsulate the target action or object state.
        3. **Structured Taxonomy:** Classify entities into defined categories (`Action`, `Object`, `Person`, `Scene`, `Audio event`).
        4. **Honest Uncertainty Calibration:** Mark confidence as `Low` or verdict as `Uncertain` when visual occlusions, motion blur, or camera angles prevent conclusive judgment.
        5. **Auditable Reviewer Rationale:** Always provide concise, factual evidence citing timestamp markers in rationales.
        6. **Data Privacy & Compliance:** Use only public, self-recorded, or explicitly licensed footage. Never commit private or sensitive clips.
        """)

    st.divider()
    st.caption("VideoLens AI · micro1 AI/ML Video Data Annotation Portfolio Project · Human Review Protocol")

if __name__ == "__main__":
    main()
