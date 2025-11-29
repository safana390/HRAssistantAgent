import streamlit as st
import subprocess
import sys
from pathlib import Path
from typing import Optional

st.set_page_config(page_title="HR Policy Agent — Hybrid", layout="centered")

st.title("HR Policy Agent")
st.markdown(
    "Ask HR questions and get answers synthesized from your uploaded company PDFs and Gemini fallback."
)

MODE = st.radio("Answering mode:", ["Hybrid (PDF first)", "PDF-only", "Gemini-only"])
question = st.text_input("Enter your question:")

col1, col2 = st.columns([1, 1])
with col1:
    ask = st.button("Ask")
with col2:
    refresh = st.button("Refresh (re-run ingest)")

# project root (two levels up from this file -> repo root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_ingest() -> tuple[int, str, str]:
    """Run the ingest script as a subprocess and return (returncode, stdout, stderr)."""
    script = PROJECT_ROOT / "ingest" / "ingest.py"
    if not script.exists():
        return (1, "", f"Ingest script not found at: {script}")
    cmd = [sys.executable, str(script), "--folder", "sample_docs", "--collection", "hr_policies"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return (proc.returncode, proc.stdout, proc.stderr)


def run_answer_script_via_subprocess(q: str) -> str:
    """
    Call answer_generator_hybrid.py as a subprocess and send the question on stdin.
    Falls back to capturing stdout/stderr and returning combined output.
    """
    script = PROJECT_ROOT / "answer_generator_hybrid.py"
    if not script.exists():
        return "Error: answer_generator_hybrid.py not found in project root."

    cmd = [sys.executable, str(script)]
    try:
        proc = subprocess.run(cmd, input=q + "\n", capture_output=True, text=True, timeout=120)
        out = proc.stdout or ""
        err = proc.stderr or ""
        if proc.returncode != 0:
            return f"Script exited with code {proc.returncode}.\n\nSTDOUT:\n{out}\n\nSTDERR:\n{err}"
        return out + ("\n\nSTDERR:\n" + err if err else "")
    except subprocess.TimeoutExpired:
        return "Error: process timed out."
    except Exception as e:
        return f"Error running script: {e}"


if refresh:
    st.info("Running ingest script to refresh documents. This may take a moment...")
    code, out, err = run_ingest()
    if code == 0:
        st.success("Ingest completed. Check terminal output for details.")
        if out:
            st.code(out[:4000])
    else:
        st.error("Ingest failed. See output below.")
        combined = (out or "") + ("\n\n" + err if err else "")
        st.code(combined or "No output captured.")


def call_answer_module(question_text: str, use_hybrid: bool) -> Optional[str]:
    """
    Try to import answer_generator_hybrid as a module and call an `answer()` function if present.
    If import or call fails, return None so caller can fall back to subprocess.
    """
    try:
        import importlib

        ag = importlib.import_module("answer_generator_hybrid")
        # Prefer an answer(question, use_hybrid=True/False) signature
        if hasattr(ag, "answer"):
            try:
                # try with use_hybrid kwarg
                return ag.answer(question_text, use_hybrid=use_hybrid)
            except TypeError:
                # try without the kwarg
                try:
                    return ag.answer(question_text)
                except Exception as e:
                    return f"Module answer() raised exception: {e}"
        # If module exposes main() only, fallback to subprocess runner
        return None
    except Exception as exc:
        return None


if ask and question.strip():
    st.markdown("---")
    st.header("Answer")
    with st.spinner("Generating answer..."):
        result_text = None
        use_hybrid = MODE == "Hybrid (PDF first)"

        # 1) Prefer direct import & function call (faster, nicer)
        result_text = call_answer_module(question, use_hybrid=use_hybrid)

        # 2) Fallback: run the script as subprocess (works if the script uses input())
        if result_text is None:
            result_text = run_answer_script_via_subprocess(question)

    if not result_text:
        st.error(
            "No answer returned. Ensure `answer_generator_hybrid.py` defines an `answer()` function "
            "or is runnable as a script that reads a question from stdin."
        )
    else:
        # show the returned output
        st.code(result_text)

        # allow user to download the text
        st.download_button("Download answer", result_text, file_name="hr_answer.txt")
        st.success("Done")

else:
    if not question.strip():
        st.info("Type a question above and press Ask. Examples:\n• 'What is the leave policy?'\n• 'How many sick days do employees get?'")
    else:
        st.warning("Press Ask to run the query.")


# Footer
st.markdown("---")
st.caption(
    "Hybrid mode: uses Chroma (PDFs) first, then Gemini fallback. Make sure your virtualenv is activated and .env contains GEMINI_API_KEY and CHROMA_DIR."
)
