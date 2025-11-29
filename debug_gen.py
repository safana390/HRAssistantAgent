# debug_sig.py
from dotenv import load_dotenv
load_dotenv()
import os, sys, inspect, traceback
from google import genai

print("Python:", sys.version)
print("CWD:", os.getcwd())

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("No GEMINI_API_KEY in .env — aborting.")
    raise SystemExit(1)

client = genai.Client(api_key=api_key)
print("Gemini client created.")

# show what models object exposes
print("\n--- client.models attributes ---")
for name in sorted([n for n in dir(client.models) if not n.startswith("_")]):
    print(name)
    
# inspect generate_content if present
if hasattr(client.models, "generate_content"):
    fn = client.models.generate_content
    print("\n--- generate_content signature ---")
    try:
        sig = inspect.signature(fn)
        print("Signature:", sig)
    except Exception as e:
        print("Could not get signature via inspect:", e)
    print("\n--- generate_content docstring (first 1000 chars) ---")
    print((fn.__doc__ or "No docstring")[:1000])
    # show the raw repr/type
    print("\nType:", type(fn))
    print("Repr (first 1000 chars):", repr(fn)[:1000])

    # Try to call safely by building positional args if signature shows parameters.
    print("\n--- Attempt to call generate_content with safe fallbacks ---")
    PROMPT = "Hello from debug_sig. Please respond with 'OK debug'."
    try:
        # If the signature has positional-only or positional-or-keyword params, try to call with a simple mapping:
        pnames = [p.name for p in sig.parameters.values()]
        print("Parameter names:", pnames)
    except Exception as e:
        print("Could not list parameters:", e)
        pnames = []

    # Attempt a few dynamic calls depending on parameter names
    attempts = []

    # 1) If first param looks like 'model', try positional (model, ???)
    if len(pnames) >= 1 and pnames[0].lower().startswith("model"):
        attempts.append(("positional_model_prompt",
                         lambda: fn("models/gemini-2.5-flash", PROMPT)))
    # 2) If 'model' and something like 'text' or 'prompt' present as names, pass as kwargs
    kwargs = {}
    for key in ("model","text","prompt","input","content","messages"):
        if key in pnames:
            kwargs[key] = PROMPT
    if kwargs:
        attempts.append(("kwargs_based", lambda: fn(**{k:v for k,v in kwargs.items()}))

)
    # 3) try a single positional prompt if signature is single-arg
    if len(pnames) == 1:
        attempts.append(("single_positional", lambda: fn(PROMPT)))
    # 4) try calling without args (some SDKs return an error but we can see response)
    attempts.append(("no_args", lambda: fn()))

    # run attempts
    for desc, call in attempts:
        print("\nAttempt:", desc)
        try:
            resp = call()
            print(" -> Success. Type:", type(resp))
            # print small info if possible
            try:
                import json
                if hasattr(resp, "__dict__"):
                    d = resp.__dict__
                    print("  resp.__dict__ keys:", list(d.keys())[:10])
                elif isinstance(resp, dict):
                    print("  dict keys:", list(resp.keys())[:30])
                else:
                    print("  repr:", repr(resp)[:1000])
            except Exception as e:
                print("  (failed to inspect resp):", e)
            break
        except TypeError as e:
            print(" -> TypeError:", e)
            traceback.print_exc()
        except Exception as e:
            print(" -> Exception:", type(e).__name__, e)
            traceback.print_exc()
else:
    print("client.models does not have generate_content; list client.models methods above and tell me which one looks like generation.")
