import sys

# CrewAI prints emoji to the console; Windows terminals default to a legacy encoding
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")
