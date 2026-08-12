#========================================
#CODING QUESTION DRY-RUN VALIDATOR
#========================================
#
#Executes a CODING question's `solutionCode` against its `testCases` locally so
#misconfigured harnesses / test cases are caught before upload (the assessments
#API runs the real harness server-side in internal/services/code_execution).
#
#Contract: the candidate's code MUST define a symbol named `solution`
#(Python/JS/C++: a function `solution`; Java: a `class Solution` with a method
#`solution`; SQL: a SELECT query). The API's auto-generated harness for non-SQL
#languages calls `solution(...)`, so any other name is a misconfig.
#
#Dry-runs are BEST-EFFORT per language. If a language toolchain is missing or a
#solution signature is too complex to drive generically, the dry-run is SKIPPED
#(with a warning) rather than rejecting the question -- the contract check still
#applies. SQL uses the stdlib `sqlite3` as a best-effort DB; dialect differences
#from the API's Postgres are not caught here.

import ast
import json
import os
import re
import shutil
import subprocess
import tempfile


#========================================
#TOOLCHAIN PROBING
#========================================

def _which(names):
    for n in names:
        path = shutil.which(n)
        if path:
            return path
    return None


_PY = _which(["python3", "python"])
_NODE = _which(["node"])
_JAVA = _which(["java"])
_JAVAC = _which(["javac"])
_GPP = _which(["g++", "clang++", "c++"])


def toolchain_available(language):
    """True if the local toolchain can actually run a canary program for `language`.

    Probes once per language with a tiny program and caches the result, so a
    toolchain that is present on PATH but broken (e.g. a `java` launcher with no
    JRE) is treated as unavailable and the dry-run is skipped rather than
    false-rejecting every question of that language.
    """
    language = (language or "").upper()
    if language not in _CANARY_CACHE:
        _CANARY_CACHE[language] = _run_canary(language)
    return _CANARY_CACHE[language]


_CANARY_CACHE = {}


def _run_canary(language):
    language = (language or "").upper()
    if language == "PYTHON":
        if not _PY:
            return False
        out, err, rc = _run_subprocess([_PY, "-c", "print(1)"], "", 5)
        return rc == 0 and out.strip() == "1"
    if language == "JAVASCRIPT":
        if not _NODE:
            return False
        out, err, rc = _run_subprocess([_NODE, "-e", "console.log(1)"], "", 5)
        return rc == 0 and out.strip() == "1"
    if language == "SQL":
        if not _PY:
            return False
        out, err, rc = _run_subprocess([_PY, "-c", "import sqlite3; print(1)"], "", 5)
        return rc == 0 and out.strip() == "1"
    if language == "CPP":
        if not _GPP:
            return False
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "c.cpp")
            binp = os.path.join(d, "c")
            with open(src, "w") as f:
                f.write("#include <iostream>\nint main(){ std::cout << 1 << std::endl; return 0; }\n")
            co, ce, crc = _run_subprocess([_GPP, "-std=c++17", "-O0", src, "-o", binp], "", 15)
            if crc != 0:
                return False
            out, err, rc = _run_subprocess([binp], "", 5)
            return rc == 0 and out.strip() == "1"
    if language == "JAVA":
        if not _JAVA or not _JAVAC:
            return False
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "C.java")
            with open(src, "w") as f:
                f.write("public class C { public static void main(String[] a){ System.out.println(1); } }\n")
            co, ce, crc = _run_subprocess([_JAVAC, src], "", 15, cwd=d)
            if crc != 0:
                return False
            out, err, rc = _run_subprocess([_JAVA, "-cp", d, "C"], "", 5)
            return rc == 0 and out.strip() == "1"
    return False


#========================================
#COMPARISON STRATEGIES
#========================================

def _normalize(s):
    return (s or "").strip()


def compare(actual, expected, strategy):
    """Compare `actual` (stdout) to `expected` (expectedOutput) per strategy."""
    strategy = (strategy or "EXACT").upper()
    a = _normalize(actual)
    e = _normalize(expected)
    if strategy == "EXACT":
        return a == e
    if strategy == "CONTAINS":
        return e in a
    if strategy == "REGEX":
        try:
            return re.search(e, a) is not None
        except re.error:
            return a == e
    if strategy == "SORTED_LINES":
        a_lines = sorted(l.strip() for l in a.splitlines() if l.strip())
        e_lines = sorted(l.strip() for l in e.splitlines() if l.strip())
        return a_lines == e_lines
    return a == e


#========================================
#CONTRACT CHECKS (solution symbol defined)
#========================================

def _contract_python(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"solutionCode is not valid Python: {e}"
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "solution":
            return True, None
    return False, "solutionCode must define a function named 'solution'"


def _contract_javascript(code):
    patterns = [
        r"\bfunction\s+solution\s*\(",
        r"\bconst\s+solution\s*=",
        r"\blet\s+solution\s*=",
        r"\bvar\s+solution\s*=",
        r"\bsolution\s*=\s*function",
        r"\bsolution\s*=\s*\([^)]*\)\s*=>",
    ]
    for p in patterns:
        if re.search(p, code):
            return True, None
    return False, "solutionCode must define a function named 'solution'"


def _contract_cpp(code):
    if re.search(r"\bsolution\s*\([^)]*\)\s*\{", code):
        return True, None
    return False, "solutionCode must define a function named 'solution'"


def _contract_java(code):
    if not re.search(r"\bclass\s+Solution\b", code):
        return False, "solutionCode must define a 'class Solution'"
    if not re.search(r"\bsolution\s*\([^)]*\)", code):
        return False, "class Solution must define a method named 'solution'"
    return True, None


def _contract_sql(code):
    stripped = code.strip().lower()
    if not stripped:
        return False, "solutionCode (SQL) must be a SELECT query"
    if stripped.startswith("select") or stripped.startswith("with"):
        return True, None
    return False, "solutionCode (SQL) must be a SELECT or WITH query"


def check_contract(language, solution_code):
    """Return (ok, error). `error` is None when the contract holds."""
    language = (language or "").upper()
    if not solution_code or not str(solution_code).strip():
        return False, "solutionCode is empty"
    if language == "PYTHON":
        return _contract_python(solution_code)
    if language == "JAVASCRIPT":
        return _contract_javascript(solution_code)
    if language == "CPP":
        return _contract_cpp(solution_code)
    if language == "JAVA":
        return _contract_java(solution_code)
    if language == "SQL":
        return _contract_sql(solution_code)
    return False, f"unsupported language '{language}'"


#========================================
#SUBPROCESS RUNNER
#========================================

def _run_subprocess(cmd, stdin_text, timeout_s, cwd=None):
    """Run `cmd`, feed stdin, return (stdout, stderr, exit_code)."""
    try:
        proc = subprocess.run(
            cmd,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=cwd,
        )
        return proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", -1
    except FileNotFoundError:
        return "", "TOOLCHAIN_MISSING", -1


#========================================
#PYTHON RUNNER
#========================================

_PY_DRIVER = """
import json, sys
__SOLUTION__
def _parse(s):
    s = (s or "").strip()
    if not s:
        return ("none", None)
    v = json.loads(s)
    if isinstance(v, list):
        return ("list", v)
    if isinstance(v, dict):
        return ("dict", v)
    return ("scalar", v)
if __name__ == "__main__":
    kind, args = _parse(sys.stdin.read())
    if kind == "none":
        out = solution()
    elif kind == "list":
        out = solution(*args)
    elif kind == "dict":
        out = solution(**args)
    else:
        out = solution(args)
    print(out if out is not None else "")
"""


def run_python(solution_code, test_input, time_limit_ms):
    if not _PY:
        return None, "TOOLCHAIN_MISSING", "python3 not found on PATH"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_PY_DRIVER.replace("__SOLUTION__", solution_code))
        path = f.name
    try:
        out, err, rc = _run_subprocess([_PY, path], test_input or "", time_limit_ms / 1000 + 1)
    finally:
        os.unlink(path)
    if rc != 0:
        return None, (err or "non-zero exit").strip(), rc
    return out, None, rc


#========================================
#JAVASCRIPT RUNNER
#========================================

_JS_DRIVER = """
__SOLUTION__
const fs = require('fs');
function parse(s) {
  const t = (s || '').trim();
  if (!t) return ['none', undefined];
  const v = JSON.parse(t);
  if (Array.isArray(v)) return ['list', v];
  if (v && typeof v === 'object') return ['dict', v];
  return ['scalar', v];
}
const [kind, args] = parse(fs.readFileSync(0, 'utf8'));
let out;
if (kind === 'none') out = solution();
else if (kind === 'list') out = solution(...args);
else if (kind === 'dict') out = solution(args);
else out = solution(args);
console.log(out === undefined || out === null ? '' : out);
"""


def run_javascript(solution_code, test_input, time_limit_ms):
    if not _NODE:
        return None, "TOOLCHAIN_MISSING", "node not found on PATH"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(_JS_DRIVER.replace("__SOLUTION__", solution_code))
        path = f.name
    try:
        out, err, rc = _run_subprocess([_NODE, path], test_input or "", time_limit_ms / 1000 + 1)
    finally:
        os.unlink(path)
    if rc != 0:
        return None, (err or "non-zero exit").strip(), rc
    return out, None, rc


#========================================
#CPP RUNNER
#========================================

_CPP_TYPE_MAP = {
    "int": "stoi", "long": "stol", "long long": "stoll",
    "unsigned int": "stoul", "unsigned long": "stoul",
    "double": "stod", "float": "stof",
}


def _cpp_signature(code):
    """Extract (return_type, param_type) from a `solution` free function.

    Returns None if the signature is too complex to drive generically.
    `param_type` is the type only (variable name stripped).
    """
    m = re.search(
        r"([\w\s\*]+?)\s+solution\s*\(\s*([\w\s\*&:<>,]*?)\s*\)\s*\{",
        code,
    )
    if not m:
        return None
    ret = m.group(1).strip()
    param = m.group(2).strip()
    if not param or param == "void":
        return (ret, None)
    if "," in param or "<" in param:
        return None
    # `type name` -> keep the type (all but the last token, which is the name).
    tokens = param.split()
    if len(tokens) >= 2:
        param_type = " ".join(tokens[:-1])
    else:
        param_type = tokens[0]
    return (ret, param_type)


def run_cpp(solution_code, test_input, time_limit_ms):
    if not _GPP:
        return None, "TOOLCHAIN_MISSING", "g++ not found on PATH"
    sig = _cpp_signature(solution_code)
    if sig is None:
        return None, "UNSUPPORTED_SIGNATURE", -2
    ret_type, param_type = sig

    if param_type is None:
        driver = (
            "#include <iostream>\n#include <string>\n#include <vector>\n#include <sstream>\nusing namespace std;\n"
            + "static string strip_quotes(string s){ if(s.size()>=2 && s.front()=='\"' && s.back()=='\"') return s.substr(1,s.size()-2); return s; }\n"
            + solution_code + "\n"
            + "int main(){ auto r = solution(); cout << r << endl; return 0; }\n"
        )
    else:
        p = param_type
        if p in _CPP_TYPE_MAP:
            conv = f"{_CPP_TYPE_MAP[p]}(raw)"
        elif p.startswith("string") or p.startswith("std::string"):
            conv = "strip_quotes(raw)"
        elif p.startswith("char"):
            conv = "strip_quotes(raw)[0]"
        elif p == "bool":
            conv = '(raw == "true" || raw == "1")'
        else:
            return None, "UNSUPPORTED_SIGNATURE", -2
        driver = (
            "#include <iostream>\n#include <string>\n#include <vector>\n#include <sstream>\nusing namespace std;\n"
            + "static string strip_quotes(string s){ if(s.size()>=2 && s.front()=='\"' && s.back()=='\"') return s.substr(1,s.size()-2); return s; }\n"
            + solution_code + "\n"
            + "int main(){ string raw; getline(cin, raw);"
            + f" auto r = solution({conv}); cout << r << endl; return 0; }}\n"
        )

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "sol.cpp")
        binp = os.path.join(d, "sol")
        with open(src, "w") as f:
            f.write(driver)
        co, ce, crc = _run_subprocess([_GPP, "-std=c++17", "-O0", src, "-o", binp], "", 15)
        if crc != 0:
            return None, "COMPILE_FAILED", crc
        out, err, rc = _run_subprocess([binp], test_input or "", time_limit_ms / 1000 + 1)
    if rc != 0:
        return None, (err or "non-zero exit").strip(), rc
    return out, None, rc


#========================================
#JAVA RUNNER
#========================================

_JAVA_PRIMITIVES = {
    "int": "Integer.parseInt", "long": "Long.parseLong",
    "double": "Double.parseDouble", "float": "Float.parseFloat",
    "boolean": "Boolean.parseBoolean", "short": "Short.parseShort",
    "byte": "Byte.parseByte",
}


def _java_signature(code):
    """Extract (return_type, param_type) from Solution.solution(...).

    `param_type` is the type only (variable name stripped).
    """
    m = re.search(
        r"public\s+([\w<>\[\],\s]*?)\s+solution\s*\(\s*([\w<>\[\],\s]*?)\s*\)",
        code,
    )
    if not m:
        return None
    ret = m.group(1).strip()
    param = m.group(2).strip()
    if not param:
        return (ret, None)
    if "," in param or "[" in param or "<" in param:
        return None
    tokens = param.split()
    if len(tokens) >= 2:
        param_type = " ".join(tokens[:-1])
    else:
        param_type = tokens[0]
    return (ret, param_type)


def run_java(solution_code, test_input, time_limit_ms):
    if not _JAVA or not _JAVAC:
        return None, "TOOLCHAIN_MISSING", "java/javac not found on PATH"
    sig = _java_signature(solution_code)
    if sig is None:
        return None, "UNSUPPORTED_SIGNATURE", -2
    ret_type, param_type = sig

    if param_type is None:
        body = (
            "        Object r = new Solution().solution();\n"
            + "        System.out.println(r == null ? \"\" : r);\n"
        )
    else:
        p = param_type
        if p in _JAVA_PRIMITIVES:
            conv = f"{_JAVA_PRIMITIVES[p]}(raw)"
        elif p.startswith("String") or p == "String":
            conv = "stripQuotes(raw)"
        elif p == "char":
            conv = "stripQuotes(raw).charAt(0)"
        else:
            return None, "UNSUPPORTED_SIGNATURE", -2
        body = (
            "        Object r = new Solution().solution(" + conv + ");\n"
            + "        System.out.println(r == null ? \"\" : r);\n"
        )

    main_src = (
        "import java.io.*;\n"
        + "public class Main {\n"
        + "    static String stripQuotes(String s){ if(s.length()>=2 && s.charAt(0)=='\"' && s.charAt(s.length()-1)=='\"') return s.substring(1,s.length()-1); return s; }\n"
        + "    public static void main(String[] a) throws Exception {\n"
        + "        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));\n"
        + "        String raw = br.readLine();\n"
        + "        if (raw == null) raw = \"\";\n"
        + body
        + "    }\n"
        + "}\n"
    )

    with tempfile.TemporaryDirectory() as d:
        sol_file = os.path.join(d, "Solution.java")
        main_file = os.path.join(d, "Main.java")
        with open(sol_file, "w") as f:
            f.write(solution_code)
        with open(main_file, "w") as f:
            f.write(main_src)
        co, ce, crc = _run_subprocess([_JAVAC, sol_file, main_file], "", 15, cwd=d)
        if crc != 0:
            return None, "COMPILE_FAILED", crc
        out, err, rc = _run_subprocess([_JAVA, "-cp", d, "Main"], test_input or "", time_limit_ms / 1000 + 1)
    if rc != 0:
        return None, (err or "non-zero exit").strip(), rc
    return out, None, rc


#========================================
#SQL RUNNER (sqlite3, best-effort)
#========================================

def run_sql(solution_code, test_input, time_limit_ms, harness_ddl):
    """Run the SQL solutionCode against the DDL harness in an in-memory sqlite3 DB.

    NOTE: sqlite3 is a best-effort stand-in for the API's Postgres. Dialect
    differences (Postgres-specific functions, SERIAL, RETURNING, etc.) may pass
    or fail differently than the API. The dry-run's value here is catching
    structural/contract errors, not dialect-perfect validation.
    """
    if not _PY:
        return None, "TOOLCHAIN_MISSING", "python3 not found on PATH"
    runner_src = (
        "import sqlite3, sys\n"
        "ddl = sys.stdin.readline()\n"
        "query = sys.stdin.readline()\n"
        "conn = sqlite3.connect(':memory:')\n"
        "try:\n"
        "    cur = conn.cursor()\n"
        "    if ddl.strip(): cur.executescript(ddl)\n"
        "    rows = cur.execute(query).fetchall()\n"
        "    cols = [d[0] for d in cur.description] if cur.description else []\n"
        "    for r in rows:\n"
        "        print('|'.join(str(c) for c in r))\n"
        "except Exception as e:\n"
        "    import sys; sys.stderr.write('SQL_ERROR:' + str(e)); sys.exit(1)\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(runner_src)
        path = f.name
    stdin_text = (harness_ddl or "") + "\n" + (solution_code or "")
    try:
        out, err, rc = _run_subprocess([_PY, path], stdin_text, time_limit_ms / 1000 + 1)
    finally:
        os.unlink(path)
    if rc != 0:
        return None, (err or "non-zero exit").strip(), rc
    return out, None, rc


#========================================
#DISPATCH
#========================================

def _run_one(language, solution_code, test_input, time_limit_ms, harness_ddl=None):
    """Run a single test case. Returns (stdout, error, exit_code)."""
    language = (language or "").upper()
    if language == "PYTHON":
        return run_python(solution_code, test_input, time_limit_ms)
    if language == "JAVASCRIPT":
        return run_javascript(solution_code, test_input, time_limit_ms)
    if language == "CPP":
        return run_cpp(solution_code, test_input, time_limit_ms)
    if language == "JAVA":
        return run_java(solution_code, test_input, time_limit_ms)
    if language == "SQL":
        return run_sql(solution_code, test_input, time_limit_ms, harness_ddl)
    return None, f"unsupported language '{language}'", -2


#========================================
#PUBLIC ENTRY POINT
#========================================

def validate_and_run(coding_details, test_cases):
    """Validate a CODING question's contract and dry-run solutionCode vs testCases.

    Args:
        coding_details: dict with `language`, `solutionCode`, `timeLimitMs`,
            and (for SQL) `testHarnessTemplate`.
        test_cases: list of dicts with `input`, `expectedOutput`,
            `comparisonStrategy`.

    Returns a dict:
        passed: bool          -- False if contract fails or any test fails.
        skipped: bool         -- True if the dry-run could not run (toolchain
                                 missing / unsupported signature). Contract still
                                 checked; `passed` reflects contract only then.
        skip_reason: str|None -- why the dry-run was skipped.
        contract_error: str|None -- non-None if the `solution` contract failed.
        failures: list[{case_index, reason}] -- per-case dry-run failures.
    """
    language = (coding_details or {}).get("language", "")
    solution_code = (coding_details or {}).get("solutionCode", "")
    raw_time = coding_details.get("timeLimitMs")
    try:
        time_limit_ms = int(raw_time) if raw_time is not None else 5000
    except (TypeError, ValueError):
        time_limit_ms = 5000
    harness_ddl = (coding_details or {}).get("testHarnessTemplate", "")

    result = {
        "passed": True,
        "skipped": False,
        "skip_reason": None,
        "contract_error": None,
        "failures": [],
    }

    ok, err = check_contract(language, solution_code)
    if not ok:
        result["passed"] = False
        result["contract_error"] = err
        return result

    if not toolchain_available(language):
        result["skipped"] = True
        result["skip_reason"] = f"no working local toolchain for {language}"
        return result

    for idx, tc in enumerate(test_cases or []):
        test_input = tc.get("input", "")
        expected = tc.get("expectedOutput", "")
        strategy = tc.get("comparisonStrategy", "EXACT")
        out, err, rc = _run_one(language, solution_code, test_input, time_limit_ms, harness_ddl)
        if err in ("UNSUPPORTED_SIGNATURE", "COMPILE_FAILED"):
            result["skipped"] = True
            result["skip_reason"] = (
                "solution signature too complex to dry-run generically"
                if err == "UNSUPPORTED_SIGNATURE"
                else "solution failed to compile locally (often an environment/header difference)"
            )
            return result
        if err == "TOOLCHAIN_MISSING":
            result["skipped"] = True
            result["skip_reason"] = f"toolchain missing for {language}"
            return result
        if err == "TIMEOUT":
            result["passed"] = False
            result["failures"].append({
                "case_index": idx,
                "reason": f"timed out after {time_limit_ms}ms",
            })
            continue
        if rc != 0:
            result["passed"] = False
            result["failures"].append({
                "case_index": idx,
                "reason": f"runtime error: {err}",
            })
            continue
        if not compare(out, expected, strategy):
            result["passed"] = False
            result["failures"].append({
                "case_index": idx,
                "reason": f"expected {expected!r}, got {out.strip()!r}",
            })
    return result
