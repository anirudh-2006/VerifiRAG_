# src/verifier.py — complete updated version
import sympy as sp
import re
from sympy import (symbols, sin, cos, exp, integrate, simplify,
                   trigsimp, I, oo, pi, Piecewise, Abs, expand,
                   nsimplify, factor, cancel)
from dataclasses import dataclass
from typing import Optional, Any

t, omega = symbols('t omega', real=True)
s = symbols('s')           # complex for Laplace (re(s) must stay symbolic)
z = symbols('z')           # complex for Z-transform
T = symbols('T', real=True, positive=True)
j = I

@dataclass
class StepResult:
    step_number: int
    description: str
    expression_given: str
    expression_parsed: Optional[sp.Expr]
    previous_parsed: Optional[sp.Expr]
    is_valid: Optional[bool]
    error_message: str = ""
    equivalence_checked: bool = False

def _convert_to_piecewise(expr: str) -> str:
    """
    Convert natural-language piecewise expressions to SymPy Piecewise syntax.
    Handles patterns like:
      "1 if t > 0 else 0"
      "1 for -T/4 < t < T/4, -1 for T/4 < t < 3*T/4"
      "exp(-t) for t >= 0, 0 for t < 0"
    """
    # Pattern 1: Python ternary  "value if cond else other"
    m = re.match(r"^(.+?)\s+if\s+(.+?)\s+else\s+(.+)$", expr.strip())
    if m:
        val, cond, other = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        cond = cond.replace("and", "&").replace("or", "|")
        return f"Piecewise(({val}, {cond}), ({other}, True))"

    # Pattern 2: comma-separated branches with "for" / "if"
    # e.g. "1 for -T/4 < t < T/4, -1 for T/4 < t < 3*T/4"
    branches = [b.strip() for b in expr.split(",") if b.strip()]
    if len(branches) >= 2:
        pieces = []
        for branch in branches:
            # strip any leading assignment like "x(t) = " or "x = "
            branch = re.sub(r"^[A-Za-z_]\w*(?:\([^)]*\))?\s*=\s*", "", branch)
            bm = re.match(r"^(.+?)\s+(?:for|if)\s+(.+)$", branch)
            if bm:
                val = bm.group(1).strip()
                cond = bm.group(2).strip()
                cond = cond.replace("and", "&").replace("or", "|")
                # convert chained comparison "a < t < b" to "And(a < t, t < b)"
                chain = re.match(r"^(.+?)\s*([<>]=?)\s*(\w+)\s*([<>]=?)\s*(.+)$", cond)
                if chain:
                    cond = f"And({chain.group(1)} {chain.group(2)} {chain.group(3)}, {chain.group(3)} {chain.group(4)} {chain.group(5)})"
                pieces.append(f"({val}, {cond})")
        if pieces:
            return "Piecewise(" + ", ".join(pieces) + ")"

    # Pattern 3: single "value for condition" — just extract the value
    sm = re.match(r"^(.+?)\s+(?:for|if)\s+(.+?)(?:\s+else\s+.*)?$", expr.strip())
    if sm:
        return sm.group(1).strip()

    return expr


def preprocess_expression(expr_str: str) -> str:
    expr = expr_str.strip()

    # ── evaluation-at notation: expr | (var=val) ──
    # Must run very early, before '=' handling misinterprets "(t=1)".
    # e.g. "(exp(-I*omega*t)/(-I*omega)) | (t=1) - (...) | (t=-1)"
    _eval_at = re.compile(r'\)\s*\|\s*\(\s*(\w+)\s*=\s*([^)]+)\)')
    if '|' in expr and _eval_at.search(expr):
        expr = _eval_at.sub(r').subs(\1, \2)', expr)

    # Special case: assignment with ternary on RHS, e.g. "ak = 0 if ... else ..."
    # Must run before generic '=' handling because ternary conditions often contain '=='.
    ternary_assign = re.match(r"^\s*[A-Za-z_]\w*(?:\([^)]*\))?\s*=\s*(.+\s+if\s+.+\s+else\s+.+)\s*$", expr)
    if ternary_assign:
        expr = ternary_assign.group(1).strip()

    # Handle multi-branch piecewise before generic '=' logic so strings like
    # "1 for ..., x(t) = -1 for ..." are not misinterpreted as equalities.
    if (' for ' in expr or ' if ' in expr) and ',' in expr:
        maybe_pw = _convert_to_piecewise(expr)
        if maybe_pw.startswith("Piecewise("):
            return maybe_pw

    # remove left-hand side of assignments (protect ==, <=, >=, !=)
    if "=" in expr and not expr.strip().startswith("Piecewise"):
        if not re.search(r"[!<>]=|==", expr):
            parts = expr.split("=", 1)
            lhs = parts[0].strip()
            rhs = parts[1].strip()
            # strip simple assignments like X(jw) = ...
            if re.match(r"^[A-Za-z_]\w*(\(.*\))?$", lhs):
                expr = rhs
            else:
                # If this is a mathematical equality, convert to an expression.
                # Example: exp(-I*k*w*T/2) = (-1)**k  ->  exp(...) - ((-1)**k)
                expr = f"({lhs}) - ({rhs})"

    # strip ALL sympy.X patterns in one pass
    expr = re.sub(r"sympy\.(\w+)", r"\1", expr)

    # remove kwargs that frequently break parsing in generated expressions
    # e.g., fourier_transform(..., name='dtft')
    expr = re.sub(r",\s*name\s*=\s*'[^']*'\s*", "", expr)
    expr = re.sub(r',\s*name\s*=\s*"[^"]*"\s*', "", expr)

    # replace subscript notation x[n] → x_n, x[n-k] → x_nk (opaque symbol)
    def _sub_bracket(m):
        name, idx = m.group(1), m.group(2).strip()
        if re.match(r'^\w+$', idx):
            return f'{name}_{idx}'
        # Complex index like n-k: collapse to an opaque symbol so it parses.
        idx_clean = re.sub(r'[^a-zA-Z0-9]', '', idx)
        return f'{name}_{idx_clean}'
    expr = re.sub(r"(\w+)\[([^\]]+)\]", _sub_bracket, expr)

    # normalise imaginary unit
    for old, new in [
        ("1j",   "I"),
        ("-1j",  "-I"),
        ("(-1j", "(-I"),
        ("2j",   "2*I"),
        ("-2j",  "-2*I"),
        (" j*",  " I*"),
        ("(j*",  "(I*"),
        ("(-j*", "(-I*"),
        ("/j",   "/I"),
        ("/ j",  "/ I"),
    ]:
        expr = expr.replace(old, new)

    expr = expr.replace("^", "**")

    # ── Re(...) / Im(...) → re(...) / im(...) for SymPy ──
    # Must run before func-call stripping so Re(s) isn't caught as signal name.
    expr = re.sub(r'\bRe\(', 're(', expr)
    expr = re.sub(r'\bIm\(', 'im(', expr)

    # ── function-call notation → plain symbols ──
    # LLMs write x(t), H(z), X(s), y(tau), etc.  SymPy's implicit
    # multiplication would turn these into x*t, H*z ... which is wrong.
    # We strip the argument list, keeping only the function name as a symbol.
    # Specific patterns first, then a general regex catch-all.
    for old, new in [
        ("X1(jomega)", "X1"), ("X(jomega)", "X"),
        ("x1(t)", "x1"),      ("H(jomega)", "H"),
    ]:
        expr = expr.replace(old, new)
    # General: F(var) where F is a single uppercase or lowercase letter or
    # short name (x, y, h, H, X, Y, x_sigma, etc.) and var is a known
    # independent variable or comma-separated list thereof.
    _known_vars = r't|omega|s|z|n|k|tau|u|Omega|jw|j\s*omega|I\*omega|I\*Omega'
    _func_call_re = re.compile(
        r'\b([A-Za-z_]\w{0,12})\(\s*(' + _known_vars + r')(?:\s*[,\-+]\s*(?:' + _known_vars + r'))?\s*\)'
    )
    def _strip_func_call(m):
        fname = m.group(0)
        # Don't strip real SymPy / math functions
        _keep = {'sin','cos','exp','log','sqrt','Abs','abs','Piecewise','And',
                 'Sum','Integral','integrate','summation','Heaviside',
                 'FourierTransform','LaplaceTransform','conjugate','arg',
                 're','im','sign','Max','Min','ceiling','floor','Symbol',
                 'delta','DiracDelta','subs'}
        if m.group(1) in _keep:
            return fname
        # Don't strip if preceded by '.' (method call like x.subs(t, tau))
        start = m.start()
        if start > 0 and expr[start - 1] == '.':
            return fname
        return m.group(1)
    expr = _func_call_re.sub(_strip_func_call, expr)

    # ── 'and' / 'or' → bitwise operators for SymPy ──
    # Works both inside and outside Piecewise conditions.
    if re.search(r'\band\b', expr, re.IGNORECASE):
        expr = re.sub(r'\band\b', '&', expr, flags=re.IGNORECASE)
    if re.search(r'\bor\b', expr, re.IGNORECASE):
        expr = re.sub(r'\bor\b', '|', expr, flags=re.IGNORECASE)

    # evaluate .subs() calls
    if ".subs(" in expr:
        try:
            local_dict = _local_dict()
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations
            evaluated = parse_expr(expr, local_dict=local_dict,
                                   transformations=standard_transformations)
            return str(evaluated)
        except Exception:
            pass

    if ' if ' in expr or ' for ' in expr:
        expr = _convert_to_piecewise(expr)
        if expr.startswith("Piecewise("):
            return expr

    return expr

def _local_dict():
    """Shared symbol/function dictionary for parsing."""
    n, k, m = sp.symbols('n k m', integer=True)
    a, r, sigma = sp.symbols('a r sigma', real=True)
    Omega = sp.Symbol('Omega', real=True)
    return {
        # symbols
        't': t, 'omega': omega, 's': s, 'z': z, 'T': T,
        'n': n, 'k': k, 'm': m, 'a': a, 'r': r,
        'sigma': sigma, 'Omega': Omega,
        # imaginary
        'I': I, 'j': I,
        # trig and exponential
        'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp,
        'pi': sp.pi, 'oo': sp.oo, 'inf': sp.oo,
        # calculus
        'integrate': sp.integrate, 'summation': sp.summation,
        'Sum': sp.Sum, 'Integral': sp.Integral,
        # complex
        're': sp.re, 'im': sp.im,
        'conjugate': sp.conjugate, 'arg': sp.arg,
        'Abs': sp.Abs, 'abs': sp.Abs,
        # other functions
        'sqrt': sp.sqrt, 'log': sp.log,
        'Heaviside': sp.Heaviside,
        'Piecewise': sp.Piecewise,
        'And': sp.And,
        # named variables / signal symbols
        'x1': sp.Symbol('x1'), 'X1': sp.Symbol('X1'),
        'X': sp.Symbol('X'),   'H': sp.Symbol('H'),
        'x': sp.Symbol('x'),   'Y': sp.Symbol('Y'),
        'y': sp.Symbol('y'),   'u': sp.Symbol('u'),
        'x_n': sp.Symbol('x_n'), 'h_n': sp.Symbol('h_n'),
        'y_n': sp.Symbol('y_n'), 'u_n': sp.Symbol('u_n'),
        'h_k': sp.Symbol('h_k'), 'x_k': sp.Symbol('x_k'),
        'X_dtft': sp.Symbol('X_dtft'),
        'x_sigma': sp.Symbol('x_sigma'),
        'T0': sp.Symbol('T0', positive=True),
        't0': sp.Symbol('t0', real=True),
        'tau': sp.Symbol('tau', real=True),
        'delta': sp.DiracDelta,
        'DiracDelta': sp.DiracDelta,
        'FourierTransform': sp.FourierTransform,
        'LaplaceTransform': sp.LaplaceTransform,
        'pole_location': sp.Symbol('pole_location'),
        'pole_real_part': sp.Symbol('pole_real_part'),
    }

_LEADING_ASSIGN_RE = re.compile(
    r"""^\s*
        (?:[A-Za-z_][A-Za-z0-9_]*      # bare identifier, e.g. X_dtft
           (?:\([^()=]{0,40}\))?       #   optionally a function arg list, e.g. H(z)
        )
        \s*=(?!=)\s*                   # a single '=' (not '==')
    """,
    re.VERBOSE,
)


def _strip_leading_assignment(expr_str: str) -> str:
    """
    Strip a single leading 'NAME = ' or 'NAME(args) = ' prefix from a final
    answer string so the RHS can be parsed on its own. Does nothing if the
    string doesn't start with such a prefix.
    """
    if not expr_str:
        return expr_str
    m = _LEADING_ASSIGN_RE.match(expr_str)
    if not m:
        return expr_str
    return expr_str[m.end():]


# ---------------------------------------------------------------------------
# Final-answer parser
# ---------------------------------------------------------------------------
# Unlike `safe_parse` (used for derivation-step verification), this parser
# registers common signal names (X, H, x, Y, ...) as SymPy *Functions* so
# that e.g. X(I*omega) stays as an applied function object rather than being
# treated as implicit multiplication (X*I*omega) or being stripped down to X.

_SIGNAL_FUNC_NAMES = [
    'X', 'H', 'Y', 'M', 'x', 'y', 'h', 'm',
    'X1', 'X2', 'x1', 'x2',
    'X_z', 'X_dtft', 'H_d', 'M_FT',
]


def _final_local_dict():
    """Symbol/function dictionary for final-answer parsing."""
    base = _local_dict()
    for name in _SIGNAL_FUNC_NAMES:
        base[name] = sp.Function(name)
    return base


def _preprocess_final(expr_str: str) -> str:
    """Light preprocessing for final-answer strings.

    Much less aggressive than `preprocess_expression` — we do NOT strip
    function-call arguments (the whole point is to preserve X(I*omega)).
    """
    expr = expr_str.strip()

    expr = re.sub(r"sympy\.(\w+)", r"\1", expr)
    expr = expr.replace("^", "**")
    expr = re.sub(r'\bRe\(', 're(', expr)
    expr = re.sub(r'\bIm\(', 'im(', expr)

    # x[n] → x_n
    def _sub_bracket(m):
        name, idx = m.group(1), m.group(2).strip()
        if re.match(r'^\w+$', idx):
            return f'{name}_{idx}'
        idx_clean = re.sub(r'[^a-zA-Z0-9]', '', idx)
        return f'{name}_{idx_clean}'
    expr = re.sub(r"(\w+)\[([^\]]+)\]", _sub_bracket, expr)

    for old, new in [
        ("1j",   "I"), ("-1j",  "-I"), ("(-1j", "(-I"),
        ("2j",   "2*I"), ("-2j",  "-2*I"),
        (" j*",  " I*"), ("(j*",  "(I*"), ("(-j*", "(-I*"),
        ("/j",   "/I"), ("/ j",  "/ I"),
    ]:
        expr = expr.replace(old, new)

    return expr


def safe_parse_final(expr_str: str) -> Optional[sp.Expr]:
    """Parse a final-answer expression preserving function-call semantics."""
    if not expr_str or expr_str.strip() == "":
        return None
    cleaned = _preprocess_final(expr_str)
    try:
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations,
            implicit_multiplication_application
        )
        transformations = (standard_transformations +
                          (implicit_multiplication_application,))
        result = parse_expr(cleaned, local_dict=_final_local_dict(),
                            transformations=transformations)
        if isinstance(result, bool):
            return None
        return result
    except Exception:
        return None


def _equiv_final(g: sp.Expr, e: sp.Expr) -> bool:
    """Check equivalence of two final-answer expressions.

    Handles the omega/Omega case-sensitivity issue: if the direct comparison
    fails, retry after substituting Omega → omega (or vice versa) since
    continuous and discrete frequency are often conflated by LLMs.
    """
    if check_step_equivalence(g, e):
        return True
    # Omega ↔ omega normalization
    Omega_sym = sp.Symbol('Omega', real=True)
    omega_sym = sp.Symbol('omega', real=True)
    try:
        g_sub = g.subs(Omega_sym, omega_sym)
        e_sub = e.subs(Omega_sym, omega_sym)
        if check_step_equivalence(g_sub, e_sub):
            return True
        g_sub2 = g.subs(omega_sym, Omega_sym)
        e_sub2 = e.subs(omega_sym, Omega_sym)
        if check_step_equivalence(g_sub2, e_sub2):
            return True
    except Exception:
        pass
    return False


def safe_parse(expr_str: str) -> Optional[sp.Expr]:
    if not expr_str or expr_str.strip() == "":
        return None

    cleaned = preprocess_expression(expr_str)

    try:
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations,
            implicit_multiplication_application
        )
        transformations = (standard_transformations +
                          (implicit_multiplication_application,))
        result = parse_expr(cleaned, local_dict=_local_dict(),
                            transformations=transformations)
        # Chained comparisons (a < re(s) < b) can evaluate to Python bool
        if isinstance(result, bool):
            return None
        return result
    except Exception as e:
        print(f"  [parse failed] original: '{expr_str[:60]}' "
              f"→ cleaned: '{cleaned[:60]}' → error: {e}")
        return None

import multiprocessing as _mp
import queue as _queue


def _timed(fn, seconds: int):
    """Run *fn()* with a hard wall-clock timeout (portable, no SIGALRM).

    Uses a daemon thread + queue so that SymPy's C extensions are never
    forked mid-call (which would deadlock on macOS).  If fn() doesn't
    return within *seconds*, we raise TimeoutError.
    """
    import threading

    result_q: _queue.Queue = _queue.Queue()

    def _target():
        try:
            result_q.put(("ok", fn()))
        except Exception as exc:
            result_q.put(("err", exc))

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    th.join(timeout=seconds)

    if th.is_alive():
        raise TimeoutError(f"_timed: exceeded {seconds}s budget")

    tag, payload = result_q.get_nowait()
    if tag == "err":
        raise payload
    return payload


def check_step_equivalence(expr1: sp.Expr, expr2: sp.Expr) -> bool:
    if expr1 is None or expr2 is None:
        return False

    # fast algebraic checks — no timeout needed
    try:
        diff = expand(expr1 - expr2)
        if diff == 0:
            return True
        if expand(expr1) == expand(expr2):
            return True
        if cancel(expr1 - expr2) == 0:
            return True
        if factor(expr1) == factor(expr2):
            return True
    except Exception:
        pass

    # trigsimp catches sin/cos identities cheaply
    try:
        if trigsimp(expr1 - expr2) == 0:
            return True
    except Exception:
        pass

    # Evaluate unevaluated Sums/Integrals if present
    def _try_doit(e):
        if not e.has(sp.Sum):
            return e
        d = e.doit()
        if isinstance(d, sp.Piecewise) and len(d.args) >= 1:
            return d.args[0][0]  # first branch value
        return d
    try:
        e1d = _try_doit(expr1)
        e2d = _try_doit(expr2)
        if e1d != expr1 or e2d != expr2:
            if expand(e1d - e2d) == 0 or cancel(e1d - e2d) == 0:
                return True
    except Exception:
        pass

    # numerical check — very fast, catches most cases.
    # Wrapped in a timeout because .subs() on an unevaluated Integral can
    # trigger symbolic evaluation that takes forever.
    n_sym, k_sym = sp.symbols('n k', integer=True)
    a_sym = sp.Symbol('a', real=True)
    try:
        def _num_eval():
            test_points = [
                {omega: 1.5, t: 0.5, T: 1.0, s: 0.5+0.3j, z: 0.8+0.2j,
                 n_sym: 3, k_sym: 2, a_sym: 0.6},
                {omega: 2.3, t: 0.7, T: 1.0, s: 1.1+0.7j, z: 0.6+0.4j,
                 n_sym: 5, k_sym: 1, a_sym: 0.4},
                {omega: 0.7, t: 0.3, T: 1.0, s: 0.3+1.2j, z: 0.9+0.1j,
                 n_sym: 2, k_sym: 3, a_sym: 0.8},
            ]
            for pt in test_points:
                v1 = complex(expr1.subs(list(pt.items())))
                v2 = complex(expr2.subs(list(pt.items())))
                if abs(v1 - v2) > 1e-6:
                    return False
            return True
        return _timed(_num_eval, 3)
    except TimeoutError:
        return False
    except Exception:
        pass

    # symbolic simplify — with 3 second timeout (Mac/Linux only)
    try:
        return _timed(lambda: simplify(expr1 - expr2) == 0, 3)
    except (TimeoutError, Exception):
        return False

def check_euler_identity(expr_str: str) -> bool:
    """
    Special case: verify Euler identity steps like
    exp(j*omega) - exp(-j*omega) = 2j*sin(omega)
    """
    try:
        lhs_str, rhs_str = expr_str.split("=", 1)
        lhs = safe_parse(lhs_str.strip())
        rhs = safe_parse(rhs_str.strip())
        if lhs is not None and rhs is not None:
            return check_step_equivalence(lhs, rhs)
    except Exception:
        pass
    return False

def verify_derivation(structured_output: dict, fast_mode: bool = True,
                       strict_mode: bool = False) -> list:
    """
    Verify each step of a structured derivation.

    Parameters
    ----------
    fast_mode   : if True, abstain (is_valid=None) on expensive or ambiguous
                  checks. Default behaviour for real LLM output.
    strict_mode : if True, disable the integral short-circuit and treat
                  numerical non-equivalence as a confirmed error (is_valid=False).
                  Used by the synthetic error-injection harness, where the base
                  derivation is known to be correct and any numerical mismatch
                  between consecutive steps is a real mathematical error.
                  `strict_mode=True` implies `fast_mode=False`.
    """
    if strict_mode:
        fast_mode = False

    steps = structured_output.get("steps", [])
    results = []
    prev_expr = None

    for step in steps:
        step_num    = step.get("step_number", 0)
        description = step.get("description", "")
        expr_str    = step.get("expression", "")
        operation   = step.get("operation", "")

        # Non-symbolic set/region statements (e.g., ROC.contains(...)) are
        # currently outside this verifier's symbolic scope.
        if "ROC.contains" in expr_str or " in ROC" in expr_str:
            result = StepResult(
                step_number=step_num,
                description=description,
                expression_given=expr_str,
                expression_parsed=None,
                previous_parsed=prev_expr,
                is_valid=None,
                error_message="Region-of-convergence condition — review manually"
            )
            results.append(result)
            continue

        # Equation-first check:
        # many derivation steps are of form "lhs = rhs" and should be
        # validated internally before sequential (prev -> current) checking.
        raw_stripped = expr_str.strip()
        if "=" in raw_stripped and not re.search(r"[!<>]=|==", raw_stripped):
            parts = raw_stripped.split("=", 1)
            lhs_raw = parts[0].strip()
            rhs_raw = parts[1].strip() if len(parts) > 1 else ""

            # Assignment-like equations should not be validated as identities.
            # Example: X_dtft = Sum(...)
            is_assignment_like = bool(
                re.match(r"^[A-Za-z_]\w*(?:\([^)]*\))?$", lhs_raw)
            )
            if is_assignment_like:
                current_expr = safe_parse(expr_str)
                result = StepResult(
                    step_number=step_num,
                    description=description,
                    expression_given=expr_str,
                    expression_parsed=current_expr,
                    previous_parsed=prev_expr,
                    is_valid=None
                )
                if current_expr is None:
                    result.is_valid = None
                    result.error_message = "Could not parse — review manually"
                elif prev_expr is None:
                    result.is_valid = True
                    result.error_message = "First step"
                else:
                    result.equivalence_checked = True
                    if check_step_equivalence(prev_expr, current_expr):
                        result.is_valid = True
                    elif fast_mode:
                        result.is_valid = None
                        result.error_message = (
                            f"Could not verify step {step_num} — fast mode, review manually"
                        )
                    else:
                        result.is_valid = None
                        result.error_message = "Could not verify — review manually"
                results.append(result)
                if current_expr is not None:
                    prev_expr = current_expr
                continue

            lhs_parsed = safe_parse(lhs_raw)
            rhs_parsed = safe_parse(rhs_raw)

            if lhs_parsed is not None and rhs_parsed is not None:
                eq_valid = check_step_equivalence(lhs_parsed, rhs_parsed)
                result = StepResult(
                    step_number=step_num,
                    description=description,
                    expression_given=expr_str,
                    expression_parsed=rhs_parsed,
                    previous_parsed=prev_expr,
                    is_valid=True if eq_valid else (None if fast_mode else False),
                    equivalence_checked=True,
                    error_message="" if eq_valid else (
                        "Equation sides not verified equivalent — review manually"
                        if fast_mode else "Equation sides are not equivalent"
                    )
                )
                results.append(result)
                # For chained derivations, continue from RHS expression.
                prev_expr = rhs_parsed
                continue

        # Fallback: parse as single expression and compare with previous step.
        current_expr = safe_parse(expr_str)

        # detect natural language mixed with math — skip gracefully
        is_text_expression = (
            current_expr is None and
            any(word in expr_str.lower() for word in
                [' for ', ' if ', ' where ', ' such that ', ' when '])
            and not expr_str.strip().startswith('sympy')
        )
        if is_text_expression:
            result = StepResult(
                step_number=step_num,
                description=description,
                expression_given=expr_str,
                expression_parsed=None,
                previous_parsed=prev_expr,
                is_valid=None,
                error_message="Natural language expression — not verifiable symbolically"
            )
            results.append(result)
            continue

        result = StepResult(
            step_number=step_num,
            description=description,
            expression_given=expr_str,
            expression_parsed=current_expr,
            previous_parsed=prev_expr,
            is_valid=None
        )

        if current_expr is None:
            result.is_valid = None
            result.error_message = "Could not parse — review manually"

        elif prev_expr is None:
            result.is_valid = True
            result.error_message = "First step"

        else:
            result.equivalence_checked = True
            # Only skip if the parsed AST actually contains unevaluated
            # Integral nodes (not just the word "integrate" in the source).
            prev_has_int = hasattr(prev_expr, 'has') and prev_expr.has(sp.Integral)
            curr_has_int = hasattr(current_expr, 'has') and current_expr.has(sp.Integral)
            if (prev_has_int or curr_has_int) and not strict_mode:
                # Before giving up, try comparing integrands when both are
                # Piecewise-wrapped integrals (common for Fourier/Laplace defs).
                matched_integrand = False
                try:
                    p_ints = list(prev_expr.atoms(sp.Integral)) if prev_has_int else []
                    c_ints = list(current_expr.atoms(sp.Integral)) if curr_has_int else []
                    if len(p_ints) == 1 and len(c_ints) == 1:
                        p_ig = p_ints[0].args[0]  # integrand
                        c_ig = c_ints[0].args[0]
                        if check_step_equivalence(p_ig, c_ig):
                            matched_integrand = True
                except Exception:
                    pass
                if matched_integrand:
                    result.is_valid = True
                    result.error_message = "Integrand equivalence confirmed"
                else:
                    result.is_valid = None
                    result.error_message = "Contains integral — skipped for speed, review manually"
                results.append(result)
                if current_expr is not None:
                    prev_expr = current_expr
                continue

            if check_step_equivalence(prev_expr, current_expr):
                result.is_valid = True
            elif fast_mode:
                result.is_valid = None
                result.error_message = (
                    f"Could not verify step {step_num} — fast mode, review manually"
                )
            else:
                # try evaluating prev (handles integrals) — timeout-bounded
                evaluated_matched = False
                try:
                    evaluated = _timed(lambda: sp.simplify(prev_expr), 3)
                    if check_step_equivalence(evaluated, current_expr):
                        result.is_valid = True
                        evaluated_matched = True
                except (TimeoutError, Exception):
                    pass

                if not evaluated_matched:
                    # Try a numerical mismatch test. If we can evaluate both
                    # expressions at concrete points and they disagree, this
                    # is a confirmed error (no legitimate rewrite preserves
                    # equality while changing numerical value).
                    try:
                        test_points = [(1.5, 0.5), (2.3, 0.7), (0.9, 0.3)]

                        def _collect():
                            pv_list, cv_list = [], []
                            for w, tv in test_points:
                                pv = complex(prev_expr.subs(
                                    [(omega, w), (t, tv), (T, 1.0)]))
                                cv = complex(current_expr.subs(
                                    [(omega, w), (t, tv), (T, 1.0)]))
                                pv_list.append(pv)
                                cv_list.append(cv)
                            return pv_list, cv_list

                        prev_vals, curr_vals = _timed(_collect, 3)

                        # sign-error diagnostic first (more informative)
                        if all(abs(p + c) < 1e-8 for p, c in
                               zip(prev_vals, curr_vals)):
                            result.is_valid = False
                            result.error_message = (
                                f"SIGN ERROR detected in step {step_num}: "
                                f"expression has opposite sign to previous step"
                            )
                        elif any(abs(p - c) > 1e-6 for p, c in
                                 zip(prev_vals, curr_vals)):
                            if strict_mode:
                                result.is_valid = False
                                result.error_message = (
                                    f"Numerical mismatch in step {step_num}: "
                                    f"value differs from previous step"
                                )
                            else:
                                # Without strict_mode we still abstain, because
                                # legitimate rewrites (substitution, integration,
                                # definition) can legitimately change numerical
                                # value vs the symbolic prior step.
                                result.is_valid = None
                                result.error_message = (
                                    f"Could not verify step {step_num} — "
                                    f"review manually"
                                )
                        else:
                            result.is_valid = None
                            result.error_message = (
                                f"Could not verify step {step_num} — "
                                f"review manually"
                            )
                    except Exception:
                        result.is_valid = None
                        result.error_message = "Could not verify — review manually"

        results.append(result)
        if current_expr is not None:
            prev_expr = current_expr

    return results

def categorize_step(r: "StepResult") -> str:
    """
    Classify a verified step into a coarse category for separated metrics.
    Returns one of:
      'passed'        — equivalence check passed
      'failed'        — equivalence check confirmed a math error
      'parse_fail'    — preprocessing/parsing failed
      'natural_lang'  — natural-language expression, not verifiable
      'roc'           — region-of-convergence condition, not verifiable
      'integral_skip' — contains integral, skipped for speed
      'first_step'    — first step, trivially marked valid
      'unclear'       — everything else (fast-mode punt, internal failure, etc.)
    """
    msg = (r.error_message or "").lower()
    if r.is_valid is True and "first step" in msg:
        return "first_step"
    if r.is_valid is True:
        return "passed"
    if r.is_valid is False:
        return "failed"
    if "could not parse" in msg:
        return "parse_fail"
    if "natural language" in msg:
        return "natural_lang"
    if "region-of-convergence" in msg or "roc" in msg:
        return "roc"
    if "contains integral" in msg:
        return "integral_skip"
    return "unclear"


def _normalise_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _split_top_level_commas(s: str) -> list[str]:
    """Split a string on commas that are at parenthesis-depth 0."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch in "([{":
            depth += 1
            current.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def _split_composite_generated(generated: str, expected_parts_count: int) -> list[str]:
    """
    Heuristic splitter for generated composite answers such as:
      "1/(s+2); Re(s) > -2"
      "1/(s+2) and Re(s) > -2"
      "1/(s+2), Re(s) > -2"     (top-level comma — won't split inside Sum(...))
    """
    raw = (generated or "").strip()
    if not raw:
        return []
    if ";" in raw:
        return [p.strip() for p in raw.split(";") if p.strip()]
    if " and " in raw:
        return [p.strip() for p in raw.split(" and ") if p.strip()]
    # Top-level comma split, but only accept it if the split produces the
    # number of parts the grading contract expects (otherwise we'd wrongly
    # split "Sum(x, (n, 0, oo))" when the expected is a single-part answer
    # — though top-level comma avoids this specific case by skipping commas
    # inside parentheses).
    if "," in raw:
        pieces = _split_top_level_commas(raw)
        if len(pieces) == expected_parts_count:
            return pieces
    if expected_parts_count > 1:
        return [raw]
    return [raw]


def _normalise_inequality_notation(expr: str) -> str:
    """Rewrite student/LLM-ish notation into sympy-parseable form.

    - ``Re(s)``, ``Im(s)`` → ``re(s)``, ``im(s)``
    - ``|x|``              → ``Abs(x)`` (only when the bars are balanced and
                            don't straddle a comparison operator)
    """
    if not expr:
        return expr
    out = expr.replace("Re(", "re(").replace("Im(", "im(")
    # Replace balanced |...| pairs with Abs(...). Walk left-to-right so we
    # correctly handle e.g. "|z| > 1" or "|re(s)| <= 1".
    result = []
    i = 0
    while i < len(out):
        ch = out[i]
        if ch == "|":
            j = out.find("|", i + 1)
            if j == -1:
                result.append(ch)
                i += 1
                continue
            inner = out[i + 1 : j]
            # Reject if the inner spans a comparison operator — that would
            # mean these bars aren't a matched pair (e.g. in "|x| > |y|").
            if any(op in inner for op in ("<", ">", "=")):
                result.append(ch)
                i += 1
                continue
            result.append(f"Abs({inner})")
            i = j + 1
        else:
            result.append(ch)
            i += 1
    return "".join(result)


def _match_inequality(generated_final: str, expected_value: str, expected_type: str) -> dict:
    """
    Compare inequality targets.
    Tries symbolic parsing first; if parsing fails, falls back to normalized text.
    """
    generated_norm = _normalise_inequality_notation(generated_final or "")
    expected_norm = _normalise_inequality_notation(expected_value or "")
    g = safe_parse(generated_norm)
    e = safe_parse(expected_norm)
    if g is not None and e is not None:
        g_s = str(g)
        e_s = str(e)
        return {
            "status": "match" if g_s == e_s else "mismatch",
            "generated": generated_final,
            "expected": expected_value,
            "expected_type": expected_type,
            "generated_parsed": g_s,
            "expected_parsed": e_s,
        }
    # Fallback for textual inequalities.
    gn = _normalise_text(generated_norm)
    en = _normalise_text(expected_norm)
    return {
        "status": "match" if gn == en else "mismatch",
        "generated": generated_final,
        "expected": expected_value,
        "expected_type": expected_type,
        "generated_normalized": gn,
        "expected_normalized": en,
    }


def _resolve_expected_answer(expected_answer: Any) -> dict:
    """
    Backward-compatible expected-answer resolver.

    Accepted formats:
      1) legacy string (treated as symbolic expression target)
      2) dict, e.g.
         {"type": "symbolic", "expression": "..."}
         {"type": "text", "value": "..."}
    """
    if isinstance(expected_answer, str):
        return {
            "type": "symbolic",
            "value": expected_answer,
            "raw": expected_answer,
        }
    if isinstance(expected_answer, dict):
        ans_type = (expected_answer.get("type") or "symbolic").lower()
        if ans_type == "composite":
            parts = expected_answer.get("parts") or []
            return {
                "type": "composite",
                "parts": parts,
                "aggregation": expected_answer.get("aggregation", "all"),
                "raw": expected_answer,
            }
        value = (
            expected_answer.get("expression")
            or expected_answer.get("value")
            or expected_answer.get("expected")
            or ""
        )
        return {
            "type": ans_type,
            "value": value,
            "raw": expected_answer,
        }
    return {"type": "symbolic", "value": "", "raw": expected_answer}


def check_final_answer_match(generated_final: str, expected_answer: Any):
    """
    Check whether the LLM's final_result matches the expected answer target.
    Supports legacy string expectations and structured expected-answer dicts.

    status values:
      'match'          — parsed both and equivalent
      'mismatch'       — parsed both but not equivalent
      'unparseable'    — at least one side failed to parse
      'no_expected'    — no expected answer target provided in the query
      'unsupported_expected_type' — expected-answer type is unknown
    """
    resolved = _resolve_expected_answer(expected_answer)
    expected_type = resolved["type"]
    expected_value = resolved.get("value", "")

    if expected_type == "composite":
        parts = resolved.get("parts", [])
        if not parts:
            return {
                "status": "no_expected",
                "generated": generated_final,
                "expected": "",
                "expected_type": expected_type,
            }
        aggregation = (resolved.get("aggregation") or "all").lower()
        # "any" = accept if any single part matches.
        # Clean the generated string first (strip assignment, split on
        # top-level commas / semicolons) and test each segment against each
        # expected alternative.
        if aggregation == "any":
            cleaned_gen = _strip_leading_assignment(generated_final)
            # Collect candidate segments from different split strategies.
            candidates = {cleaned_gen}  # always try the whole thing
            for seg in _split_composite_generated(cleaned_gen, 1):
                candidates.add(_strip_leading_assignment(seg))
            for seg in _split_top_level_commas(cleaned_gen):
                candidates.add(_strip_leading_assignment(seg))
            part_results = []
            for i, p in enumerate(parts):
                part_name = p.get("name", f"part_{i+1}")
                best = None
                for cand in candidates:
                    res = check_final_answer_match(cand, p)
                    if res.get("status") == "match":
                        best = res
                        break
                    if best is None or res.get("status") != "unparseable":
                        best = res
                best["name"] = part_name
                part_results.append(best)
            statuses = [pr.get("status") for pr in part_results]
            if any(st == "match" for st in statuses):
                top = "match"
            else:
                # "any": tried all candidate segments vs all alternatives;
                # no match -> mismatch (do not surface unparseable as top).
                top = "mismatch"
        else:
            generated_parts = _split_composite_generated(generated_final, len(parts))
            part_results = []
            for i, p in enumerate(parts):
                part_generated = generated_parts[i] if i < len(generated_parts) else ""
                part_name = p.get("name", f"part_{i+1}")
                res = check_final_answer_match(part_generated, p)
                res["name"] = part_name
                part_results.append(res)
            statuses = [p.get("status") for p in part_results]
            if any(s == "unparseable" for s in statuses):
                top = "unparseable"
            elif any(s in {"mismatch", "unsupported_expected_type"} for s in statuses):
                top = "mismatch"
            elif all(s == "match" for s in statuses):
                top = "match"
            else:
                top = "mismatch"
        return {
            "status": top,
            "expected_type": expected_type,
            "aggregation": resolved.get("aggregation", "all"),
            "parts": part_results,
            "generated": generated_final,
            "expected": resolved.get("raw"),
        }

    if not expected_value:
        return {
            "status": "no_expected",
            "generated": generated_final,
            "expected": expected_value,
            "expected_type": expected_type,
        }

    if expected_type in {"text", "string", "verbatim"}:
        match = _normalise_text(generated_final) == _normalise_text(expected_value)
        return {
            "status": "match" if match else "mismatch",
            "generated": generated_final,
            "expected": expected_value,
            "expected_type": expected_type,
            "generated_normalized": _normalise_text(generated_final),
            "expected_normalized": _normalise_text(expected_value),
        }

    if expected_type in {"inequality", "relation"}:
        return _match_inequality(generated_final, expected_value, expected_type)

    if expected_type not in {"symbolic", "sympy"}:
        return {
            "status": "unsupported_expected_type",
            "generated": generated_final,
            "expected": expected_value,
            "expected_type": expected_type,
        }

    # Use the final-answer parser that preserves function-call semantics
    # (X(I*omega) stays as an applied function, not stripped to X).
    generated_for_parse = _strip_leading_assignment(generated_final)
    e = safe_parse_final(expected_value)

    # Build a list of candidate generated strings: the full thing, then each
    # top-level comma-separated segment (for when the LLM tacks on extra
    # info like "X(I*omega), Re(s) > 0").
    candidates = [generated_for_parse]
    if "," in generated_for_parse:
        segments = _split_top_level_commas(generated_for_parse)
        if len(segments) > 1:
            for seg in segments:
                seg_clean = _strip_leading_assignment(seg)
                if seg_clean not in candidates:
                    candidates.append(seg_clean)

    # Try each candidate until one matches.
    best_g = None
    best_parsed_str = None
    for cand in candidates:
        g = safe_parse_final(cand)
        if g is None:
            continue
        if best_g is None:
            best_g = g
            best_parsed_str = str(g)
        if e is not None:
            try:
                if _equiv_final(g, e):
                    return {
                        "status": "match",
                        "generated": generated_final,
                        "expected": expected_value,
                        "expected_type": expected_type,
                        "generated_parsed": str(g),
                        "expected_parsed": str(e),
                    }
            except Exception:
                continue

    if best_g is None or e is None:
        return {
            "status": "unparseable",
            "generated": generated_final,
            "expected": expected_value,
            "expected_type": expected_type,
            "generated_parsed": best_parsed_str,
            "expected_parsed": str(e) if e is not None else None,
        }
    return {
        "status": "mismatch",
        "generated": generated_final,
        "expected": expected_value,
        "expected_type": expected_type,
        "generated_parsed": best_parsed_str,
        "expected_parsed": str(e),
    }


def print_verification_report(structured_output: dict, results: list):
    concept = structured_output.get("concept", "Unknown")
    print(f"\n{'='*60}")
    print(f"VERIFICATION REPORT: {concept}")
    print(f"{'='*60}")

    passed  = sum(1 for r in results if r.is_valid is True)
    failed  = sum(1 for r in results if r.is_valid is False)
    unclear = sum(1 for r in results if r.is_valid is None)

    print(f"Steps verified: {len(results)}")
    print(f"  ✓ Passed:       {passed}")
    print(f"  ✗ Failed:       {failed}")
    print(f"  ? Needs review: {unclear}")
    print()

    for r in results:
        status = "✓" if r.is_valid is True else ("✗" if r.is_valid is False else "?")
        print(f"Step {r.step_number} {status}: {r.description}")
        print(f"  Expression: {r.expression_given[:80]}")
        if r.error_message:
            print(f"  Note: {r.error_message}")
        print()

    print(f"Final result: {structured_output.get('final_result', '')}")
    print(f"{'='*60}\n")