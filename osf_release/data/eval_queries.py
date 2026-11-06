# data/eval_queries.py
# 30 evaluation queries spanning different concept paths.
# Q01–Q10: original benchmark (mixed difficulty, FT/LT/ZT/FS/conv).
# Q11–Q30: v1 expansion covering the remaining concepts
#          (differential_equations, sampling_theorem, transfer_function,
#          frequency_response, modulation, filters) and exercising all four
#          grader paths (symbolic, inequality, composite-all, composite-any).

EVAL_QUERIES = [
    {
        "id": "Q01",
        "question": "Derive the Fourier transform of a rectangular pulse signal.",
        "expected_concepts": ["fourier_transform"],
        "expected_prereqs": ["fourier_series", "complex_numbers"],
        "expected_answer": {"type": "symbolic", "expression": "2*sin(omega)/omega"},
        "expected_final_result": "2*sin(omega)/omega",
        "difficulty": "intermediate"
    },
    {
        "id": "Q02",
        "question": "What is the Fourier transform of a unit impulse signal delta(t)?",
        "expected_concepts": ["fourier_transform"],
        "expected_prereqs": ["fourier_series", "complex_numbers"],
        "expected_answer": {"type": "symbolic", "expression": "1"},
        "expected_final_result": "1",
        "difficulty": "intermediate"
    },
    {
        "id": "Q03",
        "question": "Derive the relationship between Laplace and Fourier transforms.",
        "expected_concepts": ["laplace_transform", "fourier_transform"],
        "expected_prereqs": ["complex_numbers", "differential_equations"],
        "expected_answer": {"type": "symbolic", "expression": "X(I*omega)"},
        "expected_final_result": "X(I*omega)",
        "difficulty": "intermediate"
    },
    {
        "id": "Q04",
        "question": "Explain the convolution theorem for Fourier transforms.",
        "expected_concepts": ["fourier_transform", "convolution"],
        "expected_prereqs": ["fourier_series", "complex_numbers"],
        # Accept both the shorthand (X1*X2) and the applied-function form
        # (X1(j*omega)*X2(j*omega)) — textbooks use both and the LLM is
        # free to emit either.
        "expected_answer": {
            "type": "composite",
            "aggregation": "any",
            "parts": [
                {"name": "shorthand", "type": "symbolic", "expression": "X1*X2"},
                {
                    "name": "applied_form",
                    "type": "symbolic",
                    "expression": "X1(I*omega)*X2(I*omega)",
                },
            ],
        },
        "expected_final_result": "X1(I*omega)*X2(I*omega)",
        "difficulty": "advanced"
    },
    {
        "id": "Q05",
        "question": "Derive the Z-transform of a causal exponential signal x[n] = a^n * u[n].",
        "expected_concepts": ["z_transform"],
        "expected_prereqs": ["complex_numbers", "sampling_theorem"],
        "expected_answer": {"type": "symbolic", "expression": "z/(z-a)"},
        "expected_final_result": "z/(z-a)",
        "difficulty": "intermediate"
    },
    {
        "id": "Q06",
        "question": "What is the time shifting property of the Fourier transform?",
        "expected_concepts": ["fourier_transform"],
        "expected_prereqs": ["complex_numbers", "fourier_series"],
        "expected_answer": {
            "type": "symbolic",
            "expression": "exp(-I*omega*t0)*X(I*omega)"
        },
        "expected_final_result": "exp(-I*omega*t0)*X(I*omega)",
        "difficulty": "intermediate"
    },
    {
        "id": "Q07",
        "question": "Derive the frequency response of a discrete time system.",
        "expected_concepts": ["frequency_response", "z_transform"],
        "expected_prereqs": ["z_transform", "fourier_transform", "complex_numbers"],
        "expected_answer": {"type": "symbolic", "expression": "H(exp(I*Omega))"},
        "expected_final_result": "H(exp(I*Omega))",
        "difficulty": "advanced"
    },
    {
        "id": "Q08",
        "question": "Explain the relationship between the Z-transform and the DTFT.",
        "expected_concepts": ["z_transform", "fourier_transform"],
        "expected_prereqs": ["complex_numbers", "sampling_theorem"],
        # Either the function-name shorthand X(e^{jΩ}) or the equivalent
        # definitional sum form is an acceptable final answer. Both are
        # standard textbook expressions of the Z→DTFT relationship, and the
        # symbolic verifier cannot rewrite one into the other without a
        # definition axiom.
        "expected_answer": {
            "type": "composite",
            "aggregation": "any",
            "parts": [
                {"name": "shorthand", "type": "symbolic", "expression": "X(exp(I*Omega))"},
                {
                    "name": "definition",
                    "type": "symbolic",
                    "expression": "Sum(x_n * exp(-I*Omega*n), (n, -oo, oo))",
                },
            ],
        },
        "expected_final_result": "X(exp(I*Omega))",
        "difficulty": "advanced"
    },
    {
        "id": "Q09",
        "question": "Derive the Fourier series coefficients for a periodic square wave.",
        "expected_concepts": ["fourier_series"],
        "expected_prereqs": ["complex_numbers", "differential_equations"],
        "expected_answer": {
            "type": "symbolic",
            "expression": "sin(k*pi/2)/(k*pi/2)"
        },
        "expected_final_result": "ak = sin(k*pi/2)/(k*pi/2)",
        "difficulty": "intermediate"
    },
    {
        "id": "Q10",
        # Pinned to a specific signal so the composite expected answer
        # (transform + ROC) is well-defined. Previously the question was
        # generic ("What is the region of convergence of the Laplace
        # transform?"), which let the LLM pick an arbitrary example and
        # guaranteed a mismatch against any fixed expected answer.
        "question": (
            "Compute the Laplace transform of x(t) = exp(-2*t) * u(t) and "
            "state its region of convergence."
        ),
        "expected_concepts": ["laplace_transform"],
        "expected_prereqs": ["complex_numbers", "differential_equations"],
        "expected_answer": {
            "type": "composite",
            "aggregation": "all",
            "parts": [
                {"name": "transform", "type": "symbolic", "expression": "1/(s+2)"},
                {"name": "roc", "type": "inequality", "expression": "re(s) > -2"},
            ],
        },
        "expected_final_result": "1/(s+2), Re(s) > -2",
        "difficulty": "intermediate"
    },

    # ------------------------------------------------------------------
    # Q11–Q30: v1 benchmark expansion
    # ------------------------------------------------------------------

    {
        "id": "Q11",
        "question": (
            "Compute the Fourier transform of the two-sided decaying "
            "exponential x(t) = exp(-a*|t|) with a > 0."
        ),
        "expected_concepts": ["fourier_transform"],
        "expected_prereqs": ["complex_numbers", "differential_equations"],
        "expected_answer": {
            "type": "symbolic",
            "expression": "2*a/(a**2 + omega**2)",
        },
        "expected_final_result": "2*a/(a**2 + omega**2)",
        "difficulty": "intermediate",
    },
    {
        "id": "Q12",
        "question": "Compute the Fourier transform of x(t) = cos(omega_0 * t).",
        "expected_concepts": ["fourier_transform"],
        "expected_prereqs": ["complex_numbers", "fourier_series"],
        # Either the two-impulse form or the pi-scaled two-impulse form
        # is standard; accept both via composite "any".
        "expected_answer": {
            "type": "composite",
            "aggregation": "any",
            "parts": [
                {
                    "name": "pi_scaled_pair",
                    "type": "symbolic",
                    "expression": (
                        "pi*DiracDelta(omega - omega_0) + "
                        "pi*DiracDelta(omega + omega_0)"
                    ),
                },
                {
                    "name": "factored_pair",
                    "type": "symbolic",
                    "expression": (
                        "pi*(DiracDelta(omega - omega_0) + "
                        "DiracDelta(omega + omega_0))"
                    ),
                },
            ],
        },
        "expected_final_result": (
            "pi*(DiracDelta(omega - omega_0) + DiracDelta(omega + omega_0))"
        ),
        "difficulty": "intermediate",
    },
    {
        "id": "Q13",
        "question": (
            "Compute the inverse Fourier transform of "
            "X(j*omega) = 2*pi*delta(omega)."
        ),
        "expected_concepts": ["fourier_transform"],
        "expected_prereqs": ["complex_numbers", "fourier_series"],
        "expected_answer": {"type": "symbolic", "expression": "1"},
        "expected_final_result": "1",
        "difficulty": "intermediate",
    },
    {
        "id": "Q14",
        "question": (
            "State the Fourier transform linearity property: express the "
            "transform of a*x1(t) + b*x2(t) in terms of X1(j*omega) and "
            "X2(j*omega)."
        ),
        "expected_concepts": ["fourier_transform"],
        "expected_prereqs": ["complex_numbers", "fourier_series"],
        "expected_answer": {
            "type": "symbolic",
            "expression": "a*X1(I*omega) + b*X2(I*omega)",
        },
        "expected_final_result": "a*X1(j*omega) + b*X2(j*omega)",
        "difficulty": "intermediate",
    },
    {
        "id": "Q15",
        "question": (
            "Compute the Laplace transform of x(t) = u(t) (the unit step) "
            "and state its region of convergence."
        ),
        "expected_concepts": ["laplace_transform"],
        "expected_prereqs": ["complex_numbers", "differential_equations"],
        "expected_answer": {
            "type": "composite",
            "aggregation": "all",
            "parts": [
                {"name": "transform", "type": "symbolic", "expression": "1/s"},
                {"name": "roc", "type": "inequality", "expression": "re(s) > 0"},
            ],
        },
        "expected_final_result": "1/s, Re(s) > 0",
        "difficulty": "intermediate",
    },
    {
        "id": "Q16",
        "question": (
            "Compute the Laplace transform of x(t) = t*u(t) and state its "
            "region of convergence."
        ),
        "expected_concepts": ["laplace_transform"],
        "expected_prereqs": ["complex_numbers", "differential_equations"],
        "expected_answer": {
            "type": "composite",
            "aggregation": "all",
            "parts": [
                {"name": "transform", "type": "symbolic", "expression": "1/s**2"},
                {"name": "roc", "type": "inequality", "expression": "re(s) > 0"},
            ],
        },
        "expected_final_result": "1/s**2, Re(s) > 0",
        "difficulty": "intermediate",
    },
    {
        "id": "Q17",
        "question": (
            "State the Laplace differentiation property: express the Laplace "
            "transform of dx/dt in terms of X(s) and x(0)."
        ),
        "expected_concepts": ["laplace_transform"],
        "expected_prereqs": ["differential_equations", "complex_numbers"],
        "expected_answer": {
            "type": "symbolic",
            "expression": "s*X(s) - x(0)",
        },
        "expected_final_result": "s*X(s) - x(0)",
        "difficulty": "advanced",
    },
    {
        "id": "Q18",
        "question": (
            "Compute the Laplace transform of x(t) = sin(omega_0*t)*u(t)."
        ),
        "expected_concepts": ["laplace_transform"],
        "expected_prereqs": ["complex_numbers", "differential_equations"],
        "expected_answer": {
            "type": "symbolic",
            "expression": "omega_0/(s**2 + omega_0**2)",
        },
        "expected_final_result": "omega_0/(s**2 + omega_0**2)",
        "difficulty": "intermediate",
    },
    {
        "id": "Q19",
        "question": "Compute the Z-transform of x[n] = delta[n] (the unit impulse).",
        "expected_concepts": ["z_transform"],
        "expected_prereqs": ["complex_numbers", "sampling_theorem"],
        "expected_answer": {"type": "symbolic", "expression": "1"},
        "expected_final_result": "1",
        "difficulty": "intermediate",
    },
    {
        "id": "Q20",
        "question": (
            "Compute the Z-transform of x[n] = u[n] (the discrete unit step) "
            "and state its region of convergence."
        ),
        "expected_concepts": ["z_transform"],
        "expected_prereqs": ["complex_numbers", "sampling_theorem"],
        "expected_answer": {
            "type": "composite",
            "aggregation": "all",
            "parts": [
                {"name": "transform", "type": "symbolic", "expression": "z/(z - 1)"},
                {"name": "roc", "type": "inequality", "expression": "Abs(z) > 1"},
            ],
        },
        "expected_final_result": "z/(z-1), |z| > 1",
        "difficulty": "intermediate",
    },
    {
        "id": "Q21",
        "question": (
            "Find the DC Fourier-series coefficient a_0 of a symmetric square "
            "wave that equals +A on the first half of each period and -A on "
            "the second half."
        ),
        "expected_concepts": ["fourier_series"],
        "expected_prereqs": ["complex_numbers"],
        "expected_answer": {"type": "symbolic", "expression": "0"},
        "expected_final_result": "0",
        "difficulty": "intermediate",
    },
    {
        "id": "Q22",
        "question": (
            "A causal LTI system obeys y'(t) + a*y(t) = x(t) with a > 0 and "
            "zero initial conditions. Compute its transfer function H(s)."
        ),
        "expected_concepts": ["transfer_function", "laplace_transform"],
        "expected_prereqs": ["differential_equations", "complex_numbers"],
        "expected_answer": {"type": "symbolic", "expression": "1/(s + a)"},
        "expected_final_result": "1/(s+a)",
        "difficulty": "advanced",
    },
    {
        "id": "Q23",
        "question": (
            "Compute the convolution y(t) = x(t) * h(t) for t >= 0, where "
            "x(t) = u(t) and h(t) = exp(-a*t)*u(t) with a > 0. Give y(t) as "
            "a closed-form expression valid for t >= 0."
        ),
        "expected_concepts": ["convolution"],
        "expected_prereqs": ["fourier_transform", "complex_numbers"],
        "expected_answer": {
            "type": "symbolic",
            "expression": "(1 - exp(-a*t))/a",
        },
        "expected_final_result": "(1 - exp(-a*t))/a",
        "difficulty": "advanced",
    },
    {
        "id": "Q24",
        "question": (
            "For a causal LTI system with impulse response "
            "h(t) = exp(-a*t)*u(t), a > 0, give the transfer function H(s) "
            "and its region of convergence."
        ),
        "expected_concepts": ["transfer_function", "laplace_transform"],
        "expected_prereqs": ["convolution", "laplace_transform", "complex_numbers"],
        "expected_answer": {
            "type": "composite",
            "aggregation": "all",
            "parts": [
                {"name": "transform", "type": "symbolic", "expression": "1/(s + a)"},
                {"name": "roc", "type": "inequality", "expression": "re(s) > -a"},
            ],
        },
        "expected_final_result": "1/(s+a), Re(s) > -a",
        "difficulty": "intermediate",
    },
    {
        "id": "Q25",
        "question": (
            "Compute the Z-transform H(z) of h[n] = (1/2)**n * u[n]."
        ),
        "expected_concepts": ["z_transform", "transfer_function"],
        "expected_prereqs": ["complex_numbers", "sampling_theorem"],
        # Two standard equivalent forms (positive-power vs negative-power).
        "expected_answer": {
            "type": "composite",
            "aggregation": "any",
            "parts": [
                {
                    "name": "positive_power",
                    "type": "symbolic",
                    "expression": "z/(z - Rational(1,2))",
                },
                {
                    "name": "negative_power",
                    "type": "symbolic",
                    "expression": "1/(1 - Rational(1,2)/z)",
                },
            ],
        },
        "expected_final_result": "z/(z - 1/2)",
        "difficulty": "intermediate",
    },
    {
        "id": "Q26",
        "question": (
            "Given H(s) = 1/(s + a) with a > 0, derive the magnitude "
            "|H(j*omega)| of its frequency response."
        ),
        "expected_concepts": ["frequency_response", "transfer_function"],
        "expected_prereqs": ["laplace_transform", "complex_numbers"],
        "expected_answer": {
            "type": "symbolic",
            "expression": "1/sqrt(omega**2 + a**2)",
        },
        "expected_final_result": "1/sqrt(omega**2 + a**2)",
        "difficulty": "intermediate",
    },
    {
        "id": "Q27",
        "question": (
            "Given H(z) = 1/(1 - r*z**(-1)) with |r| < 1, write the "
            "frequency response H(e**(j*Omega))."
        ),
        "expected_concepts": ["frequency_response", "z_transform"],
        "expected_prereqs": ["z_transform", "complex_numbers"],
        "expected_answer": {
            "type": "symbolic",
            "expression": "1/(1 - r*exp(-I*Omega))",
        },
        "expected_final_result": "1/(1 - r*exp(-j*Omega))",
        "difficulty": "advanced",
    },
    {
        "id": "Q28",
        "question": (
            "State the Nyquist sampling rate (in samples per second) for a "
            "continuous-time signal whose highest frequency component is "
            "f_max Hz."
        ),
        "expected_concepts": ["sampling_theorem"],
        "expected_prereqs": ["fourier_transform"],
        "expected_answer": {"type": "symbolic", "expression": "2*f_max"},
        "expected_final_result": "2*f_max",
        "difficulty": "intermediate",
    },
    {
        "id": "Q29",
        "question": (
            "Give the transfer function H(s) of an ideal first-order "
            "continuous-time low-pass filter with cutoff frequency omega_c "
            "and unity DC gain."
        ),
        "expected_concepts": ["filters", "transfer_function"],
        "expected_prereqs": ["laplace_transform", "frequency_response"],
        "expected_answer": {
            "type": "composite",
            "aggregation": "any",
            "parts": [
                {
                    "name": "product_form",
                    "type": "symbolic",
                    "expression": "omega_c/(s + omega_c)",
                },
                {
                    "name": "ratio_form",
                    "type": "symbolic",
                    "expression": "1/(1 + s/omega_c)",
                },
            ],
        },
        "expected_final_result": "omega_c/(s + omega_c)",
        "difficulty": "intermediate",
    },
    {
        "id": "Q30",
        "question": (
            "Consider DSB-SC amplitude modulation with signal "
            "s(t) = m(t)*cos(omega_c*t), where M(j*omega) is the Fourier "
            "transform of m(t). Compute the Fourier transform S(j*omega) of "
            "s(t) in terms of M."
        ),
        "expected_concepts": ["modulation", "fourier_transform"],
        "expected_prereqs": ["fourier_transform", "frequency_response"],
        "expected_answer": {
            "type": "symbolic",
            "expression": (
                "(M(I*(omega - omega_c)) + M(I*(omega + omega_c)))/2"
            ),
        },
        "expected_final_result": (
            "(M(j*(omega - omega_c)) + M(j*(omega + omega_c)))/2"
        ),
        "difficulty": "advanced",
    },
]