\section{Python Code Guidelines}

\subsection{Scope}
Apply to all Python code, scripts, and notebooks in this workspace.

\subsection{Style \& Structure}
\begin{itemize}
    \item Follow PEP 8 for formatting and naming.
    \item Prefer small, testable functions with clear responsibilities.
    \item Use type hints for public functions and complex data structures.
    \item Keep modules cohesive; avoid circular imports.
    \item Prefer explicit data flow over hidden state.
    \item Never allow for ambiguous behaviour, I want one version that allows only one input, but with this input there are guaranteed to be no errors. Of course this has to come with good testing but it saves much code
\end{itemize}

\subsection{Scientific / Physics Documentation}
\begin{itemize}
    \item Always document the \textbf{physics being implemented}, not just the code mechanics.
    \item For non-trivial equations, state:
    \begin{itemize}
        \item the physical quantity being computed,
        \item the approximation being used,
        \item the source or reference equation,
        \item the assumptions under which the implementation is valid.
    \end{itemize}
    \item Comments should explain \textbf{why} a formula, sign, prefactor, basis choice, or frame transformation is correct.
    \item Do not add comments that merely restate obvious code line by line.
\end{itemize}

\subsubsection{Required for physics-heavy code}
\begin{itemize}
    \item Explicitly document units for all public inputs/outputs.
    \item Explicitly document conventions such as:
    \begin{itemize}
        \item lab frame vs.\ rotating frame,
        \item site basis vs.\ eigenbasis,
        \item positive- vs.\ negative-frequency parts,
        \item Fourier transform sign conventions,
        \item ordering of states/manifolds,
        \item normalization conventions.
    \end{itemize}
\end{itemize}