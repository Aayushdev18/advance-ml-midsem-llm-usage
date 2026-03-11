import nbformat as nbf
import os

os.makedirs('partB', exist_ok=True)
os.makedirs('partB/data', exist_ok=True)
os.makedirs('partB/results', exist_ok=True)

# Task 1.1
nb = nbf.v4.new_notebook()
text_1_1 = """# Task 1.1 Core Contribution / Architecture

## Step-by-Step Method Description

*   **Step 1: Compute Fourier Transform of Kernel**
    *   Description: The method begins by computing the Fourier transform $p(\omega)$ of the shift-invariant kernel function $k(x, y) = k(x-y)$. By Bochner's theorem, this yields a probability distribution from which random directions can be sampled.
    *   Reference: Equation (2) and Section 3 of the paper.
    *   Purpose: This step maps the continuous, deterministic kernel function to a probability distribution, which allows formulating the kernel evaluation as an expected value over random projections.
*   **Step 2: Sample Random Directions and Phase Shifts**
    *   Description: $D$ independent and identically distributed (i.i.d.) samples $\omega_1, \dots, \omega_D \in \mathbb{R}^d$ are drawn from the probability distribution $p(\omega)$, and $D$ i.i.d. phase shifts $b_1, \dots, b_D \in \mathbb{R}$ are drawn uniformly from the interval $[0, 2\pi]$.
    *   Reference: Algorithm 1 ("Random Fourier Features").
    *   Purpose: Generating random projection vectors and phase shifts that will be used to construct the low-dimensional randomized coordinate space.
*   **Step 3: Construct Random Fourier Features**
    *   Description: An input vector $x$ is mapped to a low-dimensional space $z(x) \in \mathbb{R}^D$ using the randomly sampled vectors. Specifically, $z(x) = \sqrt{\frac{2}{D}} [\cos(\omega_1^T x + b_1), \dots, \cos(\omega_D^T x + b_D)]^T$.
    *   Reference: Algorithm 1 ("Random Fourier Features").
    *   Purpose: This feature map explicitly non-linearizes the input data into a relatively low-dimensional Euclidean space (where $D$ is the number of features), such that the inner product of two transformed points strictly approximates their original kernel evaluation $z(x)^T z(y) \approx k(x,y)$.
*   **Step 4: Train Linear Machine Learning Algorithm**
    *   Description: The original training set features $X$ are transformed using $z(x)$, yielding $Z$. Then, a fast linear method (like Ridge Regression or Linear SVM) is trained on $Z$ directly instead of computing the heavy pairwise $N \\times N$ kernel matrix.
    *   Reference: Section 1 (Introduction) and Section 5 (Experiments).
    *   Purpose: Output the final predictive model utilizing linear operations $O(D)$ times rather than quadratic operations involving the original dimension or dataset scale.

## Final Summary Sentence

This paper solves the computationally prohibitive scaling problem of exact Kernel Machines (which require operations over massive dense $N \\times N$ kernel matrices), and the authors claim their randomized feature projection approach achieves similar accuracy as state-of-the-art non-linear kernel exact methods while operating in drastically smaller execution time and dimensions because they convert dense kernel algebra into efficient linear algebra.
"""
nb['cells'] = [nbf.v4.new_markdown_cell(text_1_1)]
nbf.write(nb, 'partB/task_1_1.ipynb')

# Task 1.2
nb = nbf.v4.new_notebook()
text_1_2 = """# Task 1.2 Key Assumptions

## Assumption 1
*   **Assumption:** The target kernel $k(x,y)$ must be continuous, positive definite, and symmetrically shift-invariant, such that $k(x,y) = k(x-y)$.
*   **Why the method needs it:** The proposed contribution of mapping data to random Fourier features completely relies on Bochner's theorem, which requires continuous positive definite functions to explicitly guarantee that their Fourier transform $p(\omega)$ evaluates to a proper integrable, non-negative probability measure that can be validly sampled from.
*   **Violation scenario:** Trying to use Random Fourier Features on non-stationary, non-shift-invariant sequence kernels like Dynamic Time Warping (DTW) or graph node kernels where $k(x,y) \\neq k(x-y)$. In such cases, there is no generic shift-invariant formula for generating proper Fourier directions $p(\omega)$ to draw variables from.
*   **Paper reference:** Section 3 (Random Fourier Features) and explicitly mentioned in "Theorem 1 (Bochner [13])".

## Assumption 2
*   **Assumption:** The underlying problem boundary or function is somewhat smooth, and does not require an overwhelmingly high randomized feature dimensionality $D$ relative to the number of samples $N$.
*   **Why the method needs it:** The method is built on the premise that $z(x)^Tz(y)$ approximates $k(x,y)$ reasonably bounds uniform convergence without needing an infinite $D$. If the problem highly fluctuates, capturing sufficient curvature to surpass exact models strictly requires driving $D$ extremely high until it overtakes $N$, rendering the method computationally irrelevant compared to kernel matrices.
*   **Violation scenario:** The dataset contains sharply irregular boundaries, such as densely grouped non-smooth clusters with extreme overlap (e.g. dense checkerboard patterns). Such a dataset necessitates tens of thousands of support vectors or unbounded dimensions $D$ forcing random features to perform worse or take longer than explicit SVM.
*   **Paper reference:** Claim 1 (Uniform convergence of Fourier features) detailing the requirement that the problem's geometric diameter and convergence bound depend on $D = \Omega(d^2 \log \\frac{1}{\epsilon})$. Section 5 references this on the Forest Cover dataset where RFF Fourier features fail under unsmooth boundaries.

## Assumption 3
*   **Assumption:** Expected inner products generated from the transformed Fourier features have low variation and concentrate fast towards the true continuous evaluation of $k(x,y)$.
*   **Why the method needs it:** Because the mapping uses an explicit sample average over $D$ finite features $z(x)^T z(y) = \\frac{1}{D} \sum_{j=1}^{D} z_{\omega_j}(x) z_{\omega_j}(y)$ to act as a stable stochastic estimator of the expected value of the kernel's Fourier domain. Hoeffding’s inequality secures this stability only if empirical deviations fall uniformly quickly as $D$ progresses.
*   **Violation scenario:** High dimensional manifolds or ill-scaled heterogeneous numeric variables where the second moment (curvature) scalar $\\sigma_p^2$ is extremely high (indicating severe volatility in the Fourier space). Concentration bounds become loose, meaning the inner products fluctuate highly randomly unless $D$ gets ridiculously scaled.
*   **Paper reference:** Section 3 discussion beneath Claim 1 relating Hoeffding’s inequality and the explicit curvature scalar $\\sigma_p^2$ defined as the trace of the Hessian of $k$ at $0$.
"""
nb['cells'] = [nbf.v4.new_markdown_cell(text_1_2)]
nbf.write(nb, 'partB/task_1_2.ipynb')

# Task 1.3
nb = nbf.v4.new_notebook()
text_1_3 = """# Task 1.3 What the Paper Claims to Improve

*   **Identified baseline method:** The main baseline compared against is exact Support Vector Machines (SVMs) and Support Vector Data-Description (SVDD) schemes that operate on dense kernel/Gram matrices. The Core Vector Machine (CVM) is also highlighted as a state-of-the-art fast SVM approach to specifically compare against.
*   **Limitation of baseline:** Exact kernel machines process data by computing operations across the entire Gram matrix of all pairs of datapoints, which scales poorly $O(N^3)$ processing and $O(N^2)$ storage. These decomposition methods simply bottleneck and drop efficiently when scaling to datasets encompassing millions of entries.
*   **Proposed overcoming method:** The authors circumvent implicitly lifting instances to infinite dimensions via the kernel trick kernel evaluations, and instead explicitly map data to a predefined low-dimensional $D$ feature representation utilizing $z(x)$. Since the inner product of vectors inside this mapped explicit space strictly approximates the explicit kernel matrix results, researchers can swap from computing expensive quadratic kernel machines $O(Nd)$ over to executing simple, massively scalable linear architectures like fast Ridge Regression $O(D+d)$.
*   **Underperforming condition:** The paper's random geometric mapping relies strictly on smooth approximation bounds across boundaries. Based on the reading, if a highly un-smooth problem requires memorizing local neighborhoods perfectly (e.g. requiring nearest neighbor properties), exact SVM scaling with Support Vectors will naturally retain locality boundaries, while Random Fourier Features will fail. One would either have to use explicit exact models with heavily localized kernel scopes or inject a randomized partitioning grid approach (Random Binning) rather than Fourier features. 
"""
nb['cells'] = [nbf.v4.new_markdown_cell(text_1_3)]
nbf.write(nb, 'partB/task_1_3.ipynb')

print("Question 1 Notebooks fully populated and successfully written.")
