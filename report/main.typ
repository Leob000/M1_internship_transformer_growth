#import "template_clean_math_paper.typ": *

#let date = datetime.today().display("[month repr:long] [day], [year]")
#show: template.with(
  title: "Internship report, Attention growing networks",
  authors: (
    (name: "Léo Burgund"),
    // (name: "Author 1", affiliation-id: 1, orcid: "0000-0000-0000-0000"),
  ),
  affiliations: (
    // (id: 1, name: "Affiliation 1, Address 1"),
    // (id: "*", name: "Corresponding author")
  ),
  date: date,
  heading-color: rgb("#000000"),
  link-color: rgb("#000082"),
  // abstract: [This is my abstract...],
  // keywords: ("First keyword", "Second keyword", "etc."),
  // AMS: ("65M70", "65M12"),
)

#let colMath(x, color) = text(fill: color)[$#x$]
#let ip(x, y) = $lr(angle.l #x, #y angle.r)$

= Nomenclature
== Dimensions
- $b$ Mini-batch size
- $d_e$ Embedding dimension
- $d_s$ Sequence length
- $d_k$ Query/Keys dimension
- $d_v$ Value dimension
- $h$ Number of heads

== Matrix operations in an attention block
We will first place ourselves in the case where $b=1$, we study only one instance.

In the case of multi head attention, for each head $i = 1,...,h$, we have:
- Input $X in RR^(d_s times d_e)$
- $W_Q_i in RR^(d_e times d_k / colMath(h,#red)), Q_i:= X W_Q_i in RR^(d_s times d_k / colMath(h,#red))$
- $W_K_i in RR^(d_e times d_k / colMath(h,#red)), K_i:= X W_K_i in RR^(d_s times d_k / colMath(h,#red))$
- $S_i := (Q_i K_i^top ) / sqrt(d_k/colMath(h,#red)) in RR^(d_s times d_s)$
- $A_i := "softmax"_"row" (S)$
- $W_V_i in RR^(d_e times d_v / colMath(h,#red)), V_i:= X W_V_i in RR^(d_s times d_v / colMath(h,#red))$
- $H_i := A_i V_i in RR^(d_s times d_v /colMath(h,#red))$, $H=[H_1,...,H_h] in RR^(d_s times d_v)$
- $W_O in RR^(d_v times d_e)$
- Output $Y:= H W_O + X in RR^(d_s times d_e)$

#remark[
  The number of parameters to learn
  $
    (underbrace(2(d_e (d_k)/h), W_Q_i\,W_K_i) + underbrace(d_e d_v/h ,W_V_i))h + underbrace(d_v d_e , W_O)
  $
  is the same for any $h in NN_+^*$.
]

#remark[
  We can easily consider the bias by augmenting the matrices:
  $
    X' = [X | bold(1)] in RR^(d_s times (d_e+1))\
    H' = [H | bold(1)] in RR^(d_s times (d_v + 1))
  $
  And adding a row of parameters to $W_Q_i, W_K_i, W_V_i, W_O$. For example:
  $
    W'_Q_i= vec(W_Q_i, (b^Q)^top ) in RR^((d_e + 1 ) times d_k / h).
  $
]

= Problem
We study the case where $h=1$.

We are interested in growing the $d_k$ dimension.
We consider the first order approximation, using the functional gradient,
$
  cal(L)(f + diff f(d theta, d cal(A)) )= cal(L) (f)+ ip(nabla_f cal(L)(f), diff f(diff theta, diff cal(A)))+o(norm(diff f(diff theta, diff cal(A)))).
$
// TODO trouver comment ref equations en typst

To avoid the softmax's non linearity, we will consider the gradient with respect to the matrix $S$, just before the softmax.
// TODO develop why

We then have
$
  cal(L)(S + diff S )= cal(L) (S)+ ip(nabla_S cal(L)(S), diff S)+o(norm(diff S))
$
with
$
  diff S= X(W_Q+diff W_Q)(W_K + diff W_K)^top X^top - X W_Q W_K^top X^top.
$

We have the following optimization problem:
$
  arg min_(diff S) ip(nabla_S cal(L)(S), diff S), "such that" norm(diff S) <= gamma
$

// TODO justify gamma
// TODO vérifier les signes
// TODO espace plus grand en typst après la virgule?


// TODO justifier passage de l'un à l'autre
$
  arg min_(diff W_Q, diff W_K) norm(B - X(W_Q+diff W_Q)(W_K+ diff W_K)^top X^top )^2_F \
  "with" B :=nabla_S cal(L)(S) + X W_Q W_K^top X^top
$
// TODO rescale par gamma pour la solution
// Precise Frobenius

Which is a low rank regression limited by $d_k$ (if $d_k< d_e$). $B$ is known.

// TODO parler des deux méthodes possibles
We can approximate $X underbrace((W_Q+diff W_Q), d_e times d_k)underbrace((W_K+ diff W_K)^top, d_k times d_e) X^top$ with a truncated SVD, taking the first $d_k$ singular values.

// TODO is it always?

If we want to grow the inner dimension of the attention matrix by $p$ neurons, we can instead approximate by taking the first $d_k':= d_(k)+p$ singular values.

Hence, instead of approximating a matrix $underbrace((W_Q+diff W_Q), d_e times d_k)underbrace((W_K+ diff W_K)^top, d_k times d_e)$, we approximate

$
  underbrace(Z, d_e times d_e) = underbrace(circle(W)_Q, d_e times (d_k')) underbrace(circle(W)_K^top, (d_k') times d_e) = [W_Q + diff W_Q | underbrace(tilde(W)_Q, d_e times p)][W_K + diff W_K | underbrace(tilde(W)_K, d_e times p)]^top
$
with $"rank"(Z) <= d_k'$ (we make the hypothesis that $d_k' < d_e$).

We then have the optimization problem

$
  arg min_(Z) norm(B - X Z X^top )^2_F space "subject to" "rank"(Z) <= d_k'.
$
Which is a low rank regression problem, limited by $d_k'$.

// TODO Considérer le scaling gamma
// TODO Trouver quelle est la solution, scaled avec gamma, faire la preuve; faire attention aux hypothèses sur les rangs, notamment d_k <= d_e ?

Let $f$ such that
$
  f(Z) = norm(B - X Z X^top )^2_F,
$
$f$ is convex.

We have
$
  nabla_Z f=-2 X^top (B-X Z X^top)X,
$
so
$
  nabla_Z f = 0 <==> X^top X Z^star X^top X = X^top B X.
$<optimZ>

In the case where $d_e <= d_s$ and $"rank"(X)=d_e$, then $X^top X$ is non-singular, and we have the solution

$
  Z^star = (X^top X)^(-1) X^top B X (X^top X)^(-1).
$

In the general case,
#set align(center)
#rect(inset: 5pt)[
  $
    Z^star = X^+ B (X^+)^top,
  $
]
#set align(left)
with $X^+$ the pseudoinverse (Moore-Penrose).

#proof[
  Suppose $Z^star  = X^+ B(X^+ ) ^top$ . Then,
  $
    X^top X Z^star X^top X &= X^top X X^+ B(X^+ )^top X^top X \
    &= X^top X X^+ B ( X^top X X^+ )^top \.
  $
  We have
  $
    X^top X X^+ &= X^top (X X^+ )^top & "by definition of the pseudoinverse"\
    &= X^top (X^+ )^top X^top \
    &= (X X^+ X )^top \
    &= X^top & "by definition" \.
  $
  Then
  $
    X^top X Z^star X^top X &= X^top B X \
  $
  we have verified equation @eq:optimZ.
]

== Factorization
We now have $Z^star$, which is equal to $circle(W)_Q circle(W) _K^top $, and want to factorize it to find $circle(W) _Q$ and $circle(W) _K$.

If we had $d_k' >= d_e$, we could use the trivial factorization $circle(W)_Q=Z^star, circle(W)_K=I_d_e$.

However, as most of the time $d_k' < d_e$, we have to approximate the factorization.

// TODO Develop theorem

According to the Eckart–Young–Mirsky theorem, the best approximations $breve(W) _Q$ and $breve( W) _K$ to get $breve(W) _Q breve(W) _K^top  approx Z^star$ with $"rank"(breve(W) _Q breve(W)_K^top ) = d_k'$ is obtained with a truncated SVD.

Indeed, we have
$
  Z^star = U Sigma V^top, space Sigma="diag"(sigma_1 >= sigma_2 >=...>= sigma_d_e).
$
We keep the $d_k'$ largest singular values
$
  U_k'=[u_1,...,u_d_k'], space V_k'=[v_1,...,v_d_k'], space Sigma_k'="diag"(sigma_1,...,sigma_d_k').
$
We get
$
  Z^star_k'=U_k' Sigma_k' V_k'^top,space "rank"(Z^star_k')=d_k'\
  breve(W)_Q=U_k' Sigma_k'^(1 / 2), space breve(W)_K= V_k' Sigma^(1 / 2)_k'.
$

#remark[
  $
    min_(breve(W)_Q,breve(W)_K) norm(B-X breve(W)_Q breve(W)_K^top X^top)^2_F= sum_(i> d_k') sigma_i^2, space "subject to" "rank"(breve(W)_Q breve(W)_K^top) <= d_k'
  $
]

#remark[
  For implementation:

  Keep the matrices apart, for example for the weight matrix of $Q$ :
  $
    breve(W)_Q=W'_Q+diff W'_Q + W^("new")_Q
  $
  with (remind that $d_k'=d_k +p$ )
  $
    limits(W'_Q)_(d_e times ( d_k+p )) =mat(delim: "[", augment:#3,
        w_1,...,w_k,bold(0)_1,...,bold(0)_p;
    ) \
    limits(diff W'_Q)_(d_e times ( d_k+p )) =mat(delim: "[", augment:#3,
        diff w_1,...,diff w_k,bold(0)_1,...,bold(0)_p;
    ) \
    limits(W^("new") _Q)_(d_e times ( d_k+p )) =mat(delim: "[", augment:#3,
        bold(0) _1,...,bold(0) _k,w^("new") _1,...,w^("new") _p;
    ) \
  $
  with any vector $w in RR^(d_e) $, and $bold(0) in RR^(d_e)$ the 0 vector.

  If we wanted to account for the bias, it's the same but include a new last row for each matrix, each vector has one more element.
]

== Summary
$
  Z &= X^+ (gradient_(S) cal(L)(S)+ X W_Q W_K^top X^top )(X^+ )^top \
$
and
$
  U_k' Sigma_k' V_k'^top = "SVD"_("trunc" k') (Z)
$
$
  breve(W)_Q=U_k' Sigma_k'^(1 / 2), space breve(W)_K= V_k' Sigma^(1 / 2)_k'.
$

== Notes on Computing
=== Mini-batch
Note: The "Mini-batch size" can refer either to the machine batch size taken in by the GPU which can optimize computations, or the statistical batch size used to estimate a statistic (this is important as a machine batch may not be of size large enough to get a good estimation of a statistic). In this section, the mini-batch will refer to the statistical batch.
// TODO verif rapport théo

Let $b$ be the mini-batch size, and $i in {1,...,b}$.

As $Z$ depends on $gradient_(S) cal(L)(S) $, the "quality" of the new weight matrices is dependant on $b$.
// TODO develop why it is dependent on the mini-batch size


To account for the batch, we identified two possibilities:

// TODO rewrite au propre ça
1. For each instance, calculate $Z_(i) $, get the empirical mean $dash(Z)_b$ then do $"SVD"(dash(Z)_b)$ to find $breve(W) _Q,breve(W)_K$.
$
  dash(Z)_(b) =EE_X [Z_(i) ] \
  U_k' Sigma_k'V_k'^top = "SVD"_("trunc" k') (dash(Z)_(b) ) \
  breve(W)_Q=U_k' Sigma_k'^(1 / 2), space breve(W)_K= V_k' Sigma^(1 / 2)_k'.
$
We do one SVD per mini-batch.

2. For each instance, calculate $Z_(i) $, do $"SVD"(Z_i) $ to get $breve(W) _(Q,i),breve(W) _(K,i)  $, then get the empirical means $dash(breve(W) ) _Q, dash(breve(W) ) _K$.
$
  U_(k',i) Sigma_( k',i)V_( k',i)^top = "SVD"_("trunc" k') (Z_(i) ) \
  breve(W)_(Q,i) = U_(k',i) Sigma^(1 / 2)_(k',i) , space breve(W)_(K,i) =V_(k',i) Sigma^(1 / 2)_(k',i)\
  breve(W)_Q = dash(breve(W) )_Q = EE_X [breve(W)_(Q,i) ], space breve(W)_K = dash(breve(W) )_K = EE_X [breve(W)_(K,i) ].
$
Here, we do one SVD for each instance.

Note: This is not counting the SVD we will have to do to find $X^+ $.

// TODO Which method to choose?
We choose the first method as it requires only one SVD, which may require less computational ressources, and may truncate (through the SVD) less valuable "information" away, as the SVD is applied after the mean.

=== Computing $Z_i$
We denote different ways to compute $Z_(i) $.

$
  Z = X^+ (gradient_(S) cal(L)(S)+ X W_Q W_K^top X^top )(X^+ )^top
$<Z1>
$
  Z= X^+ gradient_(S) cal(L)(S) (X^+ )^top +X^+ X W_Q W_K^top X^+ X
$<Z2>

$Z_(i) $ can either be computed by using @eq:Z1, or @eq:Z2, which can be further decomposed.

- If $"rank"(X) = d_e$ :
$
  X^+ =(X^top X)^(-1) X^top => X^+ X=I_d_e
$
$
  Z= X^+ gradient_(S) cal(L)(S) (X^+ )^top + W_Q W_K^top
$
- If $"rank"(X)=d_s $ :
$
  X^+ =X^top (X X^top )^(-1) => X^+ X=X^top (X^top X)^(-1) X
$
$
  Z= X^+ gradient_(S) cal(L)(S) (X^+ )^top + X^top (X X^top )^(-1) X W_Q W_K^top X^top (X X^top )^(-1) X
$
- In the general case , with $r="rank"(X) $ :
$
  Z= X^+ gradient_(S) cal(L)(S) (X^+ )^top +
  V_X mat(
    underbrace(I_r, r times r) , 0;0,underbrace(0, (e-r)times(e-r) )
) V_X^top
  W_Q W_K^top
  V_X mat(
    underbrace(I_r, r times r) , 0;0,underbrace(0, (e-r)times(e-r) )
) V_X^top .
$
#proof[
  We compute the SVD for $X$,
  $
    underbrace(X, d_s times d_e) =underbrace(U_X, d_s times d_s) underbrace(Sigma_X, d_s times d_e) underbrace(V_X^top, d_e times d_e) , space X^+ =V_X underbrace(Sigma^(+) _X, d_e times d_s) U_X^top
  $
  $
    X^+ X=V_X Sigma^(+)_X U_X^top U_X Sigma_X V_X^top = V_X Sigma^(+)_X Sigma_X V_X^top
  $
  with
  $
    Sigma^(+)_X Sigma_X= mat(
        underbrace(Sigma^(-1) _r, r times r) ,0;0,underbrace(0, (e-r)times(s-r)  )
    )
    mat(
        underbrace(Sigma_r, r times r) , 0;0, underbrace(0, (s-r) times(e-r) )
    ) =
    mat(
        underbrace(I_r, r times r),0;0, underbrace(0, (e-r) times(e-r) )
    )
  $
]

// TODO Voir steph si c'est utile ou non


// $
//   alpha=1
// $<refeq>
// @eq:refeq
//
// #theorem(title: "Example Theorem")[
//   this is a thm
// ]<refthm>
// This is its ref @refthm.
//
// To get a bibliography, we also add a citation @Cooley65.
//
// #bibliography("bibliography.bib")
//
// // Create appendix section
// #show: appendices
// = test
// #lorem(10)
