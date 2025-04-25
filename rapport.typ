#import "template.typ": *
#import "theorems.typ": *
#show: thmrules

#show: ams-article.with(
  title: [Internship report, Attention growing networks],
  authors: (
    (
      name: "Léo Burgund",
      // department: [Department of Mathematics],
      // organization: [University of Exampleville],
      // location: [Tennessee, TN 59341],
      email: "leo.burgund@gmail.com",
      // url: "math.ue.edu/~jdoe"
    ),
  ),
  // abstract: lorem(100),
  bibliography: bibliography("refs.bib"),
)


= Nomenclature
== Dimensions
- $b$ Batch
- $d_e$ Embedding dimension
- $d_s$ Sequence length
- $d_k$ Query/Keys dimension
- $d_v$ Value dimension
- $h$ Number of heads

== Matrix
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
  nabla_Z f = 0 <==> X^top X Z X^top X = X^top B X.
$

In the case where $d_e <= d_s$ and $"rank"(X)=d_e$, then $X^top X$ is non-singular, and we have the solution

$
  Z^star = (X^top X)^(-1) X^top B X (X^top X)^(-1).
$

In the general case,
$
  Z^star = X^+ B (X^+)^top,
$
with $X^+ = (X^top X)^(-1) X^top$ the Moore-Penrose inverse.

If we had $d_k' >= d_e$, we could use the trivial factorization $circle(W)_Q=Z^star, circle(W)_K=I_d_e$.

As most of the time $d_k' < d_e$, we have to approximate the factorization.

// TODO Develop theorem
According to the Eckart–Young–Mirsky theorem, the best approximation $Z^star_k'$ of $X^+ B(X^+)^top$ with $"rank"(Z^star_k') = d_k'$ is obtained with a truncated SVD.

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
  circle(W)_Q^star=U_k' Sigma_k'^(1 / 2), space circle(W)_K^star= V_k' Sigma^(1 / 2).
$

#remark[
  $
    min_(circle(W_Q),circle(W_K)) norm(B-X circle(W_Q)circle(W_K)^top X^top)^2_F= sum_(i> d_k') sigma_i^2, space "subject to" "rank"(circle(W_Q)circle(W_K)^top) <= d_k'
  $
]

#remark[
  For implementation:

  Keep the matrices apart, for example for the weight matrix of $Q$ :
  $
    circle(W)_Q=W'_Q+diff W'_Q + W^("new")_Q
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
  Z &= X^+ B(X^+ )^top \
  &= X^+ (gradient_(S) cal(L)(S)+ X W_Q W_K^top X^top )(X^+ )^top \
$

// TODO FIXME CONTINUER Summary, PUIS CAS OU SE SIMPLIFIE, CHERCHER MEILLEUR TRUCS optimization info, PUIS FAIRE AVEC ESPERANCE, identifier pourquoi nécessitée de l'espérance, possibilité d'optimization info?

